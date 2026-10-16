#!/usr/bin/env python3
"""Run ST-MoE on the reference experiment's 36 paired test inputs."""
import argparse
import collections
import csv
import json
import pathlib
import re

import moe_stmoe as m

REFERENCE = pathlib.Path(r"D:\another_project\llama-stream-moe-main\experiments\moe-markov")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reference_inputs(reference):
    ref = m.read_json(reference / "infinity-evaluation-108-manifest.json")
    source = reference / "samples.json"
    if (ref["slots"], ref["tokens"], ref["repetitions"]) != (180, 256, 1) or m.digest(source) != ref["samples_sha256"]:
        raise ValueError("reference settings or sample hash mismatch")
    samples = {r["id"]: r for r in m.read_json(source)}
    expected = [f"{dataset}-test-{i:04d}" for dataset in ("gsm8k", "arc") for i in range(18)]
    if ref["sample_ids"] != expected:
        raise ValueError("reference must contain exactly the first 18 inputs per dataset")
    selected = [samples[sid] for sid in expected]
    if collections.Counter(r["dataset"] for r in selected) != {"gsm8k": 18, "arc": 18}:
        raise ValueError("unexpected reference dataset mix")
    return ref, samples, selected


def prepare(args):
    reference = args.reference.resolve()
    ref, samples, selected = reference_inputs(reference)
    metadata = m.read_json(reference / "markov.json")
    model_hash = m.digest(args.model)
    if model_hash != ref["model_sha256"] or model_hash != metadata["model_sha256"]:
        raise ValueError("reference GGUF SHA-256 mismatch")
    meta = m.layout(args.model)
    if meta["layer_count"] != metadata["layers"] or meta["experts"] != metadata["experts"]:
        raise ValueError("reference/model dimensions mismatch")
    env = m.environment(args, model_hash)
    args.root.mkdir(parents=True, exist_ok=True)
    trace_paths, calibration, evidence = [], [], []
    for stem in metadata["calibration_samples"]:
        status_path = reference / "calibration" / (stem + ".status.json")
        trace_path = reference / "calibration" / (stem + ".jsonl")
        status = m.read_json(status_path)
        row = samples[status["sample"]]
        command = status["command"]
        prompt_path = pathlib.Path(command[command.index("-f") + 1])
        if status["returncode"] or row["split"] != "train" or status["policy"] != "hotness":
            raise ValueError("invalid calibration run")
        if prompt_path.read_text(encoding="utf-8") != row["prompt"] or command[command.index("-n") + 1] != "256":
            raise ValueError("calibration prompt/settings mismatch")
        if m.digest(trace_path) != metadata["traces"][trace_path.name]:
            raise ValueError("reference calibration trace changed")
        trace_paths.append(trace_path)
        calibration.append(m.identity(row))
        evidence.append(dict(sample=m.identity(row), trace_sha256=metadata["traces"][trace_path.name],
                             status_sha256=m.digest(status_path)))
    if len(calibration) != 200 or collections.Counter(r["dataset"] for r in calibration) != {"gsm8k": 100, "arc": 100}:
        raise ValueError("expected 100 independent calibration inputs per dataset")
    m.ensure_disjoint(calibration, [m.identity(r) for r in selected])
    provenance = dict(reference=str(reference), reference_manifest_sha256=m.digest(reference / "infinity-evaluation-108-manifest.json"),
                      calibration_metadata_sha256=m.digest(reference / "markov.json"),
                      calibration_binaries=m.read_json(reference / "calibration-build.json"),
                      wrapper_sha256=m.digest(__file__), evidence=evidence,
                      execution_overrides=["warmup disabled", "debug logging for sampled token IDs",
                                           "ST-MoE uses CPU remap and validates complete decode observations"])
    m.pin(args.root / "reference-manifest.json", dict(version=1, environment=env, provenance=provenance,
          samples=[m.identity(r) for r in selected], calibration_samples=calibration))
    cct_path = args.root / "cct.json"
    if cct_path.exists():
        cct = m.read_json(cct_path)
        if cct["environment"] != env or cct["calibration_source"] != provenance:
            raise ValueError("existing CCT provenance mismatch")
    else:
        print("Building spatial CCT from 200 verified reference training traces", flush=True)
        pairs, tokens = m.build_cct(meta, trace_paths)
        m.save(cct_path, dict(version=1, model_sha256=model_hash, layer_count=meta["layer_count"],
               k=meta["k"], layers=meta["layers"], pairs=pairs, calibration_tokens=tokens,
               calibration_samples=calibration, environment=env, calibration_source=provenance,
               trace_sha256={p.name: metadata["traces"][p.name] for p in trace_paths}))
    m.pin(args.root / "evaluation-manifest.json", dict(version=1, environment=env,
          cct_sha256=m.digest(cct_path), samples=[m.identity(r) for r in selected]))
    return selected, env


def compare_pair(root, sid):
    paths = [m.artifacts(root / "evaluation" / (sid + "-" + mode)) for mode in ("baseline", "stmoe")]
    if paths[0]["stdout"].read_bytes() != paths[1]["stdout"].read_bytes():
        raise ValueError(f"output mismatch: {sid}")
    if list(m.route_rows(paths[0]["trace"])) != list(m.route_rows(paths[1]["trace"])):
        raise ValueError(f"routing mismatch: {sid}")
    tokens = [re.findall(r"n_decoded = (\d+), n_remaining = -?\d+, next token:\s+(\d+)",
                        p["stderr"].read_text(encoding="utf-8", errors="replace")) for p in paths]
    if not tokens[0] or tokens[0] != tokens[1]:
        raise ValueError(f"sampled token mismatch: {sid}")


def report(args):
    m.report(args)
    root = args.root
    summary = m.read_json(root / "summary.json")
    manifest = m.read_json(root / "evaluation-manifest.json")
    rows, observations = [], []
    for sample in manifest["samples"]:
        paired = []
        for mode in ("baseline", "stmoe"):
            stem = root / "evaluation" / (sample["id"] + "-" + mode)
            paths = m.artifacts(stem)
            log = paths["stderr"].read_text(encoding="utf-8", errors="replace")
            counts = re.findall(r"moe_cache phase=decode policy=hotness hit=(\d+) miss=(\d+) loading=(\d+) cold=(\d+) all_hit=(\d+) tokens=(\d+)", log)
            loads = re.findall(r"moe_cache loads=(\d+) bytes=(\d+) rule_us=(\d+) stall_us=(\d+)", log)
            perf = re.findall(r"moe_cache generated=(\d+) decode_tps=([\d.]+) prompt_tps=([\d.]+)", log)
            if not counts or not loads or not perf:
                raise ValueError("missing runtime cache/performance counters")
            hit, miss, loading, cold, all_hit, tokens = map(int, counts[-1])
            nloads, nbytes, rule, stall = map(int, loads[-1])
            generated, decode_tps, prompt_tps = perf[-1]
            row = dict(sample=sample["id"], dataset=sample["dataset"], mode=mode,
                       cache_hit=hit, cache_miss=miss, cache_hit_rate=hit/(hit+miss),
                       cache_loading=loading, cache_cold=cold, cache_all_hit=all_hit, cache_observations=tokens,
                       loads=nloads, bytes=nbytes, rule_us=rule, stall_us=stall, generated=int(generated),
                       decode_tps=float(decode_tps), prompt_tps=float(prompt_tps),
                       wall_s=m.read_json(str(stem) + ".status.json")["wall_s"])
            paired.append(row)
            rows.append(row)
        invariant = ("cache_hit", "cache_miss", "cache_loading", "cache_cold", "cache_all_hit", "cache_observations", "loads", "bytes", "generated")
        if any(paired[0][key] != paired[1][key] for key in invariant):
            raise ValueError(f"cache/load behavior changed: {sample['id']}")
        events = []
        for mode in ("baseline", "stmoe"):
            trace = root / "evaluation" / (sample["id"] + "-" + mode + ".jsonl")
            with trace.open(encoding="utf-8") as f:
                events.append([r for line in f if (r := json.loads(line)).get("event") == "replace"])
        if events[0] != events[1]:
            raise ValueError("cache replacement events changed")
        trace = root / "evaluation" / (sample["id"] + "-stmoe.jsonl")
        with trace.open(encoding="utf-8") as f:
            for line in f:
                event = json.loads(line)
                if event.get("event") == "stmoe_evaluate":
                    observations.append(dict(sample=sample["id"], dataset=sample["dataset"], target=event["target"],
                         predicted=len(event["predicted"]), actual=len(event["actual"]), hit=len(event["intersection"]),
                         all_hit=int(len(event["intersection"]) == len(event["actual"])), observations=1))
    summary["reference"] = m.read_json(root / "reference-manifest.json")
    summary["cache_and_loads_equal"] = True
    summary["replacement_events_equal"] = True
    summary["inference_runs"] = len(rows)
    summary["inference"] = rows
    m.save(root / "summary.json", summary)
    write_csv(root / "samples.csv", summary["runs"])
    write_csv(root / "inference.csv", rows)
    grouped = collections.defaultdict(collections.Counter)
    for row in observations:
        grouped[row["dataset"], row["target"]].update({k: row[k] for k in m.COUNTS})
    write_csv(root / "dataset_layers.csv", [dict(dataset=d, target=l, **m.metrics(c)) for (d, l), c in sorted(grouped.items())])
    lines = ["# ST-MoE: GSM8K and ARC, 18 inputs each", "",
             "36 paired inputs, 72 inference runs, one repetition; up to 256 generated tokens per run.",
             "CCT initialization uses 200 independent reference training traces (100 per dataset).", "",
             "| Dataset | Coverage | Precision | All-hit rate | Mean predicted experts |",
             "|---|---:|---:|---:|---:|"]
    for name, row in [("total", summary["total"]), *summary["datasets"].items()]:
        lines.append(f"| {name} | {row['coverage']:.4%} | {row['precision']:.4%} | {row['all_hit_rate']:.4%} | {row['mean_predicted']:.3f} |")
    lines += ["", "| Dataset | Mode | Cache hit rate | Mean decode token/s | Mean process seconds |",
              "|---|---|---:|---:|---:|"]
    for dataset in ("gsm8k", "arc"):
        for mode in ("baseline", "stmoe"):
            selected = [r for r in rows if r["dataset"] == dataset and r["mode"] == mode]
            hit = sum(r["cache_hit"] for r in selected)
            miss = sum(r["cache_miss"] for r in selected)
            lines.append(f"| {dataset} | {mode} | {hit/(hit+miss):.4%} | {sum(r['decode_tps'] for r in selected)/len(selected):.3f} | {sum(r['wall_s'] for r in selected)/len(selected):.3f} |")
    lines += ["", "Generated token IDs, text, actual routes, online prediction replay, cache counters,",
              "replacement events, demand loads and transferred bytes were verified for every pair.",
              "This measures prediction accuracy; no real prefetch is performed. GPU expert computation remains enabled.",
              "Timing includes CPU observation, synchronization, debug logging and model hashing; it is not a prefetch speedup measurement.",
              "The OS page cache is not cleared. Calibration traces were produced by the reference binary; provenance and hashes are saved.",
              "Per-sample prediction metrics: samples.csv; per-run cache/timing metrics: inference.csv; per-dataset layers: dataset_layers.csv.", ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("evaluate", "report"))
    parser.add_argument("--reference", type=pathlib.Path, default=REFERENCE)
    parser.add_argument("--root", type=pathlib.Path, default=m.REPO / "experiments/moe-stmoe-reference-36")
    parser.add_argument("--model", type=pathlib.Path, default=m.MODEL)
    parser.add_argument("--exe", type=pathlib.Path, default=m.REPO / "build/bin/Release/llama-cli.exe")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.slots, args.tokens, args.gpu_layers = 180, 256, 999
    if args.action == "report":
        report(args)
        return
    selected, env = prepare(args)
    for i, sample in enumerate(selected):
        for mode in ("baseline", "stmoe") if i % 2 == 0 else ("stmoe", "baseline"):
            m.run_one(args, sample, args.root / "evaluation", mode, env,
                      args.root / "cct.json" if mode == "stmoe" else None)
        compare_pair(args.root, sample["id"])
        print(f"Verified pair {i+1}/36: {sample['id']}", flush=True)
    report(args)


if __name__ == "__main__":
    main()
