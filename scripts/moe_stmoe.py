#!/usr/bin/env python3
"""Private ST-MoE 4.2 prediction-only calibration, paired evaluation and replay."""
import argparse
import collections
import copy
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
MODEL = pathlib.Path(r"D:\llama-cpp\models\Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf")
COUNTS = ("predicted", "actual", "hit", "all_hit", "observations")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prompt_digest(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def save(path, value):
    pathlib.Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def pin(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"manifest mismatch; choose a new --root: {path}")
    else:
        save(path, value)


def layout(model):
    sys.path.insert(0, str(REPO / "gguf-py"))
    from gguf import GGUFReader
    reader = GGUFReader(model, "r")
    arch = reader.fields["general.architecture"].contents()
    nl = int(reader.fields[arch + ".block_count"].contents())
    ne = int(reader.fields[arch + ".expert_count"].contents())
    k = int(reader.fields[arch + ".expert_used_count"].contents())
    nextn = reader.fields.get(arch + ".nextn_predict_layers")
    active_layers = nl - (int(nextn.contents()) if nextn else 0)
    layers = sorted({int(match[1]) for t in reader.tensors
                     if (match := re.fullmatch(r"blk\.(\d+)\.ffn_(?:gate|up|down|gate_up)_exps\.weight", t.name))
                     and int(match[1]) < active_layers})
    if len(layers) < 2 or not 0 < k <= ne:
        raise ValueError("requires at least two MoE layers and valid expert counts")
    return dict(layer_count=nl, experts=ne, k=k, layers=layers)


def samples(path, split, limit):
    rows = read_json(path)
    if not isinstance(rows, list) or limit < 1:
        raise ValueError("samples must be an array; per-dataset must be positive")
    seen, prompts, counts, selected = set(), set(), collections.Counter(), []
    for row in rows:
        sid = row["id"]
        if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", sid) or sid in seen:
            raise ValueError("duplicate/unsafe sample ID")
        seen.add(sid)
        if row["split"] != split or counts[row["dataset"]] >= limit:
            continue
        ph = prompt_digest(row["prompt"])
        if ph in prompts:
            raise ValueError("duplicate selected prompt")
        prompts.add(ph)
        counts[row["dataset"]] += 1
        selected.append(row)
    if not selected:
        raise ValueError(f"no {split} samples")
    return selected


def identity(row):
    return dict(id=row["id"], dataset=row["dataset"], prompt_sha256=prompt_digest(row["prompt"]))


def ensure_disjoint(calibration, evaluation):
    ids = {r["id"] for r in calibration}
    prompts = {r["prompt_sha256"] for r in calibration}
    if any(r["id"] in ids or r["prompt_sha256"] in prompts for r in evaluation):
        raise ValueError("calibration and evaluation samples overlap")


def environment(args, model_hash):
    exe = args.exe.resolve()
    return dict(model_sha256=model_hash, binaries={p.name: digest(p) for p in
                [exe, *sorted(exe.parent.glob("*.dll"))]}, script_sha256=digest(__file__),
                slots=args.slots, tokens=args.tokens, gpu_layers=args.gpu_layers,
                batch=2048, ubatch=512, seed=0, temperature=0, log_verbosity=5)


def route_rows(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "event" not in row:
                yield row


def decode_tokens(path, meta):
    layers, k, ne = meta["layers"], meta["k"], meta["experts"]
    current, previous_steps = {}, {}
    for row in route_rows(path):
        if row["phase"] != "decode":
            continue
        il, ids = row["layer"], row["ids"]
        if il != layers[len(current)] or len(ids) != k or len(set(ids)) != k or any(
                type(e) is not int or not 0 <= e < ne for e in ids):
            raise ValueError(f"incomplete/out-of-order layers or invalid experts: {path}")
        if il in previous_steps and row["step"] != previous_steps[il] + 1:
            raise ValueError("non-consecutive layer trace steps")
        previous_steps[il] = row["step"]
        current[il] = sorted(ids)
        if len(current) == len(layers):
            yield current
            current = {}
    if current:
        raise ValueError(f"incomplete final decode token: {path}")


def build_cct(meta, traces):
    layers, ne, k = meta["layers"], meta["experts"], meta["k"]
    counts = [[collections.Counter() for _ in range(ne)] for _ in layers[1:]]
    observed = 0
    for path in traces:
        count = 0
        for token in decode_tokens(path, meta):
            count += 1
            for i, (source, target) in enumerate(zip(layers, layers[1:])):
                for e in token[source]:
                    counts[i][e].update(token[target])
        if not count:
            raise ValueError(f"no decode calibration tokens: {path}")
        observed += count
    pairs = []
    for i, (source, target) in enumerate(zip(layers, layers[1:])):
        rows = []
        for row in counts[i]:
            top = sorted(range(ne), key=lambda e: (-row[e], e))[:k] if row else []
            rows.append([[e, 2] for e in top])
        pairs.append(dict(source=source, target=target, experts=ne, rows=rows))
    return pairs, observed


def predict(rows, source, history):
    score = collections.Counter()
    for e in source:
        for f, confidence in rows[e]:
            score[f] += confidence
    for e in history:
        score[e] += 2
    return sorted(e for e, confidence in score.items() if confidence >= 2)


def update(rows, source, actual):
    actual = sorted(actual)
    for e in source:
        row = rows[e]
        for entry in row:
            if entry[0] in actual:
                entry[1] = min(3, entry[1] + 1)
            elif entry[1]:
                entry[1] -= 1
            else:
                replacement = next((f for f in actual if f not in {x[0] for x in row}), None)
                if replacement is not None:
                    entry[:] = [replacement, 2]


def metrics(counts):
    p, a, h, all_hit, observations = (counts.get(k, 0) for k in COUNTS)
    return dict(predicted=p, actual=a, hit=h, miss=a-h, false_positive=p-h,
                all_hit=all_hit, observations=observations, coverage=h/a if a else None,
                precision=h/p if p else None, all_hit_rate=all_hit/observations if observations else None,
                mean_predicted=p/observations if observations else None)


def replay(path, cct):
    tables = copy.deepcopy(cct["pairs"])
    history, totals, expected = {}, collections.defaultdict(collections.Counter), []
    meta = dict(layers=cct["layers"], k=cct["k"], experts=tables[0]["experts"])
    tokens = 0
    for step, token in enumerate(decode_tokens(path, meta)):
        tokens += 1
        for table in tables:
            source, target, rows = table["source"], table["target"], table["rows"]
            prediction = predict(rows, token[source], history.get(target, []))
            intersection = sorted(set(prediction) & set(token[target]))
            expected.append(dict(event="stmoe_predict", source=source, target=target,
                                 step=step, predicted=prediction))
            expected.append(dict(event="stmoe_evaluate", source=source, target=target,
                                 step=step, predicted=prediction, actual=token[target], intersection=intersection))
            totals[target].update(predicted=len(prediction), actual=len(token[target]), hit=len(intersection),
                                  all_hit=int(len(intersection) == len(token[target])), observations=1)
            update(rows, token[source], token[target])
        history = token
    with open(path, encoding="utf-8") as f:
        actual_events = [r for line in f if (r := json.loads(line)).get("event", "").startswith("stmoe_")]
    if not tokens or actual_events != expected:
        raise ValueError(f"online prediction/evaluation differs from independent replay: {path}")
    return totals


def artifacts(stem):
    return {kind: pathlib.Path(str(stem) + suffix) for kind, suffix in
            (("trace", ".jsonl"), ("stdout", ".stdout.txt"), ("stderr", ".stderr.txt"))}


def run_one(args, sample, folder, mode, env, cct_path=None):
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / (sample["id"] + "-" + mode)
    prompt = pathlib.Path(str(stem) + ".prompt.txt")
    paths = artifacts(stem)
    command = [str(args.exe.resolve()), "-m", str(args.model.resolve()), "-c", "8192",
               "-ngl", str(args.gpu_layers), "--reasoning", "off", "--no-warmup",
               "--moe-stream-cache", f"{args.slots}s", "--moe-stream-io-threads", "6",
               "--temp", "0", "--seed", "0", "--no-mmap", "-b", "2048", "-ub", "512",
               "--log-verbosity", "5",
               "-n", str(args.tokens), "--single-turn", "--no-display-prompt", "--no-show-timings",
               "-f", str(prompt.resolve()), "--moe-stream-policy", "hotness",
               "--moe-stream-trace", str(paths["trace"].resolve())]
    if cct_path:
        command += ["--moe-stream-stmoe-cct", str(cct_path.resolve())]
    status_path = pathlib.Path(str(stem) + ".status.json")
    spec = dict(sample=identity(sample), mode=mode, command=command, environment=env,
                cct_sha256=digest(cct_path) if cct_path else None)
    if status_path.exists():
        status = read_json(status_path)
        if status["spec"] != spec:
            raise ValueError(f"run identity mismatch: {stem}")
        if status["returncode"] == 0:
            if status["artifacts"] != {k: digest(p) for k, p in paths.items()}:
                raise ValueError(f"run artifact mismatch: {stem}")
            return stem
    prompt.write_text(sample["prompt"], encoding="utf-8")
    child_env = dict(os.environ)
    for name in list(child_env):
        if name.startswith(("LLAMA_ARG_", "LLAMA_MOE_STREAM_")):
            child_env.pop(name)
    start = time.monotonic()
    print("running", stem.name, flush=True)
    with paths["stdout"].open("wb") as out, paths["stderr"].open("wb") as err:
        process = subprocess.run(command, stdout=out, stderr=err, env=child_env, timeout=600)
    save(status_path, dict(spec=spec, returncode=process.returncode, wall_s=time.monotonic()-start,
                           artifacts={k: digest(p) for k, p in paths.items() if p.exists()}))
    if process.returncode:
        raise RuntimeError(f"inference failed: {paths['stderr']}")
    return stem


def calibrate(args):
    selected = samples(args.samples, "train", args.per_dataset)
    meta = layout(args.model)
    env = environment(args, digest(args.model))
    args.root.mkdir(parents=True, exist_ok=True)
    manifest = dict(version=1, environment=env, samples=[identity(r) for r in selected], layout=meta)
    pin(args.root / "calibration-manifest.json", manifest)
    traces = [artifacts(run_one(args, row, args.root / "calibration", "calibration", env))["trace"]
              for row in selected]
    pairs, tokens = build_cct(meta, traces)
    cct = dict(version=1, model_sha256=env["model_sha256"], layer_count=meta["layer_count"],
               k=meta["k"], layers=meta["layers"], pairs=pairs, calibration_tokens=tokens,
               calibration_samples=manifest["samples"], environment=env,
               trace_sha256={p.name: digest(p) for p in traces})
    pin(args.root / "cct.json", cct)
    print("calibrated", tokens, "decode tokens;", len(pairs), "layer pairs", flush=True)


def evaluate(args):
    cct_path = args.root / "cct.json"
    cct = read_json(cct_path)
    selected = samples(args.samples, "test", args.per_dataset)
    ensure_disjoint(cct["calibration_samples"], [identity(r) for r in selected])
    env = environment(args, digest(args.model))
    if env != cct["environment"]:
        raise ValueError("calibration/evaluation environment mismatch; recalibrate in a new --root")
    manifest = dict(version=1, environment=env, cct_sha256=digest(cct_path),
                    samples=[identity(r) for r in selected])
    pin(args.root / "evaluation-manifest.json", manifest)
    for row in selected:
        for mode in ("baseline", "stmoe"):
            run_one(args, row, args.root / "evaluation", mode, env,
                    cct_path if mode == "stmoe" else None)
    report(args)


def report(args):
    cct_path = args.root / "cct.json"
    cct = read_json(cct_path)
    manifest = read_json(args.root / "evaluation-manifest.json")
    if manifest["cct_sha256"] != digest(cct_path) or manifest["environment"] != cct["environment"]:
        raise ValueError("evaluation/calibration identity mismatch")
    ensure_disjoint(cct["calibration_samples"], manifest["samples"])
    total = collections.Counter()
    datasets, layers, rows = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter), []
    for sample in manifest["samples"]:
        stems = [args.root / "evaluation" / (sample["id"] + "-" + mode) for mode in ("baseline", "stmoe")]
        paths = [artifacts(stem) for stem in stems]
        for mode, stem, files in zip(("baseline", "stmoe"), stems, paths):
            status = read_json(str(stem) + ".status.json")
            spec = status["spec"]
            if status["returncode"] or spec["environment"] != manifest["environment"] or spec["sample"] != sample or (
                    spec["mode"] != mode or spec["cct_sha256"] != (manifest["cct_sha256"] if mode == "stmoe" else None)):
                raise ValueError("invalid run provenance")
            if status["artifacts"] != {k: digest(p) for k, p in files.items()}:
                raise ValueError("modified run artifacts")
        if paths[0]["stdout"].read_bytes() != paths[1]["stdout"].read_bytes():
            raise ValueError(f"generated output mismatch: {sample['id']}")
        logs = [p["stderr"].read_text(encoding="utf-8", errors="replace") for p in paths]
        token_ids = [re.findall(r"n_decoded = (\d+), n_remaining = -?\d+, next token:\s+(\d+)", log)
                     for log in logs]
        if not token_ids[0] or token_ids[0] != token_ids[1]:
            raise ValueError(f"generated token ID mismatch/missing tokens: {sample['id']}")
        for log, ids in zip(logs, token_ids):
            generated = re.findall(r"moe_cache generated=(\d+)", log)
            if not generated or len(ids) != int(generated[-1]) or [int(i) for i, _ in ids] != list(range(1, len(ids)+1)):
                raise ValueError("incomplete generated token ID log")
        if list(route_rows(paths[0]["trace"])) != list(route_rows(paths[1]["trace"])):
            raise ValueError(f"actual routing mismatch: {sample['id']}")
        per_layer = replay(paths[1]["trace"], cct)
        log = logs[1]
        summaries = {r["target"]: r for text in re.findall(r"stmoe_summary (\{[^\r\n]+\})", log)
                     if (r := json.loads(text))}
        sample_counts = collections.Counter()
        for layer, counts in per_layer.items():
            expected = dict(target=layer, **metrics(counts))
            if summaries.get(layer) != expected:
                raise ValueError(f"runtime summary/replay mismatch: {sample['id']} layer {layer}")
            layers[layer].update(counts)
            sample_counts.update(counts)
        if summaries.get(-1) != dict(target=-1, **metrics(sample_counts)):
            raise ValueError("runtime aggregate/replay mismatch")
        total.update(sample_counts)
        datasets[sample["dataset"]].update(sample_counts)
        rows.append(dict(sample=sample["id"], dataset=sample["dataset"], generated_tokens=len(token_ids[1]),
                         **metrics(sample_counts)))
    result = dict(version=1, model_sha256=cct["model_sha256"], cct_sha256=manifest["cct_sha256"],
                  samples=len(rows), calibration_tokens=cct["calibration_tokens"],
                  generated_output_equal=True, generated_token_ids_equal=True,
                  routing_equal=True, online_replay_equal=True,
                  total=metrics(total), datasets={k: metrics(v) for k, v in datasets.items()},
                  layers={str(k): metrics(v) for k, v in layers.items()}, runs=rows)
    save(args.root / "summary.json", result)
    lines = ["# ST-MoE prediction-only smoke report", "",
             f"Calibration decode tokens: {cct['calibration_tokens']}; test samples: {len(rows)}.", "",
             "Generated token IDs, outputs and actual routes match the baseline. Online predictions and counters match replay.", "",
             "| Group | Coverage | Precision | All-hit rate | Mean predicted | Observations |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, value in [("total", result["total"]), *result["datasets"].items()]:
        fmt = lambda x: "null" if x is None else f"{x:.4f}"
        lines.append(f"| {name} | {fmt(value['coverage'])} | {fmt(value['precision'])} | "
                     f"{fmt(value['all_hit_rate'])} | {fmt(value['mean_predicted'])} | {value['observations']} |")
    lines += ["", "Counts and per-layer results are in summary.json. This small run validates implementation;",
              "it does not establish general prediction accuracy or prefetch performance.",
              "No predicted expert loads are issued. The OS page cache is not cleared.", ""]
    (args.root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result["total"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("calibrate", "evaluate", "report"))
    parser.add_argument("--root", type=pathlib.Path, default=REPO / "experiments/moe-stmoe")
    parser.add_argument("--model", type=pathlib.Path, default=MODEL)
    parser.add_argument("--exe", type=pathlib.Path, default=REPO / "build/bin/Release/llama-cli.exe")
    parser.add_argument("--samples", type=pathlib.Path, default=REPO / "experiments/moe-fate-v2/samples.json")
    parser.add_argument("--per-dataset", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--slots", type=int, default=180)
    parser.add_argument("--gpu-layers", type=int, default=999)
    args = parser.parse_args()
    if args.tokens < 2 or args.slots < 1:
        parser.error("tokens must be at least 2 and slots must be positive")
    {"calibrate": calibrate, "evaluate": evaluate, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
