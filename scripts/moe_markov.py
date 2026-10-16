#!/usr/bin/env python3
"""Private Markov cache experiment. Python 3 and numpy; downloads use stdlib."""
import argparse
import collections
import csv
import hashlib
import json
import pathlib
import re
import struct
import subprocess
import time
import urllib.parse
import urllib.request
import sys
import functools

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments/moe-markov/python-deps"))

import numpy as np

MODEL = r"D:\llama-cpp\models\Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf"
DATASETS = (("gsm8k", "openai/gsm8k", "main"),
            ("arc", "allenai/ai2_arc", "ARC-Challenge"))


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def prepare(root):
    samples, sources = [], []
    for short, repo, config in DATASETS:
        revision = fetch("https://huggingface.co/api/datasets/" + repo)["sha"]
        for split, count in (("train", 100), ("test", 50)):
            export_revision = fetch("https://huggingface.co/api/datasets/" + repo + "/revision/refs%2Fconvert%2Fparquet")["sha"]
            info = fetch("https://huggingface.co/api/datasets/" + repo + "/parquet/" + config + "/" + split)
            if not info:
                raise RuntimeError("dataset has no parquet export")
            import pandas as pd
            frames = []
            pinned_exports = []
            for index, url in enumerate(info):
                url = ("https://huggingface.co/datasets/" + repo + "/resolve/" + export_revision
                       + f"/{config}/{split}/{index:04d}.parquet")
                pinned_exports.append(url)
                name = root / f"{short}-{split}-{index}.parquet"
                urllib.request.urlretrieve(url, name)
                frames.append(pd.read_parquet(name))
            if fetch("https://huggingface.co/api/datasets/" + repo)["sha"] != revision:
                raise RuntimeError("dataset changed during acquisition; retry")
            rows = pd.concat(frames, ignore_index=True).iloc[:count].to_dict("records")
            for i, row in enumerate(rows):
                question = row["question"]
                if short == "arc":
                    choices = row["choices"]
                    question += "\n" + "\n".join(f"{a}. {b}" for a, b in zip(choices["label"], choices["text"]))
                samples.append(dict(id=f"{short}-{split}-{i:04}", dataset=short,
                                    split=split, prompt=question + "\nExplain briefly, then give the answer."))
            sources.append(dict(repo=repo, config=config, split=split, revision=revision,
                                export_revision=export_revision,
                                exports=pinned_exports, files={p.name: digest(p) for p in root.glob(f"{short}-{split}-*.parquet")}))
    save(root / "samples.json", samples)
    save(root / "sources.json", sources)


def read_trace(path, events=False):
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if events or "event" not in row:
                yield row


def calibrate(root, model, expected=200):
    traces = sorted((root / "calibration").glob("*.jsonl"))
    if len(traces) != expected:
        raise RuntimeError(f"expected {expected} calibration traces, got {len(traces)}")
    nl, ne = model_dims(model)
    counts = np.zeros((nl, ne, ne), np.float64)
    for path in traces:
        previous = {}
        for row in read_trace(path):
            if row["phase"] != "decode":
                continue
            l, ids, w = row["layer"], row["ids"], np.asarray(row["weights"], np.float64)
            if l in previous:
                old_ids, old_w = previous[l]
                counts[l][np.ix_(old_ids, ids)] += old_w[:, None] * w[None, :]
            previous[l] = ids, w
    p = (counts + .1) / (counts.sum(-1, keepdims=True) + .1 * ne)
    sha = digest(model)
    with open(root / "markov.bin", "wb") as f:
        f.write(struct.pack("<8sII32s", b"MOEMKV01", nl, ne, bytes.fromhex(sha)))
        f.write(p.astype("<f4").tobytes())
    save(root / "markov.json", dict(version=1, model_sha256=sha, layers=nl, experts=ne,
                                   smoothing=.1, scope="adjacent decode tokens within sample",
                                   sources=json.loads((root / "sources.json").read_text(encoding="utf-8")),
                                   calibration_samples=[p.stem for p in traces],
                                   samples_sha256=digest(root / "samples.json") if (root / "samples.json").exists() else None,
                                   traces={t.name: digest(t) for t in traces}, matrix_sha256=digest(root / "markov.bin")))


def model_dims(model):
    # Read metadata only, with the repository's existing GGUF reader.
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "gguf-py"))
    from gguf import GGUFReader
    r = GGUFReader(model, "r")
    arch = r.fields["general.architecture"].contents()
    return int(r.fields[arch + ".block_count"].contents()), int(r.fields[arch + ".expert_count"].contents())


def run_one(root, exe, model, sample, policy, folder, repeat=0, slots=180, tokens=256):
    target = root / folder
    target.mkdir(exist_ok=True)
    stem = target / f"{sample['id']}-{policy}-{repeat}"
    status_path = stem.with_suffix(".status.json")
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status["returncode"] == 0:
            return
    prompt = stem.with_suffix(".prompt.txt")
    prompt.write_text(sample["prompt"], encoding="utf-8")
    command = [str(exe), "-m", str(model), "-c", "8192", "-ngl", "999", "--reasoning", "off",
               "--moe-stream-cache", f"{slots}s", "--moe-stream-io-threads", "6", "--temp", "0", "--seed", "0",
               "--no-mmap", "-b", "2048", "-ub", "512", "-n", str(tokens), "--single-turn", "--no-display-prompt", "--no-show-timings", "-f", str(prompt),
               "--moe-stream-policy", policy, "--moe-stream-trace", str(stem.with_suffix(".jsonl"))]
    if policy == "chain":
        command += ["--moe-stream-markov", str(root / "markov.bin")]
    if policy == "moe-infinity":
        command += ["--moe-stream-eam", str(root / "eam.bin")]
    start = time.time()
    print(stem.name, flush=True)
    with open(stem.with_suffix(".stdout.txt"), "wb") as out, open(stem.with_suffix(".stderr.txt"), "wb") as err:
        process = subprocess.run(command, stdout=out, stderr=err)
    save(status_path, dict(command=command, returncode=process.returncode, wall_s=time.time()-start,
                           sample=sample["id"], policy=policy, repeat=repeat,
                           binaries=binary_manifest(str(exe))))
    if process.returncode:
        raise RuntimeError(f"inference failed; inspect {stem}.stderr.txt")


@functools.lru_cache(maxsize=None)
def binary_manifest(exe):
    exe = pathlib.Path(exe)
    return {p.name: digest(p) for p in [exe, *sorted(exe.parent.glob("*.dll"))]}


def compare(root, folder, allow_partial=False):
    target = root / folder
    rows = []
    for status_path in sorted(target.glob("*.status.json")):
        status = json.loads(status_path.read_text())
        if status["returncode"]:
            raise RuntimeError("failed inference in comparison")
        stem = pathlib.Path(str(status_path).removesuffix(".status.json"))
        log = stem.with_suffix(".stderr.txt").read_text(encoding="utf-8", errors="replace")
        matches = re.findall(r"moe_cache phase=decode policy=\w+ hit=(\d+) miss=(\d+) loading=(\d+) cold=(\d+) all_hit=(\d+) tokens=(\d+)", log)
        if not matches:
            raise RuntimeError(f"missing counters: {stem}")
        hit, miss, loading, cold, all_hit, tokens = map(int, matches[-1])
        aux = re.findall(r"moe_cache loads=(\d+) bytes=(\d+) rule_us=(\d+) stall_us=(\d+)", log)
        perf = re.findall(r"moe_cache generated=(\d+) decode_tps=([\d.]+) prompt_tps=([\d.]+)", log)
        if not aux or not perf:
            raise RuntimeError(f"missing performance counters: {stem}")
        loads, transferred, rule_us, stall_us = map(int, aux[-1])
        generated, tps, prompt_tps = perf[-1]
        if not hit + miss:
            raise RuntimeError("no decode observations")
        rows.append(dict(sample=status["sample"], policy=status["policy"], repeat=status["repeat"],
                         hit=hit, miss=miss, rate=hit/(hit+miss), loading=loading, cold=cold,
                         all_hit=all_hit, token_layers=tokens, wall_s=status["wall_s"],
                         loads=loads, bytes=transferred, rule_us=rule_us, stall_us=stall_us,
                         generated=int(generated), decode_tps=float(tps), prompt_tps=float(prompt_tps)))
    grouped = collections.defaultdict(dict)
    for row in rows:
        grouped[row["sample"], row["repeat"]][row["policy"]] = row
    excluded = []
    if allow_partial:
        excluded = [list(key) for key, pair in grouped.items() if set(pair) != {"hotness", "chain"}]
        grouped = {key: pair for key, pair in grouped.items() if set(pair) == {"hotness", "chain"}}
        rows = [row for row in rows if (row["sample"], row["repeat"]) in grouped]
    if not rows:
        raise RuntimeError("no complete A/B pairs")
    for (sample, repeat), pair in grouped.items():
        if set(pair) != {"hotness", "chain"}:
            raise RuntimeError(f"incomplete pair: {sample}")
        for policy in ("hotness", "chain"):
            status_path = target / f"{sample}-{policy}-{repeat}.status.json"
            binaries = json.loads(status_path.read_text())["binaries"]
            if policy == "hotness":
                baseline_binaries = binaries
            elif binaries != baseline_binaries:
                raise RuntimeError(f"A/B binary mismatch: {sample}")
        a, b = (target / f"{sample}-{p}-{repeat}" for p in ("hotness", "chain"))
        if a.with_suffix(".stdout.txt").read_bytes() != b.with_suffix(".stdout.txt").read_bytes():
            raise RuntimeError(f"output mismatch: {sample}")
        ta, tb = list(read_trace(a.with_suffix(".jsonl"))), list(read_trace(b.with_suffix(".jsonl")))
        if ta != tb:
            raise RuntimeError(f"routing mismatch: {sample}")
        for policy, path in (("hotness", a), ("chain", b)):
            replay = replay_decode(path.with_suffix(".jsonl"), root / "markov.bin", policy)
            if (replay["hit"], replay["miss"]) != (pair[policy]["hit"], pair[policy]["miss"]):
                raise RuntimeError(f"replay counter mismatch: {sample} {policy}")
    with open(target / "samples.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_sample = collections.defaultdict(list)
    for (sample, _), pair in grouped.items():
        by_sample[sample].append((pair["chain"]["rate"] - pair["hotness"]["rate"]) * 100)
    delta = np.array([np.mean(v) for v in by_sample.values()])
    rng = np.random.default_rng(0)
    clusters = np.array([
        [sum(r[field] for r in rows if r["sample"] == sample and r["policy"] == policy)
         for policy in ("hotness", "chain") for field in ("hit", "miss")]
        for sample in by_sample], dtype=np.float64)
    boot = []
    for _ in range(10000):
        h0, m0, h1, m1 = clusters[rng.integers(len(clusters), size=len(clusters))].sum(0)
        boot.append(100*(h1/(h1+m1) - h0/(h0+m0)))
    ci = np.quantile(boot, [.025, .975])
    totals = {}
    for policy in ("hotness", "chain"):
        selected = [r for r in rows if r["policy"] == policy]
        totals[policy] = sum(r["hit"] for r in selected) / sum(r["hit"]+r["miss"] for r in selected)
    conclusion = "\u6709\u7a33\u5b9a\u63d0\u5347" if ci[0] > 0 else "\u4e0b\u964d" if ci[1] < 0 else "\u8bc1\u636e\u4e0d\u8db3"
    report = dict(rates=totals, delta_pp=(totals["chain"]-totals["hotness"])*100,
                  paired_mean_delta_pp=float(delta.mean()), ci95_pp=ci.tolist(), conclusion=conclusion,
                  samples=len(delta), runs=len(rows))
    metadata = json.loads((root / "markov.json").read_text(encoding="utf-8"))
    report["model_sha256"] = metadata["model_sha256"]
    report["matrix_sha256"] = digest(root / "markov.bin")
    report["batch"] = 2048
    report["ubatch"] = 512
    report["os_page_cache_cleared"] = False
    report["stopped_early"] = allow_partial
    report["excluded_unpaired"] = excluded
    report["repetitions"] = sorted({r["repeat"] for r in rows})
    report["ci_estimator"] = "paired sample cluster bootstrap of aggregate expert-request hit-rate difference"
    report["performance"] = {
        policy: {metric: float(np.mean([r[metric] for r in rows if r["policy"] == policy]))
                 for metric in ("decode_tps", "prompt_tps", "loads", "bytes", "stall_us", "rule_us", "wall_s")}
        for policy in ("hotness", "chain")}
    report["performance_change_percent"] = {
        metric: (report["performance"]["chain"][metric] / report["performance"]["hotness"][metric] - 1) * 100
        if report["performance"]["hotness"][metric] else None
        for metric in report["performance"]["hotness"]}
    report["datasets"] = {}
    for dataset in ("gsm8k", "arc"):
        selected = [r for r in rows if r["sample"].startswith(dataset + "-")]
        rates = {policy: sum(r["hit"] for r in selected if r["policy"] == policy) /
                 sum(r["hit"]+r["miss"] for r in selected if r["policy"] == policy)
                 for policy in ("hotness", "chain")}
        report["datasets"][dataset] = dict(rates=rates, delta_pp=100*(rates["chain"]-rates["hotness"]))
    save(target / "summary.json", report)
    (target / "report.md").write_text(
        "# \u4e13\u5bb6\u7f13\u5b58\u5bf9\u7167\u7ed3\u679c\n\n" + f"\u7ed3\u8bba\uff1a{conclusion}\u3002\n\n"
        + f"\u5b9e\u9645\u6837\u672c: {len(delta)}; \u63a8\u7406\u6b21\u6570: {len(rows)}; stopped_early={allow_partial}; excluded_unpaired={excluded}.\n\n"
        + f"Hotness: {totals['hotness']:.4%}\uff1bChain: {totals['chain']:.4%}\u3002\n\n"
        + f"\u603b\u4f53\u5dee\u503c: {report['delta_pp']:.4f} pp; \u6309\u6837\u672c\u5206\u7ec4\u914d\u5bf9 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]; \u6837\u672c\u7b49\u6743\u5747\u503c\u5dee\u503c: {delta.mean():.4f} pp.\n\n"
        + "\u6027\u80fd\u4e0e\u5206\u6570\u636e\u96c6\u7ed3\u679c\uff1a\n\n```json\n" + json.dumps({k: report[k] for k in ("performance", "performance_change_percent", "datasets")}, ensure_ascii=False, indent=2) + "\n```\n\n"
        + "\u4ec5\u7edf\u8ba1\u8bf7\u6c42\u5f00\u59cb\u65f6\u5df2\u9a7b\u7559\u7684 decode \u4e13\u5bb6\u3002Prefill \u4f7f\u7528\u539f\u7b56\u7565\u3002\u7ed3\u8bba\u9650\u4e8e\u672c\u6a21\u578b\u3001180 \u69fd\u7f13\u5b58\u4e0e\u6307\u5b9a\u8d1f\u8f7d\u3002\n",
        encoding="utf-8")


def count_ranks(slots, states, hotness, last_use, ids, result):
    resident = [i for i, e in enumerate(slots) if e >= 0 and states[i] == 2]
    resident.sort(key=lambda i: (-hotness[slots[i]], -last_use[i], slots[i]))
    rank = {slots[i]: r for r, i in enumerate(resident)}
    loading = {slots[i] for i in range(len(slots)) if slots[i] >= 0 and states[i] == 1}
    for e in ids:
        if e not in rank:
            result["miss"] += 1
            result["loading"] += e in loading
        elif rank[e] < 180:
            result["counts"][rank[e]] += 1
        else:
            result["overflow"] += 1


def replay_decode(trace_path, matrix_path, policy, rank_stats=None):
    with open(matrix_path, "rb") as f:
        magic, nl, ne, _ = struct.unpack("<8sII32s", f.read(48))
        if magic != b"MOEMKV01":
            raise RuntimeError("invalid replay matrix")
        matrix = np.frombuffer(f.read(), dtype="<f4").reshape(nl, ne, ne)
    layers = {}
    hit = miss = 0
    expected = collections.deque()
    for row in read_trace(trace_path, events=True):
        event = row.get("event")
        if event == "initial":
            if any(state == 1 for state in row["states"]):
                raise RuntimeError("replay requires quiescent decode start")
            layers[row["layer"]] = row
            continue
        if event == "replace":
            actual = (row["layer"], row["call"], row["slot"], row["old"], row["expert"])
            if not expected or expected.popleft() != actual:
                raise RuntimeError(f"replay replacement mismatch: {trace_path} {actual}")
            continue
        if row["phase"] != "decode":
            continue
        if expected:
            raise RuntimeError("missing replacement events")
        l, call = row["layer"], row["call"]
        s = layers[l]
        ids = list(dict.fromkeys(row["ids"]))
        active = set(ids)
        slots = s["slots"]
        if rank_stats is not None:
            histogram = rank_stats.setdefault(l, dict(counts=[0]*180, miss=0, loading=0,
                                                    overflow=0, slots=len(slots)))
            count_ranks(slots, s["states"], s["hotness"], s["last_use"], row["ids"], histogram)
        hit += sum(e in slots for e in row["ids"])
        miss += sum(e not in slots for e in row["ids"])
        for e in ids:
            s["hotness"][e] = min(s["hotness"][e] + 1, 2**32 - 2)
        if s["decay"] and call % s["decay"] == 0:
            for state in layers.values():
                state["hotness"] = [h >> 1 for h in state["hotness"]]
        score = np.zeros(ne, np.float32)
        if policy == "chain":
            for e, w in zip(row["ids"], row["weights"]):
                score += np.float32(w) * matrix[l, e]
        for e in ids:
            if e in slots:
                continue
            if -1 in slots:
                slot = slots.index(-1)
            else:
                choices = [i for i, resident in enumerate(slots) if resident not in active]
                key = (lambda i: (score[slots[i]], slots[i])) if policy == "chain" else (
                    lambda i: (s["hotness"][slots[i]], s["last_use"][i], i))
                slot = min(choices, key=key)
            expected.append((l, call, slot, slots[slot], e))
            slots[slot] = e
            s["states"][slot] = 2
            s["use_counter"] += 1
            s["last_use"][slot] = s["use_counter"]
        for e in row["ids"]:
            slot = slots.index(e)
            s["use_counter"] += 1
            s["last_use"][slot] = s["use_counter"]
    if expected:
        raise RuntimeError("incomplete replacement trace")
    return dict(hit=hit, miss=miss)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("action", choices=["prepare", "calibrate", "collect", "smoke", "evaluate", "report"])
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("experiments/moe-markov"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--exe", type=pathlib.Path, default=pathlib.Path("build/bin/Release/llama-cli.exe"))
    ap.add_argument("--allow-partial", action="store_true", help="report only completed pairs after an interrupted experiment")
    args = ap.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.action == "prepare":
        prepare(root)
        return
    if args.action == "calibrate":
        calibrate(root, args.model)
        return
    if args.action == "report":
        compare(root, "evaluation", allow_partial=args.allow_partial)
        return
    samples = json.loads((root / "samples.json").read_text(encoding="utf-8"))
    if args.action == "collect":
        for s in samples:
            if s["split"] == "train":
                run_one(root, args.exe.resolve(), args.model, s, "hotness", "calibration")
        return
    samples = [s for s in samples if s["split"] == "test"]
    smoke = args.action == "smoke"
    if smoke:
        samples = [s for s in samples if int(s["id"].split("-")[-1]) < 2]
    for repeat in range(1 if smoke else 3):
        for i, s in enumerate(samples):
            order = ("hotness", "chain") if (i+repeat) % 2 == 0 else ("chain", "hotness")
            for policy in order:
                run_one(root, args.exe.resolve(), args.model, s, policy, "smoke" if smoke else "evaluation", repeat)
    compare(root, "smoke" if smoke else "evaluation")


if __name__ == "__main__":
    main()
