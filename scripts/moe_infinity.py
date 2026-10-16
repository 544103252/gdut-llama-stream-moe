#!/usr/bin/env python3
"""Private per-layer MoE-Infinity experiment using held-out request EAMs."""
import argparse
import collections
import json
import pathlib
import re
import struct

import moe_markov as m
import numpy as np

POLICIES = ("hotness", "chain", "moe-infinity")
HEADER = struct.Struct("<8sIII32s")


def select(history, query):
    flat = history.reshape(len(history), -1).astype(np.float64)
    scores = flat @ query.ravel() / np.linalg.norm(flat, axis=1)
    best = int(np.argmax(scores))
    return best if scores[best] > 0 else -1


def insert(history, matrix, capacity):
    if not 1 <= capacity <= 65536 or not np.any(matrix):
        raise ValueError("invalid capacity or empty EAM")
    if len(history) < capacity:
        history.append(matrix.copy())
    else:
        # Zero similarity still replaces the earliest entry.
        best = select(np.asarray(history), matrix)
        history[max(best, 0)] = matrix.copy()


def load(path):
    with open(path, "rb") as f:
        header = f.read(HEADER.size)
        if len(header) != HEADER.size:
            raise ValueError("truncated EAM header")
        magic, nl, ne, count, digest = HEADER.unpack(header)
        if magic != b"MOEEAM01" or not nl or not 1 <= ne <= 65536 or not 1 <= count <= 65536:
            raise ValueError("invalid EAM header")
        data = f.read()
    if len(data) != nl * ne * count * 4:
        raise ValueError("EAM size mismatch")
    history = np.frombuffer(data, dtype="<u4").reshape(count, nl, ne)
    if np.any(~history.any(axis=(1, 2))):
        raise ValueError("empty historical EAM")
    return history, digest.hex()


def calibrate(root, model, capacity=1024):
    nl, ne = m.model_dims(model)
    samples = {s["id"]: s for s in json.loads((root / "samples.json").read_text(encoding="utf-8"))}
    paths = sorted((root / "calibration").glob("*.jsonl"))
    expected = {s for s, value in samples.items() if value["split"] == "train"}
    observed = set()
    history = []
    for path in paths:
        stem = str(path).removesuffix(".jsonl")
        status = json.loads(pathlib.Path(stem + ".status.json").read_text(encoding="utf-8"))
        sample = status["sample"]
        if status["returncode"] or sample not in expected or sample in observed:
            raise ValueError("incomplete, duplicate or non-training calibration request")
        command = status["command"]
        if pathlib.Path(command[command.index("-m") + 1]).resolve() != pathlib.Path(model).resolve():
            raise ValueError("calibration model mismatch")
        observed.add(sample)
        prompt = pathlib.Path(stem + ".prompt.txt").read_text(encoding="utf-8")
        if prompt != samples[sample]["prompt"]:
            raise ValueError("calibration prompt mismatch")
        matrix = np.zeros((nl, ne), dtype=np.uint64)
        steps = collections.defaultdict(list)
        prefill_layers = set()
        for row in m.read_trace(path):
            if row["phase"] != "decode":
                prefill_layers.add(row["layer"])
                continue
            l = row["layer"]
            if not 0 <= l < nl or any(not 0 <= e < ne for e in row["ids"]):
                raise ValueError("invalid calibration routing")
            np.add.at(matrix[l], row["ids"], 1)
            steps[l].append(row["step"])
        if not steps or len({len(v) for v in steps.values()}) != 1:
            raise ValueError("incomplete calibration iterations")
        if prefill_layers and set(steps) != prefill_layers:
            raise ValueError("missing decode layer")
        for values in steps.values():
            if values != list(range(values[0], values[0] + len(values))):
                raise ValueError("non-contiguous calibration trace")
        if matrix.max() > np.iinfo(np.uint32).max:
            raise ValueError("EAM count overflow")
        insert(history, matrix.astype("<u4"), capacity)
    if observed != expected or not history:
        raise ValueError("missing calibration requests")
    digest = m.digest(model)
    markov_meta = root / "markov.json"
    if markov_meta.exists() and json.loads(markov_meta.read_text(encoding="utf-8"))["model_sha256"] != digest:
        raise ValueError("historical model digest mismatch")
    with open(root / "eam.bin", "wb") as f:
        f.write(HEADER.pack(b"MOEEAM01", nl, ne, len(history), bytes.fromhex(digest)))
        f.write(np.asarray(history, dtype="<u4").tobytes())
    m.save(root / "eam.json", dict(version=1, capacity=capacity, entries=len(history),
                                 layers=nl, experts=ne, model_sha256=digest,
                                 traces={p.name: m.digest(p) for p in paths}))


def replay(trace_path, eam_path):
    history, _ = load(eam_path)
    _, nl, ne = history.shape
    query = np.zeros((nl, ne), dtype=np.uint32)
    layers = {}
    expected = collections.deque()
    prediction = None
    previous_layer = -1
    hit = miss = 0
    for row in m.read_trace(trace_path, events=True):
        event = row.get("event")
        if event == "eam":
            if prediction is not None or expected:
                raise ValueError("missing routing or replacement event")
            prediction = row
            continue
        if event == "initial":
            if any(v == 1 for v in row["states"]):
                raise ValueError("replay requires quiescent decode start")
            layers[row["layer"]] = row
            continue
        if event == "replace":
            actual = tuple(row[key] for key in ("layer", "call", "slot", "old", "expert"))
            if not expected or expected.popleft() != actual:
                raise ValueError("replacement replay mismatch")
            continue
        if row["phase"] != "decode":
            continue
        if expected or prediction is None:
            raise ValueError("missing prediction/replacement")
        l, call = row["layer"], row["call"]
        if l <= previous_layer:
            query.fill(0)
        previous_layer = l
        np.add.at(query[l], row["ids"], 1)
        best = select(history, query)
        score = None
        fallback = "no_match"
        if best >= 0:
            counts = history[best, l].astype(np.float64)
            fallback = "zero_row"
            if counts.sum():
                score = (counts / counts.sum()).astype(np.float32)
                fallback = "none"
        if (prediction["layer"], prediction["call"], prediction["match"], prediction["fallback"]) != (l, call, best, fallback):
            raise ValueError("EAM prediction replay mismatch")
        prediction = None
        s = layers[l]
        slots = s["slots"]
        ids = list(dict.fromkeys(row["ids"]))
        hit += sum(e in slots for e in row["ids"])
        miss += sum(e not in slots for e in row["ids"])
        for e in ids:
            s["hotness"][e] = min(s["hotness"][e] + 1, 2**32 - 2)
        if s["decay"] and call % s["decay"] == 0:
            for state in layers.values():
                state["hotness"] = [v >> 1 for v in state["hotness"]]
        for e in ids:
            if e in slots:
                continue
            choices = [i for i, resident in enumerate(slots) if resident not in ids]
            key = (lambda i: (score[slots[i]], slots[i])) if score is not None else (
                lambda i: (s["hotness"][slots[i]], s["last_use"][i], i))
            slot = slots.index(-1) if -1 in slots else min(choices, key=key)
            expected.append((l, call, slot, slots[slot], e))
            slots[slot] = e
            s["use_counter"] += 1
            s["last_use"][slot] = s["use_counter"]
        for e in row["ids"]:
            s["use_counter"] += 1
            s["last_use"][slots.index(e)] = s["use_counter"]
    if expected or prediction is not None or not hit + miss:
        raise ValueError("incomplete decode trace")
    return dict(hit=hit, miss=miss)


def compare(root, folder):
    target = root / folder
    grouped = collections.defaultdict(dict)
    for path in target.glob("*.status.json"):
        status = json.loads(path.read_text(encoding="utf-8"))
        if status["returncode"]:
            raise ValueError("failed inference")
        stem = pathlib.Path(str(path).removesuffix(".status.json"))
        grouped[status["sample"], status["repeat"]][status["policy"]] = (stem, status)
    if not grouped:
        raise ValueError("no completed comparisons")
    smoke = folder == "infinity-smoke"
    samples = json.loads((root / "samples.json").read_text(encoding="utf-8"))
    manifest_path = root / (folder + "-manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    sample_ids = manifest.get("sample_ids", [s["id"] for s in samples if s["split"] == "test"
                              and (not smoke or int(s["id"].split("-")[-1]) < 2)])
    expected = {(sample, repeat) for sample in sample_ids
                for repeat in range(manifest.get("repetitions", 1 if smoke else 3))}
    if set(grouped) != expected:
        raise ValueError("missing or unexpected comparison requests")
    runs = []
    for key, group in grouped.items():
        if set(group) != set(POLICIES):
            raise ValueError("incomplete three-policy comparison")
        outputs, routes = [], []
        commands = []
        manifests = []
        for policy in POLICIES:
            stem, status = group[policy]
            command = status["command"][:]
            for flag in ("--moe-stream-policy", "--moe-stream-trace", "--moe-stream-markov", "--moe-stream-eam", "-f"):
                if flag in command:
                    offset = command.index(flag)
                    del command[offset:offset + 2]
            commands.append(command)
            manifests.append(status["binaries"])
            outputs.append(stem.with_suffix(".stdout.txt").read_bytes())
            trace = stem.with_suffix(".jsonl")
            routes.append([(r["layer"], r["phase"], r["ids"], r["weights"]) for r in m.read_trace(trace)])
            log = stem.with_suffix(".stderr.txt").read_text(encoding="utf-8", errors="replace")
            phases = {}
            for phase in ("prefill", "decode"):
                found = re.findall(r"moe_cache phase=" + phase + r" policy=[\w-]+ hit=(\d+) miss=(\d+)", log)
                if not found:
                    raise ValueError("missing phase counters")
                h, miss = map(int, found[-1])
                phases[phase] = dict(hit=h, miss=miss, rate=h/(h+miss) if h+miss else None)
            checked = replay(trace, root / "eam.bin") if policy == "moe-infinity" else m.replay_decode(trace, root / "markov.bin", policy)
            if any(checked[k] != phases["decode"][k] for k in ("hit", "miss")):
                raise ValueError("replay counters disagree")
            aux = re.findall(r"moe_cache loads=(\d+) bytes=(\d+) rule_us=(\d+) stall_us=(\d+)", log)
            perf = re.findall(r"moe_cache generated=(\d+) decode_tps=([\d.]+) prompt_tps=([\d.]+)", log)
            if not aux or not perf:
                raise ValueError("missing performance counters")
            metrics = dict(zip(("loads", "bytes", "rule_us", "stall_us"), map(int, aux[-1])))
            generated, decode_tps, prompt_tps = perf[-1]
            metrics.update(generated=int(generated), decode_tps=float(decode_tps), prompt_tps=float(prompt_tps),
                           decode_ms_per_token=1000/float(decode_tps) if float(decode_tps) else None)
            runs.append(dict(sample=key[0], repeat=key[1], policy=policy, phases=phases, **metrics))
        if any(c != commands[0] for c in commands[1:]) or any(b != manifests[0] for b in manifests[1:]):
            raise ValueError("comparison build or arguments mismatch")
        if any(o != outputs[0] for o in outputs[1:]) or any(r != routes[0] for r in routes[1:]):
            raise ValueError("output or routing mismatch")
    aggregate = {}
    for policy in POLICIES:
        selected = [r for r in runs if r["policy"] == policy]
        phases = {}
        for phase in ("prefill", "decode"):
            h = sum(r["phases"][phase]["hit"] for r in selected)
            misses = sum(r["phases"][phase]["miss"] for r in selected)
            phases[phase] = dict(hit=h, miss=misses, rate=h/(h+misses) if h+misses else None)
        aggregate[policy] = dict(phases=phases, means={
            field: float(np.mean([r[field] for r in selected]))
            for field in ("loads", "bytes", "rule_us", "stall_us", "decode_tps", "prompt_tps")})
    m.save(target / "summary.json", dict(runs=runs, aggregate=aggregate,
                                        samples=len({r["sample"] for r in runs}),
                                        repetitions=sorted({r["repeat"] for r in runs}),
                                        eam_sha256=m.digest(root / "eam.bin"),
                                        markov_sha256=m.digest(root / "markov.bin"),
                                        os_page_cache_cleared=False, replay_verified=True))


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("action", choices=("calibrate", "smoke", "evaluate", "report"))
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("experiments/moe-markov"))
    ap.add_argument("--model", default=m.MODEL)
    ap.add_argument("--exe", type=pathlib.Path, default=pathlib.Path("build/bin/Release/llama-cli.exe"))
    ap.add_argument("--capacity", type=int, default=1024)
    ap.add_argument("--slots", type=int, default=180)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--limit-per-dataset", type=int, default=0, help="test samples per dataset; zero selects all")
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--folder", help="separate output folder name within the experiment root")
    args = ap.parse_args()
    root = args.root.resolve()
    if args.action == "calibrate":
        calibrate(root, args.model, args.capacity)
        return
    folder = args.folder or ("infinity-smoke" if args.action == "smoke" else "infinity-evaluation")
    if pathlib.Path(folder).name != folder or args.limit_per_dataset < 0 or args.repetitions < 1:
        raise ValueError("invalid output folder, sample limit or repetition count")
    if args.action != "report":
        if args.slots < 1 or args.tokens < 2:
            raise ValueError("positive slots and at least two generated tokens are required")
        samples = json.loads((root / "samples.json").read_text(encoding="utf-8"))
        smoke = args.action == "smoke"
        selected = []
        counts = collections.Counter()
        for sample in samples:
            if sample["split"] != "test" or (smoke and int(sample["id"].split("-")[-1]) >= 2):
                continue
            dataset = sample["dataset"]
            if args.limit_per_dataset and counts[dataset] >= args.limit_per_dataset:
                continue
            counts[dataset] += 1
            selected.append(sample)
        if not selected:
            raise ValueError("no selected test samples")
        repetitions = 1 if smoke else args.repetitions
        manifest = dict(model_sha256=m.digest(args.model), binaries=m.binary_manifest(str(args.exe.resolve())),
                        slots=args.slots, tokens=args.tokens, eam_sha256=m.digest(root / "eam.bin"),
                        markov_sha256=m.digest(root / "markov.bin"),
                        sample_ids=[s["id"] for s in selected], repetitions=repetitions,
                        samples_sha256=m.digest(root / "samples.json"))
        manifest_path = root / (folder + "-manifest.json")
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("experiment configuration changed; choose a new root")
        else:
            m.save(manifest_path, manifest)
        for repeat in range(repetitions):
            for i, sample in enumerate(selected):
                offset = (i + repeat) % 3
                for policy in POLICIES[offset:] + POLICIES[:offset]:
                    m.run_one(root, args.exe.resolve(), args.model, sample, policy, folder, repeat,
                              slots=args.slots, tokens=args.tokens)
    compare(root, folder)


if __name__ == "__main__":
    main()
