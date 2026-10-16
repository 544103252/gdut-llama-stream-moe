"""Summarize a validated three-policy live experiment and runtime rank counters."""
import argparse
import collections
import csv
import json
import pathlib
import re

import moe_markov as m
import numpy as np


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_ci(runs, policy, baseline):
    samples = sorted({r["sample"] for r in runs})
    clusters = []
    for sample in samples:
        cluster = []
        for p in (baseline, policy):
            selected = [r for r in runs if r["sample"] == sample and r["policy"] == p]
            cluster += [sum(r["phases"]["decode"][key] for r in selected) for key in ("hit", "miss")]
        clusters.append(cluster)
    clusters = np.asarray(clusters, np.float64)
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(10000):
        h0, m0, h1, m1 = clusters[rng.integers(len(clusters), size=len(clusters))].sum(0)
        draws.append(100*(h1/(h1+m1)-h0/(h0+m0)))
    h0, m0, h1, m1 = clusters.sum(0)
    low, high = np.quantile(draws, [.025, .975])
    return dict(delta_pp=100*(h1/(h1+m1)-h0/(h0+m0)), ci95_pp=[float(low), float(high)],
                conclusion="increase" if low > 0 else "decrease" if high < 0 else "insufficient_evidence")


def report(root, folder):
    target = root / folder
    summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / (folder + "-manifest.json")).read_text(encoding="utf-8"))
    if not summary["replay_verified"]:
        raise ValueError("unverified inference results")
    runs = summary["runs"]
    policies = ("hotness", "chain", "moe-infinity")
    flat, histograms = [], {}
    for run in runs:
        row = {k: v for k, v in run.items() if k != "phases"}
        for phase, counters in run["phases"].items():
            row.update({phase + "_" + k: v for k, v in counters.items()})
        flat.append(row)
        stem = target / f"{run['sample']}-{run['policy']}-{run['repeat']}"
        text = stem.with_suffix(".stderr.txt").read_text(encoding="utf-8", errors="replace")
        matches = re.findall(r"moe_rank layer=(\d+) phase=(\w+) policy=([\w-]+) slots=(\d+) miss=(\d+) loading=(\d+) overflow=(\d+) counts=([0-9,]+)", text)
        latest = {(int(r[0]), r[1]): r for r in matches}
        if not latest:
            raise ValueError("missing runtime rank statistics")
        totals = collections.defaultdict(lambda: [0, 0])
        for (layer, phase), r in latest.items():
            _, _, policy, slots, miss, loading, overflow, counts = r
            bins = list(map(int, counts.split(",")))
            if len(bins) != 180 or policy != run["policy"]:
                raise ValueError("invalid rank counter record")
            slots, miss, loading, overflow = map(int, (slots, miss, loading, overflow))
            if any(bins[min(slots,180):]):
                raise ValueError("nonzero bins beyond cache capacity")
            totals[phase][0] += sum(bins) + overflow
            totals[phase][1] += miss
            key = policy, phase, layer
            aggregate = histograms.setdefault(key, dict(counts=[0]*180, miss=0, loading=0,
                                                        overflow=0, slots=slots, runs=0))
            aggregate["counts"] = [a+b for a,b in zip(aggregate["counts"], bins)]
            for field, value in (("miss",miss),("loading",loading),("overflow",overflow)):
                aggregate[field] += value
            aggregate["runs"] += 1
        for phase, (hit, miss) in totals.items():
            if (hit, miss) != (run["phases"][phase]["hit"], run["phases"][phase]["miss"]):
                raise ValueError("rank counters disagree with live phase counters")
    write_csv(target / "samples.csv", flat)
    wide, top = [], []
    for (policy, phase, layer), s in sorted(histograms.items()):
        hits = sum(s["counts"]) + s["overflow"]
        requests = hits + s["miss"]
        base = dict(policy=policy, phase=phase, layer=layer, slots=s["slots"], runs=s["runs"],
                    hits=hits, miss=s["miss"], loading=s["loading"], overflow=s["overflow"], requests=requests,
                    hit_rate=hits/requests if requests else 0)
        wide.append(dict(base, **{f"rank_{i:03d}": n for i,n in enumerate(s["counts"],1)}))
        for n in (10,30,60,120,180):
            count = sum(s["counts"][:n])
            top.append(dict(base, rank=n, cumulative_count=count,
                            fraction_all_requests=count/requests if requests else 0,
                            fraction_hits=count/hits if hits else 0))
    write_csv(target / "per_layer_180_counts.csv", wide)
    write_csv(target / "per_layer_top_n.csv", top)
    summary["comparisons"] = {policy + "_vs_" + baseline: paired_ci(runs, policy, baseline)
        for policy, baseline in (("chain","hotness"),("moe-infinity","hotness"),("moe-infinity","chain"))}
    summary["datasets"] = {}
    for dataset in sorted({r["sample"].split("-")[0] for r in runs}):
        selected = [r for r in runs if r["sample"].startswith(dataset + "-")]
        summary["datasets"][dataset] = {policy: sum(r["phases"]["decode"]["hit"] for r in selected if r["policy"]==policy) /
            sum(r["phases"]["decode"]["hit"]+r["phases"]["decode"]["miss"] for r in selected if r["policy"]==policy)
            for policy in policies}
    summary["manifest"] = manifest
    summary["rank_counters_verified"] = True
    m.save(target / "summary.json", summary)
    lines = ["# MoE-Infinity \u4e13\u5bb6\u7f13\u5b58\u547d\u4e2d\u7387\u5bf9\u7167\u7ed3\u679c", "",
             f"{summary['samples']} paired inputs; {len(runs)} live inference runs; repetitions={summary['repetitions']}.", "",
             "| \u7b56\u7565 | Decode \u9a7b\u7559\u547d\u4e2d\u7387 | Prefill \u9a7b\u7559\u547d\u4e2d\u7387 | \u5e73\u5747 decode tokens/s |",
             "| --- | ---: | ---: | ---: |"]
    for policy in policies:
        data = summary["aggregate"][policy]
        lines.append(f"| {policy} | {data['phases']['decode']['rate']:.4%} | {data['phases']['prefill']['rate']:.4%} | {data['means']['decode_tps']:.3f} |")
    lines += ["", "| \u5bf9\u6bd4 | \u547d\u4e2d\u7387\u5dee\u503c\uff08\u767e\u5206\u70b9\uff09 | \u914d\u5bf9\u6837\u672c bootstrap 95% CI | \u7ed3\u8bba |",
              "| --- | ---: | --- | --- |"]
    for name, value in summary["comparisons"].items():
        lines.append(f"| {name} | {value['delta_pp']:+.4f} | [{value['ci95_pp'][0]:+.4f}, {value['ci95_pp'][1]:+.4f}] | {value['conclusion']} |")
    lines += ["", "Bootstrap\uff1a10,000 \u6b21\u91cd\u91c7\u6837\uff0cseed=0\uff1b\u6309\u8f93\u5165\u6837\u672c\u6210\u7ec4\u914d\u5bf9\u62bd\u6837\uff0c\u8ba1\u7b97\u603b\u4f53\u4e13\u5bb6\u8bf7\u6c42\u547d\u4e2d\u7387\u5dee\u503c\u3002",
              "\u4e09\u79cd\u7b56\u7565\u91c7\u7528\u76f8\u540c\u6784\u5efa\u3001\u8f93\u5165\u3001\u7f13\u5b58\u5bb9\u91cf\u53ca\u751f\u6210\u53c2\u6570\u3002\u751f\u6210\u6587\u672c\u3001\u8def\u7531\u3001\u9884\u6d4b\u5339\u914d\u548c\u9010\u6b21 decode \u66ff\u6362\u5747\u5df2\u901a\u8fc7\u56de\u653e\u6838\u5bf9\u3002",
              "\u72ec\u7acb\u6821\u51c6\u4f7f\u7528 200 \u6761\u8bad\u7ec3\u8f93\u5165\uff1b\u8bc4\u6d4b\u53d6 GSM8K\u3001ARC \u5404\u524d 18 \u6761\u6d4b\u8bd5\u8f93\u5165\uff0c\u6bcf\u79cd\u7b56\u7565\u5404\u8fd0\u884c\u4e00\u6b21\uff0c\u7f13\u5b58 180 \u69fd\uff0c\u6700\u591a\u751f\u6210 256 token\u3002",
              "\u672c\u6b21\u6570\u636e\u96c6\u914d\u6bd4\u4e3a 18:18\uff0c\u4e0e\u6b64\u524d 50 \u6761 GSM8K\u300110 \u6761 ARC \u7684\u914d\u6bd4\u4e0d\u540c\uff1b\u7b56\u7565\u6548\u679c\u4f7f\u7528\u672c\u8f6e\u76f8\u540c\u6837\u672c\u4f5c\u5bf9\u7167\u3002",
              "\u672a\u6e05\u7a7a OS \u6587\u4ef6\u7f13\u5b58\u3002\u9884\u6d4b\u7b56\u7565\u5728\u542f\u52a8\u65f6\u6821\u9a8c\u5b8c\u6574 GGUF \u54c8\u5e0c\uff0c\u56e0\u6b64\u541e\u5410\u91cf\u662f\u672c\u6d41\u7a0b\u4e0b\u7684\u5b9e\u6d4b\u503c\uff0c\u4e0d\u80fd\u5355\u72ec\u5f52\u56e0\u4e8e\u7f13\u5b58\u66ff\u6362\u7b56\u7565\u3002",
              "\u6392\u540d\u7edf\u8ba1\u4f7f\u7528\u66f4\u65b0\u524d\u7684\u9a7b\u7559\u5feb\u7167\uff0c\u6309\u70ed\u5ea6\u964d\u5e8f\u3001LRU \u964d\u5e8f\u6392\u5e8f\uff0c\u56fa\u5b9a 180 \u6876\uff0c\u672a\u547d\u4e2d\u5355\u5217\u3002Prefill \u5bf9\u6bcf\u4e2a token/\u4e13\u5bb6\u8ba1\u6570\uff0c\u91c7\u7528\u5b9e\u9645 micro-batch \u5165\u53e3\u5feb\u7167\u3002\u6ca1\u6709\u8bbf\u95ee\u7684\u5c42\uff0c\u5176\u6876\u8ba1\u6570\u5747\u4e3a\u96f6\u3002",
              "\u6587\u4ef6\uff1asamples.csv\u3001per_layer_180_counts.csv\u3001per_layer_top_n.csv\u3001summary.json\uff1b\u540c\u76ee\u5f55\u4fdd\u7559\u539f\u59cb\u8f93\u51fa\u3001\u8f68\u8ff9\u548c\u8fd0\u884c\u72b6\u6001\u3002", ""]
    (target / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"aggregate":summary["aggregate"],"comparisons":summary["comparisons"]}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("experiments/moe-markov"))
    ap.add_argument("--folder", default="infinity-evaluation-108")
    args = ap.parse_args()
    report(args.root.resolve(), args.folder)
