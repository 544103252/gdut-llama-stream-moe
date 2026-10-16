"""Reconstruct pre-update decode cache rank histograms from validated traces."""
import argparse
import collections
import csv
import json
import pathlib
import os

import moe_markov as m


def export(root):
    evaluation = root / "evaluation"
    with (evaluation / "samples.csv").open(encoding="utf-8", newline="") as f:
        runs = list(csv.DictReader(f))
    totals = {}
    for run in runs:
        policy = run["policy"]
        trace = evaluation / f"{run['sample']}-{policy}-{run['repeat']}.jsonl"
        stats = {}
        counters = m.replay_decode(trace, root / "markov.bin", policy, rank_stats=stats)
        if counters != dict(hit=int(run["hit"]), miss=int(run["miss"])):
            raise RuntimeError("replay totals differ from inference")
        if sum(sum(s["counts"]) + s["overflow"] for s in stats.values()) != counters["hit"]:
            raise RuntimeError("rank hits do not conserve resident hits")
        if sum(s["miss"] for s in stats.values()) != counters["miss"]:
            raise RuntimeError("rank misses do not conserve demand misses")
        for layer, s in stats.items():
            key = policy, layer
            out = totals.setdefault(key, dict(counts=[0]*180, miss=0, loading=0, overflow=0,
                                              slots=s["slots"], runs=0))
            if out["slots"] != s["slots"]:
                raise RuntimeError("cache slot count changed within aggregation")
            out["counts"] = [a+b for a, b in zip(out["counts"], s["counts"])]
            for field in ("miss", "loading", "overflow"):
                out[field] += s[field]
            out["runs"] += 1
    destination = evaluation / "ranks"
    destination.mkdir(exist_ok=True)
    long_rows, wide_rows, top_rows = [], [], []
    for (policy, layer), s in sorted(totals.items()):
        hits = sum(s["counts"]) + s["overflow"]
        requests = hits + s["miss"]
        base = dict(policy=policy, layer=layer, phase="decode", slots=s["slots"],
                    runs=s["runs"], requests=requests, hits=hits, miss=s["miss"],
                    loading=s["loading"], overflow=s["overflow"])
        cumulative = 0
        for rank, count in enumerate(s["counts"], 1):
            cumulative += count
            row = dict(base, rank=rank, count=count, cumulative_count=cumulative,
                       fraction_all_requests=cumulative/requests if requests else 0,
                       fraction_hits=cumulative/hits if hits else 0)
            long_rows.append(row)
            if rank in (10, 30, 60, 90, 120, 150, 180):
                top_rows.append(row)
        wide_rows.append(dict(base, **{f"rank_{i:03d}": count for i, count in enumerate(s["counts"], 1)}))
    for name, rows in (("per_layer_rank.csv", long_rows), ("per_layer_180_counts.csv", wide_rows),
                       ("per_layer_top_n.csv", top_rows)):
        with (destination / name).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    overall = {}
    for policy in ("hotness", "chain"):
        selected = [s for (p, _), s in totals.items() if p == policy]
        counts = [sum(s["counts"][r] for s in selected) for r in range(180)]
        miss = sum(s["miss"] for s in selected)
        overflow = sum(s["overflow"] for s in selected)
        requests = sum(counts) + overflow + miss
        overall[policy] = dict(counts=counts, miss=miss, overflow=overflow, requests=requests,
                               top_n={str(n): sum(counts[:n])/requests for n in (10,30,60,90,120,150,180)})
    metadata = dict(source="exact replay of completed live inference traces", phase="decode",
                    ranking="resident experts by descending decayed hotness, descending LRU, ascending expert ID",
                    timing="before the token's hotness/LRU updates, decay, loads or eviction",
                    counting="one observation per (token, selected expert)", buckets=180,
                    runs=len(runs), samples=len({r["sample"] for r in runs}),
                    sample_csv_sha256=m.digest(evaluation / "samples.csv"),
                    matrix_sha256=m.digest(root / "markov.bin"),
                    layers=sorted({l for _, l in totals}), overall=overall)
    m.save(destination / "summary.json", metadata)
    write_report(destination, metadata)
    plot(destination, totals, overall)
    print(f"Validated {len(runs)} live runs; exported {len(totals)} policy/layer histograms.")


def write_report(destination, metadata):
    lines = ["# \u6bcf\u5c42\u4e13\u5bb6\u7f13\u5b58\u6392\u540d\u8bbf\u95ee\u7edf\u8ba1", "",
             f"{metadata['samples']} \u7ec4\u6210\u5bf9\u6837\u672c\uff0c{metadata['runs']} \u6b21\u771f\u5b9e\u63a8\u7406\uff1b\u4ec5\u7edf\u8ba1 decode\u3002", "",
             "\u7b2c 1 \u540d\u6700\u70ed\uff1b\u70ed\u5ea6\u76f8\u540c\u65f6\u6700\u8fd1\u4f7f\u7528\u8005\u4f18\u5148\uff1b\u4ecd\u76f8\u540c\u65f6\u6309\u4e13\u5bb6 ID \u5347\u5e8f\u3002",
             "\u5728\u672c\u6b21 token \u7684\u70ed\u5ea6\u3001LRU \u66f4\u65b0\u3001\u8870\u51cf\u548c\u7f3a\u5931\u52a0\u8f7d\u4e4b\u524d\u53d6\u6392\u540d\u5feb\u7167\u3002",
             "\u6bcf\u4e2a (token, \u9009\u4e2d\u4e13\u5bb6) \u8ba1\u4e00\u6b21\uff1b\u672a\u547d\u4e2d\u4e0d\u8fdb\u5165\u6392\u540d\u6876\u3002", "",
             "| \u7f13\u5b58\u524d N \u540d | Hotness\uff1a\u5360\u5168\u90e8\u8bbf\u95ee | Chain\uff1a\u5360\u5168\u90e8\u8bbf\u95ee |",
             "| ---: | ---: | ---: |"]
    for n in (10,30,60,90,120,150,180):
        lines.append(f"| {n} | {metadata['overall']['hotness']['top_n'][str(n)]:.4%} | {metadata['overall']['chain']['top_n'][str(n)]:.4%} |")
    lines += ["", "per_layer_180_counts.csv\uff1a\u6bcf\u79cd\u7b56\u7565\u3001\u6bcf\u5c42\u4e00\u884c\uff0c\u5305\u542b\u5b8c\u6574\u7684 180 \u4e2a\u8ba1\u6570\uff1b",
              "per_layer_rank.csv\uff1a\u6bcf\u4e2a\u6392\u540d\u7684\u6b21\u6570\u53ca\u7d2f\u8ba1\u5360\u6bd4\uff1b",
              "per_layer_top_n.csv\uff1a\u524d N \u540d\u7d2f\u8ba1\u7edf\u8ba1\uff1bsummary.json\uff1a\u7edf\u8ba1\u53e3\u5f84\u4e0e\u6c47\u603b\uff1bdistribution.png\uff1a\u5206\u5e03\u56fe\u3002", "",
              "fraction_all_requests \u7684\u5206\u6bcd\u542b\u672a\u547d\u4e2d\uff1bfraction_hits \u7684\u5206\u6bcd\u53ea\u542b\u7f13\u5b58\u547d\u4e2d\u3002",
              "\u5df2\u6709\u8f68\u8ff9\u65e0\u6cd5\u6062\u590d prefill \u5165\u53e3\u6392\u540d\uff0c\u56e0\u6b64\u672c\u6b21\u56de\u653e\u53ea\u7edf\u8ba1 decode\u3002",
              "\u65b0\u589e\u8fd0\u884c\u65f6 moe_rank \u65e5\u5fd7\u5206\u522b\u8bb0\u5f55 prefill \u548c decode\u3002Prefill \u4f7f\u7528\u5b9e\u9645",
              "micro-batch \u5165\u53e3\u5feb\u7167\uff0c\u4f46\u9010\u4e2a token/\u4e13\u5bb6\u8ba1\u6570\uff0c\u91cd\u590d\u4e13\u5bb6\u4e0d\u4f1a\u5408\u5e76\u3002", ""]
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")


def plot(destination, totals, overall):
    config = destination / ".mplconfig"
    config.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config.resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    x = np.arange(1, 181)
    for policy, s in overall.items():
        counts = np.asarray(s["counts"])
        axes[0].plot(x, 100*counts/s["requests"], label=policy, linewidth=1.2)
        axes[1].plot(x, 100*np.cumsum(counts)/s["requests"], label=policy, linewidth=1.5)
    axes[0].set_ylabel("Requests at rank (% of all requests)")
    axes[1].set_ylabel("Requests in top N (% of all requests)")
    for ax in axes:
        ax.set_xlabel("Cache rank / N (1 = hottest)")
        ax.set_xlim(1, 180)
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(destination / "distribution.png", dpi=160)
    plt.close(fig)
    layers = sorted({l for _, l in totals})
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True, constrained_layout=True)
    images = []
    data = []
    for policy in ("hotness", "chain"):
        data.append(np.asarray([np.cumsum(totals[policy, l]["counts"]) /
                    (sum(totals[policy, l]["counts"]) + totals[policy,l]["miss"] + totals[policy,l]["overflow"])
                    for l in layers])*100)
    for ax, policy, values in zip(axes, ("hotness", "chain"), data):
        images.append(ax.imshow(values, aspect="auto", vmin=0, vmax=100, cmap="viridis"))
        ax.set_title(policy)
        ax.set_xticks([0,29,59,89,119,149,179], [1,30,60,90,120,150,180])
        ax.set_yticks(range(len(layers)), layers, fontsize=8)
        ax.set_xlabel("Top N cache ranks")
    axes[0].set_ylabel("Layer (zero-based)")
    fig.colorbar(images[0], ax=axes, label="Cumulative requests (% of all requests)")
    fig.savefig(destination / "per_layer_cumulative.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("experiments/moe-markov"))
    export(ap.parse_args().root.resolve())
