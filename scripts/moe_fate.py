#!/usr/bin/env python3
"""Private Fate 4.4 comparison, pinned to the reference 108-run experiment."""
import argparse
import collections
import json
import hashlib
import os
import pathlib
import re
import sys

REFERENCE = pathlib.Path(r"D:\another_project\llama-stream-moe-main\experiments\moe-markov")
sys.path.insert(0, str(REFERENCE / "python-deps"))
import moe_markov as m
import moe_infinity_report as reports
import numpy as np

POLICIES = ("hotness", "fate-hotness", "fate-arc")
VERSION = 2


class Arc:
    def __init__(self, state):
        self.q = {name: list(state['arc_' + name]) for name in ('t1', 't2', 'b1', 'b2', 'pending')}
        self.p = state['arc_p']

    def access(self, expert, slots, active):
        t1, t2, b1, b2, pending = (self.q[k] for k in ('t1', 't2', 'b1', 'b2', 'pending'))
        c = len(slots)
        if expert in slots:
            first = expert in pending
            for q in (t1, t2, pending):
                if expert in q:
                    q.remove(expert)
            (t1 if first else t2).insert(0, expert)
            return slots.index(expert)
        in_b1, in_b2 = expert in b1, expert in b2
        if in_b1:
            self.p = min(c, self.p + max(1, len(b2)//len(b1)))
        elif in_b2:
            self.p = max(0, self.p - max(1, len(b1)//len(b2)))
        if -1 in slots:
            slot = slots.index(-1)
        else:
            recent = bool(t1) and (len(t1) > self.p or (in_b2 and len(t1) == self.p))
            queues = (t1, t2) if recent else (t2, t1)
            victim = next((e for q in queues for e in reversed(q) if e not in active), None)
            if victim is None:
                raise ValueError('no ARC victim')
            slot = slots.index(victim)
        old = slots[slot]
        if old >= 0:
            recent = old in t1
            discard = not in_b1 and not in_b2 and recent and len(t1) == c
            for q in (t1, t2, pending):
                if old in q:
                    q.remove(old)
            if not discard:
                (b1 if recent else b2).insert(0, old)
        for q in (b1, b2):
            if expert in q:
                q.remove(expert)
        (t2 if in_b1 or in_b2 else t1).insert(0, expert)
        while len(t1) + len(b1) > c and b1:
            b1.pop()
        while sum(map(len, (t1, t2, b1, b2))) > 2*c:
            (b2 if b2 else b1).pop()
        return slot


def replay(path, policy):
    layers, arcs = {}, {}
    expected = collections.deque()
    hit = miss = 0
    for row in m.read_trace(path, events=True):
        event = row.get('event')
        if event == 'capacity':
            continue
        if event == 'initial':
            if 1 in row['states']:
                raise ValueError('decode starts with a pending load')
            layers[row['layer']] = row
            if policy == 'fate-arc':
                arcs[row['layer']] = Arc(row)
            continue
        if event == 'replace':
            actual = tuple(row[k] for k in ('layer', 'call', 'slot', 'old', 'expert'))
            if not expected or expected.popleft() != actual:
                raise ValueError(f'replacement replay mismatch: {path}: {actual}')
            continue
        if row['phase'] != 'decode':
            continue
        if expected:
            raise ValueError('missing replacements')
        layer, call, ids = row['layer'], row['call'], row['ids']
        s = layers[layer]
        slots = s['slots']
        hit += sum(e in slots for e in ids)
        miss += sum(e not in slots for e in ids)
        for e in dict.fromkeys(ids):
            s['hotness'][e] = min(2**32 - 2, s['hotness'][e] + 1)
        if s['decay'] and call % s['decay'] == 0:
            for state in layers.values():
                state['hotness'] = [h >> 1 for h in state['hotness']]
        active = set(ids)
        # Reservations precede all final per-access LRU stamps in the runtime.
        for e in ids if policy == 'fate-arc' else dict.fromkeys(ids):
            if policy == 'fate-arc':
                slot = arcs[layer].access(e, slots, active)
            elif e in slots:
                continue
            else:
                choices = [i for i, old in enumerate(slots) if old not in active]
                slot = slots.index(-1) if -1 in slots else min(choices, key=lambda i:
                    (s['hotness'][slots[i]], s['last_use'][i], i))
            if slots[slot] != e:
                expected.append((layer, call, slot, slots[slot], e))
                slots[slot] = e
                s['use_counter'] += 1
                s['last_use'][slot] = s['use_counter']
        for e in ids:
            s['use_counter'] += 1
            s['last_use'][slots.index(e)] = s['use_counter']
    if expected or not hit + miss:
        raise ValueError('incomplete decode replay')
    return dict(hit=hit, miss=miss)


def normalize(command):
    command = list(command)[1:]
    for flag in ('--moe-stream-policy', '--moe-stream-trace', '--moe-stream-markov', '--moe-stream-eam', '-f'):
        if flag in command:
            i = command.index(flag)
            del command[i:i+2]
    return command


def capacities(log):
    return [(int(l), int(s), int(b)) for l, s, b in re.findall(
        r'moe_capacity layer=(\d+) slots=(\d+) expert_bytes=(\d+)', log)]


def plan_capacity(model, shallow=4, uniform=180):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'gguf-py'))
    from gguf import GGUFReader
    reader = GGUFReader(model, 'r')
    arch = reader.fields['general.architecture'].contents()
    experts = int(reader.fields[arch + '.expert_count'].contents())
    top_k = int(reader.fields[arch + '.expert_used_count'].contents())
    minimum = min(experts, 3*top_k)
    sizes = collections.Counter()
    for tensor in reader.tensors:
        match = re.fullmatch(r'blk\.(\d+)\.ffn_(?:gate|up|down|gate_up)_exps\.weight', tensor.name)
        if match and len(tensor.shape) >= 3 and int(tensor.shape[2]) == experts:
            sizes[int(match[1])] += tensor.n_bytes//experts
    if not sizes:
        raise ValueError('no streamable experts')
    budget = sum(sizes.values())*uniform
    slots = {layer: minimum for layer in sorted(sizes)}
    used = sum(sizes[layer]*slots[layer] for layer in slots)
    if used > budget:
        raise ValueError('insufficient minimum budget')
    layers = list(slots)
    for layer in layers[:shallow]:
        extra = min(experts - slots[layer], (budget - used)//sizes[layer])
        slots[layer] += extra
        used += extra*sizes[layer]
    while True:
        progress = False
        for layer in layers[shallow:]:
            if slots[layer] < experts and sizes[layer] <= budget - used:
                slots[layer] += 1
                used += sizes[layer]
                progress = True
        if not progress:
            break
    return dict(capacity_table=[[layer, slots[layer], sizes[layer]] for layer in layers],
                budget_bytes=budget, allocated_bytes=used, wave_capacity=(min(slots.values()) - top_k)//2)


def run_request(root, exe, model, sample, policy, folder, manifest):
    stem = root / folder / f"{sample['id']}-{policy}-0"
    status_path = stem.with_suffix('.status.json')
    suffixes = ('.prompt.txt', '.stdout.txt', '.stderr.txt', '.jsonl')
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding='utf-8'))
        if status['returncode'] == 0:
            if status['binaries'] != manifest['binaries'] or normalize(status['command']) != manifest['normalized_command']:
                raise ValueError('resume build or arguments mismatch')
            hashes = status.get('artifacts', {})
            for suffix in suffixes:
                path = stem.with_suffix(suffix)
                if not path.exists() or hashes.get(suffix) != m.digest(path):
                    raise ValueError('resume artifact missing or changed')
    m.run_one(root, exe, model, sample, policy, folder, slots=180, tokens=manifest['tokens'])
    status = json.loads(status_path.read_text(encoding='utf-8'))
    status['artifacts'] = {suffix: m.digest(stem.with_suffix(suffix)) for suffix in suffixes}
    m.save(status_path, status)


def verify_triplet(root, folder, sample):
    outputs, routes = [], []
    for policy in POLICIES:
        stem = root / folder / f'{sample}-{policy}-0'
        outputs.append(stem.with_suffix('.stdout.txt').read_bytes())
        trace = stem.with_suffix('.jsonl')
        routes.append([(r['layer'], r['phase'], r['ids'], r['weights']) for r in m.read_trace(trace)])
        checked = replay(trace, policy)
        log = stem.with_suffix('.stderr.txt').read_text(encoding='utf-8', errors='replace')
        observed = re.findall(r'moe_cache phase=decode policy=[\w-]+ hit=(\d+) miss=(\d+)', log)
        if not observed or tuple(map(int, observed[-1])) != (checked['hit'], checked['miss']):
            raise ValueError(f'incremental replay counter mismatch: {sample} {policy}')
    if outputs[1:] != outputs[:-1] or routes[1:] != routes[:-1]:
        raise ValueError(f'incremental output or routing mismatch: {sample}')


def compare(root, folder):
    target = root / folder
    manifest = json.loads((root / (folder + '-manifest.json')).read_text(encoding='utf-8'))
    runs, rank_rows = [], []
    common_capacity = None
    for sample in manifest['sample_ids']:
        outputs, routes, capacity = [], [], []
        for policy in POLICIES:
            stem = target / f'{sample}-{policy}-0'
            status = json.loads(stem.with_suffix('.status.json').read_text(encoding='utf-8'))
            if status['returncode'] or status['binaries'] != manifest['binaries']:
                raise ValueError('failed run or build mismatch')
            if normalize(status['command']) != manifest['normalized_command']:
                raise ValueError('inference arguments mismatch')
            if hashlib.sha256(stem.with_suffix('.prompt.txt').read_text(encoding='utf-8').encode('utf-8')).hexdigest() != manifest['prompts'][sample]:
                raise ValueError('input prompt mismatch')
            for suffix, expected_digest in status.get('artifacts', {}).items():
                if m.digest(stem.with_suffix(suffix)) != expected_digest:
                    raise ValueError('run artifact changed')
            outputs.append(stem.with_suffix('.stdout.txt').read_bytes())
            trace = stem.with_suffix('.jsonl')
            routes.append([(r['layer'], r['phase'], r['ids'], r['weights']) for r in m.read_trace(trace)])
            log = stem.with_suffix('.stderr.txt').read_text(encoding='utf-8', errors='replace')
            phases = {}
            for phase in ('prefill', 'decode'):
                found = re.findall(r'moe_cache phase=' + phase + r' policy=[\w-]+ hit=(\d+) miss=(\d+)', log)
                if not found:
                    raise ValueError('missing counters')
                h, miss = map(int, found[-1])
                phases[phase] = dict(hit=h, miss=miss, rate=h/(h+miss) if h+miss else None)
            checked = replay(trace, policy)
            if any(checked[k] != phases['decode'][k] for k in ('hit', 'miss')):
                raise ValueError('replay counter mismatch')
            aux = re.findall(r'moe_cache loads=(\d+) bytes=(\d+) rule_us=(\d+) stall_us=(\d+)', log)
            perf = re.findall(r'moe_cache generated=(\d+) decode_tps=([\d.]+) prompt_tps=([\d.]+)', log)
            if not aux or not perf:
                raise ValueError('missing performance counters')
            metrics = dict(zip(('loads', 'bytes', 'rule_us', 'stall_us'), map(int, aux[-1])))
            generated, decode_tps, prompt_tps = perf[-1]
            metrics.update(generated=int(generated), decode_tps=float(decode_tps), prompt_tps=float(prompt_tps),
                decode_ms_per_token=1000/float(decode_tps) if float(decode_tps) else None, wall_s=status['wall_s'])
            runs.append(dict(sample=sample, repeat=0, policy=policy, phases=phases, **metrics))
            latest = {(int(r[0]), r[1]): r for r in re.findall(
                r'moe_rank layer=(\d+) phase=(\w+) policy=([\w-]+) slots=(\d+) miss=(\d+) loading=(\d+) overflow=(\d+) counts=([0-9,]+)', log)}
            totals = collections.defaultdict(lambda: [0, 0])
            if not latest:
                raise ValueError('missing layer ranks')
            for (layer, phase), r in latest.items():
                slots, misses, loading, overflow = map(int, r[3:7])
                bins = list(map(int, r[7].split(',')))
                if len(bins) != 180 or any(bins[min(slots, 180):]) or r[2] != policy:
                    raise ValueError('invalid rank buckets')
                hits = sum(bins) + overflow
                totals[phase][0] += hits
                totals[phase][1] += misses
                rank_rows.append(dict(sample=sample, policy=policy, phase=phase, layer=layer, slots=slots,
                    hit=hits, miss=misses, loading=loading, overflow=overflow,
                    hit_rate=hits/(hits+misses) if hits+misses else 0,
                    **{f'rank_{i:03d}': n for i, n in enumerate(bins, 1)}))
            for phase in ('prefill', 'decode'):
                if totals[phase] != [phases[phase]['hit'], phases[phase]['miss']]:
                    raise ValueError('rank conservation failure')
            if policy != 'hotness':
                snapshot = next((r for r in m.read_trace(trace, events=True) if r.get('event') == 'capacity'), None)
                if snapshot is None:
                    raise ValueError('missing runtime capacity snapshot')
                sizes = {layer: size for layer, _, size in manifest['capacity_table']}
                table = [(layer, slots, sizes[layer]) for layer, slots in enumerate(snapshot['slots']) if slots]
                budget = [(str(snapshot['budget']), str(snapshot['allocated']), str(snapshot['shallow']))]
                if not table or not budget or int(budget[-1][1]) > int(budget[-1][0]):
                    raise ValueError('invalid capacity budget')
                if sum(slots*size for _, slots, size in table) != int(budget[-1][1]):
                    raise ValueError('capacity byte accounting mismatch')
                capacity.append((table, budget[-1]))
        if outputs[1:] != outputs[:-1] or routes[1:] != routes[:-1]:
            raise ValueError(f'output or routing mismatch: {sample}')
        if capacity[0] != capacity[1] or (common_capacity is not None and capacity[0] != common_capacity):
            raise ValueError('Fate capacity differs across runs')
        common_capacity = capacity[0]
    expected_statuses = {f'{sample}-{policy}-0.status.json' for sample in manifest['sample_ids'] for policy in POLICIES}
    if {p.name for p in target.glob('*.status.json')} != expected_statuses:
        raise ValueError('unexpected run status files')
    if [list(row) for row in common_capacity[0]] != manifest['capacity_table'] or int(common_capacity[1][0]) != manifest['budget_bytes'] or int(common_capacity[1][1]) != manifest['allocated_bytes']:
        raise ValueError('runtime capacity disagrees with pinned plan')
    aggregate = {}
    for policy in POLICIES:
        selected = [r for r in runs if r['policy'] == policy]
        phases = {}
        for phase in ('prefill', 'decode'):
            h = sum(r['phases'][phase]['hit'] for r in selected)
            miss = sum(r['phases'][phase]['miss'] for r in selected)
            phases[phase] = dict(hit=h, miss=miss, rate=h/(h+miss) if h+miss else None)
        aggregate[policy] = dict(phases=phases, means={k: float(np.mean([r[k] for r in selected if r[k] is not None]))
            for k in ('loads', 'bytes', 'rule_us', 'stall_us', 'decode_tps', 'prompt_tps', 'decode_ms_per_token', 'wall_s')})
    comparisons = {a + '_vs_' + b: reports.paired_ci(runs, a, b) for a, b in
        (('fate-hotness', 'hotness'), ('fate-arc', 'hotness'), ('fate-arc', 'fate-hotness'))}
    datasets = {}
    for dataset in ('gsm8k', 'arc'):
        datasets[dataset] = {}
        for policy in POLICIES:
            selected = [r for r in runs if r['policy'] == policy and r['sample'].startswith(dataset + '-')]
            h = sum(r['phases']['decode']['hit'] for r in selected)
            miss = sum(r['phases']['decode']['miss'] for r in selected)
            datasets[dataset][policy] = h/(h+miss)
    summary = dict(runs=runs, aggregate=aggregate, comparisons=comparisons, datasets=datasets,
        samples=len(manifest['sample_ids']), repetitions=[0], manifest=manifest,
        os_page_cache_cleared=False, replay_verified=True, rank_counters_verified=True,
        output_and_routing_verified=True)
    m.save(target / 'summary.json', summary)
    flat = [dict(sample=r['sample'], policy=r['policy'], **{k: v for k, v in r.items() if k not in ('sample', 'policy', 'phases')},
        **{f'{phase}_{k}': v for phase, values in r['phases'].items() for k, v in values.items()}) for r in runs]
    reports.write_csv(target / 'samples.csv', flat)
    reports.write_csv(target / 'per_run_layer_180_counts.csv', rank_rows)
    grouped = collections.defaultdict(list)
    for row in rank_rows:
        grouped[row['policy'], row['phase'], row['layer']].append(row)
    layer_rows = []
    for (policy, phase, layer), rows in sorted(grouped.items()):
        h, miss = sum(r['hit'] for r in rows), sum(r['miss'] for r in rows)
        layer_rows.append(dict(policy=policy, phase=phase, layer=layer, slots=rows[0]['slots'], runs=len(rows),
            hit=h, miss=miss, hit_rate=h/(h+miss) if h+miss else 0,
            **{k: sum(r[k] for r in rows) for k in ('loading', 'overflow', *[f'rank_{i:03d}' for i in range(1, 181)])}))
    reports.write_csv(target / 'per_layer_180_counts.csv', layer_rows)
    lines = ['# Fate 4.4 expert cache hit-rate comparison', '',
        'Primary metric: decode resident expert cache hit rate. Prefill, dataset and layer hit rates are reported separately; speed and latency are secondary observations.', '', f"{len(runs)} live runs; {len(manifest['sample_ids'])} paired inputs; one repetition.", '',
        '| Policy | Decode hit rate | Prefill hit rate | Mean decode tokens/s | Mean decode ms/token |',
        '| --- | ---: | ---: | ---: | ---: |']
    for policy, data in aggregate.items():
        lines.append(f"| {policy} | {data['phases']['decode']['rate']:.4%} | {data['phases']['prefill']['rate']:.4%} | {data['means']['decode_tps']:.3f} | {data['means']['decode_ms_per_token']:.3f} |")
    lines += ['', '| Comparison | Decode hit delta (pp) | Paired bootstrap 95% CI (pp) |', '| --- | ---: | --- |']
    for name, value in comparisons.items():
        lines.append(f"| {name} | {value['delta_pp']:+.4f} | {value['ci95_pp']} |")
    lines += ['', '| Dataset | Hotness decode hit rate | Fate-hotness decode hit rate | Fate-ARC decode hit rate |', '| --- | ---: | ---: | ---: |']
    for dataset, rates in datasets.items():
        lines.append(f"| {dataset} | {rates['hotness']:.4%} | {rates['fate-hotness']:.4%} | {rates['fate-arc']:.4%} |")
    lines += ['', 'Bootstrap: 10,000 paired sample draws, seed 0. Dataset decode rates are in summary.json.',
        'Output, routing, every decode replacement and rank counter conservation were verified.',
        'The first four streamed MoE layers are favored; this boundary is an experimental assumption, not calibrated for this model.',
        'Both Fate variants use the same byte budget and per-layer capacities. Hotness retains uniform allocation.',
        f"All policies use a common prefill wave capacity of {manifest['wave_capacity']} experts to avoid capacity-dependent floating-point reduction order changing router selections. This is the recorded execution override relative to the reference experiment; inference CLI arguments are otherwise unchanged.",
        'Minimum capacities preserve the existing prefill wave execution requirements. Wave preload admissions do not count as real ARC accesses.',
        'The OS page cache was not cleared. Throughput is an observation of this procedure, not an isolated causal effect of replacement.',
        'This is section 4.4 cache management only: no cross-layer prediction, prefetch or adaptive quantization.', '']
    (target / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(runs=len(runs), aggregate=aggregate, comparisons=comparisons), indent=2))


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('action', choices=('smoke', 'evaluate', 'report'))
    ap.add_argument('--reference', type=pathlib.Path, default=REFERENCE)
    ap.add_argument('--root', type=pathlib.Path, default=pathlib.Path('experiments/moe-fate-v2'))
    ap.add_argument('--exe', type=pathlib.Path, default=pathlib.Path('build/bin/Release/llama-cli.exe'))
    args = ap.parse_args()
    root, reference, exe = args.root.resolve(), args.reference.resolve(), args.exe.resolve()
    folder = 'fate-smoke' if args.action == 'smoke' else 'fate-evaluation-108'
    if args.action == 'report':
        compare(root, folder)
        return
    ref_manifest = reference / 'infinity-evaluation-108-manifest.json'
    ref = json.loads(ref_manifest.read_text(encoding='utf-8'))
    if ref['slots'] != 180 or ref['tokens'] != 256 or ref['repetitions'] != 1 or len(ref['sample_ids']) != 36:
        raise ValueError('reference does not describe the expected 108-run experiment')
    source = reference / 'samples.json'
    if m.digest(source) != ref['samples_sha256']:
        raise ValueError('reference inputs changed')
    samples = {s['id']: s for s in json.loads(source.read_text(encoding='utf-8'))}
    ids = ref['sample_ids']
    if collections.Counter(samples[i]['dataset'] for i in ids) != {'gsm8k': 18, 'arc': 18}:
        raise ValueError('reference dataset mix changed')
    if args.action == 'smoke':
        ids = [i for i in ids if i.endswith(('0000', '0004'))]
    ref_status = reference / 'infinity-evaluation-108' / f'{ids[0]}-hotness-0.status.json'
    command = json.loads(ref_status.read_text(encoding='utf-8'))['command']
    model = command[command.index('-m') + 1]
    if m.digest(model) != ref['model_sha256']:
        raise ValueError('reference model changed')
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'samples.json').exists() and m.digest(root / 'samples.json') != m.digest(source):
        raise ValueError('local inputs changed')
    if not (root / 'samples.json').exists():
        (root / 'samples.json').write_bytes(source.read_bytes())
    tokens = 8 if args.action == 'smoke' else 256
    normalized = normalize(command)
    normalized[normalized.index('-n') + 1] = str(tokens)
    manifest = dict(fate_version=VERSION, policies=POLICIES, shallow_layers=4, slots=180, tokens=tokens,
        sample_ids=ids, repetitions=1, normalized_command=normalized, model_sha256=ref['model_sha256'],
        binaries=m.binary_manifest(str(exe)), samples_sha256=m.digest(source), reference_manifest_sha256=m.digest(ref_manifest),
        prompts={i: hashlib.sha256(samples[i]['prompt'].encode('utf-8')).hexdigest() for i in ids})
    manifest.update(plan_capacity(model))
    fixed_wave = str(manifest['wave_capacity'])
    if os.environ.get('LLAMA_MOE_STREAM_WAVE_CAP', fixed_wave) != fixed_wave:
        raise ValueError('incompatible common wave capacity override')
    os.environ['LLAMA_MOE_STREAM_WAVE_CAP'] = fixed_wave
    manifest['environment'] = {k: v for k, v in os.environ.items() if k.startswith('LLAMA_MOE_STREAM_')}
    manifest['sources'] = {str(path): m.digest(path) for path in [pathlib.Path(__file__), pathlib.Path(m.__file__), pathlib.Path(reports.__file__)]}
    manifest = json.loads(json.dumps(manifest))
    manifest_path = root / (folder + '-manifest.json')
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if old != manifest:
            raise ValueError('experiment configuration changed; use a new root')
    else:
        m.save(manifest_path, manifest)
    for i, sample_id in enumerate(ids):
        offset = i % 3
        for policy in POLICIES[offset:] + POLICIES[:offset]:
            run_request(root, exe, model, samples[sample_id], policy, folder, manifest)
        verify_triplet(root, folder, sample_id)
    compare(root, folder)


if __name__ == '__main__':
    main()
