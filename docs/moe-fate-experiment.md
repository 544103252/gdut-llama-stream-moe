# Fate 4.4 expert cache experiment

Private fork experiment. The default remains `hotness`; `chain` and
`moe-infinity` retain their existing configuration and behavior.

## Policies

```
--moe-stream-cache 180s --moe-stream-policy fate-hotness --moe-stream-shallow-layers 4
--moe-stream-cache 180s --moe-stream-policy fate-arc --moe-stream-shallow-layers 4
```

Both Fate policies use shallow-favoring capacity allocation. `fate-hotness`
retains the original layer-local hotness/LRU replacement; `fate-arc` uses
layer-local ARC. This implements section 4.4 only, without cross-layer gate
prediction, cross-layer prefetching, or dynamic quantization. Four is the
number of streamed MoE layers, not a model-specific calibrated boundary.
All clients must be rebuilt after the model parameter struct change.

For an explicit slot count, the byte budget equals that count times the sum
of the actual bytes per expert over all streamable layers. A byte budget is
used directly when no slot count is supplied. Each layer first receives
`min(num_experts, 3 * top_k)` slots to preserve the existing prefill wave
requirements. Remaining bytes fill the shallow layers in order, then add
one slot per deep layer in repeated layer-order rounds. Integer leftovers
are assigned in layer order; no allocation may exceed the byte budget.
Expert sizes include all streamed weight tensors and can differ by layer.
Insufficient minimum capacity fails explicitly. Tensor registration is
checked against the capacity plan before allocating buffers.

The `3 * top_k` reservation is an adaptation to this branch's execution
architecture: unlike a literal zero-capacity interpretation of the paper,
deep layers always retain sufficient execution capacity. Existing budgets
and allocation behavior are unchanged for the three original policies.

ARC maintains T1/T2 resident queues, B1/B2 ghost queues and integer target p.
B1 hits increase p, B2 hits decrease it, using integer queue-size ratios
with a minimum adjustment of one. Ties follow queue order. Candidate search
skips active and LOADING slots, then tries the other queue. If both queues
are blocked, the existing load-completion wait is used. Planning uses a
copy of ARC state and commits it only when the complete plan succeeds.

Real routing references update ARC in token/router-position order within
each execution wave. Prefill still executes the existing first-use expert
waves; references in different waves therefore follow wave execution order.
A speculative next-wave admission is recorded as pending in T1 and does not
adapt p or count as a real reference. Its first demand access clears pending
without promoting it to T2; subsequent references promote normally. Ghosts
contain expert IDs only, with bounded metadata and no retained weights.

Both new policies use CPU cache planning, including when optional GPU
planning is requested; model computation still uses the configured backend.
Rank histograms retain their hotness/LRU interpretation for all policies,
including ARC. Shallow caches above 180 slots use the existing overflow bin.
Rule time includes ARC planning and demand accesses, but is not a complete
cross-policy CPU overhead benchmark.

## Reference-matched comparison

Run from the repository root with Python and numpy available:

```powershell
./build/bin/Release/test-moe-markov.exe
python scripts/test_moe_markov.py
python scripts/test_moe_infinity.py
python scripts/test_moe_fate.py
python scripts/moe_fate.py smoke
python scripts/moe_fate.py evaluate
python scripts/moe_fate.py report
```

The reference defaults to
`D:\another_project\llama-stream-moe-main\experiments\moe-markov`.
`--reference`, `--root`, and `--exe` select alternative locations. Reference
files are read only. No network acquisition or calibration is needed.

Evaluation reuses the exact 36 sample IDs from the reference
`infinity-evaluation-108-manifest.json`, with 18 GSM8K and 18 ARC inputs.
It runs hotness, fate-hotness and fate-arc once per input, rotating order
by input index, for 108 fresh-process runs. The generation limit is 256,
with normal EOS termination. Model, prompt text, context, backend, seed,
temperature, batch sizes and I/O threads match the reference commands.
Smoke uses inputs 0000 and 0004 per dataset and eight generated tokens, in a separate
folder; it is not included in the formal 108-run count.

The local root defaults to `experiments/moe-fate-v2`. Manifests pin the
reference manifest, model and input hashes, executable/DLL hashes, scripts,
streaming environment overrides, budget and expected capacity table before
inference. The runtime capacity snapshot must agree with the independent
GGUF metadata plan. Completed runs resume only with unchanged configuration
and intact prompt, stdout, stderr and trace hashes. A changed configuration
requires a new root. Failed runs are not counted as completed.

## Common prefill partition

Different cache capacities otherwise change prefill wave boundaries and the
floating-point reduction order. A diagnostic run found equal generated text
but one differing prefill expert ID on GSM8K input 0004. That batch was stopped
and excluded. The final comparison sets `LLAMA_MOE_STREAM_WAVE_CAP=81` for all
three policies, including fully cached shallow layers, so expert wave
partitions and accumulation order agree. The value is derived from the
smallest Fate layer capacity and top-k, and is pinned in the manifest.
Without this optional override, normal execution retains its original wave
selection. The reference inference CLI arguments are unchanged; this extra
execution control is the explicit deviation required for a paired comparison
with identical routes. Earlier diagnostic runs are not formal results.

## Primary metric and verification

The primary metric is the **decode resident expert cache hit rate**:
`hit / (hit + miss)` over routed expert requests. LOADING is a miss. Prefill
uses the callback-entry residency snapshot for every token/expert reference,
consistent with the existing counters. Aggregate rates sum request counts;
they are not an unweighted mean of per-run percentages.

The report checks all paired outputs and routes, independently replays every
decode replacement (including ARC from the initial queue snapshot), and
conserves prefill/decode rank counts against live counters. Formal reporting
requires all 36 triplets. Results include per-dataset and per-layer hit rates,
paired differences versus hotness and between Fate variants, and sample-level
paired bootstrap 95% intervals (10,000 draws, seed 0).

Outputs in `fate-evaluation-108`:

- `report.md` and `summary.json`: hit rates, paired confidence intervals and secondary metrics.
- `samples.csv`: per-run phase counts, hit rates, loads, bytes, rule/stall time, throughput and latency.
- `per_layer_180_counts.csv`: aggregated per-layer/phase rates and rank counts.
- `per_run_layer_180_counts.csv`: per-run layer/phase counts and ranks.
- Original prompts, stdout, stderr, JSONL traces and status/artifact hashes.

The OS page cache is not cleared. Speed and latency are secondary observations
of this procedure, not evidence isolating the cache policy's causal effect.
Historical experiment results are not mixed into the paired comparison.
