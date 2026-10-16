# Markov expert cache experiment

Private experiment, based on markov-moe-cache-final v1.0.0. No upstream
submission is prepared. The default policy remains `hotness`.

`--moe-stream-policy chain --moe-stream-markov PATH` enables weighted first-order
Markov replacement for single-token decode. Multi-token prefill and wave
preloading use the existing policy. Empty slots win first; active and loading
slots cannot be evicted; equal scores select the lowest expert ID.
Both policies protect the complete active set before handling misses. The
hotness baseline includes this shared protection fix and is built from this
workspace, not the external original executable.

The binary matrix is little-endian: eight bytes `MOEMKV01`, uint32 layer count,
uint32 expert count, 32 raw GGUF SHA-256 bytes, then float32 probabilities in
layer/source/destination order. Every row must sum to one within 1e-4.
Split GGUF files are currently rejected. Metadata is in a sibling JSON file;
the runtime validates the dimensions, probabilities and actual GGUF digest.
The model parameter struct has three new pointer fields; rebuild all clients.
Windows hashing uses the system CNG SHA-256 provider; other platforms use the
SHA-256 implementation already shipped in this repository. Every process still
hashes the actual GGUF bytes; no timestamp-based digest shortcut is used.

`--moe-stream-trace PATH` writes JSONL observations before demand reservations.
The sample ID is the trace filename recorded in the run manifest. Layers have
independent step counters. Calibration uses adjacent decode rows within one
sample and layer, with weighted outer-product counts and smoothing 0.1.

The `moe_cache` counters count each routed expert request against the residency
snapshot at callback entry. LOADING is a miss and also appears in `loading`.
`all_hit` counts token-layer observations whose experts are all resident.
Prefill counters use the start-of-ubatch snapshot. Legacy counters retain their
existing interpretation, including loading entries as hits. Loads, bytes and
stall time are totals across phases; rule time excludes tracing and I/O.
`cold` counts each never-before-requested expert once per layer, including in
multi-token prefill. Trace initialization snapshots and replacement events let
the CPU replay check every decode eviction, not just aggregate rates.

Build with Visual Studio 18 2026, Vulkan enabled, Release and eight jobs.
Run `test-moe-markov` before collecting data. With Python, numpy, pandas,
pyarrow and PyYAML installed, execute these commands from the repository root:

```powershell
python scripts/moe_markov.py prepare
python scripts/moe_markov.py collect
python scripts/moe_markov.py calibrate
python scripts/moe_markov.py smoke
python scripts/moe_markov.py evaluate
```

Artifacts live in `experiments/moe-markov`. Runs resume only completed samples.
Do not reuse that directory after changing the executable, model or inputs;
choose a new `--root` and repeat acquisition and calibration. Calibration and
evaluation samples are disjoint. Test data never updates the matrix.

Every request starts a fresh process. Smoke uses two test samples per dataset;
evaluation uses 50 per dataset with three paired repetitions and alternating
policy order. Decode generation stops at EOS or 256 generated tokens. Output
and routing differences fail the comparison. Reports include aggregate hit
rates and a paired sample bootstrap with 10,000 draws, seed 0. A CI crossing
zero means insufficient evidence, even if the observed difference is positive.

The OS file cache is not cleared between processes. Chain validation reads the
whole GGUF before inference, so process wall time includes this work and decode
throughput can also reflect OS cache warming. Report throughput as an observed
measurement, not an isolated attribution to replacement. The hit-rate comparison
is checked against identical routing and exact replacement replay.

`scripts/run_moe_markov.ps1` runs the complete build, tests and experiment in
order; use `-Python` and `-CMake` for explicit runtime paths, or `-SkipBuild` to
use an already verified build. Stage logs are saved under the experiment root.
After intentionally stopping early, run `python scripts/moe_markov.py report
--allow-partial` to report only completed pairs. The report records excluded
unpaired requests and the actual sample and repetition counts.

## Fixed cache rank buckets

Each streamed layer keeps 180 rank counters separately for prefill and decode.
Only RESIDENT experts rank: descending decayed route hotness, descending last
use, then ascending expert ID. Ranking precedes the callback's hotness/LRU
updates, decay, reservations and demand loads. Every token/expert pair counts,
including repeated experts within a micro-batch; prefill uses the real entry
snapshot shared by that micro-batch. LOADING and absent experts count as misses,
with LOADING also reported separately. Empty slots do not rank. Unused buckets
stay zero; a cache larger than 180 records hits beyond rank 180 as overflow.
The histogram never changes routing, hotness, LRU or eviction decisions.

`llama_moe_stream_print_stats` emits `moe_rank` lines with the layer, phase,
policy, actual slot count, misses, loading misses, overflow and 180 counts.
`python scripts/moe_rank_report.py` reconstructs decode rankings from the
completed live runs listed in `evaluation/samples.csv`, checks every replacement
event and conserves live hit/miss totals. Results are under `evaluation/ranks`.
Old traces cannot recover prefill rankings, so retrospective results cover
decode only; future runtime logs cover both phases. Cumulative fractions are
provided relative to all requests and relative to resident hits.
The offline rank report uses numpy and matplotlib; neither is a runtime
dependency of llama. Figures include the overall rank distribution and a
per-layer cumulative heatmap.

## GPU planner compatibility

The chain and moe-infinity policies, and any routing trace, use CPU remap even
when GPU decode, continuous decode or shadow verification is requested through
environment variables. A startup message reports this fallback. Ordinary
hotness without tracing retains the branch's optional Vulkan GPU decode path.
CPU route/rank counters are unavailable during GPU decode; use the backend
profile counters for that mode, or disable GPU decode for policy comparisons.
Rebuild all clients after migrating the model parameter fields.
