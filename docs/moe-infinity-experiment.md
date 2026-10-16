# Per-layer MoE-Infinity replacement experiment

This private experiment adapts the request EAM matching and replacement rule
from sections 4.5/4.7 and appendix B.1 of MoE-Infinity. It is not a full
reproduction: cache slots remain fixed per layer, and prediction-based
prefetching and online learning are not implemented. The layer multiplier
in Algorithm 1 is constant within each victim candidate set and does not
affect its ordering.

The default remains `hotness`. `chain` retains its existing scoring and
configuration. To enable the new policy:

```powershell
./build/bin/Release/llama-cli.exe -m MODEL.gguf --moe-stream-cache 180s --moe-stream-policy moe-infinity --moe-stream-eam experiments/moe-markov/eam.bin
```

Every single-token decode forward clears an iteration EAM. Each streamed
layer adds one count for every selected expert ID, independent of router
weights. The partial iteration EAM is compared with complete historical
decode request EAMs using flattened cosine similarity. The nearest entry
is selected; equal similarities keep the earliest collection index. This
single-match choice resolves the paper's unspecified number of matches.
The selected row is normalized to probabilities. The lowest probability
resident expert is evicted, with lowest expert ID breaking ties. No positive
match or an empty predicted row uses the original hotness/LRU rule instead.
Empty slots win first; LOADING and active slots remain protected.

Prefill and wave preloading use hotness. Only a single sequence and one
context per streamed model are supported. No request begin/end API is
needed because the collection is read-only and the query is iteration-level.
All clients must be rebuilt after adding `moe_stream_eam` to model parameters.

## Historical collection

Reuse the acquisition and calibration collection steps documented in
`moe-markov-experiment.md`, then run:

```powershell
python scripts/moe_infinity.py calibrate
python scripts/moe_infinity.py smoke
python scripts/moe_infinity.py evaluate
python scripts/moe_infinity.py report
```

`--root`, `--model`, and `--exe` select inputs; `--capacity` defaults to
1024. Calibration requires exactly one successful trace for every training
sample in `samples.json`. Only decode activations are accumulated. The
collection is filled in sorted trace filename order; once full, the entry
with the greatest cosine similarity to the incoming request is replaced.
Ties choose the earliest collection index. Counts are uint32 and overflow
is rejected. Test requests never update the collection.

The little-endian binary layout is eight bytes `MOEEAM01`, uint32 layer
count, uint32 expert count, uint32 entry count, 32 raw model SHA-256 bytes,
then uint32 counts in entry/layer/expert order. The runtime rejects invalid
versions, dimensions, entry counts outside 1..65536, empty matrices,
truncation, trailing data, model digest mismatch, and split GGUF. Individual
zero rows are legal and activate fallback. The sibling `eam.json` records
the model digest and calibration trace digests.

## Verification and reports

```powershell
./build/bin/Release/test-moe-markov.exe
python scripts/test_moe_markov.py
python scripts/test_moe_infinity.py
```

The experiment runs all three policies with identical arguments and rotates
their order. `--slots` defaults to 180 and `--tokens` to 256. Smoke uses two
test requests per dataset and one repetition; evaluation uses all test
requests and three repetitions. For a shorter functional smoke, use
`--tokens 8`; use a separate root for a different configuration or build.

Results are written to `infinity-smoke/summary.json` or
`infinity-evaluation/summary.json`. Reports verify identical output and
routing, independently replay every decode replacement, and retain per-run
prefill/decode hit rates, bytes, loads, rule/stall time, throughput and
decode milliseconds per token. `eam` trace events record the selected entry
and fallback reason before the corresponding routing observation. Existing
hotness rank histogram semantics are unchanged.

The OS page cache is not cleared. Both predictive policies hash the full
model at startup; hotness does not. Throughput measurements are observations
under this procedure, not evidence isolating replacement performance. A
short smoke establishes functional agreement, not a performance improvement.

## Bounded three-policy comparison

To run one repetition on the first 18 test inputs from each dataset (36 inputs,
108 live runs), using a separate folder and the original 256-token limit:

```powershell
python scripts/moe_infinity.py evaluate --limit-per-dataset 18 --repetitions 1 --folder infinity-evaluation-108
python scripts/moe_infinity_report.py --folder infinity-evaluation-108
```

The manifest pins sample IDs, input-file checksum, build hashes, model digest,
cache slots, generation limit and calibration artifacts. All three policies
are rerun with the same build; no old inference results are mixed into this
comparison. Defaults without these options retain the original full evaluation.
The report adds paired sample bootstrap intervals (10,000 draws, seed 0),
per-dataset decode hit rates, per-run CSV and runtime per-layer rank counts for
prefill and decode. Rank hits and misses must conserve the phase counters.
The balanced 18:18 input mix differs from the earlier 50:10 two-policy run;
use the baselines from this bounded run when assessing the new strategy.

## GPU planner compatibility

The chain and moe-infinity policies, and any routing trace, use CPU remap even
when GPU decode, continuous decode or shadow verification is requested through
environment variables. A startup message reports this fallback. Ordinary
hotness without tracing retains the branch's optional Vulkan GPU decode path.
CPU route/rank counters are unavailable during GPU decode; use the backend
profile counters for that mode, or disable GPU decode for policy comparisons.
Rebuild all clients after migrating the model parameter fields.
