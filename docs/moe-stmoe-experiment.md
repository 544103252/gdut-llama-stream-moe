# ST-MoE 4.2 prediction-only experiment

Private experiment implementing the CCT and HT strategy in section 4.2 of
ST-MoE. Predictions never issue weight loads or modify cache replacement.
The experiment measures routing prediction accuracy, not latency hiding or
prefetch performance. Existing cache hit counters keep their original meaning.

## Algorithm

Calibration uses only decode routes from independent training samples. For
each adjacent MoE layer pair, count every source/target expert co-activation
and retain the target Top-K per source expert. Ties use ascending expert ID.
Unobserved source rows remain empty. Initial confidence is 2.

At source-layer routing, sum CCT confidence over all selected source experts,
add 2 for each expert in the target layer's previous-token HT, and predict
every expert with score >= 2. The set is deduplicated and is not capped at K.
This follows the paper's prose and equation (1); Algorithm 2's HT assignment
would instead overwrite spatial scores and is not used here.

At target-layer routing, evaluate the saved prediction before updating CCT.
For each selected source row, actual candidates increment confidence up to 3;
absent candidates decrement a positive confidence. An absent candidate whose
confidence was already zero is replaced by the lowest-ID actual target expert
not currently in that row, with confidence 2. Process stored entries in order
and recompute eligibility after each replacement. With no eligible replacement,
retain the entry at zero. Empty calibration rows remain empty.

HT is replaced with actual routes only after all layers for a decode token
have been observed. Prefill does not update CCT or HT. The first decode token
has an empty HT and is included in statistics. The first MoE layer is excluded
from target statistics because it has no source layer.

## Runtime interface and restrictions

`--moe-stream-stmoe-cct PATH` enables prediction-only testing independently of
`--moe-stream-policy`. `--moe-stream-trace PATH` also records `stmoe_predict`
and `stmoe_evaluate` events with source, target, decode step, predicted IDs,
actual IDs and intersection. Normal routing observations are retained.
`stmoe_summary` log lines provide per-target and aggregate counts and ratios;
target -1 denotes the aggregate. Zero denominators are JSON null.

Only a single sequence, one-token decode and one sample per process are
supported. Use `--no-warmup`; do not use prompt-cache/session restoration,
speculative decoding, multi-context execution, or server request reuse.
All routed MoE layers must be streamed and observed in order. Non-MoE layers
may lie between consecutive MoE layers. Missing layers, duplicate routes and
K mismatches fail rather than silently reducing the evaluated layer set.

GPU routing planning, continuous planning and shadow planning are disabled
when prediction testing is enabled. Expert computation can still use Vulkan
or another GPU backend. Decode synchronizes the scheduler to validate complete
observations, so throughput here must not be interpreted as prefetch speedup.

The model parameter struct adds a `moe_stream_stmoe_cct` pointer; rebuild all
clients. Calibration JSON version 1 contains `model_sha256`, `layer_count`,
`k`, ordered `layers` and `pairs`. Each pair contains `source`, `target`,
`experts` and one `rows` array per source expert. Each row is empty or contains
K `[expert_id, confidence]` entries. The runtime validates dimensions, ranges,
unique candidates and the actual GGUF SHA-256. Split GGUF is unsupported.

## Reproduce the small validation

Build the existing Release/Vulkan configuration with Visual Studio and run:

```powershell
cmake --build build --config Release --target llama-cli test-moe-markov -j 8
& ./build/bin/Release/test-moe-markov.exe
python -m unittest discover -s scripts -p test_moe_stmoe.py
python scripts/moe_stmoe.py calibrate
python scripts/moe_stmoe.py evaluate
python scripts/moe_stmoe.py report
```

Python calibration requires numpy and PyYAML for the repository GGUF reader;
the predictor/replay itself uses only the standard library. Defaults use the
local Qwen3.5-35B-A3B Q4_K_M model, the existing `moe-fate-v2/samples.json`,
one training sample and one test sample per dataset, 16 generated tokens and
180 cache slots per layer. Override `--model`, `--exe`, `--samples`,
`--per-dataset`, `--tokens`, `--slots`, `--gpu-layers`, and `--root` as needed.
Use the same settings for calibration and evaluation.

Samples are a JSON array of objects with `id`, `dataset`, `split` (`train` or
`test`) and `prompt`. The script selects the first N rows per dataset/split.
It rejects calibration/test overlap by both sample ID and prompt hash.

Results live under `experiments/moe-stmoe`, separate from previous experiments.
Manifests pin model, executable/DLLs, script, settings, sample identities and
calibration hash. Completed runs resume only if artifact hashes also match.
Use a new root after changing any pinned input. Each sample runs in a fresh
process. Evaluation runs a paired hotness baseline and prediction-enabled
hotness run, requires identical sampled token IDs, byte-identical generated
output and actual routes, and
checks every prediction, evaluation and summary against independent replay.
Debug logging records every sampled token ID, including the final token.

`summary.json` contains global, dataset, target-layer and sample counts:

- Coverage: intersection / actual experts (primary prediction hit rate).
- Precision: intersection / predicted experts.
- All-hit rate: fully covered token-layer observations / observations.
- Mean predicted: predicted experts / observations.
- Raw predicted, actual, hit, miss, false-positive and all-hit counts.

`report.md` summarizes the smoke run. A tiny calibration/evaluation validates
correctness only; it does not establish representative prediction accuracy.
The OS page cache is not cleared, and no actual expert prefetch is performed.

## Verified local smoke result

The run under `experiments/moe-stmoe-smoke` used the defaults above: two
independent training samples (30 decode tokens), two test samples and 39
adjacent MoE layer pairs. It evaluated 1,170 token-layer observations:

| Metric | Result |
|---|---:|
| Actual expert requests | 9,360 |
| Predicted experts | 24,689 |
| Intersection | 6,467 |
| Coverage | 69.09% |
| Precision | 26.19% |
| All-hit rate | 7.52% |
| Mean predicted experts | 21.10 |

Sampled token IDs, generated text and actual routing matched the paired
baseline. Every online prediction and counter matched offline replay.
Replacement events, demand-load counts and transferred bytes also matched
the baseline. These results validate the implementation with very small
calibration data; they are not a full dataset evaluation.

Recheck the saved artifacts without inference:

```powershell
python scripts/moe_stmoe.py report --root experiments/moe-stmoe-smoke
```

## Reference 36-input comparison

To use exactly the same GSM8K and ARC test inputs as the
`llama-stream-moe-main` 108-run reference experiment:

```powershell
python scripts/moe_stmoe_reference.py evaluate
python scripts/moe_stmoe_reference.py report
```

This driver pins the reference sample and model hashes, imports the 200
independent training traces after checking their recorded SHA-256 hashes,
and builds a fresh spatial CCT from those routes. The original calibration
binary provenance is retained separately from the current evaluation binary.
It does not import the reference Markov replacement matrix as a CCT.

Evaluation uses the first 18 test inputs per dataset, 256 generated tokens
maximum, 180 slots, temperature 0, seed 0, and one repetition. Each input runs
both current hotness and current hotness with ST-MoE prediction, for 72 live
runs. Pair order alternates. Warmup is disabled and debug token logging is
enabled for verification. Results are separate from the initial smoke run,
under `experiments/moe-stmoe-reference-36`.

The final report verifies sampled tokens, output, routing, replay, resident
cache counters, replacement events, load counts and transferred bytes.
`samples.csv`, `inference.csv`, `dataset_layers.csv` and `summary.json`
provide prediction metrics and observed timing/cache metrics. Timing includes
CPU observation, synchronization, debug logging and model hashing; it does
not measure any real prefetch speedup.
