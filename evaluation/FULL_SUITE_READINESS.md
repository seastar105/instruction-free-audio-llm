# Full 6K/20K evaluation readiness

Audit date: 2026-08-26. This plan covers the complete selected public suites for
both projector checkpoints. Generation uses one persistent `vllm serve` process
per checkpoint and HTTP requests; scoring is a later, independent stage.

## Exact selection and size

| Benchmark | Selected subsets | Rows | Snapshot bytes |
| --- | ---: | ---: | ---: |
| VoiceBench | 9 | 13,313 | 9,739,727,179 |
| standalone MMSU | 1 | 5,000 | 1,466,894,219 |
| MMAU public test-mini | 1 | 1,000 | 1,213,407,227 |
| MMAU-Pro | 1 | 5,305 | 47,507,201,472 |
| MMAR | 1 | 1,000 | 2,984,276,670 |
| VoiceBench-JA | 4 | 1,694 | 1,256,113,802 |
| KVoiceBench | 9 | 7,306 | 4,639,811,428 |
| KMMAU | 9 | 2,204 | 4,489,373,187 |
| **Total per checkpoint** | **35 subset runs** | **36,822** | **73,296,805,184** |

The two checkpoints therefore require 73,644 generation rows. The byte total
includes MMAU-Pro's 47,506,239,410-byte `data.zip` and MMAR's
2,983,578,514-byte `mmar-audio.tar.gz`. It excludes small README/config files.

Locally present at audit time:

- VoiceBench `mmsu`: 3,074 rows, 2,140,740,285 bytes.
- MMAU test-mini: 1,000 rows, 1,213,407,227 bytes.

Approximately 69,942,657,672 bytes (65.14 GiB) are still missing before
extraction. Reserve substantially more space for the two extracted archives and
the Hugging Face cache; 150 GiB free is a conservative pre-download check.

The VoiceBench selection is its nine standard reported tasks:
`alpacaeval_full`, `commoneval`, `wildvoice`, `sd-qa`, `mmsu`, `openbookqa`,
`bbh`, `ifeval`, and `advbench`. The Hub also contains duplicate/diagnostic
configurations (`alpacaeval`, `mtbench`, and speaker variants); these are not
part of the standard aggregate and are intentionally excluded.

MMAU means the complete reproducible **public labeled** `MMAU-test-mini`
(1,000 rows). The official full MMAU test has 10,000 rows with held-out labels
and is evaluated through the benchmark submission service. It cannot be
reproduced by the local scorer without submission access/private labels.

## Pinned sources

Dataset and scorer commits are in `benchmarks.yaml`. All eight scorer checkouts
are present under `evaluation-scorers/` at those detached commits. Dataset
downloads must continue through `hf` and the immutable revisions in the
manifest:

```bash
source .venv-evaluation/bin/activate
audio-lfm-eval list  # inspect the pinned subsets in benchmarks.yaml

# Run once for every subset printed by `audio-lfm-eval list`.
audio-lfm-eval download --benchmark voicebench --subset alpacaeval_full --output-root evaluation-data

# After the large archive downloads finish:
audio-lfm-eval unpack --benchmark mmau-pro --data-root evaluation-data
audio-lfm-eval unpack --benchmark mmar --data-root evaluation-data
```

`download` deliberately accepts one subset at a time and calls `hf download`
with two workers to limit WSL host memory. Repeated downloads resume from the
Hub cache.

## Checkpoint export and generation

The 6K checkpoint exists, but its export did not exist at audit time. The 20K
checkpoint/export must wait for training to finish.

```bash
audio-lfm export-vllm \
  --config configs/wavcaps_paraspeech_whisper_small_lfm2_expanded_3epoch.yaml \
  --checkpoint runs/wavcaps-paraspeech-whisper-small-lfm2-expanded-3epoch-16k/checkpoint-00006000 \
  --output-dir exports/wavcaps-paraspeech-lfm2-6k-vllm

audio-lfm export-vllm \
  --config configs/wavcaps_paraspeech_whisper_small_lfm2_expanded_3epoch.yaml \
  --checkpoint runs/wavcaps-paraspeech-whisper-small-lfm2-expanded-3epoch-16k/checkpoint-00020000 \
  --output-dir exports/wavcaps-paraspeech-lfm2-20k-vllm
```

Run every benchmark through one server lifetime for each export. Separate
output roots prevent checkpoint results from overwriting each other:

```bash
source .venv-evaluation/bin/activate
export VLLM_PLUGINS=audio_lfm2

audio-lfm-eval run-suite \
  --model-export exports/wavcaps-paraspeech-lfm2-6k-vllm \
  --model-name audio-lfm-6k \
  --data-root evaluation-data \
  --output-root evaluation-runs/6k \
  --benchmark voicebench --benchmark mmsu --benchmark mmau \
  --benchmark mmau-pro --benchmark mmar --benchmark voicebench-ja \
  --benchmark kvoicebench --benchmark kmmau

audio-lfm-eval run-suite \
  --model-export exports/wavcaps-paraspeech-lfm2-20k-vllm \
  --model-name audio-lfm-20k \
  --data-root evaluation-data \
  --output-root evaluation-runs/20k \
  --benchmark voicebench --benchmark mmsu --benchmark mmau \
  --benchmark mmau-pro --benchmark mmar --benchmark voicebench-ja \
  --benchmark kvoicebench --benchmark kmmau
```

Generation is resumable by sample ID. Do not run either server while training
holds the GPU.

### Audio preflight

The current plugin accepts three ordered audio items and chunks each logical
item into independent 30-second Whisper windows without dropping its final
partial window. Before the full run, retain the configured 32K context and
720-second per-item ceiling: MMAU-Pro has 456 multi-audio rows (430 with two
items and 26 with three), while KMMAU reaches 664.73 seconds. Generation's
preflight sums projected tokens across every item and fails before serving if a
complete row cannot fit; it never crops or silently skips audio.

Observed duration facts useful for validating that preflight:

- MMAU-Pro: up to 10 minutes; 1,219 rows are labeled long/ultra-long and 456
  rows contain multiple audio items.
- MMAR: exactly 3 of 1,000 rows exceed 30 seconds; maximum 56 seconds.
- standalone MMSU: maximum 35.915 seconds.
- VoiceBench-JA maxima by subset: 36.468, 36.468, 32.907, and 78.215 seconds.
- KVoiceBench maxima include 163.76 seconds (`wildvoice-test`/`mmsu`) and
  37.76 seconds (`bbh-test`).
- KMMAU maxima include 664.73 seconds (`number_of_speakers`) and about 600
  seconds in `fact_extraction`/`topic_summary`.

The 47.5 GB MMAU-Pro archive must be present before its exact aggregate-context
preflight can be completed from audio headers.

## Independent scoring

Create the lightweight scorer environment after all pinned checkouts exist:

```bash
# Set this to 0 for the smaller non-Japanese environment. Full-suite scoring
# needs the larger Japanese FlexEval stack, including M-IFEval dependencies.
INSTALL_FLEXEVAL=1 bash evaluation/scripts/create_scorer_env.sh

# Separate GPU-heavy local-model scorer environment.
bash evaluation/scripts/create_mmau_pro_scorer_env.sh
```

The scorer environment uses `uv`, installs KEval from the pinned Raon-Eval
checkout, and includes VoiceBench's `qa-metrics`/NLTK imports. The current local
`.venv-evaluation-scorers` must be recreated because those executables were not
installed at audit time.

For each of the 35 generated subset directories and each checkpoint, first
print/inspect the exact plan, then add `--execute`:

```bash
source .venv-evaluation-scorers/bin/activate
audio-lfm-eval score \
  --benchmark voicebench --subset alpacaeval_full \
  --output-root evaluation-runs/6k \
  --scorer-root evaluation-scorers \
  --data-root evaluation-data \
  --execute
```

Each scoring directory receives `scoring_manifest.json`, containing dataset and
scorer commits, the full command, and the explicit judge provider/model without
the credential value. KEval also receives a locally materialized
`ground-truth.jsonl` from the pinned generation rows, avoiding its default
moving-branch Hub reload.

Scorer requirements:

| Suite/subsets | Scorer | External dependency |
| --- | --- | --- |
| VoiceBench open + SD-QA (7,919 rows/checkpoint) | official `api_judge.py`, 3 samples/request | `OPENAI_API_KEY`, `gpt-4o-mini` |
| VoiceBench remaining five | official programmatic evaluators | NLTK, `qa-metrics` import |
| standalone MMSU | `mmsu_evaluation.py` | CPU only |
| MMAU test-mini | `evaluation.py` | CPU only |
| MMAR standard answer accuracy | `code/evaluation.py` | CPU only |
| VoiceBench-JA Elyza + Spoken-Elyza (70 rows/checkpoint) | FlexEval ChatLLMScore | `OPENAI_API_KEY`, `gpt-4o-2024-08-06` |
| VoiceBench-JA M-IFEval + JamC-QA | FlexEval programmatic metrics | Japanese/FlexEval packages |
| KVoiceBench seven judge-backed subsets (6,514 rows/checkpoint) | pinned KEval | `OPENAI_API_KEY`, explicit paper-comparable `gpt-5.4` |
| KVoiceBench AdvBench + IFEval | pinned KEval programmatic metrics | CPU only |
| all KMMAU (2,204 rows/checkpoint) | pinned KEval semantic yes/no judge | `OPENAI_API_KEY`, explicit `gpt-5.4` |
| MMAU-Pro | official comprehensive scorer | local GPU models and NLTK data |

MMAU-Pro's 625 open-ended rubric prompts are judged together with
`Qwen/Qwen2.5-7B-Instruct` through batched vLLM. The 87
instruction-following rows use programmatic NLTK checks. Closed-ended
NV-Embed evaluation is deliberately omitted and recorded as such in both the
scoring manifest and result JSON; it is not included in the evaluated-row
denominator. This phase must not overlap generation or training and does not
require an API key.

Every scorer uses the complete generated-row denominator. An unparseable model
answer is incorrect, and an unparseable judge verdict contributes zero; neither
case is filtered out or replaced with a random guess/default midpoint.

MMAR's optional rubric/reasoning evaluation is not the benchmark's standard
answer-accuracy path. If requested separately, it requires `OPENAI_API_KEY`,
hardcodes `gpt-4o-2024-11-20`, uses five raters, and requires reasoning fields
that the one-response generation contract does not currently produce.

VoiceBench SD-QA has a separate security limitation. Its official `qa_metrics.PEDANT`
package unconditionally downloads two pickle files from mutable GitHub URLs and
deserializes them with `joblib.load`. Those assets were not fetched during this
setup. The safe default runs the official GPT majority-vote portion and writes
`panda: null`, `complete_official_sdqa: false`, and an explicit omission reason.
The untouched official scorer reports both GPT and PANDA metrics, so full
SD-QA remains blocked unless a separately reviewed, hash-pinned PANDA path is
approved.

Supply-chain audit details (files were downloaded to `/tmp` for hashing/static
inspection only and were never deserialized):

- installed package: `qa_metrics==0.2.17`, upstream
  `https://github.com/zli12321/qa_metrics`; the repository publishes no Git
  tags, so package version 0.2.17 is not bound to an asset commit;
- resolved upstream `master` during the audit:
  `26623c6ca3313e7a58c48fd6ce2b8579eb0c742e`;
- `lr_classifier`: 753,600 bytes, Git blob
  `ed2403c19941e42c1d6f335c4ba7d1ab3dc3971b`, locally computed SHA-256
  `9c3dd5998745ece1cfb6cb16b953358f794e8716d63a08f3b212d8bc32a59b8b`;
- `tf-idf_vectorizer`: 2,966,139 bytes, Git blob
  `fd1bf7d90eb5ad6155d43c72bde9cab5b35e2bd2`, locally computed SHA-256
  `e886c9ba3ddb15a4b693b075dfe19cfad36dc74e7f8f195b8d9e98e989f2ba4f`.

GitHub's API publishes the immutable blob IDs and they matched local
`git hash-object`, but upstream does not publish SHA-256 checksums. Static
`pickletools` inspection found expected scikit-learn, NumPy, SciPy, and Joblib
constructors, then stopped at Joblib's embedded raw array blocks; therefore it
was not a complete opcode audit and is not sufficient to authorize loading.

Across both checkpoints, API-backed scoring covers 33,414 row evaluations;
VoiceBench additionally requests three judge completions per row. Confirm API
quota/cost before starting stage 2. Judge-backed scores are nondeterministic, so
record the scorer manifests and compare like-for-like judge versions.

## Validation status

Static validation does not start vLLM or touch the GPU:

```bash
UV_CACHE_DIR=.runtime/uv-cache uv run --project evaluation --extra dev ruff check evaluation
UV_CACHE_DIR=.runtime/uv-cache uv run --project evaluation --extra dev mypy evaluation/src
UV_CACHE_DIR=.runtime/uv-cache uv run --project evaluation --extra dev pytest -q evaluation/tests
```

At audit time, the isolated command passed 27 tests and skipped the two
Torch/plugin tests because Torch deliberately is not installed in the
lightweight `uv` project. Ruff and strict Mypy also passed. The plugin's six CPU
contract tests passed separately in the Torch-enabled project environment; no
GPU server was started.
