# Repository guide for coding agents

This file is the operational entry point for automated coding work. Read it
before changing the repository. Use `PLAN.md` for the original implementation
contract, `README.md` for user workflows, and `evaluation/README.md` for the
benchmark contract.

## What this repository does

Audio LFM trains only a temporal projector between frozen
`openai/whisper-small` features and frozen
`LiquidAI/LFM2.5-1.2B-Instruct`. The production data mixture is WavCaps plus
ParaSpeechCaps-Base, with offline response-expanded targets. Training uses
worker-local WebDataset shuffling, packing, FLAC decoding, and Whisper log-Mel
preprocessing. The CUDA process receives ready packed batches and performs only
H2D, Whisper encoding, projector/LFM forward, backward, and optimization.

The vLLM export is intentionally projector-only. It depends on immutable base
Whisper and LFM revisions recorded in `export_manifest.json`; do not treat its
small `model.safetensors` as a standalone full-model checkpoint.

## Source-of-truth order

When documentation appears inconsistent, use this order and repair the stale
document in the same change:

1. Tests and typed runtime contracts in `src/audio_lfm/` and
   `evaluation/src/audio_lfm_eval/`.
2. The selected YAML configuration and its serialized run manifest.
3. `PLAN.md` and `CAPTION_EXPAND.md` for non-negotiable design requirements.
4. `README.md`, `evaluation/README.md`, and readiness reports.

Do not infer production settings from smoke or regression configurations.
The main completed run uses
`configs/wavcaps_paraspeech_whisper_small_lfm2_expanded_3epoch.yaml`.

## Repository map

- `src/audio_lfm/`: training, data, model, checkpoint, generation, and vLLM
  plugin code.
- `configs/`: typed experiment configurations and prompts.
- `scripts/`: environment setup, shard mirroring, metadata planning, and
  operational launchers.
- `tests/`: credential-free synthetic unit and integration tests; GPU/private
  tests are marked.
- `evaluation/`: isolated two-stage benchmark harness and its own `uv.lock`.
- `evaluation-scorers/`: pinned upstream scorer checkouts. Avoid broad cleanup
  or formatting here.
- `reports/`: small reproducibility summaries suitable for source control.
- `runs/`, `exports/`, `evaluation-data/`, and `evaluation-runs/`: local runtime
  artifacts; do not add these wholesale to Git.

## Environment and package rules

Use `uv` for every Python environment and dependency operation. Do not use
bare `pip`, Poetry, Conda, or ad-hoc dependency installs. Torch is deliberately
absent from the root dependency list so `uv sync --inexact` preserves the
machine's CUDA-matched Torch build.

Keep incompatible workloads isolated:

- `.venv` or `.venv-training`: training and CPU development.
- `.venv-vllm`: caption expansion and standalone vLLM plugin work.
- `.venv-evaluation`: vLLM benchmark generation and lightweight harness tests.
- `.venv-evaluation-scorers`: official CPU/API scorer dependencies.
- `.venv-mmau-pro`: optional isolated MMAU-Pro scoring dependencies.

On restricted hosts set `UV_CACHE_DIR=.runtime/uv-cache`. Never run training,
vLLM generation, or a GPU judge concurrently on the single production GPU.

Common setup and checks:

```bash
uv sync --extra dev --inexact
UV_CACHE_DIR=.runtime/uv-cache uv run --no-sync pytest -q
UV_CACHE_DIR=.runtime/uv-cache uv run --no-sync ruff format --check .
UV_CACHE_DIR=.runtime/uv-cache uv run --no-sync ruff check .
UV_CACHE_DIR=.runtime/uv-cache uv run --no-sync mypy

UV_CACHE_DIR=.runtime/uv-cache uv run --project evaluation --extra dev \
  pytest -q evaluation/tests
UV_CACHE_DIR=.runtime/uv-cache uv run --project evaluation --extra dev \
  ruff check evaluation
UV_CACHE_DIR=.runtime/uv-cache uv run --project evaluation --extra dev \
  mypy evaluation/src
```

Use the narrowest relevant checks while iterating, then run the complete
credential-free suites before publication. GPU, private-data, and full
benchmark runs are explicit operations, never implicit test fallbacks.

## Architecture invariants

Preserve these unless the user explicitly changes the experiment:

- Only `projector.*` parameters are trainable. Whisper runs under
  `torch.no_grad()`; the frozen LFM must remain in autograd so gradients reach
  continuous audio embeddings.
- Input audio is validated as mono 16 kHz FLAC. Never silently resample,
  downmix, re-encode, crop, or rewrite dataset audio.
- Audio is split into 30-second blocks and every block is padded to exactly 30
  seconds for Whisper. Only effective frames survive. Four-frame stacking gives
  12.5 Hz; effective projected length is rounded from real audio duration.
- DataLoader workers own shard/sample shuffle, pack planning, selected FLAC
  reads, log-Mel extraction, token layouts, gather indices, and scatter masks.
  Do not move waveform conversion, feature extraction, or sequence packing into
  the main training process.
- Packing capacity is based on total LFM input length, not supervised-token
  count. Examples per pack are uncapped. Maintain `seq_idx`, cumulative lengths,
  reset positions, FlashAttention boundary tests, and causal-convolution
  isolation.
- Resume must reproduce worker-local shuffled pack boundaries and commit sample
  IDs only after a successful optimizer step. Distributed ranks synchronize
  exhaustion before entering model collectives.
- The hot model path must not perform CPU tensor decisions or data-dependent
  host synchronization. Worker-built indices feed CUDA gather/scatter. Keep a
  performance-sensitive alternative only when a measured benchmark shows it is
  faster.
- vLLM inference preserves long audio by the same 30-second chunking rule and
  supports ordered multi-audio prompts. Context overflow fails before requests;
  it never crops or silently skips an item.

## Dataset and target rules

Parquet catalogs are authoritative for IDs, splits, and targets; `audio_id` is
the stable join key. TAR files contain payloads and are never rewritten.
Training uses ParaSpeechCaps `train_base` plus WavCaps; validation uses
ParaSpeechCaps `dev`, and final teacher-forced evaluation uses `holdout`.

Style captions and transcripts remain separately typed in source metadata. For
offline ParaSpeechCaps response expansion, combine the same row's style caption
and transcription into one user message and generate one response target. Do
not create two training targets or concatenate them directly as the target.
WavCaps expansion uses its caption. The exact generation recipe is locked by
`CAPTION_EXPAND.md` and decoder-lock hashes.

## Training defaults and observability

The production optimizer defaults are peak LR `1e-3`, weight decay `1e-2`,
gradient norm clip `1.0`, 5% warmup, then cosine decay. Sequence capacity is
16,384 tokens. Validate actual YAML and run manifests before reporting a run.

TensorBoard distinguishes CUDA-only `input_tokens_per_second` from
`end_to_end_input_tokens_per_second`, which includes DataLoader wait. It also
logs pack utilization, loss, LR, grad norm, validation NLL, and phase timings.
Never describe supervised tokens as the packing constraint.

## Evaluation contract

Evaluation has two stages:

1. One persistent `vllm serve` process handles concurrent HTTP generation for
   multiple benchmark selections. Saved JSONL predictions are resumable and
   contain no embedded audio bytes.
2. Scoring reads saved generations independently. CPU metrics, API judges, and
   local vLLM judges must not reload the Audio LFM checkpoint.

Generation uses `temperature=0.1`, `top_k=50`, `top_p=1.0`,
`repetition_penalty=1.05`, seed 0, and `max_tokens=1024`. Max-length completions
must set `truncated: true`.

Every generated row stays in the scorer denominator. Unparseable model answers
and unparseable judge verdicts score zero. VoiceBench SD-QA PANDA is omitted
because its dependency downloads mutable pickles. MMAU-Pro closed-ended
NV-Embed is omitted by request. Both omissions must remain explicit in manifests
and published results. Regenerate compact summaries with:

```bash
UV_CACHE_DIR=.runtime/uv-cache uv run --no-sync python \
  evaluation/scripts/summarize_results.py
```

## Secrets, data, and publication safety

Use `hf auth login` and the `hf` CLI; `HF_TOKEN` is only an optional
non-interactive override. Load judge keys from `.env` without printing them.
Never put tokens, keys, Bucket credentials, audio bytes, absolute private paths,
or `.env` contents in logs, manifests, commits, or Hub uploads.

Before committing, inspect `git status --short` and add files deliberately.
Never add local venvs, caches, private data, raw benchmark snapshots,
checkpoints, TensorBoard runs, or full evaluation prediction trees. Model
artifacts and evaluation summaries belong on the model Hub; source code,
small reports, and documentation belong on GitHub.

## Change discipline

Use `rg`/`rg --files` for discovery and `apply_patch` for hand edits. Preserve
unrelated user changes in the dirty worktree. Add or update tests with behavior
changes, especially for long audio, packing boundaries, resume, scorer
denominators, and malformed outputs. Prefer explicit failures over silent
fallbacks. Report what was not tested when hardware, credentials, or private
data are unavailable.
