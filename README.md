# Audio LFM

Audio LFM is a projector-only audio-language-model training and evaluation
stack. It connects frozen `openai/whisper-small` audio features to the frozen
`LiquidAI/LFM2.5-1.2B-Instruct` decoder and optimizes only `projector.*`.
The production experiment trains on WavCaps plus ParaSpeechCaps-Base response
expansions, packs complete examples into 16K-token LFM sequences, and serves the
result through a native vLLM multimodal plugin.

```text
16 kHz mono FLAC
  -> worker CPU: 30 s chunk/pad + Whisper log-Mel + sequence packing
  -> GPU: frozen Whisper encoder -> 4x temporal stack (12.5 Hz)
  -> trainable projector -> masked scatter into packed LFM embeddings
  -> frozen LFM2.5-1.2B -> next-token loss
```

Parquet catalogs are authoritative for splits and targets, and every join uses
`audio_id`. Audio is validated and is never silently resampled, downmixed,
cropped, re-encoded, or rewritten. Training prefers a verified local shard
mirror and otherwise streams the private CaptionStew Bucket.

## Current status

The 20K-step projector checkpoint completed training and loaded successfully in
vLLM for the full two-stage evaluation harness. Generation used one persistent
`vllm serve` process per checkpoint and concurrent OpenAI-compatible HTTP
requests. The most comparable headline scores are:

| Benchmark | 6K checkpoint | 20K checkpoint | Scale |
| --- | ---: | ---: | --- |
| MMAU test-mini | 22.40 | 28.20 | accuracy, % |
| MMSU | 18.38 | 21.60 | accuracy, % |
| MMAU-Pro open + instruction following | 41.67 | 47.34 | category mean, % |
| MMAR | 25.30 | 25.40 | accuracy, % |

MMAU-Pro closed-ended NV-Embed scoring is intentionally omitted; its table
therefore covers 712 of 5,305 rows. VoiceBench SD-QA PANDA is also omitted
because the upstream dependency downloads mutable pickle files. Unparseable
model responses and judge outputs remain in every applicable denominator and
score zero. Complete per-subset results, judge identities, counts, generation
settings, and omission metadata are in
[`reports/evaluation-6k.json`](reports/evaluation-6k.json) and
[`reports/evaluation-20k.json`](reports/evaluation-20k.json).

## Start here

- Coding agents and contributors: read [`AGENTS.md`](AGENTS.md), then run the
  credential-free checks before changing behavior.
- Training implementation contract: [`PLAN.md`](PLAN.md).
- Caption-expansion prompt and recipe: [`CAPTION_EXPAND.md`](CAPTION_EXPAND.md).
- Full benchmark setup, generation, and scoring:
  [`evaluation/README.md`](evaluation/README.md).
- Completed 6K/20K evaluation audit:
  [`evaluation/FULL_SUITE_READINESS.md`](evaluation/FULL_SUITE_READINESS.md).

The repository deliberately uses separate `uv` environments for training,
vLLM generation, benchmark generation, and scoring. Do not run training and a
vLLM server or GPU judge at the same time on a single-GPU machine.

## Requirements

- Python 3.11 or 3.12 (the current reproducible setup uses 3.12)
- `uv`
- one CUDA GPU with BF16 support for model training (the production target is
  an RTX 4090 with 24 GiB)
- a CUDA-matched installation of Torch and torchaudio, installed before this
  package
- an `hf auth login` session with access to the private Bucket
- a local CaptionStew checkout with its training extra

Torch and torchaudio are intentionally absent from `pyproject.toml`: resolving
ordinary project dependencies must not replace the user's CUDA-matched builds.
Model-weight caching in `HF_HOME` is allowed and unavoidable. A complete local
audio mirror is optional and requires substantial disk space.

## Install with uv

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate

# Install the CUDA-matched Torch build appropriate for this machine first.
uv pip install torch torchaudio --torch-backend=auto

export CAPTIONSTEW_REPO=/path/to/captionstew-repository
uv pip install -e "${CAPTIONSTEW_REPO}[training]"

# --inexact preserves the separately installed CUDA Torch stack.
uv sync --extra dev --inexact
bash scripts/install_cuda_extensions.sh
```

`uv.lock` pins the complete Python dependency graph. The CUDA extension script
installs the official prebuilt `flash-attn==2.8.3` wheel for CPython 3.12,
Torch 2.10, CUDA 13.0, CXX11 ABI, and Linux x86_64; it fails explicitly on a
mismatched stack instead of compiling from source. It also pins
`causal-conv1d==1.6.2.post1` after checking CUDA and BF16.

### Environment map

| Environment | Purpose | Created by |
| --- | --- | --- |
| `.venv` / `.venv-training` | training, data planning, root tests | root `uv sync` |
| `.venv-vllm` | caption expansion and manual plugin work | `scripts/create_vllm_env.sh` |
| `.venv-evaluation` | persistent vLLM benchmark generation | `evaluation/scripts/create_evaluation_env.sh` |
| `.venv-evaluation-scorers` | official CPU and API scorers | `evaluation/scripts/create_scorer_env.sh` |
| `.venv-mmau-pro` | isolated MMAU-Pro scoring | `evaluation/scripts/create_mmau_pro_scorer_env.sh` |

On a restricted or low-memory WSL host, keep caches inside the repository and
compiler concurrency at one:

```bash
export UV_CACHE_DIR="$PWD/.runtime/uv-cache"
export MAX_JOBS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1
export TORCHINDUCTOR_COMPILE_THREADS=1
```

## Data and security

```bash
hf auth login
export CAPTIONSTEW_ROOT=/path/to/CaptionStew
export HF_HOME=/path/to/shared/huggingface-cache

hf buckets info seastar105/caption-stew
```

Authentication uses the token stored by the `hf` CLI. `HF_TOKEN` remains an
optional override for non-interactive environments. Configuration expansion
never logs secret-like environment values, and serialized configurations redact
fields whose names contain `TOKEN`, `SECRET`, `PASSWORD`, or `KEY`.

To mirror all ParaSpeechCaps-Base and WavCaps TAR shards resumably with the HF
CLI, then make training automatically prefer them:

```bash
uv run --no-sync python scripts/download_training_shards.py \
  --captionstew-root "$CAPTIONSTEW_ROOT"
```

The script uses `hf buckets sync`, verifies every catalog-referenced shard is
present and nonempty, and atomically writes a completion marker. The backend
uses local shards only when that marker and every selected file size match;
otherwise it continues streaming the private Bucket. Use `--dry-run` to inspect
the sync or `--verify-only` to rebuild the marker after an existing download.

Packed training also requires sample-exact duration metadata. Upstream duration
floats are used only to seed the table; the second command verifies every local
TAR member and reads the FLAC `STREAMINFO` header without waveform decoding:

```bash
uv run --no-sync python scripts/calculate_epoch_packing.py \
  --captionstew-root "$CAPTIONSTEW_ROOT" \
  --paraspeech-metadata /path/to/paraspeech/train_base.parquet \
  --wavcaps-metadata /path/to/WavCaps/metadata \
  --hf-cache /path/to/hf-cache \
  --output reports/mixed_epoch_packing.json \
  --duration-sidecar-output reports/mixed_training_durations.parquet \
  --sidecar-only

uv run --no-sync python scripts/build_exact_duration_sidecar.py \
  --captionstew-root "$CAPTIONSTEW_ROOT" \
  --input-sidecar reports/mixed_training_durations.parquet \
  --output reports/mixed_training_durations_exact.parquet
```

The exact sidecar is FLAC-hash-bound and stores the verified sample count plus
the FLAC/JSON byte ranges inside each uncompressed TAR. Workers therefore plan
with lightweight references and read only the examples selected for the next
pack. Training fails if decoded audio differs from the sidecar.
FLAC containers whose STREAMINFO declares an unknown length are checked with a
libsndfile header query. Containers with no discoverable audio frames are marked
`flac_empty` and skipped before packing; the current mixed corpus contains six.

To regenerate the real-data packing report from exact sample counts:

```bash
uv run --no-sync python scripts/calculate_epoch_packing.py \
  --captionstew-root "$CAPTIONSTEW_ROOT" \
  --paraspeech-metadata /path/to/paraspeech/train_base.parquet \
  --wavcaps-metadata /path/to/WavCaps/metadata \
  --hf-cache /path/to/hf-cache \
  --exact-duration-sidecar reports/mixed_training_durations_exact.parquet \
  --output reports/mixed_epoch_packing_exact.json
```

That report includes both the capacity/planning-window sweep and a runtime-faithful
two-worker simulation: WDS shard shuffle, worker shard split, WDS sample shuffle,
mixed-dataset interleave, and independent worker-local packing. No example-count
cap is applied.

ParaSpeechCaps uses exactly:

- training: `train_base`
- periodic validation: `dev`
- explicit final evaluation: `holdout`

`test` is rejected because its upstream IDs overlap holdout. Style captions and
transcripts remain separately typed. The direct-caption baseline uses style
captions only. For the production expanded-response experiment, the same
ParaSpeechCaps row's style caption and transcription become one user message
and produce exactly one response target; they are not concatenated directly as
the training target and do not become separate generated samples. WavCaps uses
its caption as the expansion input.

## Preflight and inspection

Packed training is gated by attention and causal-convolution isolation tests.
Failure aborts training; the program does not silently switch attention
implementations or disable packing.

```bash
uv run --no-sync audio-lfm preflight \
  --config configs/paraspeech_whisper_lfm2.yaml \
  --check-private-data

uv run --no-sync audio-lfm inspect-data \
  --config configs/paraspeech_whisper_lfm2.yaml \
  --num-samples 256

uv run --no-sync audio-lfm inspect-model \
  --config configs/paraspeech_whisper_lfm2.yaml

uv run --no-sync audio-lfm test-packing \
  --config configs/paraspeech_whisper_lfm2.yaml
```

`inspect-data` keeps only aggregate statistics, not waveform arrays. The
default long-audio policy is `skip`; cropping, when explicitly configured,
records the policy and sample offset.

## Tests

The ordinary suite builds synthetic FLAC, TAR, and Parquet inputs locally and
does not need credentials.

```bash
uv run --no-sync pytest -q
uv run --no-sync pytest -q -m gpu
uv run --no-sync pytest -q -m private_data
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync mypy
```

GPU and private-data tests are deliberately separate. A production run should
not begin until the GPU packing forward/backward gates pass on the target
machine.

## Train and resume

```bash
uv run --no-sync audio-lfm train \
  --config configs/paraspeech_whisper_lfm2_smoke.yaml

# Production WavCaps + ParaSpeechCaps expanded-response run.
uv run --no-sync audio-lfm train \
  --config configs/wavcaps_paraspeech_whisper_small_lfm2_expanded_3epoch.yaml

uv run --no-sync audio-lfm train \
  --config configs/wavcaps_paraspeech_whisper_small_lfm2_expanded_3epoch.yaml \
  --resume \
  runs/wavcaps-paraspeech-whisper-small-lfm2-expanded-3epoch-16k/checkpoint-00006000
```

The direct PyTorch loop accumulates summed loss until the configured input-token
budget, then divides projector gradients by the total supervised-token count.
Frozen LFM forward execution remains in autograd so gradients can reach the
continuous audio embeddings. Checkpoints contain projector tensors, optimizer,
scheduler, RNG, manifest, and committed IDs—never frozen Whisper or LFM weights.

The production optimizer uses peak LR `1e-3`, weight decay `1e-2`, gradient
norm clipping at `1.0`, 5% linear warmup, and cosine decay. A 16,384-token pack
capacity constrains total LFM sequence length; it does not target a number of
supervised tokens or cap the number of examples in a pack.

For the Whisper frontend, WebDataset performs shard and sample shuffle separately
inside every worker. Each worker owns one tokenizer, one CPU feature extractor,
one 2,048-example packing buffer, and its disjoint WDS shard stream. Local shards
are represented by exact TAR byte ranges, so the planning buffer contains no
FLAC payloads. It compiles targets and best-fit packs from the exact duration
sidecar before reading or decoding audio.
Only examples selected for the next pack are then FLAC-validated, split into
30-second blocks, padded to exactly 30 seconds, and converted to log-Mel features.
The worker also constructs the concatenated features, effective-frame masks, and
packed token layout, then yields one complete `HostAudioBatch` through the bounded
DataLoader result queue. `prefetch_factor` therefore counts packed batches, not
individual examples or planning windows.

The main training process performs no sample preparation or sequence packing. It
only receives a ready host batch, transfers it to CUDA, runs the fixed-length
Whisper encoder, 4x stack/projector, packed LFM forward/backward, and optimizer.
Thus a 45-second item is encoded as padded 30-second and 15-second blocks but
contributes only 45 seconds of effective audio embeddings.

All fixed Whisper blocks are encoded as a dense batch. Invalid padded frames are
zeroed before 4x stacking, worker-built indices gather effective projected frames,
and worker-built layout masks scatter the audio payload into the LFM embedding
sequence. The forward uses PyTorch's CUDA `masked_scatter`; its fixed-index custom
backward avoids the stock data-dependent backward synchronization. This path is
kept behind a measured throughput gate.
The worker also emits fixed supervised-token indices, so selective LM-head loss
uses `index_select` instead of synchronizing boolean indexing.
When Whisper is compiled, its final block microbatch is padded on the batch axis
to the configured fixed block count; padded outputs are discarded. This keeps a
single static Inductor graph while preserving every real block's effective slice.
The production configuration also compiles the LFM backbone. The projector stays
eager because arbitrary per-audio time lengths trigger a PyTorch Inductor
`CantSplit` in RMSNorm backward; this is an explicit, tested module-level choice,
not a global compile fallback.

TensorBoard logs `data_wait_seconds`, `h2d_seconds`, `whisper_seconds`,
`projector_lfm_forward_seconds`, `backward_seconds`, and `optimizer_seconds`.
`pack_utilization` is the actual LFM input length divided by the configured
capacity across all packs in the optimizer update.
`input_tokens_per_second` measures packed CUDA compute after a plan is ready;
`end_to_end_input_tokens_per_second` additionally charges the update for waiting
on the next worker-built packed batch. Both throughput metrics count all input
tokens sent to LFM, not only supervised target tokens.

Resume regenerates deterministic worker-local shuffled planning windows and skips
only complete previously committed packs before TAR reads or log-Mel work,
preserving uninterrupted pack boundaries. IDs are committed only after a
successful optimizer step. A shared
stop event drains worker completion signals on an intentional early stop. In
distributed training, every rank performs an exhaustion all-reduce before each
packed-batch yield; if any rank is exhausted, all ranks start the next epoch
together instead of entering an unmatched model collective.

## Evaluate and generate

Teacher-forced evaluation uses the same packed path and scores every official
style-caption reference.

```bash
uv run --no-sync audio-lfm evaluate \
  --config configs/paraspeech_whisper_lfm2.yaml \
  --checkpoint runs/paraspeech-whisper-small-lfm2-projector/checkpoint-best \
  --split dev

uv run --no-sync audio-lfm evaluate \
  --config configs/paraspeech_whisper_lfm2.yaml \
  --checkpoint runs/paraspeech-whisper-small-lfm2-projector/checkpoint-best \
  --split holdout

uv run --no-sync audio-lfm generate \
  --config configs/paraspeech_whisper_lfm2.yaml \
  --checkpoint runs/paraspeech-whisper-small-lfm2-projector/checkpoint-best \
  --split dev
```

Generation is non-packed and writes incrementally committed JSONL/Parquet
predictions with source provenance. Caption or transcript text is not logged by
the trainer.

### Full benchmark evaluation

The independent harness under `evaluation/` covers VoiceBench, MMAU, MMSU,
MMAU-Pro, MMAR, KVoiceBench, KMMAU, and VoiceBench-JA. Stage 1 launches one
persistent `vllm serve` process, sends concurrent HTTP requests for any number
of selected benchmarks, and writes resumable predictions. Stage 2 reads only
those predictions and runs the pinned official CPU, API-judge, or local-vLLM
scorer. This separation avoids repeated model loads and prevents scoring
dependencies from contaminating the inference environment.

```bash
bash evaluation/scripts/create_evaluation_env.sh
source .venv-evaluation/bin/activate
export VLLM_PLUGINS=audio_lfm2

audio-lfm-eval run-suite \
  --model-export exports/wavcaps-paraspeech-lfm2-20k-vllm \
  --model-name audio-lfm-20k \
  --data-root evaluation-data \
  --output-root evaluation-runs/20k \
  --benchmark voicebench --benchmark mmsu --benchmark mmau \
  --benchmark mmau-pro --benchmark mmar --benchmark voicebench-ja \
  --benchmark kvoicebench --benchmark kmmau
```

Scoring credentials are loaded only in stage 2. Do not place API keys in YAML,
command arguments, manifests, or committed files. See
[`evaluation/README.md`](evaluation/README.md) for exact scorer environments and
commands. Rebuild the publication-safe result files after scoring with:

```bash
UV_CACHE_DIR=.runtime/uv-cache uv run --no-sync python \
  evaluation/scripts/summarize_results.py
```

## Optional dMel frontend

Install `uv pip install dmel` into the already prepared Torch environment and
use `configs/paraspeech_dmel_lfm2.yaml`. The dMel frontend returns deterministic
integer codes; code/channel embeddings and temporal patching stay inside the
trainable projector namespace and reuse the same packing/training interfaces.

## Optional vLLM generation environment

vLLM owns a tightly coupled Torch/CUDA stack, so it runs in a separate venv.

```bash
export HF_HOME=/path/to/shared/huggingface-cache
export VLLM_CACHE_ROOT=/path/to/vllm-cache
export VLLM_PLUGINS=audio_lfm2
export CAPTIONSTEW_REPO=/path/to/captionstew-repository
bash scripts/create_vllm_env.sh
source .venv-vllm/bin/activate

audio-lfm export-vllm \
  --config configs/wavcaps_paraspeech_whisper_small_lfm2_expanded_3epoch.yaml \
  --checkpoint runs/wavcaps-paraspeech-whisper-small-lfm2-expanded-3epoch-16k/checkpoint-00020000 \
  --output-dir exports/wavcaps-paraspeech-lfm2-20k-vllm

audio-lfm preflight-vllm \
  --export-dir exports/wavcaps-paraspeech-lfm2-20k-vllm

audio-lfm evaluate-vllm \
  --config configs/vllm_eval.yaml \
  --split dev

audio-lfm evaluate-vllm \
  --config configs/vllm_eval.yaml \
  --split holdout \
  --allow-final-evaluation
```

The export is intentionally small: it contains the tokenizer/config and mapped
`multi_modal_projector.*` tensors only. Immutable secondary revisions provide
native vLLM LFM2 and frozen Whisper weights. The added `<|audio|>` placeholder
is outside the frozen tokenizer vocabulary. It may occupy an otherwise-unused
row inside LFM's padded embedding table; vLLM overwrites every placeholder
position with a projected audio embedding. The LFM embedding table and
`vocab_size` are not resized. Each audio item is an independent vLLM request,
and evaluation resumes from complete atomic Parquet parts keyed by `audio_id`.

Response expansion is never automatic. It uses the exact fixed listener system
message from `CAPTION_EXPAND.md`, sends a WavCaps caption or the combined
ParaSpeechCaps style caption and same-audio transcription as the user message,
and leaves all TAR payloads untouched. Generation runs through vLLM with
LiquidAI's recommended model-card settings (`temperature=0.1`, `top_k=50`,
`repetition_penalty=1.05`, sampling enabled) and the requested
`max_tokens=1024`; a max-length completion is stored with `truncated: true`.

```bash
hf download LiquidAI/LFM2.5-1.2B-Instruct \
  --revision 0f604ada3f766f9f257460c4c9f0b5d6f69d431b

audio-lfm expand-responses \
  --catalog-dir /path/to/ParaSpeechCaps-Base/targets/source=official \
  --output-dir runs/caption-expansion/ParaSpeechCaps-Base \
  --dataset ParaSpeechCaps-Base \
  --source-target-types style_caption,transcription \
  --combine-paraspeech-sources \
  --model-path /path/to/downloaded/model/snapshot \
  --model-revision 0f604ada3f766f9f257460c4c9f0b5d6f69d431b \
  --request-batch-size 8192 \
  --max-num-seqs 512 \
  --max-num-batched-tokens 32768 \
  --max-tokens 1024
```

Generated records are atomically committed as resumable Parquet parts. They
retain the official source target ID and type, immutable decoder identity,
prompt and recipe hashes, finish reason, token count, and typed truncation flag.

For paper-like expanded-response training, use
`configs/paraspeech_whisper_lfm2_expanded.yaml`. It uses Whisper large-v2, 4x
frame stacking (12.5 Hz), a frozen LFM2.5-1.2B decoder, and trains only the
projector. The audio training and HF/vLLM evaluation prompt contains one audio
placeholder as a user message and no system message, caption, or transcript.
