# Audio LFM benchmark evaluation

This directory is an isolated, two-stage evaluation stack for:

- VoiceBench
- MMAU (`test-mini`, the public labeled split)
- MMSU
- MMAU-Pro
- MMAR
- KVoiceBench
- KMMAU
- VoiceBench-JA

Every Hugging Face dataset and official scorer is pinned to an immutable commit
in `benchmarks.yaml`. Dataset snapshots are downloaded only by the `hf` CLI.
Generation does not import or call vLLM in-process.

## Environments

The generation environment is independent of `.venv-training` and the caption
expansion environment. Builds and compiler pools default to one worker for WSL
host-memory safety.

```bash
bash evaluation/scripts/create_evaluation_env.sh
source .venv-evaluation/bin/activate
audio-lfm-eval list
```

The setup uses `uv`, installs `vllm[audio]==0.27.1`, installs this evaluation
project from its lockfile, and exposes the main repository's vLLM plugin as an
editable no-dependency install. It does not install CaptionStew.

Python 3.12 is the default for this environment. FlashInfer 0.6.16 contains a
type annotation that is valid on 3.12 but is evaluated incompatibly on 3.11.
On WSL the launcher also selects vLLM's legacy model runner because CUDA UVA is
not exposed there, and disables the FlashInfer sampler JIT that assumes a
system `/usr/local/cuda/nvcc`. These fallbacks are applied only on a Microsoft
kernel; native Linux keeps vLLM's normal defaults.

Official scorers have different dependency stacks. First clone their pinned
source, then optionally create a scorer-only environment:

```bash
audio-lfm-eval sync-scorers --output-root evaluation-scorers
bash evaluation/scripts/create_scorer_env.sh
```

`INSTALL_FLEXEVAL=1` adds the substantially larger VoiceBench-JA FlexEval stack.
MMAU-Pro's official NV-Embed/Qwen judges are intentionally not installed by the
lightweight scorer setup; run those in a separate scoring GPU window.

`sync-scorers` checks out each immutable upstream revision and then applies the
small versioned patches in `src/audio_lfm_eval/scorer_patches/`. These overlays
implement the repository's full-denominator policy and MMAU-Pro metric
selection without vendoring the upstream repositories. A patch mismatch fails
explicitly instead of running a moving or silently incompatible scorer.

## Download benchmark data

Download one subset at a time so disk use is deliberate. `--max-workers 2` is
fixed in the downloader to avoid another WSL host-memory spike.

```bash
audio-lfm-eval download \
  --benchmark voicebench \
  --subset mmsu \
  --output-root evaluation-data

audio-lfm-eval download \
  --benchmark mmau \
  --subset default \
  --output-root evaluation-data
```

Repeat for the desired entries shown by `audio-lfm-eval list`. MMAR and MMAU-Pro
contain benchmark-owned archives; unpack those with validated paths:

```bash
audio-lfm-eval unpack --benchmark mmar --data-root evaluation-data
audio-lfm-eval unpack --benchmark mmau-pro --data-root evaluation-data
```

These datasets are public, so an HF token is not normally required. The `hf`
CLI will still use the current login and cache. Dataset licenses differ;
VoiceBench-JA audio is non-commercial and may not be redistributed.

## Stage 1: persistent vLLM HTTP generation

Export the trained projector with the repository's `audio-lfm export-vllm`
command first. Then select any number of benchmarks/subsets in one invocation:

```bash
audio-lfm-eval run-suite \
  --model-export exports/paraspeech-whisper-small-lfm2-vllm \
  --data-root evaluation-data \
  --output-root evaluation-runs \
  --benchmark voicebench:mmsu \
  --benchmark mmau:default \
  --benchmark mmsu:default \
  --benchmark kmmau:age
```

`run-suite` launches exactly one persistent `vllm serve` child process, waits
for `/health`, sends OpenAI-compatible `/v1/chat/completions` requests over
HTTP, runs every selection sequentially against that same loaded model, and
stops the server after all selections finish. HTTP requests within a benchmark
are concurrent so vLLM can form large continuous batches. Defaults are:

- `max_num_seqs=128`
- `max_num_batched_tokens=131072`
- `max_model_len=32768`, with up to three audio items per prompt
- 128 concurrent HTTP requests with 256 HTTP connections
- a 3,600-audio-second weighted in-flight budget, so short requests can fill
  vLLM while long/multi-audio requests cannot exhaust WSL host memory
- allowlisted local `file://` audio URLs; embedded Parquet audio is materialized
  once by content hash under each dataset's `.vllm-media-cache` directory
- `VLLM_MAX_AUDIO_DECODE_DURATION_S=720`, matching the complete-audio client
  contract rather than vLLM's lower default decode ceiling
- caption-expansion sampling: `temperature=0.1`, `top_k=50`,
  `repetition_penalty=1.05`, and `max_tokens=1024` (`top_p=1.0`, seed 0)

Edit `evaluation/configs/default.yaml` to tune these without changing the data
or scorer manifests. `vllm-server.log` records startup/runtime logs.

To reuse a manually managed server for several invocations, start `vllm serve`
with the arguments emitted by `build_server_command` (or use the equivalent
defaults in the YAML), then use `audio-lfm-eval generate --base-url ...`.
`generate` refuses to continue unless `/health` succeeds.

Each `<benchmark>/<subset>` output contains:

- `generation_manifest.json`: dataset SHA, export-config hash, transport, and
  generation settings; an incompatible resume is rejected;
- `predictions.jsonl`: fsync-committed rows with source IDs, responses,
  finish reasons, `truncated`, usage, and audio normalization provenance;
- `progress.json`: successful, failed, and previously completed counts.

Successful IDs are skipped on resume. Failed rows are retried. Base64 audio is
never written to prediction files.

### Long and multi-audio contract

The exported AudioLFM2 vLLM plugin supports up to three ordered audio items per
request and preserves each complete item up to 720 seconds. Each logical item
is split into independent 30-second Whisper windows internally, matching the
training frontend. Only effective (unpadded) encoder frames are concatenated
before projection, and one start/end boundary pair is added per logical item.
Semantically distinct MMAU-Pro clips therefore remain distinct prompt
placeholders; long clips are never cropped or flattened together.

The default 32,768-token model context accommodates the static worst case of
three 720-second audio items at the configured stack factor, plus the prompt
and output reserves. Before any HTTP requests are submitted, generation reads
the headers for every selected row and validates the sum of all audio-item token
lengths. A row that cannot fit fails the selection with a detailed error instead
of being cropped, skipped, or left as a per-row prediction failure.

## Stage 2: independent scoring

Scoring reads only saved generations. It does not start vLLM or load the Audio
LFM checkpoint. First materialize the official input and print the exact scorer
plan:

```bash
audio-lfm-eval score \
  --benchmark mmau \
  --subset default \
  --output-root evaluation-runs \
  --scorer-root evaluation-scorers
```

Add `--execute` after the scorer environment is active. Native/programmatic
paths require no judge credentials:

- MMAU public test-mini: official string matcher
- MMSU: official multiple-choice scorer
- MMAR: official answer scorer (the newer rubric scorer is optional)
- VoiceBench MCQ, BBH, IFEval, and AdvBench subsets
- KVoiceBench safety/instruction subsets where Raon-Eval uses programmatic
  metrics
- VoiceBench-JA JamC-QA and M-IFEval through their supplied FlexEval configs

Judge-backed paths are deliberately stage 2 and require credentials only when
scored:

- `OPENAI_API_KEY`: VoiceBench open/SD-QA, some Raon-Eval paths,
  VoiceBench-JA Elyza/Spoken-Elyza, and optional MMAR Rubrics;
- `OPENROUTER_API_KEY`: supported by Raon-Eval as an alternative;
- MMAU-Pro: Qwen 2.5 open-ended judging is batched through vLLM, and
  instruction-following constraints are scored programmatically. Closed-ended
  NV-Embed scoring is omitted and explicitly recorded in result metadata. Run
  the open judge as a separate GPU session.

All scorer denominators include every generated row. Unparseable model answers
and unparseable judge verdicts score zero; they are never omitted or replaced
with random/default scores.

MMAU-Pro's reported overall score is the unweighted arithmetic mean of its
normalized category scores. In the configured scope, open-ended overall judge
score divided by five and instruction-following success rate each contribute
50%, regardless of their different row counts.

Never put judge keys in YAML, manifests, prediction files, or shell history.

## Validation

Lightweight checks do not start vLLM or use the GPU:

```bash
uv run --project evaluation --extra dev pytest -q
uv run --project evaluation --extra dev ruff check evaluation
uv run --project evaluation --extra dev mypy evaluation/src
```

Before a real run, also verify the exported checkpoint in the isolated
generation environment:

```bash
VLLM_PLUGINS=audio_lfm2 audio-lfm preflight-vllm \
  --export-dir exports/paraspeech-whisper-small-lfm2-vllm
```

## LiquidAI reference checkpoints

The runnable end-to-end smoke profile is
`Qwen/Qwen2.5-Omni-3B@f75b40e3da2003cdd6e1829b1f420ca70797c34e`.
vLLM 0.27.1 natively supports its thinker path for audio input and text output.
It uses exactly the same required persistent-server architecture and sends
OpenAI-compatible content parts of the form
`{"type":"audio_url","audio_url":{"url":"data:audio/wav;base64,..."}}`.

After downloading a short benchmark subset, run a one-row infrastructure test
without committing to a full benchmark:

```bash
audio-lfm-eval run-profile-suite \
  --profile qwen25-omni-3b-smoke \
  --data-root evaluation-data \
  --output-root evaluation-runs \
  --benchmark voicebench:mmsu \
  --limit 1
```

This starts one `vllm serve` process, sends HTTP requests, saves a resumable
prediction and pinned model identity, and stops the process. Omit `--limit` and
select several benchmarks to amortize the same model load across them. It is a
GPU smoke/full-run command and is not executed by the lightweight test suite.

`model_profiles.yaml` pins the two requested public references by immutable Hub
revision (resolved with `hf models info`, not a moving branch):

- `LiquidAI/LFM2.5-Audio-1.5B@c362a0625dfe45aa588dce5f0ada28a7e5707628`
- `LiquidAI/LFM2.5-Audio-1.5B-JP@6c34b4d590f80563f8cb2939c2ebd7686d952394`

They are reference profiles, not silently substituted inference backends. Both
declare `Lfm2AudioForConditionalGeneration`; vLLM 0.27.1 does not register that
architecture, while the official examples use `liquid-audio`'s `ChatState` and
`generate_sequential`. Because this evaluation contract requires persistent
`vllm serve` plus HTTP, preflight fails explicitly until a vLLM multimodal
adapter is implemented:

```bash
audio-lfm-eval list-model-profiles
audio-lfm-eval preflight-model-profile --profile lfm25-audio-en-reference
```

The repo's trained projector export remains runnable through its `audio_lfm2`
vLLM plugin. Do not call a `liquid-audio` in-process run equivalent to the
required serving smoke test.

Published model-card values and deliberately broad reproducibility ranges are
stored alongside each profile. After independent stage-2 scoring, write a
canonical JSON object such as `{"voicebench-ja.elyza": 2.20}` and validate it:

```bash
audio-lfm-eval validate-reference \
  --profile lfm25-audio-jp-reference \
  --scores score-summary.json \
  --require-complete
```

Normal pytest validates pins, capability preflight, and range logic without a
GPU. Full numerical validation is an expensive, separately invoked benchmark:
judge-backed scores can vary, and it requires all benchmark data, a working
future vLLM adapter, GPU time, and judge credentials.
