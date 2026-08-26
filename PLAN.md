# Implementation brief: projector-only Audio LLM on LFM2.5-1.2B

Build a complete, testable training repository from an empty directory.

Do not stop at scaffolding. Implement the working data path, model, packing, trainer, checkpointing, evaluation, and tests. Use plain PyTorch plus Hugging Face Transformers. Do not use Hugging Face Trainer, TRL, LEAP, DeepSpeed, FSDP, Megatron, Lightning, bitsandbytes, CPU offload, or quantized training in the initial implementation.

The first production target is one NVIDIA RTX 4090 with 24 GiB VRAM.

The architecture is:

    remote WebDataset FLAC
        -> frozen audio frontend
        -> trainable temporal reduction + projector
        -> continuous embeddings inserted into the LFM token stream
        -> frozen LiquidAI/LFM2.5-1.2B-Instruct
        -> text next-token loss

Only the projector namespace is trainable. The projector namespace may include:

- temporal frame stacking or patching;
- projection MLP;
- trainable audio-start and audio-end vectors;
- output normalization and scaling;
- dMel code/channel embeddings in the dMel experiment.

The first baseline uses `openai/whisper-small`. Add dMel as a second frontend behind the same interface, but do not make dMel part of the first working milestone.

---

# 1. Non-negotiable constraints

## 1.1 Dataset

The dataset has already been created. Never:

- regenerate audio;
- re-encode audio;
- apply VoiceFixer;
- upsample audio;
- bulk-download TAR shards;
- cache entire TAR shards through Hugging Face;
- rewrite TAR shards to attach generated annotations.

The private Hugging Face Bucket is:

    hf://buckets/seastar105/caption-stew/CaptionStew/_webdataset/

Authentication must come only from:

    HF_TOKEN

Never store or print `HF_TOKEN`. Redact all environment variables whose names contain `TOKEN`, `SECRET`, `PASSWORD`, or `KEY`.

Use the existing dataset client:

    from captionstew.training_client import open_webdataset

The path passed to it is configured by `CAPTIONSTEW_ROOT`.

The intended installation is:

    python -m pip install -e "${CAPTIONSTEW_REPO}[training]"

where `CAPTIONSTEW_REPO` is the local source repository containing the `captionstew` package, and `CAPTIONSTEW_ROOT` is the data root passed to `open_webdataset`.

Do not copy or fork the CaptionStew client into this repository. Create a thin adapter around it.

## 1.2 Dataset format

Each WebDataset sample contains exactly:

    <audio_id>.flac
    <audio_id>.json

The audio contract is:

- FLAC;
- 16,000 Hz;
- one channel;
- variable duration.

The runtime loader must assert:

    sample["__key__"] == metadata["audio_id"]
    sample_rate == 16_000
    number_of_channels == 1

Do not silently resample or downmix.

The Parquet catalogs are separate from the TAR payloads. The stable join key is always:

    audio_id

Audio catalog:

    CaptionStew/_webdataset/<dataset>/16k-flac/parquet/audio/*.parquet

Required audio columns:

    audio_id
    dataset
    flac_sha256
    flac_size
    source_id
    splits
    target_count
    wds_key
    wds_shard

Official target catalog:

    CaptionStew/_webdataset/<dataset>/16k-flac/parquet/targets/source=official/*.parquet

Required target columns:

    audio_id
    target_id
    target_type
    text
    split
    source
    generator_model
    generator_revision
    prompt_sha256
    review_status

The official catalog does not contain a reliable duration field. Obtain duration from decoded FLAC. A small derived duration sidecar keyed by `audio_id` is allowed. Never mutate the official Parquet catalogs.

## 1.3 Initial dataset and split policy

Implement ParaSpeechCaps-Base first.

Use:

    training:   train_base
    validation: dev
    final eval: holdout

Do not create a separate ParaSpeechCaps test loader. Its upstream test IDs overlap holdout.

For ParaSpeechCaps:

- `style_caption` is the primary target;
- `transcription` is a separate typed field;
- never merge transcripts and style captions;
- never use a transcript as a style-caption target;
- never leak the transcript into the input prompt unless an explicit future experiment enables it.

During training, select one style caption per audio item per epoch. Make the selection deterministic using SHA-256 of:

    seed || epoch || audio_id

Do not use Python `hash()`.

During validation and final evaluation, evaluate all style-caption references and report:

- target-weighted mean NLL;
- audio-weighted mean NLL, where references are averaged within each audio item first.

Add WavCaps support after ParaSpeech works. For WavCaps, use only `target_type == "caption"`. Its source labels are not train/dev/test partitions. Any future random partition must be derived deterministically from SHA-256 of `audio_id`, with partition proportions supplied in configuration.

## 1.4 Frozen-model gradient semantics

The audio encoder must run as:

    encoder.eval()
    with torch.no_grad():
        audio_features = encoder(...)

Detach encoder outputs before the projector.

The LLM parameters must have:

    requires_grad = False

However, never put the LLM forward under `torch.no_grad()` during training. Gradients must flow:

    loss
      -> frozen LFM layers
      -> projected audio embeddings
      -> projector parameters

Use the LFM in training mode when gradient checkpointing is active. Assert that all configured LFM dropout probabilities are zero. If any nonzero frozen-model dropout is found, fail unless an explicit `allow_frozen_llm_dropout` option is enabled.

Only projector parameters may be passed to the optimizer. Assert this at startup and after checkpoint restoration.

## 1.5 Packing

Packing is mandatory for the production training configuration.

Packed LFM input must contain:

    inputs_embeds:  [1, total_tokens, hidden_size]
    labels:         [1, total_tokens]
    position_ids:   [1, total_tokens]
    seq_idx:        [1, total_tokens]
    cu_seq_lens_q:  [num_sequences + 1]
    cu_seq_lens_k:  [num_sequences + 1]
    max_length_q:   Python int
    max_length_k:   Python int

Rules:

- reset `position_ids` to zero for every logical example;
- increment `seq_idx` for every logical example;
- use `torch.int32` for `seq_idx` and cumulative sequence lengths;
- put cumulative sequence lengths on CUDA before the LFM call;
- do not pass `attention_mask`;
- pass `use_cache=False`;
- use FlashAttention 2;
- use an optimized causal-conv1d implementation that honors `seq_idx`;
- never rely on `position_ids` alone.

Packed training must abort if either the attention boundary test or causal-convolution boundary test fails.

---

# 2. Repository layout

Create this structure:

    .
    ├── pyproject.toml
    ├── README.md
    ├── LICENSE
    ├── .gitignore
    ├── configs/
    │   ├── paraspeech_whisper_lfm2.yaml
    │   ├── paraspeech_whisper_lfm2_smoke.yaml
    │   ├── paraspeech_dmel_lfm2.yaml
    │   └── prompts/
    │       ├── paraspeech_style_caption.txt
    │       └── response_expansion.example.txt
    ├── scripts/
    │   ├── install_cuda_extensions.sh
    │   └── run_smoke.sh
    ├── src/
    │   └── audio_lfm/
    │       ├── __init__.py
    │       ├── cli.py
    │       ├── config.py
    │       ├── environment.py
    │       ├── manifest.py
    │       ├── data/
    │       │   ├── __init__.py
    │       │   ├── types.py
    │       │   ├── catalog.py
    │       │   ├── captionstew_backend.py
    │       │   ├── decode.py
    │       │   ├── targets.py
    │       │   ├── stream.py
    │       │   ├── pack_planner.py
    │       │   └── resume_state.py
    │       ├── model/
    │       │   ├── __init__.py
    │       │   ├── audio_lfm.py
    │       │   ├── prompt_compiler.py
    │       │   ├── projector.py
    │       │   ├── packed_batch.py
    │       │   └── frontends/
    │       │       ├── __init__.py
    │       │       ├── base.py
    │       │       ├── whisper.py
    │       │       └── dmel.py
    │       ├── training/
    │       │   ├── __init__.py
    │       │   ├── engine.py
    │       │   ├── loss.py
    │       │   ├── optimizer.py
    │       │   ├── scheduler.py
    │       │   ├── checkpoint.py
    │       │   └── metrics.py
    │       ├── evaluation/
    │       │   ├── __init__.py
    │       │   ├── teacher_forced.py
    │       │   ├── generation.py
    │       │   └── predictions.py
    │       ├── overlays/
    │       │   ├── __init__.py
    │       │   └── response_expansion.py
    │       └── utils/
    │           ├── __init__.py
    │           ├── hashing.py
    │           ├── logging.py
    │           ├── rng.py
    │           └── tensors.py
    └── tests/
        ├── conftest.py
        ├── fixtures/
        │   └── build_tiny_captionstew.py
        ├── test_config.py
        ├── test_catalog.py
        ├── test_target_selection.py
        ├── test_prompt_compiler.py
        ├── test_pack_planner.py
        ├── test_packed_batch_metadata.py
        ├── test_resume_state.py
        ├── test_synthetic_webdataset.py
        ├── test_whisper_variable_length.py
        ├── test_only_projector_trainable.py
        ├── test_causal_conv_boundary_gpu.py
        ├── test_lfm_packing_forward_gpu.py
        ├── test_lfm_packing_backward_gpu.py
        ├── test_overfit_gpu.py
        └── test_private_bucket_integration.py

Mark tests requiring CUDA with:

    @pytest.mark.gpu

Mark tests requiring `HF_TOKEN` and private data with:

    @pytest.mark.private_data

All ordinary unit tests must run without credentials by building a tiny synthetic local fixture.

---

# 3. Dependencies and installation

Use Python 3.11.

Pin:

    transformers==5.15.1
    causal-conv1d==1.6.2.post1
    flash-attn==2.8.3.post1

Do not declare `torch` or `torchaudio` as ordinary `pyproject.toml` dependencies because pip must not replace the user’s CUDA-matched PyTorch installation. Document that matching CUDA builds of Torch and torchaudio must be installed first.

Core dependencies:

    pydantic>=2
    pyyaml
    typer
    rich
    numpy
    pyarrow
    pandas
    soundfile
    webdataset
    safetensors
    zstandard
    tensorboard
    huggingface-hub
    pytest
    pytest-xdist
    ruff
    mypy

Optional dependency group:

    dmel

Create `scripts/install_cuda_extensions.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA PyTorch must be installed first")

print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("BF16:", torch.cuda.is_bf16_supported())

if not torch.cuda.is_bf16_supported():
    raise SystemExit("This training configuration requires CUDA BF16 support")
PY

python -m pip install --no-build-isolation \
  "flash-attn==2.8.3.post1" \
  "causal-conv1d==1.6.2.post1"
```

Do not enable broad Hugging Face `use_kernels=True` patching in the first implementation. Use the explicitly installed `causal-conv1d` package and FlashAttention 2. Broad kernel patching can be benchmarked only after correctness is established.

At process startup, record exact versions of:

- Python;
- Torch;
- CUDA runtime;
- CUDA driver when available;
- torchaudio;
- Transformers;
- flash-attn;
- causal-conv1d;
- WebDataset;
- pyarrow;
- CaptionStew package.

---

# 4. Configuration

Use YAML loaded into strict Pydantic models.

Unknown configuration fields must be errors.

Implement `${ENV:VARIABLE_NAME}` expansion without logging the resolved value for secret-like variables.

Create this initial production configuration:

```yaml
run:
  name: paraspeech-whisper-small-lfm2-projector
  seed: 1337
  output_dir: runs/paraspeech-whisper-small-lfm2-projector
  log_every_updates: 10
  checkpoint_every_updates: 250
  eval_every_updates: 250
  keep_last_checkpoints: 3
  fail_on_oom: true
  deterministic_algorithms: false

data:
  backend: captionstew
  captionstew_root: "${ENV:CAPTIONSTEW_ROOT}"
  dataset: ParaSpeechCaps-Base
  train_split: train_base
  validation_split: dev
  final_split: holdout
  target_type: style_caption
  target_sampling: one_per_audio_per_epoch
  metadata_source: parquet
  shard_shuffle: 53
  sample_shuffle: 256
  num_workers: 2
  persistent_workers: true
  prefetch_factor: 2
  max_bad_samples: 0
  strict_audio_contract: true
  max_audio_seconds: 30.0
  long_audio_policy: skip
  duration_sidecar: null
  preserve_provenance: true

prompt:
  prompt_file: configs/prompts/paraspeech_style_caption.txt
  audio_sentinel: "<<__AUDIO_EMBEDDINGS_08E8F7E7__>>"
  system_prompt: null
  supervise_assistant_termination: true

frontend:
  kind: whisper
  model_id: openai/whisper-small
  revision: main
  dtype: bfloat16
  mode: variable_length_masked
  feature_extraction_device: cpu
  max_seconds: 30.0
  encoder_microbatch_max_padded_samples: 960000

projector:
  kind: frame_stack_mlp
  stack_factor: 5
  hidden_dim: 2048
  activation: gelu
  dropout: 0.0
  use_input_layer_norm: true
  use_output_rms_norm: true
  use_trainable_audio_boundary_vectors: true
  initialize_to_text_embedding_rms: true

llm:
  model_id: LiquidAI/LFM2.5-1.2B-Instruct
  revision: main
  dtype: bfloat16
  attention_implementation: flash_attention_2
  trust_remote_code: false
  use_cache: false
  gradient_checkpointing: true
  gradient_checkpointing_use_reentrant: false
  allow_frozen_llm_dropout: false

packing:
  enabled: true
  max_lfm_tokens: 2048
  planning_buffer_examples: 64
  max_examples_per_pack: 8
  best_fit_decreasing: true
  require_boundary_kernel_tests: true

optimization:
  optimizer: adamw
  fused: true
  learning_rate: 0.0003
  min_learning_rate: 0.00003
  weight_decay: 0.01
  beta1: 0.9
  beta2: 0.95
  epsilon: 1.0e-8
  max_grad_norm: 1.0
  warmup_updates: 500
  max_updates: 20000
  target_input_tokens_per_update: 8192
  max_microbatches_per_update: 32

evaluation:
  validation_max_audio_items: null
  final_eval_enabled_during_training: false
  generation_examples: 32
  generation_max_new_tokens: 128
  generation_do_sample: false

checkpoint:
  save_optimizer: true
  save_scheduler: true
  save_rng: true
  save_committed_audio_ids: true
  atomic_write: true
```

Create a smoke configuration that overrides:

```yaml
data:
  sample_shuffle: 16
  num_workers: 0

packing:
  max_lfm_tokens: 512
  planning_buffer_examples: 8
  max_examples_per_pack: 2

optimization:
  target_input_tokens_per_update: 512
  warmup_updates: 2
  max_updates: 20

run:
  checkpoint_every_updates: 10
  eval_every_updates: 10
```

On startup:

1. Resolve `revision: main` to the immutable Hub commit SHA.
2. Save the resolved SHA in the run manifest.
3. On resume, require the same SHA.
4. Save SHA-256 of the tokenizer chat template.
5. Save SHA-256 of the prompt file.
6. Save fingerprints of the audio and target Parquet catalogs.
7. Save the fully resolved non-secret configuration.

A real run may begin from `main`, but every checkpoint and manifest must record the immutable resolved revision.

---

# 5. Core data types

Use typed dataclasses.

```python
@dataclass(frozen=True)
class TargetRecord:
    audio_id: str
    target_id: str
    target_type: str
    text: str
    split: str
    source: str
    review_status: str


@dataclass(frozen=True)
class CatalogAudioRecord:
    audio_id: str
    dataset: str
    source_id: str
    splits: tuple[str, ...]
    wds_key: str
    wds_shard: str
    flac_sha256: str
    flac_size: int
    target_count: int


@dataclass
class RawAudioExample:
    audio_id: str
    waveform: torch.Tensor  # CPU float32 [num_samples]
    sample_rate: int
    source_id: str
    splits: tuple[str, ...]
    style_captions: tuple[TargetRecord, ...]
    transcript: TargetRecord | None
    selected_target: TargetRecord
    metadata: dict[str, Any]
    crop_start_sample: int | None
    original_num_samples: int


@dataclass(frozen=True)
class PreparedText:
    before_audio_ids: tuple[int, ...]
    after_audio_prompt_ids: tuple[int, ...]
    target_suffix_ids: tuple[int, ...]
    target_id: str
    prompt_sha256: str


@dataclass
class PreparedExample:
    raw: RawAudioExample
    text: PreparedText
    estimated_audio_embedding_length: int
    estimated_total_lfm_length: int


@dataclass
class PackPlan:
    examples: list[PreparedExample]
    estimated_total_lfm_length: int


@dataclass
class PackedBatch:
    inputs_embeds: torch.Tensor
    labels: torch.Tensor
    position_ids: torch.Tensor
    seq_idx: torch.Tensor
    cu_seq_lens_q: torch.Tensor
    cu_seq_lens_k: torch.Tensor
    max_length_q: int
    max_length_k: int
    logical_lengths: list[int]
    audio_ids: list[str]
    target_ids: list[str]
    input_token_count: int
    supervised_token_count: int
```

Validate all shapes and dtypes in `PackedBatch.__post_init__` or a dedicated validation function.

---

# 6. Catalog implementation

Implement `CatalogIndex`.

It must use `pyarrow.dataset` and read only required columns.

For ParaSpeechCaps:

1. Load the audio catalog.
2. Select rows whose `splits` list contains the requested logical split.
3. Load official target rows.
4. Group target rows by `audio_id`.
5. Preserve target typing.
6. Expose:
   - `allowed_audio_ids`;
   - `audio_by_id`;
   - `style_captions_by_id`;
   - `transcript_by_id`;
   - `selected_shards`;
   - target-count distributions;
   - split-overlap report.

Hard assertions:

- every training audio ID belongs to `train_base`;
- no training audio ID belongs to `dev`;
- no training audio ID belongs to `holdout`;
- no validation audio ID belongs to `train_base`;
- no holdout ID is loaded into the training iterator;
- a ParaSpeech training example has at least one style caption;
- at most one transcript is selected per audio ID;
- target IDs are unique;
- audio IDs are unique in the audio catalog;
- `test` is not accepted as a configured ParaSpeech evaluation split.

If the source catalogs contain unexpected overlap beyond the documented holdout/test relationship, print the overlap counts and fail rather than guessing.

Do not silently filter by `review_status`. Log its distribution. Add an optional configured allow-list, but default to no status filter for official targets.

## Deterministic target choice

Implement:

```python
def stable_reference_index(
    *,
    seed: int,
    epoch: int,
    audio_id: str,
    num_references: int,
) -> int:
    payload = f"{seed}\0{epoch}\0{audio_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    value = int.from_bytes(digest[:8], "big", signed=False)
    return value % num_references
```

Sort candidate references by `target_id` before indexing.

Unit-test that reference choice:

- is stable across processes;
- changes for at least some audio items between epochs;
- never depends on Python hash randomization.

---

# 7. WebDataset streaming

Implement `CaptionStewBackend` around:

```python
from captionstew.training_client import open_webdataset
```

The backend must:

1. Receive `captionstew_root`, dataset name, and shard shuffle configuration.
2. use the existing private-Bucket streaming implementation;
3. split shards by node and worker using the installed WebDataset version’s supported splitter mechanism;
4. filter by `CatalogIndex.allowed_audio_ids`;
5. use bounded sample shuffling;
6. decode only after bounded shuffle where the CaptionStew pipeline permits it;
7. avoid any complete-shard cache;
8. yield one `RawAudioExample` at a time.

Inspect the actual installed `open_webdataset` signature rather than assuming it. If it already inserts `split_by_node` and `split_by_worker`, use it. If it exposes splitters as arguments, pass them explicitly. If it does neither, add a minimal adapter around the shard list returned by the CaptionStew client. Do not post-hoc assign samples to workers using `audio_id % num_workers`; shard splitting must happen before TAR expansion.

Decode with `soundfile`:

```python
array, sample_rate = sf.read(
    io.BytesIO(sample["flac"]),
    dtype="float32",
    always_2d=True,
)

if sample_rate != 16_000:
    raise DataContractError(...)

if array.shape[1] != 1:
    raise DataContractError(...)

waveform = torch.from_numpy(array[:, 0].copy())
```

Do not average channels.

Parse `sample["json"]`, preserve the complete metadata dictionary, and assert the JSON `audio_id` equals WebDataset `__key__`.

Use Parquet, not TAR iteration order, to choose split and target.

Cross-check the JSON target records against the Parquet target records in:

- all synthetic tests;
- the first 100 private-data samples during `inspect-data`;
- optionally every sample when `strict_target_consistency=true`.

Do not require this cross-check to be the target source. Parquet remains authoritative.

## Long audio

Implement these policies:

- `skip`;
- `center_crop`;
- `random_crop`.

Default to `skip`.

Do not silently crop. Record:

- original duration;
- crop policy;
- crop start;
- cropped duration.

For deterministic random crop, derive the crop offset from SHA-256 of seed, epoch, and audio ID.

A style caption may refer to the whole utterance, so the initial baseline must use `skip` rather than random cropping. `inspect-data` must report the percentage excluded by the 30-second limit before training.

## Duration sidecar

Allow an optional local sidecar:

    <run_dir>/cache/durations.parquet

Schema:

    audio_id: string
    num_samples: int64
    duration_seconds: float64
    flac_sha256: string

Treat it as derived cache only. Validate `flac_sha256` before reuse. Never modify official catalogs.

---

# 8. Synthetic offline data fixture

Build a pytest fixture generator that creates:

- several short 16 kHz mono FLAC files;
- one local uncompressed WebDataset TAR;
- JSON members matching the data contract;
- an audio Parquet catalog;
- an official-target Parquet catalog;
- train_base, dev, and holdout records;
- multiple style captions for at least one item;
- a separate transcript for each ParaSpeech item.

Use it for all CPU data tests.

Also create intentionally malformed fixtures:

- wrong sample rate;
- stereo audio;
- JSON key mismatch;
- duplicate audio ID;
- missing style caption;
- transcript accidentally typed as style caption;
- overlapping train/dev IDs.

Each malformed fixture must produce a specific exception, not a generic assertion failure.

---

# 9. Prompt compiler

Do not hardcode ChatML token IDs or manually assemble LFM special tokens.

Use the exact tokenizer and:

    tokenizer.apply_chat_template(..., tokenize=False)

Do not add new tokenizer vocabulary entries and do not resize LFM embeddings.

The configured prompt file contains an audio sentinel. Create this initial example:

```text
Listen to the following audio and describe the speaker's vocal style and paralinguistic characteristics. Do not transcribe the spoken words.

<<__AUDIO_EMBEDDINGS_08E8F7E7__>>
```

Document clearly that this is a direct style-caption baseline prompt, not a claimed reproduction of an external IFAO prompt or response-expansion method.

## Compilation algorithm

For an example with target text:

1. Read the user prompt and assert it contains the audio sentinel exactly once.
2. Build prompt-only messages:
   - optional configured system message;
   - user message containing the sentinel.
3. Render prompt-only chat using:
   - `add_generation_prompt=True`;
   - `tokenize=False`.
4. Build full messages by adding the assistant target.
5. Render the full chat using:
   - `add_generation_prompt=False`;
   - `tokenize=False`.
6. Assert both rendered strings contain the audio sentinel exactly once.
7. Split the prompt-only rendering into:
   - `before_audio_text`;
   - `after_audio_prompt_text`.
8. Split the full rendering into:
   - `full_before_audio_text`;
   - `full_after_audio_text`.
9. Assert `before_audio_text == full_before_audio_text`.
10. Assert `full_after_audio_text.startswith(after_audio_prompt_text)`.
11. Define:
    - target suffix text =
      `full_after_audio_text[len(after_audio_prompt_text):]`.
12. Tokenize each component with:
    - `add_special_tokens=False`.
13. Assert:

```python
tokenizer(
    after_audio_prompt_text + target_suffix_text,
    add_special_tokens=False,
).input_ids == (after_audio_prompt_ids + target_suffix_ids)
```

If this assertion fails because of a tokenizer boundary merge, implement an offset-mapping split using the fast tokenizer. Do not silently accept differing tokenization.

The sequence produced by the multimodal builder is:

    before_audio token embeddings
    trainable audio-start vector
    projected audio frame embeddings
    trainable audio-end vector
    after_audio_prompt token embeddings
    target_suffix token embeddings

Labels are:

    -100 for all prompt and audio positions
    actual token IDs for target_suffix positions

The target suffix must include the assistant termination token emitted by the model’s chat template when `supervise_assistant_termination=true`.

Save to the run manifest:

- tokenizer repository;
- resolved tokenizer revision;
- SHA-256 of `tokenizer.chat_template`;
- SHA-256 of prompt text;
- one rendered prompt example with target text redacted.

Unit tests must prove:

- no manually hardcoded LFM control-token IDs are used;
- target labels begin exactly at the assistant response;
- audio insertion does not remove the assistant generation prefix;
- assistant termination is supervised when configured;
- changing the chat template hash prevents checkpoint resume.

---

# 10. Audio frontend interface

Create an abstract interface:

```python
class AudioFrontend(nn.Module, ABC):
    output_dim: int

    @abstractmethod
    def estimate_output_lengths(
        self,
        num_samples: torch.LongTensor,
    ) -> torch.LongTensor: ...

    @abstractmethod
    def encode(
        self,
        waveforms: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Return one detached tensor [time, output_dim] per waveform.
        """
        ...
```

Frontend outputs must be detached and must never require gradients.

The projector owns all trainable reduction after this interface.

---

# 11. Whisper-small frontend

## 11.1 Loading

Load the exact revision of:

    openai/whisper-small

Instantiate only the encoder for runtime use. It is acceptable to load `WhisperModel`, retain `.encoder`, delete the decoder, and release unused memory.

Set:

    encoder.eval()
    encoder.requires_grad_(False)

Use BF16 on CUDA.

Use the matching `WhisperFeatureExtractor`.

## 11.2 Variable-length issue

Do not call the standard `WhisperEncoder.forward` on short, longest-padded batches. The pinned Transformers implementation expects the fixed maximum Mel length.

Implement a pinned-version adapter named:

    VariableLengthWhisperEncoder

It reuses the loaded encoder’s existing modules and weights:

- `conv1`;
- `conv2`;
- learned positional embeddings;
- encoder layers;
- final layer norm.

This is an explicit code extension and must be tested against the pinned Transformers source.

## 11.3 Feature extraction

Batch waveforms with:

```python
features = feature_extractor(
    waveforms_as_numpy,
    sampling_rate=16_000,
    padding="longest",
    truncation=False,
    return_attention_mask=True,
    return_tensors="pt",
)
```

Reject inputs longer th the configured maximum before this call.

The feature attention mask is in Mel-frame units. After Whisper’s stride-2 convolution, compute each valid encoder length with the actual convolution output-length formula. Do not assume simple floor division without testing boundary lengths.

## 11.4 Variable encoder forward

Conceptually:

```python
with (
    torch.no_grad(),
    torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ),
):
    hidden = F.gelu(encoder.conv1(input_features))
    hidden = F.gelu(ender.conv2(hidden))
    hidden = hidden.transpose(1, 2)

    time = hidden.shape[1]
    hidden = hidden + encoder.embed_positions.weight[:time]

    encoder_lengths = conv_output_lengths(mel_lengths)

    # Mask only padded keys. We later slice invalid query outputs.
    additive_mask = make_additive_key_padding_mask(
        lengths=encoder_lengths,
        max_length=time,
        dtype=hidden.dtype,
        device=hidden.device,
    )

    for layer in encoder.layers:
        hidden = call_pinned_whisper_layer(
            layer,
            hidden,
            attention_mask=additive_mask,
        )

    hidden = encoder.layer_norm(hidden)

outputs = [hidden[i, : encoder_lengths[i]].detach() for i in range(batch_size)]
```

Inspect the exact return type of a Whisper encoder layer in Transformers 5.15.1 and handle it explicitly. Do not write version-agnostic reflection around arbitrary layouts.

Set encoder dropout and LayerDrop to evaluation behavior.

## 11.5 Required Whisper tests

Test 1: full-length equivalence

- create a full expected-length Mel input;
- run official `WhisperEncoder.forward`;
- run `VariableLengthWhisperEncoder`;
- use no padding mask;
- assert final valid outputs are numerically close.

Test 2: padded-batch isolation

- run short A and short B individually;
- run A and B padded together;
- assert valid outputs are close to the individual outputs.

Test 3: padded-value invariance

- modify only padded Mel values for B;
- assert valid B outputs are unchanged.

Test 4: output-length estimator

- test waveform lengths around every relevant hop and stride boundary;
- assert estimated projected length is never smaller than actual length;
- assert normal examples have exact estimates.

Provide a correctness fallback configuration:

    frontend.mode: official_fixed_30s

It may invoke the official encoder with 30-second padding. Never silently switch to it. Log a clear throughput warning when explicitly enabled.

---

# 12. Projector

Implement `FrameStackMLPProjector`.

For Whisper-small, the frontend dimension should be discovered from the loaded encoder config rather than hardcoded.

Given audio features:

    x: [T, D_encoder]

Use deterministic frame stacking with `stack_factor=5`:

1. pad only the final incomplete stack;
2. reshape to:
   `[ceil(T / 5), 5 * D_encoder]`;
3. preserve the exact valid output length.

Then apply:

    LayerNorm(5 * D_encoder)
    Linear(5 * D_encoder, hidden_dim)
    GELU
    Linear(hidden_dim, LFM_hidden_size)
    RMSNorm or numerically stable RMS normalization
    learned scalar output scale

Add trainable:

    audio_start: [LFM_hidden_size]
    audio_end:   [LFM_hidden_size]

Both belong to the projector.

Initialize the final projected-vector RMS to approximately the RMS of the frozen LFM token embedding table:

```python
with torch.no_grad():
    target_rms = llm.get_input_embeddings().weight.float().pow(2).mean().sqrt()
```

Use a learned multiplicative scalar initialized so projected vectors begin near this RMS. Store the computed initialization statistic in the run manifest.

Keep projector parameters in FP32. Under CUDA BF16 autocast, its matrix operations may execute in BF16 while AdamW maintains FP32 parameters and states.

Expose:

```python
def projected_length(self, frontend_length: int) -> int:
    return math.ceil(frontend_length / self.stack_factor)
```

Do not include trainable parameters outside the `projector` namespace.

---

# 13. dMel frontend and projector

Implement only after the Whisper baseline passes all tests.

Use the official `dmel` package for deterministic discretized log-Mel extraction. Do not reimplement its quantization initially.

`DmelFrontend` returns integer codes:

    [time, num_mel_channels]

It has no trainable parameters.

Implement `DmelProjector` under the same projector namespace. A reasonable initial design is:

1. shared dMel-bin embedding;
2. learned channel embedding;
3. combine code and channel embeddings;
4. flatten or concatenate channels within each frame;
5. deterministic temporal patching;
6. two-layer MLP into LFM hidden size;
7. output RMS normalization and learned scaling;
8. audio-start and audio-end vectors.

Configuration example:

```yaml
frontend:
  kind: dmel
  sample_rate: 16000

projector:
  kind: dmel_patch_mlp
  dmel_bin_embedding_dim: 16
  temporal_patch_size: 8
  hidden_dim: 2048
  use_trainable_audio_boundary_vectors: true
```

Do not feed every channel/bin as an independent LFM token. That would make sequence length impractical on one 4090.

The dMel implementation must satisfy the same `PreparedExample`, packing, LFM, loss, and checkpoint interfaces as Whisper.

---

# 14. Online packing planner

Packing occurs independently inside each DataLoader worker after WDS shard/sample
shuffle and prompt compilation, but before FLAC waveform decode, log-Mel
extraction, or GPU audio encoding. Exact `num_samples` comes from the
FLAC-STREAMINFO-verified duration sidecar. For complete local mirrors, that
sidecar also carries exact FLAC/JSON TAR byte ranges; the planning window stores
only those lightweight references and reads payload bytes after selecting a bin.
The main training loop must never receive individual examples or run the packer.

Each `PreparedExample` has a conservative estimated total length:

    before_audio text
    + audio_start
    + estimated projected audio length
    + audio_end
    + after_audio prompt
    + target suffix

Implement rolling best-fit-decreasing packing:

1. let WDS split shards by rank/worker and apply its bounded shuffle;
2. read up to `planning_buffer_examples` lightweight shuffled records;
3. prepare target and text tokens;
4. calculate exact logical LFM length from verified `num_samples`;
5. sort descending by estimated length;
6. place into the existing bin with the smallest remaining capacity that fits;
7. otherwise open a new bin;
8. enforce `max_examples_per_pack`;
9. decode, validate, chunk, pad, and compute CPU log-Mels only for the next bin;
10. construct audio masks, supervised-token indices, and a complete CPU
    `HostAudioBatch` in the worker;
11. yield that packed batch through the bounded DataLoader queue;
12. refill the worker-local planning buffer.

Do not require an offline global sort.

A `PackPlan` must also be compatible with the frontend encoder microbatch budget. The frontend may split audio encoding within one pack into smaller audio microbatches. This does not alter LFM packing.

After actual frontend and projector execution:

- compute exact logical lengths;
- assert total length is at most `max_lfm_tokens`;
- if a conservative estimator overestimated, proceed;
- if it underestimated and the actual pack exceeds the limit, fail with the responsible audio IDs and estimator details;
- do not truncate projected audio embeddings to make the pack fit.

Log:

- pack utilization;
- logical examples per pack;
- audio duration per pack;
- projected audio tokens per second of audio;
- text tokens per pack;
- target tokens per pack;
- planning-buffer wait time.

---

# 15. Packed batch construction

The pack planner produces a complete CPU `HostAudioBatch`. The main/GPU path performs:

1. frontend encoding under `no_grad`;
2. projector forward with gradients;
3. frozen text embedding lookup;
4. logical-sequence assembly;
5. flattened packing metadata construction.

Use the frozen embedding table under `no_grad()` for ordinary text-token embeddings:

```python
with torch.no_grad():
    text_embeds = llm.get_input_embeddings()(token_ids)
```

Do not use `no_grad()` for:

- projector;
- LFM backbone;
- LM head applied to selected hidden states.

For each logical sequence:

```python
logical_embeds = torch.cat(
    [
        embed(before_audio_ids),
        projector.audio_start[None],
        projected_audio,
        projector.audio_end[None],
        embed(after_audio_prompt_ids),
        embed(target_suffix_ids),
    ],
    dim=0,
)
```

Labels:

```python
labels = torch.full(
    (logical_length,),
    -100,
    dtype=torch.long,
    device=device,
)

labels[target_suffix_start:] = target_suffix_ids
```

Then flatten:

```python
inputs_embeds = torch.cat(logical_embeds_list, dim=0).unsqueeze(0)
labels = torch.cat(logical_labels_list, dim=0).unsqueeze(0)

position_ids = torch.cat(
    [torch.arange(length) for length in logical_lengths],
).unsqueeze(0)

seq_idx = torch.cat(
    [
        torch.full((length,), sequence_index, dtype=torch.int32)
        for sequence_index, length in enumerate(logical_lengths)
    ],
).unsqueeze(0)

cu = torch.tensor(
    [0, *itertools.accumulate(logical_lengths)],
    dtype=torch.int32,
    device=device,
)
```

Move `position_ids` to CUDA and use `torch.long`.

Set:

```python
cu_seq_lens_q = cu
cu_seq_lens_k = cu
max_length_q = max(logical_lengths)
max_length_k = max(logical_lengths)
```

No conventional attention mask is allowed.

Validate:

- total dimensions agree;
- every sequence’s positions begin at zero;
- `seq_idx` isonstant within each sequence;
- cumulative lengths end at total tokens;
- no sequence is empty;
- every sequence has at least one supervised token;
- all target IDs fit the LFM vocabulary;
- no labels from one sequence are assigned to another sequence.

---

# 16. LFM loading and model wrapper

Load:

```python
llm = AutoModelForCausalLM.from_pretrained(
    model_id,
    revision=resolved_revision,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=False,
)
```

Move it to the single CUDA device. Do not use `device_map="auto"`.

Then:

```python
llm.requires_grad_(False)
llm.config.use_cache = False

if gradient_checkpointing:
    llm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
```

Assert:

- model class is the expected LFM2 causal-LM class;
- `llm.model` exists and is the LFM backbone;
- `llm.lm_head` exists;
- LM head and token embeddings are tied if the checkpoint declares tied embeddings;
- LFM hidden size equals projector output size;
- context length is at least the configured pack length;
- attention implementation is FlashAttention 2;
- no LLM parameter requires gradients.

Use `llm.model` directly during training to avoid creating full-vocabulary logits at all sequence positions.

The model wrapper owns:

- frontend;
- projector;
- tokenizer;
- frozen LFM causal LM;
- prompt compiler;
- packed-batch builder.

---

# 17. Training forward and selective LM head

Use this logical structure:

```python
def forward_packed(self, batch: PackedBatch) -> LossOutput:
    outputs = self.llm.model(
        inputs_embeds=batch.inputs_embeds,
        attention_mask=None,
        position_ids=batch.position_ids,
        use_cache=False,
        seq_idx=batch.seq_idx,
        cu_seq_lens_q=batch.cu_seq_lens_q,
        cu_seq_lens_k=batch.cu_seq_lens_k,
        max_length_q=batch.max_length_q,
        max_length_k=batch.max_length_k,
        return_dict=True,
    )

    hidden = outputs.last_hidden_state[:, :-1, :]
    shifted_labels = batch.labels[:, 1:]

    supervised = shifted_labels.ne(-100)
    selected_hidden = hidden[supervised]
    selected_labels = shifted_labels[supervised]

    if selected_labels.numel() == 0:
        raise RuntimeError("packed batch has no supervised tokens")

    logits = self.llm.lm_head(selected_hidden)

    loss_sum = F.cross_entropy(
        logits.float(),
        selected_labels,
        reduction="sum",
    )

    return LossOutput(
        loss_sum=loss_sum,
        supervised_tokens=int(selected_labels.numel()),
        input_tokens=batch.input_token_count,
    )
```

Do not construct:

    [total_sequence_length, vocabulary_size]

logits for unsupervised positions.

Unit-test selective-logit loss against a full-logit reference on short synthetic sequences.

---

# 18. Causal-convolution and packing preflight

Implement a mandatory command:

    audio-lfm preflight --config <config>

It must run before production training.

## 18.1 Environment checks

Verify:

- CUDA is available;
- one selected CUDA device exists;
- BF16 is supported;
- FlashAttention imports successfully;
- causal-conv1d imports successfully;
- required package versions match the lock;
- sufficient free disk exists for model checkpoints and model cache;
- the private Bucket is accessible when `--check-private-data` is supplied;
- no secret is printed.

## 18.2 Direct causal-conv boundary test

Test `causal_conv1d_fn` directly with a width-3 depthwise convolution.

Construct sequence A and sequence B, concatenate them, and supply:

    seq_idx = [0 ... 0, 1 ... 1]

Compare the packed result against separately convolving A and B and concatenating their outputs.

Test both:

- forward output;
- gradient with respect to input.

Also perturb A and assert B’s packed output is unchanged.

This test must fail if a fallback implementation ignores `seq_idx`.

## 18.3 Full LFM packed-forward test

Load LFM2 in BF16 with FlashAttention 2.

Create two random logical `inputs_embeds` sequences A and B.

Run:

- A separately;
- B separaty;
- A and B packed.

Use reset position IDs and complete packing metadata.

Assert packed hidden states match separate hidden states within a documented BF16 tolerance.

Then perturb only A and verify B’s packed hidden states remain unchanged.

## 18.4 Full LFM backward-isolation test

Create trainable leaf input embeddings for A and B.

Pack them.

Define a scalar loss using only the first few B output positions.

Backpropagate.

Assert:

- B receives a nonzero gradient;
- A receives zero gradient withinumerical tolerance.

Also run a toy projector separately and packed, and compare projector gradients.

## 18.5 Failure behavior

If any packing isolation test fails:

- print the exact installed package versions;
- explain whether attention or convolution isolation failed;
- abort packed training;
- do not automatically fall back to SDPA;
- do not automatically disable packing.

An explicit `packing.enabled=false` mode may exist for debugging and reference tests, but the production configuration must remain packed.

---

# 19. Training engine

Use a direct PyTorch loop.

At startup:

```python
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
```

Modes:

```python
model.frontend.eval()
model.projector.train()
model.llm.train()
```

Assert encoder and LLM parameters have `grad is None` after every optimizer update during the smoke test.

Use:

```python
torch.autocast(
    device_type="cuda",
    dtype=torch.bfloat16,
)
```

Do not use GradScaler for BF16.

## Token-normalized gradient accumulation

Accumulate until:

    accumulated_input_tokens >= target_input_tokens_per_update

or:

    accumulated_microbatches == max_microbatches_per_update

For each microbatch:

```python
loss_output.loss_sum.backward()
accumulated_supervised_tokens += loss_output.supervised_tokens
accumulated_input_tokens += loss_output.input_tokens
```

Before the optimizer step, divide every projector gradient by the total number of supervised tokens:

```python
for parameter in projector.parameters():
    if parameter.grad is not None:
        parameter.grad.div_(accumulated_supervised_tokens)
```

Then:

1. clip projector gradient norm;
2. optimizer step;
3. scheduler step;
4. zero gradients with `set_to_none=True`;
5. mark all audio IDs from the completed update as committed;
6. increment global update.

This computes a true supervised-token mean across variable-sized microbatches.

Do not average each microbatch loss independently before accumulation.

## Optimizer

Pass only:

```python
[p for p in model.projector.parameters() if p.requires_grad]
```

to AdamW.

Use fused AdamW when supported. If fused AdamW is unavailable, log the reason and use ordinary AdamW.

At startup, print:

- trainable parameter names;
- trainable parameter count;
- frozen encoder parameter count;
- frozen LLM parameter count.

Fail if any trainable parameter name does not begin with `projector.`.

## OOM behavior

Default behavior is to fail.

On CUDA OOM:

- do not skip the pack silently;
- zero incomplete gradients;
- do not mark its audio IDs committed;
- write an OOM diagnostic containing logical lengths and audio durations;
- exit with a recommendation to lower `max_lfm_tokens` or encoder microbatch budget.

Do not implement an automatic length-changing retry in the first version.

---

# 20. Metrics

Log to stdout, JSONL, and TensorBoard.

Per optimizer update:

- update number;
- epoch;
- total loss;
- token-normalized NLL;
- perplexity with a capped exponent for display;
- input tokens;
- supervised tokens;
- logical examples;
- packs;
- pack utilization;
- projected audio tokens;
- decoded audio seconds;
- input tokens/s;
- supervised tokens/s;
- audio seconds/s;
- examples/s;
- frontend time;
- projector time;
- LFM forward/backward time;
- data wait time;
- optimizer time;
- learning rate;
- gradient norm;
- allocated CUDA memory;
- reserved CUDA memory;
- peak CUDA memory;
- long-audio skip count;
- decode failure count;
- reference-selection distribution.

Never log transcript or caption text by default. Allow opt-in sanitized sample logging.

---

# 21. Checkpointing and resume

Do not save frozen LFM or encoder weights.

Each checkpoint directory contains:

    projector.safetensors
    optimizer.pt
    scheduler.pt
    trainer_state.json
    rng_state.pt
    data_state.json
    committed_audio_ids.txt.zst
    resolved_config.yaml
    run_manifest.json

Use atomic checkpoint creation:

1. write to a temporary directory;
2. fsync relevant files;
3. rename the directory atomically.

`trainer_state.json` includes:

- global update;
- current epoch;
- input tokens processed;
- supervised tokens processed;
- audio seconds processed;
- best validation metric;
- accumulated metrics;
- checkpoint format version.

Save RNG state for:

- Python;
- NumPy;
- Torch CPU;
- Torch CUDA.

## Iterable-stream resume semantics

Exact byte-offset restoration inside remote WebDataset pipes is not required.

Implement:

    at-least-once data reading
    exactly-once committed optimizer updates within the current epoch

Maintain a set of audio IDs committed in the current epoch.

Only add IDs after a successful optimizer step.

On resume:

1. restore model, optimizer, scheduler, RNG, epoch, and committed-ID set;
2. rebuild the deterministic epoch stream;
3. filter already committed audio IDs;
4. allow remote shards and samples to be reread;
5. never retrain a committed audio ID in the same epoch.

When the epoch completes:

- advance epoch;
- clear the committed-ID set;
- change deterministic target-reference selection using the new epoch.

Store committed IDs sorted and compressed with Zstandard.

Add a test that:

1. runs several updates;
2. checkpoints;
3. reconstructs the process;
4. resumes;
5. verifies no committed audio ID is trained twice;
6. verifies the restored optimizer and scheduler states match.

On resume, fail if any of these changed:

- LFM resolved revision;
- encoder resolved revision;
- tokenizer chat-template hash;
- prompt hash;
- projector architecture;
- target type;
- split configuration;
- catalog fingerprint;
- packing semantics version.

Allow an explicit `--allow-nonreproducible-resume` override only for development, and record it prominently in the manifest.

---

# 22. Validation and final evaluation

## Teacher-forced validation

Use the same packed model path and boundary metadata.

Evaluation may run under full `torch.no_grad()` because no gradients are needed.

For each ParaSpeech audio item, evaluate every style-caption reference.

Report:

- aggregate target-level NLL;
- aggregate target-level token count;
- mean target NLL;
- per-audio mean NLL;
- mean of per-audio NLLs;
- reference-count distribution.

Do not use holdout during periodic training evaluation.

Periodic evaluation uses only `dev`.

Holdout evaluation must require an explicit command:

    audio-lfm evaluate --config ... --checkpoint ... --split holdout

Reject:

    --split test

for ParaSpeechCaps.

## Generation

Implement non-packed generation for qualitative evaluation.

For each logical item:

1. construct prompt and continuous audio embeddings;
2. run one prefill with `use_cache=True`;
3. generate subsequent text tokens autoregressively;
4. stop on the tokenizer/model EOS condition;
5. decode only generated tokens.

Initially implement greedy generation. Add sampling parameters later.

Do not pack independent examples during autoregressive generation.

Prediction records must include:

- audio_id;
- source_id;
- target IDs;
- reference texts when explicitly requested;
- generated text;
- checkpoint identifier;
- LFM model and resolved revision;
- encoder model and resolved revision;
- projector configuration;
- prompt SHA-256;
- generation parameters;
- dataset provenance and licensing fields from the source metadata;
- crop information.

Write JSONL and Parquet.

---

# 23. Response-target expansion

The ready dataset contains official targets only. Response expansion is a separate preprocessing operation.

Implement the command:

    audio-lfm expand-responses

Do not run it automatically as part of training.

The exact response-expansion prompt is not specified by the dataset contract. Therefore:

- require a user-supplied prompt file;
- do not invent or claim an IFAO reproduction prompt;
- refuse to run when the prompt remains the provided example placeholder.

When response expansion is used:

- use exactly the same LFM checkpoint and immutable revision as the frozen training decoder;
- use exactly the same tokenizer revision and chat template;
- record `generator_model`;
- record immutable `generator_revision`;
- record `prompt_sha256`;
- record `source`;
- record `review_status`;
- join outputs by `audio_id`;
- write append-only overlay records;
- never rewrite TAR files;
- never key records by batch index or TAR order.

Support either:

1. the existing CaptionStew CRUD API; or
2. direct append-only Parquet export under:

       CaptionStew/_webdataset/<dataset>/16k-flac/parquet/overlays/kind=response/

Do not overwrite existing overlay records with the same logical identity without explicit optimistic-concurrency handling.

The training target provider should support:

- `official_target`, immediately runnable;
- `response_overlay`, which fails clearly if the requested overlay is absent.

The initial production configuration uses `official_target`.

---

# 24. CLI

Use Typer and expose:

```text
audio-lfm preflight
audio-lfm inspect-data
audio-lfm inspect-model
audio-lfm test-packing
audio-lfm train
audio-lfm evaluate
audio-lfm generate
audio-lfm expand-responses
audio-lfm benchmark
```

## `preflight`

Checks environment, CUDA extensions, model loading, optional Bucket access, and all packing-isolation tests.

## `inspect-data`

Streams a configured number of examples and reports:

- split counts;
- target-type counts;
- reference-count distribution;
- transcript presence;
- duration quantiles;
- long-audio exclusion rate;
- decoded sample-rate and channel checks;
- JSON/Parquet consistency;
- duplicate audio IDs;
- worker duplication.

It must not retain waveform arrays after statistics are updated.

## `inspect-model`

Reports:

- resolved revisions;
- parameter counts;
- trainable names;
- hidden dimensions;
- embedding RMS;
- projected output-rate estimate;
- estimated sequence lengths for 1, 5, 10, 20, and 30 seconds;
- approximate checkpoint size.

## `test-packing`

Runs the required direct causal-conv and full-LFM forward/backward isolation tests.

## `benchmark`

Runs a fixed number of synthetic or private-data steps and reports separate data, encoder, projector, LFM, and optimizer timings. It must support comparing:

- unpacked;
- packed;
- Whisper official fixed-30s;
- Whisper variable-length;
- several pack-token limits.

Do not enable `torch.compile` in the initial benchmark.

---

# 25. README commands

Document this workflow:

```bash
export HF_TOKEN='<hugging-face-token>'
export CAPTIONSTEW_REPO=/path/to/captionstew-repository
export CAPTIONSTEW_ROOT=/path/to/CaptionStew

python -m pip install -e "${CAPTIONSTEW_REPO}[training]"

# Install a matching CUDA build of torch and torchaudio first.
python -m pip install -e '.[dev,whisper]'
bash scripts/install_cuda_extensions.sh

hf buckets info seastar105/caption-stew

audio-lfm preflight \
  --config configs/paraspeech_whisper_lfm2.yaml \
  --check-private-data

audio-lfm inspect-data \
  --config configs/paraspeech_whisper_lfm2.yaml \
  --num-samples 256

pytest -q
pytest -q -m gpu
pytest -q -m private_data

audio-lfm train \
  --config configs/paraspeech_whisper_lfm2_smoke.yaml

audio-lfm train \
  --config configs/paraspeech_whisper_lfm2.yaml

audio-lfm train \
  --config configs/paraspeech_whisper_lfm2.yaml \
  --resume runs/paraspeech-whisper-small-lfm2-projector/checkpoint-00002500

audio-lfm evaluate \
  --config configs/paraspeech_whisper_lfm2.yaml \
  --checkpoint runs/paraspeech-whisper-small-lfm2-projector/checkpoint-best \
  --split dev

audio-lfm evaluate \
  --config configs/paraspeech_whisper_lfm2.yaml \
  --checkpoint runs/paraspeech-whisper-small-lfm2-projector/checkpoint-best \
  --split holdout
```

Explain that model-weight caching is allowed and unavoidable, but audio TAR caching is prohibited.

---

# 26. Required tests

## CPU tests

### Configuration

- unknown fields rejected;
- environment expansion works;
- secret values are never included in serialized config or logs;
- invalid split combinations rejected.

### Catalog

- stable audio_id join;
- duplicate IDs rejected;
- style captions and transcripts remain separate;
- ParaSpeech test split rejected;
- train/dev/holdout leakage detected;
- reference selection deterministic.

### Data

- synthetic WebDataset streams;
- 16 kHz mono enforced;
- stereo rejected;
- wrong rate rejected;
- JSON key mismatch rejected;
- provenance retained;
- long-audio policies deterministic;
- duration sidecar validated by FLAC hash.

### Prompt compiler

- exact chat template used;
- audio inserted at intended location;
- target suffix labels correct;
- assistant terminator supervision correct;
- no tokenizer vocabulary modification;
- template-hash mismatch blocks resume.

### Pack planner

- no pack exceeds token limit;
- every example emitted once;
- deterministic for fixed inputs;
- utilization calculation correct;
- large example produces a singleton pack;
- oversized singleton fails rather than truncates.

### Resume

- only committed IDs skipped;
- pending accumulation IDs are replayed after crash;
- completed epoch clears committed set;
- model revision mismatch rejected.

## GPU tests

### Whisper

- custom full-length path matches official full-length encoder;
- variable-length padded batch matches individual variable-length calls;
- padding perturbation does not affect valid outputs;
- frontend outputs are detached.

### Trainable parameters

- only projector parameters require gradients;
- encoder gradients remain `None`;
- LFM parameter gradients remain `None`;
- projector gradients are nonzero.

### Causal-conv boundary

- direct forward packed/separate equivalence;
- direct backward packed/separate equivalence;
- perturbing A does not affect B.

### Full LFM packing

- packed/separate hidden-state equivalence;
- perturbing A does not affect B;
- loss on B yields zero input gradient on A;
- projector gradient packed/unpacked equivalence.

### Loss

- selective LM-head loss matches full-logit loss;
- packed label shift never crosses logical boundaries;
- target token normalization is correct across variable microbatches.

### Overfit

Build a fixed 8–16-example synthetic or private-data subset.

Train only the projector until:

- loss decreases substantially;
- projector weights change;
- frozen LFM and encoder checksums remain unchanged;
generated outputs begin to copy or approximate the tiny target set.

### Memory stability

Run at least 100 smoke updates and verify:

- allocated memory does not grow monotonically;
- no retained logits from previous steps;
- no waveform/TAR accumulation;
- checkpoint creation releases temporary tensors.

## Private integration test

When credentials exist:

- read the private Bucket without logging the token;
- decode at least 100 samples;
- verify `__key__ == audio_id`;
- verify 16 kHz mono;
- verify Parquet target lookup;
- verify no duplicate IDs across two DataLoader workers;
- verify ParaSpeech split behavior;
- verify no full TAR appears in local cache directories.

---

# 27. Development sequence

Implement in this order.

## Milestone 1: repository and synthetic data

Deliver:

- packaging;
- strict configuration;
- synthetic CaptionStew fixture;
- Parquet catalog;
- WebDataset decoding;
- split and target tests.

Do not load neural models yet.

## Milestone 2: prompt compiler and logical examples

Deliver:

- tokenizer loading;
- chat-template-based prompt compiler;
- target labels;
- deterministic reference selection;
- CPU tests.

## Milestone 3: Whisper frontend

Deliver:

- frozen Whisper loader;
- variable-length wrapper;
- output-length estimator;
- full-length equivalence and padding-isolation tests.

## Milestone 4: projector and un-packed reference model

Deliver:

- projector;
- audio/text embedding assembly;
- one-example un-packed LFM forward;
- selective LM-head loss;
- projector-gradient test.

The un-packed path remains as a correctness oracle.

## Milestone 5: packed LFM

Deliver:

- pack planner;
- packed metadata;
- direct causal-conv tests;
- full-LFM forward/backward isolation tests;
- hard preflight gate.

Do not begin long training until this milestone passes.

## Milestone 6: trainer and checkpointing

Deliver:

- token-normalized accumulation;
- optimizer and scheduler;
- metrics;
- atomic projector-only checkpoints;
- committed-ID resume semantics;
- smoke training.

## Milestone 7: evaluation

Deliver:

- all-reference dev NLL;
- explicit holdout evaluation;
- qualitative generation;
- prediction JSONL and Parquet.

## Milestone 8: dMel

Deliver:

- dMel frontend;
- dMel projector;
- same packing and trainer interfaces;
- comparison configuration.

## Milestone 9: response overlays

Deliver:

- generic user-prompt-driven expansion;
- append-only output;
- same-model/revision/template checks;
- overlay target provider.

After each milestone:

1. run formatting;
2. run static checks;
3. run relevant tests;
4. update README;
5. do not leave `TODO`, `pass`, placeholder exceptions, or mocked critical paths in production modules.

---

# 28. Definition of done

The repository is complete only when all of the following are true:

1. It installs from an empty repository after CUDA PyTorch is installed.
2. CPU tests pass without credentials.
3. GPU packing tests pass on the RTX 4090.
4. Private-data preflight streams FLAC from the HF Bucket without bulk download.
5. `HF_TOKEN` never appears in logs, manifests, checkpoints, or exceptions.
6. Parquet `audio_id` joins are used for split and target selection.
7. ParaSpeech style captions and transcripts remain separately typed.
8. Training uses `train_base`, validation uses `dev`, and final evaluation uses `holdout`.
9. No separate ParaSpeech test loader exists.
10. The encoder runs detached under `no_grad()`.
11. The frozen LFM runs with autograd enabled during training.
12. Only projector parameters are optimized.
13. Full-vocabulary logits are computed only for supervised hidden positions.
14. Packed attention cannot cross logical sequences.
15. Packed short convolution cannot cross logical sequences.
16. Forward and backward boundary tests fail under an intentionally forced fallback kernel.
17. Dynamic online packing works without offline audio preprocessing.
18. Training resumes without retraining committed audio IDs in the same epoch.
19. Checkpoints contain projector state but not frozen LFM or encoder weights.
20. A tiny subset can be overfit.
21. A 100-update smoke run has stable CUDA and host memory.
22. Dev evaluation covers all style-caption references.
23. Holdout evaluation requires an explicit command.
24. Generated predictions preserve source provenance fields.
25. dMel can replace Whisper without changing the trainer or packing interfaces.
26. The README contains exact installation, preflight, smoke, train, resume, and evaluation commands.

When installed APIs differ from this specification, inspect the pinned library source and adapt the implementation while preserving these behavioral contracts. Do not weaken boundary isolation, dataset typing, frozen-gradient semantics, or resume guarantees merely to fit a framework abstraction.

# 29. vLLM generation backend

Add a vLLM backend for high-throughput autoregressive generation.

Do not replace the existing Hugging Face evaluation implementation.

The backend division is:

    Hugging Face / PyTorch
        -> teacher-forced NLL
        -> all-reference NLL
        -> packed/unpacked correctness oracle
        -> projector and first-token parity tests

    vLLM
        -> greedy or sampled autoregressive generation
        -> batched dev/holdout prediction generation
        -> generation throughput benchmarks
        -> generation-derived text metrics

The initial vLLM implementation is offline only:

    from vllm import LLM, SamplingParams

Do not implement an OpenAI-compatible server until offline generation passes all
parity, isolation, split, resume, and memory-stability tests.

Do not manually sequence-pack evaluation examples. Submit one logical audio item
as one vLLM request. vLLM owns request batching, LFM attention KV state, and LFM
short-convolution state.

The plugin architecture is:

    AudioLfm2ForConditionalGeneration
        ├── audio_tower
        │     └── shared VariableLengthWhisperEncoder
        ├── multi_modal_projector
        │     └── exact projector used during training
        └── language_model
              └── native vLLM Lfm2ForCausalLM

Do not copy or rewrite the native vLLM LFM2 implementation.

---

# 30. Version and environment isolation

Pin the initial integration to:

    vllm==0.27.1

Use a separate environment because vLLM supplies a tightly coupled Torch/CUDA
runtime and compiled kernels.

Create:

    requirements-vllm.txt
    scripts/create_vllm_env.sh

`requirements-vllm.txt`:

```text
pyarrow
pandas
webdataset
soundfile
safetensors
zstandard
pyyaml
pydantic>=2
typer
rich
numpy
tensorboard
```

Do not list:

- torch;
- torchaudio;
- flash-attn;
- causal-conv1d;
- transformers;
- vLLM.

Install vLLM separately so its compatible dependencies are selected first.

`scripts/create_vllm_env.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${VENV_PATH:-.venv-vllm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

uv venv "${VENV_PATH}" \
  --python "${PYTHON_VERSION}" \
  --seed

# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

uv pip install \
  "vllm[audio]==0.27.1" \
  --torch-backend=auto

uv pip install -r requirements-vllm.txt

if [[ -z "${CAPTIONSTEW_REPO:-}" ]]; then
  echo "CAPTIONSTEW_REPO must point to the CaptionStew source repository" >&2
  exit 1
fi

uv pip install -e "${CAPTIONSTEW_REPO}[training]"

# Install this repository without allowing its training dependencies to replace
# the Torch/Transformers stack selected by vLLM.
uv pip install -e . --no-deps

python - <<'PY'
import importlib.metadata
import torch
import transformers
import vllm

print("vLLM:", importlib.metadata.version("vllm"))
print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if importlib.metadata.version("vllm") != "0.27.1":
    raise SystemExit("Unexpected vLLM version")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is required")

if not torch.cuda.is_bf16_supported():
    raise SystemExit("BF16 support is required")

print("GPU:", torch.cuda.get_device_name(0))
PY
```

Use the same shared Hugging Face model cache for training and vLLM so the LFM and
Whisper weights are not duplicated:

```bash
export HF_HOME=/path/to/shared/huggingface-cache
export VLLM_CACHE_ROOT=/path/to/vllm-cache
```

Model-weight caching is allowed. Audio TAR caching is not.

The vLLM preflight must record the exact versions of:

- vLLM;
- Torch;
- CUDA runtime;
- CUDA driver;
- Transformers;
- tokenizers;
- safetensors;
- CaptionStew;
- this repository's Git commit.

The training checkpoint already records its Transformers version. Compare it
against the vLLM environment. Initially require an exact match because the
shared variable-length Whisper wrapper depends on pinned Transformers behavior.
Fail rather than silently use a different frontend implementation.

---

# 31. Repository additions

Add:

```text
.
├── requirements-vllm.txt
├── configs/
│   └── vllm_eval.yaml
├── scripts/
│   └── create_vllm_env.sh
├── src/
│   └── audio_lfm/
│       └── vllm_plugin/
│           ├── __init__.py
│           ├── config.py
│           ├── export.py
│           ├── processing.py
│           ├── data_parser.py
│           ├── model.py
│           ├── weight_mapping.py
│           ├── runner.py
│           ├── parity.py
│           ├── benchmark.py
│           └── types.py
└── tests/
    └── vllm/
        ├── test_plugin_discovery.py
        ├── test_export_config.py
        ├── test_audio_token.py
        ├── test_prompt_replacement.py
        ├── test_feature_parser.py
        ├── test_weight_loading.py
        ├── test_projector_parity.py
        ├── test_hybrid_state_contract.py
        ├── test_hf_vllm_parity.py
        ├── test_request_isolation.py
        ├── test_raw_feature_parity.py
        ├── test_mm_cache_uuid.py
        ├── test_vllm_resume.py
        └── test_vllm_private_data.py
```

vLLM must remain an optional dependency. Importing the ordinary training package
must not import vLLM.

All CLI commands unrelated to vLLM must continue to work when vLLM is absent.

---

# 32. Plugin registration

Register a general vLLM plugin through Python entry points.

Add to `pyproject.toml`:

```toml
[project.entry-points."vllm.general_plugins"]
audio_lfm2 = "audio_lfm.vllm_plugin:register"
```

`src/audio_lfm/vllm_plugin/__init__.py`:

```python
from __future__ import annotations

ARCHITECTURE = "AudioLfm2ForConditionalGeneration"


def register() -> None:
    # Keep this function lightweight. Do not import Torch, Transformers, the
    # audio tower, or any module that initializes CUDA here.
    from vllm import ModelRegistry

    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            ARCHITECTURE,
            ("audio_lfm.vllm_plugin.model:AudioLfm2ForConditionalGeneration"),
        )
```

The registration function must be re-entrant.

Use the lazy string form of `ModelRegistry.register_model`. Do not import the
model class directly inside `register()`.

Filter plugin loading explicitly:

```bash
export VLLM_PLUGINS=audio_lfm2
```

The preflight must verify:

```python
from vllm import ModelRegistry

assert "AudioLfm2ForConditionalGeneration" in ModelRegistry.get_supported_archs()
```

Add a subprocess test proving that plugin discovery does not initialize CUDA in
the parent process.

---

# 33. vLLM export artifact

Implement:

```text
audio-lfm export-vllm
```

Example:

```bash
audio-lfm export-vllm \
  --config configs/paraspeech_whisper_lfm2.yaml \
  --checkpoint \
    runs/paraspeech-whisper-small-lfm2-projector/checkpoint-best \
  --output-dir \
    exports/paraspeech-whisper-small-lfm2-vllm
```

The export directory must be small. Do not copy frozen Whisper or LFM weights
into it.

Expected contents:

```text
exports/paraspeech-whisper-small-lfm2-vllm/
├── config.json
├── generation_config.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── added_tokens.json                 # when emitted by the tokenizer
├── model.safetensors                 # projector-owned parameters only
├── projector_manifest.json
└── export_manifest.json
```

The exporter must read immutable model revisions from the training run manifest.
It must never resolve `main` again during export.

## 33.1 Config strategy

Start from the exact resolved LFM2 config:

```python
base_config = AutoConfig.from_pretrained(
    text_model_id,
    revision=text_model_revision,
    trust_remote_code=False,
).to_dict()
```

Preserve:

```json
{
  "model_type": "lfm2"
}
```

Do not introduce an unknown custom `model_type`, because that would require
remote Transformers code.

Replace `architectures` with:

```json
{
  "architectures": [
    "AudioLfm2ForConditionalGeneration"
  ]
}
```

Add these fields:

```json
{
  "audio_lfm_format_version": 1,

  "text_model_id": "LiquidAI/LFM2.5-1.2B-Instruct",
  "text_model_revision": "<immutable commit SHA>",

  "audio_model_id": "openai/whisper-small",
  "audio_model_revision": "<immutable commit SHA>",

  "frontend_kind": "whisper",
  "frontend_mode": "variable_length_masked",

  "audio_sample_rate": 16000,
  "max_audio_seconds": 30.0,

  "audio_token": "<|audio|>",
  "audio_token_index": 65536,

  "audio_config": {
    "...": "exact resolved Whisper encoder configuration"
  },

  "projector_config": {
    "...": "exact training projector configuration"
  },

  "projector_checkpoint_sha256": "<sha256>",
  "training_run_manifest_sha256": "<sha256>",

  "prompt_sha256": "<sha256>",
  "base_chat_template_sha256": "<sha256>",
  "export_tokenizer_sha256": "<sha256>",

  "projected_length_formula_version": 1
}
```

The example `audio_token_index` above is illustrative. Derive the actual value
from the exported tokenizer.

The resulting config must still load as an LFM2 config:

```python
config = AutoConfig.from_pretrained(export_dir)
assert config.model_type == "lfm2"
assert config.architectures == ["AudioLfm2ForConditionalGeneration"]
```

Unknown custom fields must survive the save/load round trip.

## 33.2 Projector key mapping

The training checkpoint contains keys under:

```text
projector.*
```

The vLLM model contains the same modules under:

```text
multi_modal_projector.*
```

The exporter must map:

```text
projector.<suffix>
    ->
multi_modal_projector.<suffix>
```

This includes:

- temporal stack/reduction layers;
- projection MLP;
- normalization parameters;
- output scale;
- audio-start vector;
- audio-end vector;
- dMel embeddings in the future dMel implementation.

Save only mapped projector parameters to `model.safetensors`.

Reject:

- missing projector keys;
- unexpected trainable keys;
- duplicate mapped keys;
- non-finite tensors;
- architecture mismatch;
- hidden-size mismatch.

Record every source-to-export key mapping in `projector_manifest.json`.

## 33.3 Secondary weights

The plugin will load the frozen models through vLLM secondary weight sources:

```text
language_model.*
    <- LiquidAI/LFM2.5-1.2B-Instruct at exact commit

audio_tower.*
    <- openai/whisper-small at exact commit
```

This avoids creating a second combined 1.5B checkpoint.

The export manifest must record that its local `model.safetensors` is incomplete
by design and that the two immutable secondary sources are required.

---

# 34. Audio placeholder token

The training path inserts continuous audio embeddings directly and therefore
does not require an audio vocabulary token.

The vLLM processor does require a textual placeholder that can be expanded to
the exact audio-embedding length.

During export, add exactly one token:

```python
from transformers import AddedToken, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    text_model_id,
    revision=text_model_revision,
    trust_remote_code=False,
)

audio_token = AddedToken(
    "<|audio|>",
    special=True,
    normalized=False,
    lstrip=False,
    rstrip=False,
)

num_added = tokenizer.add_special_tokens({"additional_special_tokens": [audio_token]})

if num_added != 1:
    raise RuntimeError("Expected to add exactly one audio token")

audio_token_id = tokenizer.convert_tokens_to_ids("<|audio|>")

if tokenizer.encode(
    "<|audio|>",
    add_special_tokens=False,
) != [audio_token_id]:
    raise RuntimeError("Audio placeholder must tokenize to one token")
```

The placeholder token must be outside the frozen tokenizer vocabulary:

```python
assert audio_token_id >= original_tokenizer_size
```

LFM2 pads its embedding table beyond the tokenizer vocabulary, so the newly
added token may still have an ID below `base_config.vocab_size`. This is safe:
vLLM scatters the multimodal embeddings over every placeholder position. Do
not manufacture dummy tokenizer entries merely to force an out-of-range ID.

Do not:

- resize the LFM embedding table;
- change `config.vocab_size`;
- initialize a new text embedding row;
- train the LFM vocabulary.

The vLLM model must call:

```python
self.configure_mm_token_handling(
    self.config.vocab_size,
    [self.config.audio_token_index],
)
```

This tells vLLM that the out-of-vocabulary token marks multimodal embedding
positions and must not be passed to the normal text embedding lookup.

## Prompt parity

Build vLLM prompts from the exact same prompt file and chat template used in
training.

Algorithm:

1. Render the training prompt with its existing unique audio sentinel.
2. Assert the sentinel occurs exactly once.
3. Replace only that sentinel with `<|audio|>`.
4. Do not otherwise edit the rendered prompt.
5. Tokenize the text before and after the placeholder.
6. Compare those token IDs against the training prompt compiler.
7. Fail if either side differs.

The exported tokenizer must preserve the original chat template byte-for-byte.
Adding the placeholder token must not alter the template.

Store separate hashes for:

- original tokenizer;
- exported tokenizer;
- original chat template;
- rendered prompt;
- prompt file.

---

# 35. Shared frontend and projector implementation

Do not maintain separate mathematical implementations for training and vLLM.

Refactor these components into backend-neutral modules:

```text
audio_lfm.model.frontends.whisper_math
    - feature-frame length calculation
    - Whisper convolution output-length calculation
    - additive padding-mask construction
    - valid-output slicing

audio_lfm.model.frontends.whisper_encoder
    - VariableLengthWhisperEncoder

audio_lfm.model.projector
    - frame stack
    - FrameStackMLPProjector
    - output RMS normalization
    - learned output scale
    - audio-start and audio-end vectors
```

Both the HF training wrapper and the vLLM plugin must instantiate the same
classes.

The vLLM environment must not import:

- the HF training loop;
- FlashAttention training utilities;
- causal-conv training utilities;
- checkpoint optimizer code.

The shared Whisper encoder class must be constructible from configuration only:

```python
encoder = VariableLengthWhisperEncoder(
    whisper_config,
)
```

It must not call `from_pretrained()` itself.

Weights are supplied by vLLM's model loader.

The module parameter names must make deterministic Whisper checkpoint mapping
possible.

---

# 36. Multimodal processor and data parser

Implement:

```python
AudioLfm2ProcessingInfo
AudioLfm2DummyInputsBuilder
AudioLfm2MultiModalDataParser
AudioLfm2MultiModalProcessor
AudioLfm2Processor
```

Use the APIs and method signatures in the pinned vLLM source. Do not attempt to
write one implementation compatible with arbitrary vLLM releases.

## 36.1 Supported input modes

Support three input modes.

### Production mode: Whisper features

The WebDataset evaluation workers decode and validate audio, then run the exact
Whisper feature extractor on CPU.

One request passes:

```python
{
    "prompt": rendered_prompt,
    "multi_modal_data": {
        "audio": {
            "audio_features": feature_tensor,
            "audio_feature_length": feature_length,
            "audio_token_length": audio_token_length,
        }
    },
}
```

Recommended tensor contract:

```text
audio_features:
    CPU float32 [80, mel_frames]

audio_feature_length:
    CPU int64 scalar

audio_token_length:
    CPU int64 scalar
```

`audio_token_length` includes:

```text
1 audio-start vector
+ projected frame count
+ 1 audio-end vector
```

This mode is the production private-data evaluator because it:

- keeps strict decoding in the known CaptionStew loader;
- avoids vLLM's generic automatic channel downmix;
- moves log-Mel work into parallel DataLoader workers;
- does not persist features;
- lets the audio encoder and projector still execute inside vLLM on GPU.

### Debug mode: raw audio

Allow:

```python
{
    "multi_modal_data": {
        "audio": (
            waveform_numpy_float32,
            16000,
        )
    }
}
```

The offline runner must validate before submitting:

```python
if waveform.dtype != np.float32:
    waveform = waveform.astype(np.float32, copy=False)

if waveform.ndim != 1:
    raise AudioContractError("Expected mono 1D audio")

if sample_rate != 16_000:
    raise AudioContractError("Expected 16 kHz audio")

if waveform.size == 0:
    raise AudioContractError("Empty waveform")

if waveform.size > int(max_audio_seconds * sample_rate):
    raise AudioContractError("Audio exceeds configured maximum")
```

Do not depend on vLLM's automatic downmixing or resampling for dataset
evaluation.

Where the pinned vLLM parser API exposes raw audio before normalization, add the
same checks to the plugin parser. If it does not, mark raw mode as debug-only
and enforce the contract in the offline runner.

### Parity mode: final audio embeddings

Optionally accept final embeddings:

```text
[num_audio_tokens, lfm_hidden_size]
```

These embeddings already include the audio-start and audio-end vectors.

This mode is:

- disabled by default;
- enabled only with `enable_mm_embeds=True`;
- accepted only from trusted local code;
- never exposed in a public server;
- never written as a persistent dataset cache.

Use it to isolate whether discrepancies arise in:

- feature extraction/audio encoder/projector; or
- LFM/vLLM generation.

## 36.2 Custom data parser

Mirror the current vLLM audio-model pattern:

```python
class AudioLfm2MultiModalDataParser(MultiModalDataParser):
    def _parse_audio_data(self, data):
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={
                    "audio_features",
                    "audio_feature_length",
                    "audio_token_length",
                },
                fields_factory=lambda _: AUDIO_FIELD_CONFIG,
            )

        return super()._parse_audio_data(data)
```

Adapt exact imports and type signatures to vLLM 0.27.1.

Do not call these feature tensors “embeddings”; they are Whisper input features
and must still pass through the frozen encoder and trainable projector.

## 36.3 Field configuration

Configure every field as belonging to one independent audio item.

Conceptually:

```python
AUDIO_FIELD_CONFIG = {
    "audio_features": MultiModalFieldConfig.batched("audio"),
    "audio_feature_length": MultiModalFieldConfig.batched("audio"),
    "audio_token_length": MultiModalFieldConfig.batched("audio"),
    "audio_embeds": MultiModalFieldConfig.batched("audio"),
}
```

Inspect the pinned vLLM API and use its exact constructors.

The processed output for one audio item must be independent of other audio
items in the same preprocessing call. This is required for safe multimodal
caching.

## 36.4 Processing info

Implement:

```python
class AudioLfm2ProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self):
        return {"audio": 1}

    def get_data_parser(self):
        return AudioLfm2MultiModalDataParser(...)

    def get_hf_processor(self, **kwargs):
        return AudioLfm2Processor(
            tokenizer=self.get_tokenizer(),
            feature_extractor=...,
            config=...,
        )
```

The first implementation supports exactly one audio item per prompt.

Reject zero audio items when the prompt contains an audio placeholder.

Reject multiple audio items.

## 36.5 Dummy input builder

The dummy builder must represent the maximum configured audio length because
vLLM uses it for memory profiling.

Generate:

```text
waveform:
    float32 zeros [max_audio_seconds * 16000]

prompt:
    one prompt containing exactly one <|audio|>
```

The dummy path must produce the same maximum projected token count as the real
length estimator.

Test this for 30 seconds.

## 36.6 Prompt replacement

The user-facing rendered prompt initially contains one placeholder token:

```text
<|audio|>
```

The processor replaces it with:

```text
<|audio|> repeated audio_token_length times
```

Conceptually:

```python
def replacement(item_index: int) -> list[int]:
    length = int(audio_token_lengths[item_index])
    if length <= 2:
        raise ValueError("Invalid audio token length")
    return [audio_token_index] * length


return [
    PromptReplacement(
        modality="audio",
        target=[audio_token_index],
        replacement=replacement,
    )
]
```

The exact replacement count must equal the tensor length returned by
`embed_multimodal`.

Never use a fixed maximum placeholder count for variable-duration audio.

---

# 37. vLLM model plugin

Implement:

```python
@MULTIMODAL_REGISTRY.register_processor(
    AudioLfm2MultiModalProcessor,
    info=AudioLfm2ProcessingInfo,
    dummy_inputs=AudioLfm2DummyInputsBuilder,
)
class AudioLfm2ForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    IsHybrid,
): ...
```

The implementation is supported only for:

```text
tensor_parallel_size == 1
pipeline_parallel_size == 1
```

Fail clearly for larger values. Do not claim multi-GPU support until tested.

## 37.1 Native LFM2 reuse

Import the native vLLM class:

```python
from vllm.model_executor.models.lfm2 import (
    Lfm2ForCausalLM as NativeLfm2ForCausalLM,
)
```

Delegate all hybrid-state contracts:

```python
@classmethod
def get_mamba_state_dtype_from_config(cls, vllm_config):
    return NativeLfm2ForCausalLM.get_mamba_state_dtype_from_config(vllm_config)


@classmethod
def get_mamba_state_shape_from_config(cls, vllm_config):
    return NativeLfm2ForCausalLM.get_mamba_state_shape_from_config(vllm_config)


@classmethod
def get_mamba_state_copy_func(cls):
    return NativeLfm2ForCausalLM.get_mamba_state_copy_func()
```

Do not reimplement the short-convolution cache dimensions.

Add a test that these methods return the same result as the native LFM2 class
for the exported config.

## 37.2 Initialization

Use this structure:

```python
def __init__(
    self,
    *,
    vllm_config: VllmConfig,
    prefix: str = "",
) -> None:
    super().__init__()

    config = vllm_config.model_config.hf_config

    if vllm_config.parallel_config.tensor_parallel_size != 1:
        raise NotImplementedError("AudioLFM2 v1 supports tensor_parallel_size=1 only")

    if vllm_config.parallel_config.pipeline_parallel_size != 1:
        raise NotImplementedError("AudioLFM2 v1 supports pipeline_parallel_size=1 only")

    self.config = config
    self.vllm_config = vllm_config

    self.configure_mm_token_handling(
        config.vocab_size,
        [config.audio_token_index],
    )

    self.secondary_weights = [
        DefaultModelLoader.Source(
            model_or_path=config.text_model_id,
            revision=config.text_model_revision,
            prefix="language_model.",
        ),
        DefaultModelLoader.Source(
            model_or_path=config.audio_model_id,
            revision=config.audio_model_revision,
            prefix="audio_tower.",
        ),
    ]

    with self._mark_tower_model(vllm_config, "audio"):
        self.audio_tower = VariableLengthWhisperEncoder(config.audio_config)

        self.multi_modal_projector = FrameStackMLPProjector.from_config(
            config.projector_config
        )

    with self._mark_language_model(vllm_config):
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config,
            prefix=maybe_prefix(prefix, "language_model"),
            # Critical: avoid recursively constructing the custom outer
            # architecture.
            architectures=["Lfm2ForCausalLM"],
        )

    self.make_empty_intermediate_tensors = (
        self.language_model.make_empty_intermediate_tensors
    )
```

Verify the exact `DefaultModelLoader.Source` and
`init_vllm_registered_model` signatures in vLLM 0.27.1.

Do not replace immutable secondary revisions with the outer local model's
revision.

## 37.3 Multimodal module mapping

Implement:

```python
def get_mm_mapping(self) -> MultiModelKeys:
    return MultiModelKeys.from_string_field(
        language_model="language_model.",
        connector="multi_modal_projector.",
        tower_model="audio_tower.",
    )
```

## 37.4 Audio embedding path

Implement:

```python
def embed_multimodal(
    self,
    **kwargs: object,
) -> MultiModalEmbeddings: ...
```

Feature mode:

1. Parse lists/nested tensors for:
   - `audio_features`;
   - `audio_feature_length`;
   - `audio_token_length`.
2. Pad only inside a bounded encoder microbatch.
3. Run the shared variable-length Whisper encoder.
4. Slice every output to its exact valid encoder length.
5. Run the shared projector.
6. Prepend `audio_start`.
7. Append `audio_end`.
8. Assert output length equals `audio_token_length`.
9. Return one tensor per logical audio item.

Conceptually:

```python
result: list[torch.Tensor] = []

for encoder_batch in bounded_audio_batches:
    encoded = self.audio_tower(
        encoder_batch.features,
        encoder_batch.lengths,
    )

    for item_encoded, expected_length in zip(
        encoded,
        encoder_batch.audio_token_lengths,
        strict=True,
    ):
        projected = self.multi_modal_projector.project_frames(item_encoded)

        embeddings = torch.cat(
            [
                self.multi_modal_projector.audio_start[None],
                projected,
                self.multi_modal_projector.audio_end[None],
            ],
            dim=0,
        )

        if embeddings.shape[0] != expected_length:
            raise RuntimeError("Processor/model audio-token length mismatch")

        result.append(embeddings)

return tuple(result)
```

The encoder microbatch limit must be configurable:

```yaml
audio_encoder_microbatch_size: 4
```

Start at 4 on the RTX 4090 and benchmark 4, 8, and 16.

Embedding mode:

- validate rank 2;
- validate hidden size;
- validate finite values;
- validate expected placeholder length;
- return the supplied final embeddings without modification.

## 37.5 Text embedding insertion

Implement or inherit the standard multimodal embedding insertion pattern:

```python
def embed_input_ids(
    self,
    input_ids: torch.Tensor,
    multimodal_embeddings=None,
    *,
    is_multimodal=None,
) -> torch.Tensor:
    if multimodal_embeddings is None or is_multimodal is None:
        return super().embed_input_ids(input_ids)

    return super().embed_input_ids(
        input_ids,
        multimodal_embeddings=multimodal_embeddings,
        is_multimodal=is_multimodal,
    )
```

Do not write a custom scatter operation unless the pinned vLLM interface
requires one.

Assert that the number of multimodal placeholder positions exactly equals the
total number of returned audio vectors. Never use `min()` to conceal a mismatch.

## 37.6 LFM forward and logits

Delegate directly:

```python
def forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors=None,
    inputs_embeds: torch.Tensor | None = None,
    **kwargs: object,
):
    if intermediate_tensors is not None:
        inputs_embeds = None

    return self.language_model.model(
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds=inputs_embeds,
    )


def compute_logits(
    self,
    hidden_states: torch.Tensor,
) -> torch.Tensor | None:
    return self.language_model.compute_logits(hidden_states)
```

Use the exact native LFM2 call signature in vLLM 0.27.1.

Do not invoke an HF LFM model from inside the plugin.

---

# 38. Weight loading

Implement deterministic loading for three sources:

```text
local export:
    multi_modal_projector.*

text secondary source:
    language_model.*

audio secondary source:
    audio_tower.*
```

## 38.1 Audio mapping

The Whisper checkpoint commonly exposes encoder parameters under:

```text
model.encoder.*
```

After the secondary prefix is added, source names become:

```text
audio_tower.model.encoder.*
```

Map:

```text
audio_tower.model.encoder.<suffix>
    ->
audio_tower.<suffix>
```

Use a `WeightsMapper` or an explicit generator equivalent to:

```python
hf_to_vllm_mapper = WeightsMapper(
    orig_to_new_prefix={
        "audio_tower.model.encoder.": "audio_tower.",
    }
)
```

Explicitly skip only source modules not instantiated by this project:

```text
audio_tower.model.decoder.*
audio_tower.proj_out.*
```

Do not broadly skip `audio_tower.*`.

## 38.2 Text mapping

The text secondary source prefix should naturally produce:

```text
language_model.model.*
language_model.lm_head.*
```

matching the native wrapped model.

When LFM embeddings and LM head are tied, follow the native LFM2 loader's tied
weight behavior. Do not demand a duplicate loaded LM-head parameter when the
module is tied.

## 38.3 Projector mapping

The local export already contains:

```text
multi_modal_projector.*
```

No runtime rename should be required.

## 38.4 Loader contract

Conceptually:

```python
def load_weights(
    self,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    filtered = filter_explicit_source_only_weights(weights)

    loader = AutoWeightsLoader(
        self,
        skip_prefixes=[
            "audio_tower.model.decoder.",
            "audio_tower.proj_out.",
            # Add the tied LFM head only when required by native LFM behavior.
        ],
    )

    loaded = loader.load_weights(
        filtered,
        mapper=self.hf_to_vllm_mapper,
    )

    validate_loaded_parameter_set(
        model=self,
        loaded=loaded,
        allowed_tied_aliases=...,
    )

    return loaded
```

The preflight must prove:

- every expected audio encoder parameter was loaded exactly once;
- every expected projector parameter was loaded exactly once;
- every expected LFM parameter or allowed tied alias was loaded;
- no projector weight came from a secondary source;
- no frozen model weight came from the local projector checkpoint;
- no unexpected decoder/TTS modules were instantiated;
- all loaded tensors have the expected shape;
- the projector checkpoint SHA matches the export manifest.

Fail on a partially loaded projector.

---

# 39. Evaluation input pipeline

Reuse:

- `CatalogIndex`;
- `CaptionStewBackend`;
- strict FLAC decoder;
- split validation;
- duration/crop policy;
- prompt compiler;
- provenance handling.

Do not build a second dataset implementation for vLLM.

## 39.1 Production feature pipeline

The production flow is:

```text
private HF Bucket TAR stream
    -> strict 16 kHz mono FLAC decode
    -> target/provenance lookup by audio_id
    -> exact training prompt render
    -> exact Whisper feature extractor in DataLoader worker
    -> bounded host-memory request queue
    -> vLLM plugin audio encoder + projector
    -> native vLLM LFM2 generation
    -> atomic prediction part
```

Each DataLoader worker owns its own feature extractor instance.

Do not send padded feature batches through the queue. Store each item as:

```text
float32 [80, exact_mel_length]
```

vLLM may pad temporarily inside a bounded encoder microbatch.

Use a producer queue bounded by request chunks:

```yaml
prefetch_request_chunks: 2
```

Do not allow the producer to read the entire dev or holdout split into memory.

## 39.2 Stable multimodal UUID

Create a stable UUID from the complete semantic audio input:

```python
def make_audio_uuid(
    *,
    audio_id: str,
    flac_sha256: str,
    crop_start_sample: int | None,
    num_samples: int,
    frontend_config_sha256: str,
    projector_checkpoint_sha256: str,
) -> str:
    payload = "\0".join(
        [
            audio_id,
            flac_sha256,
            str(crop_start_sample),
            str(num_samples),
            frontend_config_sha256,
            projector_checkpoint_sha256,
        ]
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()
```

Do not use `audio_id` alone because a crop, frontend, or projector change
changes the multimodal embedding.

For one audio item, pass:

```python
"multi_modal_uuids": {
    "audio": stable_audio_uuid,
}
```

Add a compatibility test for the pinned vLLM UUID schema.

Caching is only an optimization. Prediction correctness must not depend on a
cache hit.

## 39.3 Request construction

Use already rendered prompts rather than `LLM.chat()`:

```python
request = {
    "prompt": rendered_prompt_with_one_audio_token,
    "multi_modal_data": {
        "audio": {
            "audio_features": item.audio_features,
            "audio_feature_length": item.audio_feature_length,
            "audio_token_length": item.audio_token_length,
        }
    },
    "multi_modal_uuids": {
        "audio": item.audio_uuid,
    },
}
```

Before submission, assert:

```python
assert rendered_prompt.count("<|audio|>") == 1
assert item.audio_features.ndim == 2
assert item.audio_features.shape[0] == 80
assert item.audio_feature_length > 0
assert item.audio_token_length == (
    projector.projected_length(whisper_encoder_length(item.audio_feature_length)) + 2
)
```

Do not send target captions in the prompt.

One generated caption is produced per audio item. Attach every official
style-caption reference to the output record after generation.

---

# 40. vLLM configuration

Create `configs/vllm_eval.yaml`:

```yaml
run:
  name: paraspeech-whisper-small-lfm2-vllm-eval
  seed: 1337
  output_dir: runs/vllm-eval
  prediction_part_size: 256
  write_jsonl: true
  write_parquet: true

model:
  export_dir: exports/paraspeech-whisper-small-lfm2-vllm

vllm:
  required_version: "0.27.1"
  dtype: bfloat16
  tensor_parallel_size: 1
  pipeline_parallel_size: 1

  max_model_len: 2048
  gpu_memory_utilization: 0.80
  max_num_seqs: 4
  max_num_batched_tokens: 4096

  enforce_eager: true
  enable_prefix_caching: false
  enable_mm_embeds: false

  limit_mm_per_prompt:
    audio: 1

  mm_processor_cache_gb: 1
  mm_processor_cache_type: lru

  audio_encoder_microbatch_size: 4

data:
  backend: captionstew
  captionstew_root: "${ENV:CAPTIONSTEW_ROOT}"
  dataset: ParaSpeechCaps-Base
  split: dev
  target_type: style_caption

  input_mode: whisper_features

  num_workers: 2
  persistent_workers: true
  prefetch_factor: 2
  prefetch_request_chunks: 2

  request_chunk_max_items: 64
  request_chunk_max_audio_seconds: 240.0

  max_audio_seconds: 30.0
  long_audio_policy: skip
  strict_audio_contract: true

generation:
  temperature: 0.0
  top_p: 1.0
  top_k: -1
  max_tokens: 128
  seed: 1337
  stop_token_ids: null

evaluation:
  max_audio_items: null
  include_reference_text_in_predictions: true
  include_transcript_in_predictions: false
  fail_on_duplicate_audio_id: true
```

Starting values are deliberately conservative.

After parity tests pass, benchmark:

```text
max_num_seqs:
    4, 8, 16, 32

max_num_batched_tokens:
    2048, 4096, 8192

gpu_memory_utilization:
    0.80, 0.85, 0.90

audio_encoder_microbatch_size:
    4, 8, 16

enforce_eager:
    true, false
```

Do not promote `enforce_eager=false` to the production configuration until
greedy parity and request-isolation tests pass.

Do not use tensor parallelism on one GPU.

---

# 41. Offline vLLM runner

Construct the engine once:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model=str(export_dir),
    dtype="bfloat16",
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    max_model_len=config.vllm.max_model_len,
    gpu_memory_utilization=config.vllm.gpu_memory_utilization,
    max_num_seqs=config.vllm.max_num_seqs,
    max_num_batched_tokens=config.vllm.max_num_batched_tokens,
    limit_mm_per_prompt={"audio": 1},
    enforce_eager=config.vllm.enforce_eager,
    enable_prefix_caching=config.vllm.enable_prefix_caching,
    enable_mm_embeds=config.vllm.enable_mm_embeds,
    mm_processor_cache_gb=config.vllm.mm_processor_cache_gb,
    mm_processor_cache_type=config.vllm.mm_processor_cache_type,
    trust_remote_code=False,
    load_format="safetensors",
)
```

Use:

```python
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=128,
    seed=1337,
)
```

Process bounded chunks:

```python
for request_chunk in request_stream:
    request_dicts = [build_vllm_request(item) for item in request_chunk]

    started = time.perf_counter()

    outputs = llm.generate(
        request_dicts,
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    elapsed = time.perf_counter() - started

    if len(outputs) != len(request_chunk):
        raise RuntimeError("vLLM output-count mismatch")

    for item, output in zip(
        request_chunk,
        outputs,
        strict=True,
    ):
        completion = output.outputs[0]

        write_prediction(
            item=item,
            generated_text=completion.text,
            generated_token_ids=list(completion.token_ids),
            finish_reason=completion.finish_reason,
            stop_reason=completion.stop_reason,
            cumulative_logprob=completion.cumulative_logprob,
            chunk_elapsed_seconds=elapsed,
        )
```

Add a test confirming that the offline API returns outputs in input order for
the pinned release. Continue to associate outputs with the original chunk
records rather than inferring identity from generated text.

Do not instantiate one vLLM engine per chunk.

Explicitly release the engine only at process termination.

---

# 42. Prediction format and resume

Write atomic prediction parts:

```text
runs/vllm-eval/
├── resolved_config.yaml
├── evaluation_manifest.json
├── progress.json
├── predictions/
│   ├── part-000000.parquet
│   ├── part-000001.parquet
│   └── ...
└── predictions.jsonl
```

A Parquet part is wrio a temporary path and atomically renamed after the
complete chunk succeeds.

One row represents one audio item.

Required fields:

```text
audio_id
source_id
dataset
split

generated_text
generated_token_ids
finish_reason
stop_reason
cumulative_logprob

reference_target_ids
reference_texts
reference_count

audio_duration_seconds
original_num_samples
crop_start_sample
evaluated_num_samples
flac_sha256

input_prompt_sha256
prompt_template_sha256
chat_template_sha256

projector_checkpoint_sha256
training_run_manifest_sha256

text_model_id
text_model_revision
audio_model_id
audio_model_revision

vllm_version
plugin_git_commit
export_format_version

sampling_temperature
sampling_top_p
sampling_top_k
sampling_max_tokens
sampling_seed

dataset_license
media_license
attribution
creator
source_url
acquisition_route
restrictions
origin
upstream

created_at_utc
```

Preserve any additional source provenance fields in a structured metadata
column.

Do not include transcripts by default.

## Resume

At startup:

1. Scan only complete atomic Parquet parts.
2. Reconstruct the set of completed `audio_id` values.
3. Verify there are no duplicates.
4. Verify every part has the same evaluation-manifest SHA.
5. Filter completed IDs from the streamed split.
6. Continue with the next part number.

Never mark an item complete before its prediction part is atomically committed.

On resume, fail if any of these changed:

- split;
- dataset catalog fingerprint;
- projector checkpoint SHA;
- LFM revision;
- Whisper revision;
- prompt hash;
- chat-template hash;
- tokenizer hash;
- generation parameters;
- vLLM version;
- plugin Git commit;
- frontend configuration;
- crop policy.

A `--allow-nonreproducible-resume` development override may exist, but it must
start a new prediction directory rather than append to an incompatible one.

---

# 43. Evaluation split rules

Periodic generation evaluation:

```text
ParaSpeechCaps-Base dev
```

Final generation evaluation:

```text
ParaSpeechCaps-Base holdout
```

Reject:

```text
ParaSpeechCaps-Base test
```

Require an explicit flag for holdout:

```bash
audio-lfm evaluate-vllm \
  --config configs/vllm_eval.yaml \
  --split holdout \
  --allow-final-evaluation
```

Without `--allow-final-evaluation`, reject `holdout`.

Generate once per audio item, not once per reference.

Attach all style-caption references after generation.

Do not use the transcript as:

- a prompt field;
- a generation target;
- a reference caption;
- an automatic semantic hint.

Teacher-forced all-reference NLL remains in the HF evaluator.

Do not report vLLM generation loss as if it were the HF all-reference NLL.

---

# 44. CLI additions

Add:

```text
audio-lfm export-vllm
audio-lfm preflight-vllm
audio-lfm compare-hf-vllm
audio-lfm evaluate-vllm
audio-lfm benchmark-vllm
```

Import vLLM lazily inside these commands.

## `export-vllm`

- validates training checkpoint;
- maps projector keys;
- exports tokenizer and audio placeholder;
- writes local config;
- records immutable secondary sources;
- creates hashes and manifest.

## `preflight-vllm`

Checks:

- exact vLLM version;
- plugin discovery;
- config loading;
- tokenizer placeholder behavior;
- model registry;
- model initialization;
- complete weight loading;
- hybrid-state methods;
- one raw-audio debug request;
- one feature-mode request;
- HF/vLLM projector parity;
- one greedy generation.

## `compare-hf-vllm`

Runs identical examples through:

- HF reference generation;
- vLLM plugin generation.

It compares:

- rendered prompt text;
- prompt token IDs around audio;
- Whisper input features;
- valid encoder outputs;
- projected audio embeddings;
- projected lengths;
- first generated token;
- top-k first-token log probabilities;
- complete greedy token sequence.

## `evaluate-vllm`

Streams dev or explicit holdout, generates predictions, and writes atomic output
parts.

## `benchmark-vllm`

Measures:

- remote data wait time;
- FLAC decode time;
- feature-extraction time;
- vLLM wall time;
- audio seconds/s;
- requests/s;
- input multimodal tokens/s;
- generated tokens/s;
- peak host memory;
- peak GPU memory;
- output-length distribution.

Benchmark:

- raw audio;
- precomputed Whisper features;
- HF generation;
- vLLM eager;
- vLLM non-eager after correctness approval.

---

# 45. Required vLLM tests

## 45.1 Plugin discovery

In a fresh subprocess:

- set `VLLM_PLUGINS=audio_lfm2`;
- import vLLM;
- verify architecture registration;
- verify registration can be invoked twice;
- verify plugin entry does not import the heavy model module;
- verify CUDA is not initialized by registration alone.

## 45.2 Export config

- local config loads as `Lfm2Config`;
- custom architecture is retained;
- all extra fields survive;
- text and audio revisions are immutable SHAs;
- no `auto_map` or remote custom code is required;
- config vocabulary size remains the base LFM vocabulary size.

## 45.3 Audio token

- `<|audio|>` is one token;
- token ID is outside the base vocabulary;
- LFM embeddings are not resized;
- prompt contains one token before processing;
- processed prompt contains exactly the expected repeated count;
- text tokens before and after the placeholder match the HF training compiler.

## 45.4 Feature parser

- one feature item accepted;
- missing field rejected;
- wrong Mel dimension rejected;
- non-finite values rejected;
- zero length rejected;
- mismatched token length rejected;
- multiple audio items rejected;
- different items remain independently cacheable.

## 45.5 Strict audio

The private-data evaluator must reject before vLLM:

- stereo arrays;
- `[time, channels]` arrays;
- `[channels, time]` arrays;
- non-16 kHz audio;
- empty audio;
- overlength audio;
- NaN/Inf samples.

It must not downmix or resample.

## 45.6 Weight loading

- all projector keys loaded;
- all Whisper encoder keys loaded;
- Whisper decoder keys skipped explicitly;
- native LFM parameters loaded;
- tied LM-head alias handled;
- no unexpected parameter silently ignored;
- loading a corrupt projector shape fails;
- loading a missing projector tensor fails;
- export checkpoint hash verified.

## 45.7 Projector parity

Given identical Whisper hidden states:

```text
training projector output
    ≈
vLLM plugin projector output
```

Compare:

- reduced frame count;
- projected values;
- start vector;
- end vector;
- output RMS;
- final concatenated embeddings.

Use FP32 and BF16 test cases.

## 45.8 Whisper parity

Given identical waveform:

```text
training feature extraction
    ==
vLLM production feature extraction
```

and:

```text
training variable-length Whisper output
    ≈
vLLM plugin Wer output
```

Test multiple lengths, including convolution boundary lengths.

## 45.9 LFM hybrid contract

- plugin state dtype equals native LFM2 state dtype;
- plugin state shape equals native LFM2 state shape;
- plugin state-copy function comes from native LFM2;
- generated requests never share short-convolution state.

## 45.10 HF/vLLM generation parity

Use `enforce_eager=true`, BF16, temperature zero, and fixed seed.

For at least:

- 32 synthetic examples;
- 32 private dev examples when credentials exist.

Compare:

- first-token top-k;
- first-token top-1;
- greedy token IDs for at least 32 generated tokens.

Do not normalize differing generated strings and call them equal. Compare token
IDs.

When top-1 differs, write a diagnostic containing the log-probability margin,
projected embedding difference, and first divergent layer or stage that can be
observed.

Do not automatically loosen tolerance.

## 45.11 Request isolation

Run A and B:

1. separately;
2. in one `LLM.generate([A, B])` call.

Assert B's greedy output is unchanged.

Perturb A's:

- waveform;
- length;
- prompt;
- projected token count.

Assert B remains unchanged.

This is the inference-side guard against attention or short-convolution state
leakage.

## 45.12 Raw/feature/embedding parity

For the same audio:

```text
raw audio mode
    ≈
Whisper feature mode
    ≈
final embedding mode
```

Compare projected embeddings and greedy output tokens.

Final embedding mode is tested only with trusted local tensors and
`enable_mm_embeds=tru
## 45.13 Multimodal UUID cache

- same UUID and same media gives identical output;
- changed crop gives changed UUID;
- changed projector gives changed UUID;
- changed frontend config gives changed UUID;
- cache hit does not alter output;
- cache miss does not alter output;
- no result depends on omitted media unless a documented cache hit exists.

## 45.14 Resume

Interrupt evaluation after several atomic parts.

Resume and prove:

- no completed audio ID is regenerated;
- no audio ID occurs twice;
- all expected split IDs eventually occur;
- incompatible configuration is rejected;
- partial temporary parts are ignored.

## 45.15 Long-run memory

Run at least 1,000 dev examples.

Verify:

- host memory remains bounded;
- request queue remains bounded;
- multimodal cache remains within configuration;
- CUDA memory does not grow monotonically;
- decoded waveforms are released;
- feature tensors are released after their chunk;
- no TAR file is retained locally;
- output parts are committed incrementally.

---

# 46. Benchmark promotion gates

The production backend begins with:

```text
feature input
enforce_eager=true
max_num_seqs=4
audio encoder microbatch=4
```

Promote settings only in this order.

## Gate 1: exact data parity

Pass:

- prompt parity;
- feature parity;
- length parity;
- projector parity.

## Gate 2: generation parity

Pass:

- first-token parity;
- greedy sequence parity;
- request-isolation tests.

## Gate 3: non-eager vLLM

Set:

```yaml
enforce_eager: false
```

Repeat all generation and isolation tests.

Keep it only when parity is retained.

## Gate 4: concurrency sweep

Sweep `max_num_seqs` and encoder microbatch size.

Choose the highest-throughput setting that:

- does not OOM;
- preserves output parity;
- leaves at least 1 GiB of observed GPU safety margin;
- does not cause host-memory growth.

## Gate 5: optional prefix caching

Do not enable prefix caching initially.

It may be benchmarked later only with the native LFM2-compatible hybrid cache
mode and after request-isolation tests are repeated.

Do not use prefix caching as a requirement for evaluation correctness.

---

# 47. dMel vLLM extension

Implement dMel only after Whisper vLLM generation is stable.

Keep the same outer architecture and plugin registration.

For dMel:

```text
raw waveform
    -> deterministic dMel CPU processor
    -> integer dMel codes
    -> trainable dMel projector inside vLLM
    -> native LFM2
```

The processor emits:

```text
dmel_codes
dmel_frame_length
audio_token_length
```

The model has no frozen audio tower secondary source.

The exported local checkpoint contains:

```text
multi_modal_projector.dmel_embedding.*
multi_modal_projector.channel_embedding.*
multi_modal_projector.temporal_projection.*
multi_modal_projector.output_projection.*
multi_modal_projector.audio_start
multi_modal_projector.audio_end
```

The text secondary source remains the exact LFM checkpoint.

Implement dMel through the same:

- placeholder;
- processor registry;
- prompt replacement;
- `embed_multimodal`;
- vLLM runner;
- evaluation outputs;
- resume behavior.

Do not create a second vLLM architecture name unless the outer runtime contract
cannot be shared.

Add Whisper-versus-dMel benchmark fields:

- frontend CPU time;
- encoder GPU time;
- projected audio token rate;
- request throughput;
- generated token throughput;
- peak VRAM;
- output agreement where meaningful.

---

# 48. vLLM definition of done

The vLLM extension is complete only when:

1. `vllm==0.27.1` is installed in an isolated environment.
2. The repository is discovered through `vllm.general_plugins`.
3. Plugin registration is lazy and re-entrant.
4. The exported config remains an LFM2 config.
5. The custom outer architecture is selected without `trust_remote_code`.
6. The native vLLM `Lfm2ForCausalLM` is used internally.
7. Native LFM2 hybrid-state methods are reused.
8. Frozen LFM weights are not duplicated in the export.
9. Frozen Whisper weights are not duplicated in the export.
10. The local export contains only projector-owned model tensors.
11. Text and audio secondary sources use immutable revisions.
12. The audio placeholder is one OOV tokenizer token.
13. The LFM embedding table is not resized.
14. Placeholder expansion exactly matches projected audio length.
15. Training and vLLM use the same projector implementation.
16. Training and vLLM use the same variable-length Whisper semantics.
17. Production evaluation passes precomputed in-memory Whisper features.
18. Production evaluation does not persist audio features.
19. Production evaluation rejects stereo and non-16 kHz audio.
20. Every audio item is an independent vLLM request.
21. No training-style manual sequence packing is used.
22. Request-isolation tests pass for LFM attention and short-conv state.
23. HF and vLLM projected embeddings match.
24. HF and vLLM greedy generation parity passes.
25. Dev generation uses only `dev`.
26. Holdout generation requires an explicit final-evaluation flag.
27. No independent ParaSpeechCaps `test` evaluation exists.
28. Prediction output is joined and resumed by `audio_id`.
29. All source provenance and licensing fields are retained.
30. Evaluation streams remote WebDataset shards without bulk download.
31. Interrupted evaluation resumes without duplicate predictions.
32. A 1,000-example run has bounded host and GPU memory.
33. Benchmark results separate data, feature, encoder, prefill, and decode costs.
34. HF remains the authoritative all-reference NLL evaluator.
35. vLLM is the authoritative high-throughput generation backend.
