from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from audio_lfm.utils.logging import REDACTED, is_secret_name

ENV_PATTERN = re.compile(r"\$\{ENV:([A-Za-z_][A-Za-z0-9_]*)\}")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunConfig(StrictModel):
    name: str
    seed: int = 1337
    output_dir: Path
    log_every_updates: int = Field(10, gt=0)
    checkpoint_every_updates: int = Field(250, gt=0)
    eval_every_updates: int = Field(250, gt=0)
    keep_last_checkpoints: int = Field(3, ge=1)
    fail_on_oom: bool = True
    deterministic_algorithms: bool = False


class TrainingSourceConfig(StrictModel):
    dataset: str
    splits: list[str] = Field(min_length=1)
    expansion_decoder_lock: Path | None = None
    expansion_recipe_sha256: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> TrainingSourceConfig:
        if len(self.splits) != len(set(self.splits)):
            raise ValueError("training source splits must be unique")
        if self.dataset.startswith("ParaSpeech") and self.splits != ["train_base"]:
            raise ValueError("ParaSpeechCaps training source must use train_base")
        return self


class DataConfig(StrictModel):
    backend: Literal["captionstew", "local"] = "captionstew"
    captionstew_root: Path
    dataset: str = "ParaSpeechCaps-Base"
    train_split: str = "train_base"
    validation_split: str = "dev"
    final_split: str = "holdout"
    target_type: str = "style_caption"
    target_sampling: Literal["one_per_audio_per_epoch", "all"] = (
        "one_per_audio_per_epoch"
    )
    metadata_source: Literal["parquet"] = "parquet"
    shard_shuffle: int = Field(53, ge=0)
    sample_shuffle: int = Field(256, ge=0)
    num_workers: int = Field(2, ge=0)
    persistent_workers: bool = True
    prefetch_factor: int = Field(2, gt=0)
    max_bad_samples: int = Field(0, ge=0)
    strict_audio_contract: bool = True
    strict_target_consistency: bool = False
    max_audio_seconds: float = Field(30.0, gt=0)
    long_audio_policy: Literal["skip", "center_crop", "random_crop", "chunk_pad"] = (
        "skip"
    )
    duration_sidecar: Path | None = None
    require_exact_duration_sidecar: bool = False
    require_complete_local_shards: bool = False
    preserve_provenance: bool = True
    review_status_allowlist: list[str] | None = None
    target_provider: Literal[
        "official_target", "response_overlay", "caption_expansion_overlay"
    ] = "official_target"
    expansion_decoder_lock: Path | None = None
    expansion_recipe_sha256: str | None = None
    training_sources: list[TrainingSourceConfig] | None = None

    @model_validator(mode="after")
    def validate_splits(self) -> DataConfig:
        if self.dataset.startswith("ParaSpeech"):
            splits = (self.train_split, self.validation_split, self.final_split)
            if "test" in splits:
                raise ValueError(
                    "ParaSpeechCaps split 'test' is forbidden; use holdout"
                )
            if splits != ("train_base", "dev", "holdout"):
                raise ValueError(
                    "ParaSpeechCaps requires train_base/dev/holdout split semantics"
                )
            if (
                self.target_provider == "official_target"
                and self.target_type != "style_caption"
            ):
                raise ValueError("ParaSpeechCaps baseline target_type is style_caption")
            if (
                self.target_provider != "official_target"
                and self.target_type != "audio_assistant_response"
            ):
                raise ValueError(
                    "ParaSpeechCaps response overlays require "
                    "target_type=audio_assistant_response"
                )
        if self.num_workers == 0 and self.persistent_workers:
            raise ValueError("persistent_workers requires num_workers > 0")
        if self.training_sources is not None:
            datasets = [source.dataset for source in self.training_sources]
            if len(datasets) != len(set(datasets)):
                raise ValueError("training source datasets must be unique")
        return self


class PromptConfig(StrictModel):
    mode: Literal["direct_caption_alignment", "caption_expansion_alignment"] = (
        "direct_caption_alignment"
    )
    prompt_file: Path | None = None
    audio_sentinel: str
    system_message: str | None = None
    require_no_system_message: bool = False
    user_content: Literal["prompt_file", "audio_sentinel_only"] = "prompt_file"
    supervise_assistant_termination: bool = True

    @model_validator(mode="after")
    def validate_prompt_mode(self) -> PromptConfig:
        if self.mode == "caption_expansion_alignment":
            if self.system_message is not None:
                raise ValueError(
                    "caption_expansion_alignment forbids a training-time system message"
                )
            if not self.require_no_system_message:
                raise ValueError(
                    "caption_expansion_alignment requires "
                    "require_no_system_message=true"
                )
            if self.user_content != "audio_sentinel_only":
                raise ValueError(
                    "caption_expansion_alignment requires audio_sentinel_only"
                )
            if self.prompt_file is not None:
                raise ValueError(
                    "caption_expansion_alignment must not configure a prompt file"
                )
        else:
            if self.prompt_file is None:
                raise ValueError("direct_caption_alignment requires prompt_file")
            if self.user_content != "prompt_file":
                raise ValueError(
                    "direct_caption_alignment requires user_content=prompt_file"
                )
        return self


class FrontendConfig(StrictModel):
    kind: Literal["whisper", "dmel"] = "whisper"
    model_id: str | None = "openai/whisper-small"
    revision: str | None = "main"
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    mode: Literal["variable_length_masked", "official_fixed_30s"] = (
        "variable_length_masked"
    )
    feature_extraction_device: Literal["cpu"] = "cpu"
    max_seconds: float = Field(30.0, gt=0)
    chunk_long_audio: bool = False
    encoder_microbatch_max_padded_samples: int = Field(960_000, gt=0)
    sample_rate: int = 16_000


class ProjectorConfig(StrictModel):
    kind: Literal["frame_stack_mlp", "dmel_patch_mlp"] = "frame_stack_mlp"
    stack_factor: int = Field(5, gt=0)
    hidden_dim: int = Field(2048, gt=0)
    activation: Literal["gelu"] = "gelu"
    dropout: float = Field(0.0, ge=0.0, lt=1.0)
    use_input_layer_norm: bool = True
    use_output_rms_norm: bool = True
    use_trainable_audio_boundary_vectors: bool = True
    initialize_to_text_embedding_rms: bool = True
    dmel_bin_embedding_dim: int = Field(16, gt=0)
    temporal_patch_size: int = Field(8, gt=0)


class LlmConfig(StrictModel):
    model_id: str
    revision: str = "main"
    dtype: Literal["bfloat16"] = "bfloat16"
    attention_implementation: Literal["flash_attention_2"] = "flash_attention_2"
    trust_remote_code: Literal[False] = False
    use_cache: Literal[False] = False
    gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = False
    allow_frozen_llm_dropout: bool = False


class PackingConfig(StrictModel):
    enabled: bool = True
    max_lfm_tokens: int = Field(2048, gt=0)
    max_sample_lfm_tokens: int | None = Field(None, gt=0)
    planning_buffer_examples: int = Field(64, gt=0)
    max_examples_per_pack: int | None = Field(8, gt=0)
    oversized_example_policy: Literal["error", "skip"] = "error"
    best_fit_decreasing: bool = True
    require_boundary_kernel_tests: bool = True

    @property
    def sample_lfm_token_limit(self) -> int:
        return self.max_sample_lfm_tokens or self.max_lfm_tokens

    @model_validator(mode="after")
    def validate_sample_and_batch_limits(self) -> PackingConfig:
        if self.sample_lfm_token_limit > self.max_lfm_tokens:
            raise ValueError(
                "max_sample_lfm_tokens cannot exceed the packed batch token limit"
            )
        return self


class OptimizationConfig(StrictModel):
    optimizer: Literal["adamw"] = "adamw"
    fused: bool = True
    learning_rate: float = Field(3e-4, gt=0)
    min_learning_rate: float = Field(3e-5, ge=0)
    weight_decay: float = Field(0.01, ge=0)
    beta1: float = Field(0.9, gt=0, lt=1)
    beta2: float = Field(0.95, gt=0, lt=1)
    epsilon: float = Field(1e-8, gt=0)
    max_grad_norm: float = Field(1.0, gt=0)
    warmup_updates: int = Field(500, ge=0)
    max_updates: int = Field(20_000, gt=0)
    target_input_tokens_per_update: int = Field(8192, gt=0)
    max_microbatches_per_update: int = Field(32, gt=0)
    torch_compile: bool = False
    torch_compile_backend: Literal["inductor"] = "inductor"
    torch_compile_mode: Literal[
        "default", "reduce-overhead", "max-autotune-no-cudagraphs"
    ] = "default"
    torch_compile_dynamic: bool = True
    compile_whisper_encoder: bool = True
    compile_projector: bool = True
    compile_lfm_backbone: bool = True


class EvaluationConfig(StrictModel):
    validation_max_audio_items: int | None = Field(None, gt=0)
    final_eval_enabled_during_training: Literal[False] = False
    generation_examples: int = Field(32, ge=0)
    generation_max_new_tokens: int = Field(128, gt=0)
    generation_do_sample: bool = False


class CheckpointConfig(StrictModel):
    save_optimizer: bool = True
    save_scheduler: bool = True
    save_rng: bool = True
    save_committed_audio_ids: bool = True
    atomic_write: bool = True


class AppConfig(StrictModel):
    run: RunConfig
    data: DataConfig
    prompt: PromptConfig
    frontend: FrontendConfig
    projector: ProjectorConfig
    llm: LlmConfig
    packing: PackingConfig
    optimization: OptimizationConfig
    evaluation: EvaluationConfig
    checkpoint: CheckpointConfig
    source_path: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def validate_alignment_contract(self) -> AppConfig:
        expanded = self.data.target_provider == "caption_expansion_overlay"
        if expanded != (self.prompt.mode == "caption_expansion_alignment"):
            raise ValueError(
                "caption_expansion_overlay and caption_expansion_alignment "
                "must be configured together"
            )
        if expanded:
            if self.data.expansion_decoder_lock is None:
                raise ValueError(
                    "caption expansion training requires expansion_decoder_lock"
                )
            recipe = (self.data.expansion_recipe_sha256 or "").lower()
            if len(recipe) != 64 or any(
                character not in "0123456789abcdef" for character in recipe
            ):
                raise ValueError(
                    "caption expansion training requires a 64-character "
                    "expansion_recipe_sha256"
                )
            for source in self.data.training_sources or []:
                if source.expansion_decoder_lock is None:
                    raise ValueError(
                        f"training source {source.dataset!r} requires "
                        "expansion_decoder_lock"
                    )
                source_recipe = (source.expansion_recipe_sha256 or "").lower()
                if len(source_recipe) != 64 or any(
                    character not in "0123456789abcdef" for character in source_recipe
                ):
                    raise ValueError(
                        f"training source {source.dataset!r} requires a "
                        "64-character expansion_recipe_sha256"
                    )
        chunked = self.data.long_audio_policy == "chunk_pad"
        if chunked != self.frontend.chunk_long_audio:
            raise ValueError(
                "data chunk_pad and frontend chunk_long_audio must be enabled together"
            )
        if chunked and self.data.max_audio_seconds != self.frontend.max_seconds:
            raise ValueError("data and frontend chunk durations must match exactly")
        if (
            self.optimization.torch_compile
            and self.optimization.compile_whisper_encoder
            and self.frontend.kind == "whisper"
            and self.frontend.mode != "official_fixed_30s"
        ):
            raise ValueError(
                "compiled Whisper requires official_fixed_30s input shapes"
            )
        if self.frontend.kind == "whisper" and self.data.num_workers == 0:
            raise ValueError(
                "Whisper training requires DataLoader workers for CPU log-Mel "
                "preprocessing"
            )
        return self

    def redacted_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json", exclude={"source_path"})
        return cast(dict[str, Any], _redact_config(data))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ValueError(f"Required environment variable {name!r} is not set")
        return os.environ[name]

    return ENV_PATTERN.sub(replace, value)


def _redact_config(value: Any, *, key: str = "") -> Any:
    if is_secret_name(key):
        return REDACTED
    if isinstance(value, dict):
        return {
            item_key: _redact_config(item, key=item_key)
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_config(item, key=key) for item in value]
    return value


def load_config(path: str | Path) -> AppConfig:
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    extends = raw.pop("extends", None)
    if extends is not None:
        base_path = (source.parent / str(extends)).resolve()
        base_raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        if not isinstance(base_raw, dict):
            raise ValueError("Base configuration root must be a mapping")
        if "extends" in base_raw:
            raise ValueError("Nested configuration inheritance is not supported")
        raw = _deep_merge(base_raw, raw)
    expanded = _expand_env(raw)
    config = AppConfig.model_validate(expanded)
    config.source_path = source
    return config
