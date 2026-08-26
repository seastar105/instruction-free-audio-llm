from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file

from audio_lfm.config import AppConfig
from audio_lfm.prompts import audio_training_prompt_template_sha256
from audio_lfm.utils.hashing import (
    canonical_sha256,
    sha256_file,
    sha256_text,
)
from audio_lfm.vllm_plugin.config import projector_config_from_training


def _immutable_revision(manifest: dict[str, Any], key: str) -> str:
    value = str(manifest.get(key, ""))
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"Training manifest {key} is not an immutable commit SHA")
    return value


def export_vllm_artifact(
    config: AppConfig,
    checkpoint: str | Path,
    output_dir: str | Path,
) -> Path:
    from transformers import AddedToken, AutoConfig, AutoTokenizer, GenerationConfig

    checkpoint = Path(checkpoint)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Export directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    text_revision = _immutable_revision(manifest, "llm_revision")
    audio_revision = _immutable_revision(manifest, "frontend_revision")
    base_config_object = AutoConfig.from_pretrained(
        config.llm.model_id, revision=text_revision, trust_remote_code=False
    )
    base_config = base_config_object.to_dict()
    if base_config.get("model_type") != "lfm2":
        raise ValueError("Frozen decoder config is not LFM2")
    tokenizer = AutoTokenizer.from_pretrained(
        config.llm.model_id, revision=text_revision, trust_remote_code=False
    )
    original_template = tokenizer.chat_template
    if not original_template:
        raise ValueError("LFM tokenizer lacks a chat template")
    text_tokenizer_size = len(tokenizer)
    audio_token = AddedToken(
        "<|audio|>", special=True, normalized=False, lstrip=False, rstrip=False
    )
    added = tokenizer.add_special_tokens({"additional_special_tokens": [audio_token]})
    if added != 1:
        raise RuntimeError("Expected to add exactly one audio token")
    audio_token_id = int(tokenizer.convert_tokens_to_ids("<|audio|>"))
    if tokenizer.encode("<|audio|>", add_special_tokens=False) != [audio_token_id]:
        raise RuntimeError("Audio placeholder must tokenize to one token")
    if audio_token_id < text_tokenizer_size:
        raise RuntimeError("Audio placeholder replaced a frozen tokenizer token")
    if tokenizer.chat_template != original_template:
        raise RuntimeError("Adding the audio token changed the chat template")
    tokenizer.save_pretrained(output)
    audio_config = AutoConfig.from_pretrained(
        config.frontend.model_id,
        revision=audio_revision,
        trust_remote_code=False,
    ).to_dict()
    local_checkpoint = checkpoint / "projector.safetensors"
    mapped: dict[str, Any] = {}
    mappings: dict[str, str] = {}
    with safe_open(local_checkpoint, framework="pt", device="cpu") as handle:
        source_names = handle.keys()
        for source_name in source_names:
            if not source_name.startswith("projector."):
                raise ValueError(f"Unexpected trainable checkpoint key: {source_name}")
            destination_name = "multi_modal_projector." + source_name.removeprefix(
                "projector."
            )
            if destination_name in mapped:
                raise ValueError(f"Duplicate mapped projector key: {destination_name}")
            tensor = handle.get_tensor(source_name)
            if not tensor.isfinite().all():
                raise ValueError(f"Non-finite projector tensor: {source_name}")
            mapped[destination_name] = tensor.contiguous()
            mappings[source_name] = destination_name
    if not mapped:
        raise ValueError("Checkpoint contains no projector weights")
    model_path = output / "model.safetensors"
    save_file(mapped, model_path)
    target_rms = float(manifest["projector_embedding_target_rms"])
    projector_config = projector_config_from_training(
        config.projector.model_dump(mode="json"),
        input_dim=int(audio_config["d_model"]),
        output_dim=int(base_config["hidden_size"]),
        target_rms=target_rms,
    )
    prompt_text = (
        config.prompt.audio_sentinel
        if config.prompt.mode == "caption_expansion_alignment"
        else config.prompt.prompt_file.read_text(encoding="utf-8")
    )
    audio_prompt_hash = audio_training_prompt_template_sha256(
        config.prompt.audio_sentinel
    )
    tokenizer_files = sorted(
        path for path in output.iterdir() if path.name != "model.safetensors"
    )
    tokenizer_hash = canonical_sha256(
        {path.name: sha256_file(path) for path in tokenizer_files if path.is_file()}
    )
    base_config.update(
        {
            "architectures": ["AudioLfm2ForConditionalGeneration"],
            "audio_lfm_format_version": 1,
            "text_model_id": config.llm.model_id,
            "text_model_revision": text_revision,
            "audio_model_id": config.frontend.model_id,
            "audio_model_revision": audio_revision,
            "frontend_kind": "whisper",
            "frontend_mode": config.frontend.mode,
            "audio_sample_rate": 16000,
            "max_audio_seconds": config.frontend.max_seconds,
            "audio_token": "<|audio|>",
            "audio_token_index": audio_token_id,
            "text_tokenizer_size": text_tokenizer_size,
            "audio_config": audio_config,
            "projector_config": projector_config,
            "projector_checkpoint_sha256": sha256_file(model_path),
            "training_run_manifest_sha256": sha256_file(manifest_path),
            "prompt_sha256": sha256_text(prompt_text),
            "audio_inference_has_system_message": False,
            "audio_inference_prompt_template_sha256": audio_prompt_hash,
            "expansion_recipe_sha256": config.data.expansion_recipe_sha256,
            "base_chat_template_sha256": sha256_text(original_template),
            "export_tokenizer_sha256": tokenizer_hash,
            "projected_length_formula_version": 1,
        }
    )
    (output / "config.json").write_text(
        json.dumps(base_config, indent=2, sort_keys=True), encoding="utf-8"
    )
    try:
        GenerationConfig.from_pretrained(
            config.llm.model_id, revision=text_revision
        ).save_pretrained(output)
    except OSError:
        GenerationConfig().save_pretrained(output)
    projector_manifest = {
        "format_version": 1,
        "source_to_export": mappings,
        "tensor_count": len(mapped),
        "projector_checkpoint_sha256": sha256_file(model_path),
    }
    (output / "projector_manifest.json").write_text(
        json.dumps(projector_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    export_manifest = {
        "format_version": 1,
        "local_weights_incomplete_by_design": True,
        "secondary_sources": [
            {"model_id": config.llm.model_id, "revision": text_revision},
            {"model_id": config.frontend.model_id, "revision": audio_revision},
        ],
        "projector_checkpoint_sha256": sha256_file(model_path),
        "training_run_manifest_sha256": sha256_file(manifest_path),
        "prompt_file": (
            str(config.prompt.prompt_file)
            if config.prompt.prompt_file is not None
            else None
        ),
        "prompt_text": prompt_text,
        "audio_sentinel": config.prompt.audio_sentinel,
        "audio_training_has_system_message": False,
        "audio_training_prompt_template_sha256": audio_prompt_hash,
        "audio_inference_has_system_message": False,
        "audio_inference_prompt_template_sha256": audio_prompt_hash,
        "expansion_recipe_sha256": config.data.expansion_recipe_sha256,
        "resolved_training_config": config.redacted_dict(),
    }
    (output / "export_manifest.json").write_text(
        json.dumps(export_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output
