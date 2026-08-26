from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from audio_lfm.config import load_config


def _config(tmp_path: Path) -> dict[str, object]:
    return {
        "run": {"name": "test", "output_dir": str(tmp_path / "runs")},
        "data": {"captionstew_root": "${ENV:CAPTIONSTEW_ROOT}"},
        "prompt": {
            "prompt_file": "configs/prompts/paraspeech_style_caption.txt",
            "audio_sentinel": "<<__AUDIO_EMBEDDINGS_08E8F7E7__>>",
        },
        "frontend": {},
        "projector": {},
        "llm": {"model_id": "LiquidAI/LFM2.5-1.2B-Instruct"},
        "packing": {},
        "optimization": {},
        "evaluation": {},
        "checkpoint": {},
    }


def test_environment_expansion_and_redaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAPTIONSTEW_ROOT", "/private/data")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_config(tmp_path)))
    config = load_config(path)
    assert config.data.captionstew_root == Path("/private/data")
    serialized = str(config.redacted_dict())
    assert "HF_TOKEN" not in serialized


def test_unknown_fields_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAPTIONSTEW_ROOT", "/data")
    value = _config(tmp_path)
    value["run"]["mystery"] = True  # type: ignore[index]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValidationError):
        load_config(path)


def test_paraspeech_test_split_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAPTIONSTEW_ROOT", "/data")
    value = _config(tmp_path)
    value["data"]["final_split"] = "test"  # type: ignore[index]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValidationError, match="test"):
        load_config(path)


def test_smoke_config_inherits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAPTIONSTEW_ROOT", "/data")
    config = load_config("configs/paraspeech_whisper_lfm2_smoke.yaml")
    assert config.optimization.max_updates == 20
    assert config.llm.model_id == "LiquidAI/LFM2.5-1.2B-Instruct"


def test_caption_expansion_mode_rejects_training_system_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAPTIONSTEW_ROOT", "/data")
    value = _config(tmp_path)
    value["data"].update(  # type: ignore[union-attr]
        {
            "target_provider": "caption_expansion_overlay",
            "target_type": "audio_assistant_response",
            "expansion_decoder_lock": "/tmp/decoder-lock.json",
            "expansion_recipe_sha256": "a" * 64,
        }
    )
    value["prompt"] = {
        "mode": "caption_expansion_alignment",
        "prompt_file": None,
        "audio_sentinel": "<<__AUDIO_EMBEDDINGS_08E8F7E7__>>",
        "system_message": "This must not be accepted.",
        "require_no_system_message": True,
        "user_content": "audio_sentinel_only",
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValidationError, match="forbids a training-time system"):
        load_config(path)


def test_one_thousand_step_chunked_training_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAPTIONSTEW_ROOT", "/data")
    monkeypatch.setenv("CAPTION_EXPANSION_DECODER_LOCK", "/tmp/decoder-lock.json")
    monkeypatch.setenv("CAPTION_EXPANSION_RECIPE_SHA256", "a" * 64)
    monkeypatch.setenv("WAVCAPS_EXPANSION_DECODER_LOCK", "/tmp/wavcaps-lock.json")
    monkeypatch.setenv("WAVCAPS_EXPANSION_RECIPE_SHA256", "b" * 64)
    config = load_config("configs/paraspeech_whisper_small_lfm2_expanded_1k.yaml")
    assert config.optimization.max_updates == 1000
    assert config.optimization.torch_compile
    assert config.frontend.model_id == "openai/whisper-small"
    assert config.frontend.mode == "official_fixed_30s"
    assert config.frontend.chunk_long_audio
    assert config.projector.stack_factor == 4
    assert config.packing.max_lfm_tokens == 16384
    assert config.packing.sample_lfm_token_limit == 16384
    assert config.packing.planning_buffer_examples == 2048
    assert config.packing.max_examples_per_pack is None
    assert config.packing.oversized_example_policy == "skip"
    assert config.optimization.target_input_tokens_per_update == 16384
    assert config.data.long_audio_policy == "chunk_pad"
    assert config.optimization.max_microbatches_per_update == 1
    assert config.data.training_sources is not None
    assert [source.dataset for source in config.data.training_sources] == [
        "ParaSpeechCaps-Base",
        "WavCaps",
    ]
