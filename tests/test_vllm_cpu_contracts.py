from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from audio_lfm.vllm_plugin.config import validate_export_config
from audio_lfm.vllm_plugin.data_parser import validate_feature_dict
from audio_lfm.vllm_plugin.processing import (
    dummy_waveform_num_samples,
    extract_chunked_audio_features,
    max_audio_tokens_per_item,
    replacement_token_ids,
    split_audio_waveform,
)
from audio_lfm.vllm_plugin.weight_mapping import map_weight_names


def test_feature_validation_and_variable_prompt_replacement() -> None:
    validate_feature_dict(
        {
            "audio_features": np.zeros((80, 12), np.float32),
            "audio_feature_length": 10,
            "audio_token_length": 5,
        }
    )
    assert replacement_token_ids(100, 5) == [100] * 5
    with pytest.raises(ValueError):
        replacement_token_ids(100, 2)


def test_long_and_multiple_audio_cpu_contracts() -> None:
    chunks = split_audio_waveform(
        np.zeros(65 * 16_000, dtype=np.float32), chunk_samples=30 * 16_000
    )
    assert [chunk.size for chunk in chunks] == [
        30 * 16_000,
        30 * 16_000,
        5 * 16_000,
    ]
    assert (
        max_audio_tokens_per_item(seq_len=32_768, item_count=3, stack_factor=4) == 9_002
    )
    assert dummy_waveform_num_samples(token_budget=9_002, stack_factor=4) > 0


def test_feature_extraction_preserves_every_chunk_boundary() -> None:
    from transformers import WhisperFeatureExtractor

    features, feature_length, chunk_lengths, token_length = (
        extract_chunked_audio_features(
            WhisperFeatureExtractor(),
            np.zeros(18_001, dtype=np.float32),
            sample_rate=16_000,
            chunk_samples=8_000,
            stack_factor=4,
        )
    )
    assert chunk_lengths.tolist() == [50, 50, 13]
    assert feature_length == 113
    assert features.shape == (80, 113)
    # Independent Whisper chunks produce 25 + 25 + 7 encoder frames.
    assert token_length == 17


def test_feature_validation_checks_chunk_boundaries() -> None:
    validate_feature_dict(
        {
            "audio_features": np.zeros((80, 12), np.float32),
            "audio_feature_length": 12,
            "audio_chunk_length": np.array([7, 5]),
            "audio_token_length": 5,
        }
    )
    with pytest.raises(ValueError, match="must sum"):
        validate_feature_dict(
            {
                "audio_features": np.zeros((80, 12), np.float32),
                "audio_feature_length": 12,
                "audio_chunk_length": np.array([7, 4]),
                "audio_token_length": 5,
            }
        )


def test_whisper_weight_mapping_is_explicit() -> None:
    values = list(
        map_weight_names(
            [
                ("audio_tower.model.encoder.conv1.weight", torch.ones(1)),
                ("audio_tower.model.decoder.layer.weight", torch.ones(1)),
            ]
        )
    )
    assert [name for name, _ in values] == ["audio_tower.conv1.weight"]


def test_audio_placeholder_may_use_padded_embedding_row() -> None:
    config = SimpleNamespace(
        model_type="lfm2",
        architectures=["AudioLfm2ForConditionalGeneration"],
        audio_lfm_format_version=1,
        text_model_id="LiquidAI/LFM2.5-1.2B-Instruct",
        text_model_revision="a" * 40,
        audio_model_id="openai/whisper-small",
        audio_model_revision="b" * 40,
        audio_token="<|audio|>",
        audio_token_index=64_402,
        text_tokenizer_size=64_402,
        vocab_size=65_536,
        audio_config={},
        projector_config={},
        projector_checkpoint_sha256="c" * 64,
    )
    validate_export_config(config)
    config.audio_token_index = 64_401
    with pytest.raises(ValueError, match="frozen tokenizer vocabulary"):
        validate_export_config(config)
