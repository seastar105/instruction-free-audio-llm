from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import WhisperFeatureExtractor

from audio_lfm.data.types import RawAudioExample, TargetRecord
from audio_lfm.data.worker_preprocessing import (
    AudioPreprocessingConfig,
    WhisperWorkerPreprocessor,
)
from audio_lfm.model.frontends.whisper_math import (
    split_waveform_chunks,
    whisper_encoder_lengths,
)


def _raw(seconds: int) -> RawAudioExample:
    samples = seconds * 16_000
    time = torch.arange(samples, dtype=torch.float32) / 16_000
    waveform = 0.1 * torch.sin(2 * torch.pi * 220 * time)
    target = TargetRecord(
        audio_id=str(seconds),
        target_id=f"target-{seconds}",
        target_type="audio_assistant_response",
        text="A reference response.",
        split="train_base",
        source="expanded",
        review_status="accepted",
    )
    return RawAudioExample(
        audio_id=str(seconds),
        waveform=waveform,
        sample_rate=16_000,
        source_id=str(seconds),
        splits=("train_base",),
        style_captions=(target,),
        transcript=None,
        selected_target=target,
        metadata={},
        crop_start_sample=None,
        original_num_samples=samples,
    )


@pytest.mark.parametrize(
    ("seconds", "expected_lengths"),
    [(19, (950,)), (30, (1500,)), (45, (1500, 750))],
)
def test_worker_features_match_reference_preprocessing(
    seconds: int, expected_lengths: tuple[int, ...]
) -> None:
    raw = _raw(seconds)
    config = AudioPreprocessingConfig(model_id=None, revision=None)
    actual = WhisperWorkerPreprocessor(config)(raw)

    reference_extractor = WhisperFeatureExtractor()
    chunks = split_waveform_chunks(raw.waveform, chunk_samples=30 * 16_000)
    reference = reference_extractor(
        [chunk.numpy() for chunk in chunks],
        sampling_rate=16_000,
        padding="max_length",
        max_length=30 * 16_000,
        truncation=False,
        return_attention_mask=True,
        return_tensors="pt",
    )
    reference_lengths = whisper_encoder_lengths(
        reference.attention_mask.sum(dim=-1).long()
    )

    assert torch.equal(actual.input_features, reference.input_features.float())
    assert actual.effective_encoder_lengths == expected_lengths
    assert actual.effective_encoder_lengths == tuple(reference_lengths.tolist())
    assert actual.evaluated_num_samples == seconds * 16_000
    assert actual.input_features.shape == (len(expected_lengths), 80, 3000)
    assert actual.encoder_frame_mask is not None
    assert actual.projected_frame_mask is not None
    assert actual.encoder_frame_mask.sum(dim=1).tolist() == list(expected_lengths)
    assert actual.projected_frame_mask.sum(dim=1).tolist() == [
        (length + 3) // 4 for length in expected_lengths
    ]
    assert actual.projected_frame_indices is not None
    assert actual.projected_frame_indices.numel() == sum(
        (length + 3) // 4 for length in expected_lengths
    )
    assert np.isfinite(actual.input_features.numpy()).all()
    assert not hasattr(actual, "waveform")
    repeated = WhisperWorkerPreprocessor(config)(raw)
    assert torch.equal(actual.input_features, repeated.input_features)
    assert actual.effective_encoder_lengths == repeated.effective_encoder_lengths
