from __future__ import annotations

import torch

from audio_lfm.model.frontends.whisper_math import (
    chunked_whisper_encoder_lengths,
    convolution_output_length,
    make_additive_key_padding_mask,
    split_waveform_chunks,
    whisper_encoder_lengths,
)


def test_whisper_convolution_lengths_at_boundaries() -> None:
    lengths = torch.tensor([1, 2, 3, 4, 5, 6])
    expected = torch.tensor([1, 1, 2, 2, 3, 3])
    assert torch.equal(whisper_encoder_lengths(lengths), expected)
    assert (
        convolution_output_length(
            torch.tensor([3]), kernel_size=3, stride=2, padding=1
        ).item()
        == 2
    )


def test_additive_mask_masks_only_padded_keys() -> None:
    mask = make_additive_key_padding_mask(
        lengths=torch.tensor([2]),
        max_length=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert mask.shape == (1, 1, 4, 4)
    assert torch.all(mask[..., :2] == 0)
    assert torch.all(mask[..., 2:] < -1e20)


def test_fixed_30_second_chunks_keep_only_effective_audio_lengths() -> None:
    samples = torch.tensor([19 * 16_000, 45 * 16_000])
    encoder = chunked_whisper_encoder_lengths(samples, chunk_samples=30 * 16_000)
    assert encoder.tolist() == [950, 2250]
    projected = torch.div(encoder + 3, 4, rounding_mode="floor")
    assert projected.tolist() == [238, 563]

    chunks = split_waveform_chunks(torch.zeros(45 * 16_000), chunk_samples=30 * 16_000)
    assert [chunk.numel() for chunk in chunks] == [30 * 16_000, 15 * 16_000]
