from __future__ import annotations

import torch


def convolution_output_length(
    lengths: torch.Tensor,
    *,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> torch.Tensor:
    return (
        torch.div(
            lengths + 2 * padding - dilation * (kernel_size - 1) - 1,
            stride,
            rounding_mode="floor",
        )
        + 1
    )


def whisper_encoder_lengths(mel_lengths: torch.Tensor) -> torch.Tensor:
    lengths = convolution_output_length(mel_lengths, kernel_size=3, stride=1, padding=1)
    return convolution_output_length(lengths, kernel_size=3, stride=2, padding=1)


def waveform_to_mel_lengths(
    num_samples: torch.Tensor, *, hop_length: int = 160
) -> torch.Tensor:
    return torch.div(num_samples + hop_length - 1, hop_length, rounding_mode="floor")


def chunked_whisper_encoder_lengths(
    num_samples: torch.Tensor,
    *,
    chunk_samples: int,
    hop_length: int = 160,
) -> torch.Tensor:
    """Sum effective Whisper lengths after independent fixed-size chunk forwards."""
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    full_chunks = torch.div(num_samples, chunk_samples, rounding_mode="floor")
    remainder = torch.remainder(num_samples, chunk_samples)
    full_mel = waveform_to_mel_lengths(
        torch.full_like(num_samples, chunk_samples), hop_length=hop_length
    )
    full_encoder = whisper_encoder_lengths(full_mel)
    remainder_mel = waveform_to_mel_lengths(remainder, hop_length=hop_length)
    remainder_encoder = torch.where(
        remainder > 0,
        whisper_encoder_lengths(remainder_mel),
        torch.zeros_like(remainder),
    )
    return full_chunks * full_encoder + remainder_encoder


def split_waveform_chunks(
    waveform: torch.Tensor, *, chunk_samples: int
) -> list[torch.Tensor]:
    if waveform.ndim != 1 or waveform.numel() <= 0:
        raise ValueError("waveform must be a nonempty mono tensor")
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    return list(waveform.split(chunk_samples))


def make_additive_key_padding_mask(
    *, lengths: torch.Tensor, max_length: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    positions = torch.arange(max_length, device=device)
    invalid_keys = positions.unsqueeze(0) >= lengths.to(device).unsqueeze(1)
    mask = torch.zeros(
        (lengths.shape[0], 1, max_length, max_length),
        dtype=dtype,
        device=device,
    )
    return mask.masked_fill(invalid_keys[:, None, None, :], torch.finfo(dtype).min)


def slice_valid_outputs(
    hidden: torch.Tensor, lengths: torch.Tensor
) -> list[torch.Tensor]:
    return [
        hidden[index, : int(length.item())].detach()
        for index, length in enumerate(lengths)
    ]
