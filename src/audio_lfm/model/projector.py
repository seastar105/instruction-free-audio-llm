from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(value.dtype)


def frame_stack(value: torch.Tensor, factor: int) -> torch.Tensor:
    if value.ndim != 2 or value.shape[0] <= 0:
        raise ValueError("Audio features must be nonempty [time, dimension]")
    padding = (-value.shape[0]) % factor
    if padding:
        value = F.pad(value, (0, 0, 0, padding))
    return value.reshape(value.shape[0] // factor, value.shape[1] * factor)


class FrameStackMLPProjector(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        stack_factor: int = 5,
        hidden_dim: int = 2048,
        dropout: float = 0.0,
        use_input_layer_norm: bool = True,
        use_output_rms_norm: bool = True,
        target_embedding_rms: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.stack_factor = stack_factor
        stacked = input_dim * stack_factor
        self.input_norm = (
            nn.LayerNorm(stacked) if use_input_layer_norm else nn.Identity()
        )
        self.linear_in = nn.Linear(stacked, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(hidden_dim, output_dim)
        self.output_norm = RMSNorm(output_dim) if use_output_rms_norm else nn.Identity()
        self.output_scale = nn.Parameter(torch.tensor(float(target_embedding_rms)))
        self.audio_start = nn.Parameter(torch.empty(output_dim))
        self.audio_end = nn.Parameter(torch.empty(output_dim))
        nn.init.normal_(self.audio_start, std=target_embedding_rms)
        nn.init.normal_(self.audio_end, std=target_embedding_rms)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FrameStackMLPProjector:
        allowed = {
            "input_dim",
            "output_dim",
            "stack_factor",
            "hidden_dim",
            "dropout",
            "use_input_layer_norm",
            "use_output_rms_norm",
            "target_embedding_rms",
        }
        return cls(**{key: value for key, value in config.items() if key in allowed})

    def projected_length(self, frontend_length: int) -> int:
        return math.ceil(frontend_length / self.stack_factor)

    def project_frames(self, value: torch.Tensor) -> torch.Tensor:
        original_dtype = value.dtype
        stacked = frame_stack(value, self.stack_factor)
        return self._project_stacked(stacked, original_dtype=original_dtype)

    def _project_stacked(
        self, stacked: torch.Tensor, *, original_dtype: torch.dtype
    ) -> torch.Tensor:
        projected = self.linear_out(
            self.dropout(F.gelu(self.linear_in(self.input_norm(stacked))))
        )
        projected = self.output_norm(projected)
        return projected * self.output_scale.to(dtype=original_dtype)

    def project_blocks(
        self,
        encoded_blocks: torch.Tensor,
        encoder_frame_mask: torch.Tensor,
        projected_frame_mask: torch.Tensor,
        projected_frame_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project all fixed-size Whisper blocks, then flatten valid frames."""
        if encoded_blocks.ndim != 3:
            raise ValueError("Encoded blocks must be [blocks, frames, dimension]")
        if encoder_frame_mask.shape != encoded_blocks.shape[:2]:
            raise ValueError("Encoder-frame mask shape differs from encoded blocks")
        blocks, frames, dimension = encoded_blocks.shape
        if frames % self.stack_factor:
            raise ValueError("Fixed Whisper frames must divide the stack factor")
        projected_frames = frames // self.stack_factor
        if projected_frame_mask.shape != (blocks, projected_frames):
            raise ValueError("Projected-frame mask shape is invalid")
        masked = encoded_blocks.masked_fill(~encoder_frame_mask[..., None], 0)
        stacked = masked.reshape(
            blocks, projected_frames, dimension * self.stack_factor
        )
        projected = self._project_stacked(stacked, original_dtype=encoded_blocks.dtype)
        flattened = projected.flatten(0, 1)
        if projected_frame_indices is None:
            return flattened[projected_frame_mask.flatten()]
        return flattened.index_select(0, projected_frame_indices)

    def forward(self, values: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self.project_frames(value) for value in values]


class DmelProjector(nn.Module):
    def __init__(
        self,
        *,
        num_bins: int,
        num_channels: int,
        output_dim: int,
        bin_embedding_dim: int = 16,
        temporal_patch_size: int = 8,
        hidden_dim: int = 2048,
        target_embedding_rms: float = 1.0,
    ) -> None:
        super().__init__()
        self.temporal_patch_size = temporal_patch_size
        self.output_dim = output_dim
        self.bin_embedding = nn.Embedding(num_bins, bin_embedding_dim)
        self.channel_embedding = nn.Embedding(num_channels, bin_embedding_dim)
        per_frame = num_channels * bin_embedding_dim
        self.input_norm = nn.LayerNorm(per_frame * temporal_patch_size)
        self.linear_in = nn.Linear(per_frame * temporal_patch_size, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, output_dim)
        self.output_norm = RMSNorm(output_dim)
        self.output_scale = nn.Parameter(torch.tensor(float(target_embedding_rms)))
        self.audio_start = nn.Parameter(torch.randn(output_dim) * target_embedding_rms)
        self.audio_end = nn.Parameter(torch.randn(output_dim) * target_embedding_rms)

    def projected_length(self, frontend_length: int) -> int:
        return math.ceil(frontend_length / self.temporal_patch_size)

    def project_frames(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim != 2:
            raise ValueError("dMel codes must be [time, channels]")
        channels = torch.arange(codes.shape[1], device=codes.device)
        value = (
            self.bin_embedding(codes.long()) + self.channel_embedding(channels)[None]
        )
        flattened = value.flatten(1)
        patched = frame_stack(flattened, self.temporal_patch_size)
        output = self.linear_out(F.gelu(self.linear_in(self.input_norm(patched))))
        return self.output_norm(output) * self.output_scale

    def forward(self, values: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self.project_frames(value) for value in values]
