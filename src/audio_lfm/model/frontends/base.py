from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class AudioFrontend(nn.Module, ABC):
    output_dim: int

    @abstractmethod
    def estimate_output_lengths(self, num_samples: torch.Tensor) -> torch.Tensor: ...

    def encode(self, waveforms: list[torch.Tensor]) -> list[torch.Tensor]:
        """Return one detached [time, output_dim] tensor per waveform."""
        raise NotImplementedError("This frontend requires worker-precomputed features")
