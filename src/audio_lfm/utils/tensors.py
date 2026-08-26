from __future__ import annotations

import torch


def tensor_rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().square().mean().sqrt()


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def assert_finite(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
