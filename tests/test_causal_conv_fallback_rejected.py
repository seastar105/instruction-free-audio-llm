from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from audio_lfm.model.packing_preflight import run_direct_causal_conv_boundary_test


def _fallback_ignoring_seq_idx(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    activation: object = None,
    seq_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    del activation, seq_idx
    result = F.conv1d(
        value,
        weight[:, None],
        bias,
        padding=weight.shape[1] - 1,
        groups=value.shape[1],
    )
    return result[..., : value.shape[-1]]


def test_fallback_that_ignores_boundaries_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="boundary forward isolation failed"):
        run_direct_causal_conv_boundary_test(
            device=torch.device("cpu"),
            tolerance=1e-6,
            conv_function=_fallback_ignoring_seq_idx,
        )
