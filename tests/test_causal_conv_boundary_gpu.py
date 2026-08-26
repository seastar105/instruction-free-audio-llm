from __future__ import annotations

import importlib.util

import pytest
import torch

from audio_lfm.model.packing_preflight import run_direct_causal_conv_boundary_test


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(
    importlib.util.find_spec("causal_conv1d") is None,
    reason="causal-conv1d is unavailable",
)
def test_causal_conv_forward_backward_and_perturbation_isolation() -> None:
    result = run_direct_causal_conv_boundary_test(device=torch.device("cuda"))
    assert result.max_forward_difference <= 2e-3
    assert result.max_gradient_difference <= 2e-3
