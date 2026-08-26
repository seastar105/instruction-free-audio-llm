from __future__ import annotations

import importlib.util
import os

import pytest
import torch


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(
    importlib.util.find_spec("flash_attn") is None,
    reason="FlashAttention 2 is unavailable",
)
@pytest.mark.skipif(
    importlib.util.find_spec("causal_conv1d") is None,
    reason="causal-conv1d is unavailable",
)
def test_lfm_packed_forward_and_backward_isolation() -> None:
    from audio_lfm.config import load_config
    from audio_lfm.model.loading import load_audio_lfm
    from audio_lfm.model.packing_preflight import run_lfm_packing_isolation_test

    os.environ.setdefault("CAPTIONSTEW_ROOT", "/tmp/captionstew-not-read")
    config = load_config("configs/paraspeech_whisper_lfm2_smoke.yaml")
    model, _ = load_audio_lfm(config)
    result = run_lfm_packing_isolation_test(model.llm, device=torch.device("cuda"))
    assert result.max_forward_difference <= 3e-2
    assert result.max_gradient_difference <= 3e-2
