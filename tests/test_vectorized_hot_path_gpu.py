from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from audio_lfm.model.packed_batch import (
    build_packed_layout,
    build_vectorized_packed_batch,
)
from audio_lfm.model.projector import FrameStackMLPProjector
from audio_lfm.training.loss import selective_causal_lm_loss
from tests.test_pack_planner import _example


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_vectorized_cuda_hot_path_has_no_host_scalar_sync() -> None:
    projector = FrameStackMLPProjector(
        input_dim=3,
        output_dim=8,
        stack_factor=4,
        hidden_dim=16,
        dropout=0.0,
    ).to("cuda", dtype=torch.bfloat16)
    table = torch.nn.Embedding(10, 8).to("cuda", dtype=torch.bfloat16)
    table.requires_grad_(False)
    lm_head = torch.nn.Linear(8, 10, bias=False).to("cuda", dtype=torch.bfloat16)
    encoded = torch.randn(3, 8, 3, device="cuda", dtype=torch.bfloat16)
    encoder_mask = (
        torch.arange(8, device="cuda")[None]
        < torch.tensor((7, 8, 5), device="cuda")[:, None]
    )
    projected_mask = torch.ones(3, 2, device="cuda", dtype=torch.bool)
    projected_indices = torch.arange(6, device="cuda")
    examples = [
        replace(
            _example(0, 7),
            estimated_audio_embedding_length=2,
            estimated_total_lfm_length=7,
        ),
        replace(
            _example(1, 9),
            estimated_audio_embedding_length=4,
            estimated_total_lfm_length=9,
        ),
    ]
    layout = build_packed_layout(
        examples=examples, vocabulary_size=10, max_lfm_tokens=32
    ).to(torch.device("cuda"))

    previous = torch.cuda.get_sync_debug_mode()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        torch.cuda.set_sync_debug_mode("error")
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                projected = projector.project_blocks(
                    encoded, encoder_mask, projected_mask, projected_indices
                )
                batch = build_vectorized_packed_batch(
                    layout=layout,
                    projected_audio=projected,
                    embed_tokens=table,
                    audio_start=projector.audio_start,
                    audio_end=projector.audio_end,
                )
                loss, supervised = selective_causal_lm_loss(
                    hidden_states=batch.inputs_embeds,
                    labels=batch.labels,
                    supervised_token_indices=batch.supervised_token_indices,
                    lm_head=lm_head,
                )
                assert supervised == batch.supervised_token_count
                loss.backward()
        finally:
            torch.cuda.set_sync_debug_mode(previous)
    keys = {event.key for event in profile.key_averages()}
    assert "aten::item" not in keys
    assert "aten::_local_scalar_dense" not in keys
    assert any("masked_scatter" in key for key in keys)
