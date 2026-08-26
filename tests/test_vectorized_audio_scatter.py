from __future__ import annotations

from dataclasses import replace

import torch

from audio_lfm.model.packed_batch import (
    build_packed_batch,
    build_packed_layout,
    build_vectorized_packed_batch,
)
from audio_lfm.model.projector import FrameStackMLPProjector
from tests.test_pack_planner import _example


def test_vectorized_projection_and_masked_scatter_match_per_audio_reference() -> None:
    torch.manual_seed(7)
    projector = FrameStackMLPProjector(
        input_dim=3,
        output_dim=6,
        stack_factor=4,
        hidden_dim=10,
        dropout=0.0,
    )
    encoded = torch.randn(3, 8, 3)
    encoder_lengths = (7, 8, 5)
    encoder_mask = torch.arange(8)[None] < torch.tensor(encoder_lengths)[:, None]
    projected_mask = torch.arange(2)[None] < torch.tensor((2, 2, 2))[:, None]
    projected_indices = torch.nonzero(
        projected_mask.flatten(), as_tuple=False
    ).flatten()
    vectorized_audio = projector.project_blocks(
        encoded, encoder_mask, projected_mask, projected_indices
    )
    reference_audio = [
        projector.project_frames(encoded[0, :7]),
        projector.project_frames(torch.cat([encoded[1, :8], encoded[2, :5]])),
    ]
    torch.testing.assert_close(
        vectorized_audio, torch.cat(reference_audio), atol=2e-6, rtol=2e-6
    )

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
    table = torch.nn.Embedding(10, 6)
    layout = build_packed_layout(
        examples=examples, vocabulary_size=10, max_lfm_tokens=32
    )
    vectorized_batch = build_vectorized_packed_batch(
        layout=layout,
        projected_audio=vectorized_audio,
        embed_tokens=table,
        audio_start=projector.audio_start,
        audio_end=projector.audio_end,
    )
    reference_batch = build_packed_batch(
        examples=examples,
        projected_audio=reference_audio,
        embed_tokens=table,
        audio_start=projector.audio_start,
        audio_end=projector.audio_end,
        device=torch.device("cpu"),
        vocabulary_size=10,
        max_lfm_tokens=32,
    )
    torch.testing.assert_close(
        vectorized_batch.inputs_embeds,
        reference_batch.inputs_embeds,
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.equal(vectorized_batch.labels, reference_batch.labels)
    assert torch.equal(vectorized_batch.position_ids, reference_batch.position_ids)
    assert torch.equal(vectorized_batch.seq_idx, reference_batch.seq_idx)
    vectorized_batch.inputs_embeds.square().sum().backward()
    assert all(parameter.grad is not None for parameter in projector.parameters())
    assert table.weight.grad is None
