from __future__ import annotations

import torch

from audio_lfm.model.packed_batch import build_packed_batch
from tests.test_pack_planner import _example


def test_metadata_resets_positions_and_keeps_projector_gradient() -> None:
    examples = [_example(0, 5), _example(1, 5)]
    table = torch.nn.Embedding(10, 4)
    audio = [torch.randn(1, 4, requires_grad=True) for _ in examples]
    start = torch.nn.Parameter(torch.randn(4))
    end = torch.nn.Parameter(torch.randn(4))
    batch = build_packed_batch(
        examples=examples,
        projected_audio=audio,
        embed_tokens=table,
        audio_start=start,
        audio_end=end,
        device=torch.device("cpu"),
        vocabulary_size=10,
        max_lfm_tokens=20,
    )
    assert batch.seq_idx.dtype == torch.int32
    assert batch.cu_seq_lens_q.tolist() == [0, 6, 12]
    assert batch.position_ids.tolist() == [list(range(6)) + list(range(6))]
    batch.inputs_embeds.sum().backward()
    assert all(value.grad is not None for value in audio)
    assert table.weight.grad is None
