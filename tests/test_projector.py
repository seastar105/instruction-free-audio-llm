from __future__ import annotations

import torch

from audio_lfm.model.projector import FrameStackMLPProjector


def test_frame_stack_projector_preserves_ceil_length_and_gradients() -> None:
    projector = FrameStackMLPProjector(
        input_dim=3, output_dim=4, stack_factor=5, hidden_dim=8
    )
    value = torch.randn(11, 3)
    output = projector.project_frames(value)
    assert output.shape == (3, 4)
    with_boundaries = torch.cat(
        [projector.audio_start[None], output, projector.audio_end[None]]
    )
    with_boundaries.square().mean().backward()
    assert all(parameter.grad is not None for parameter in projector.parameters())
