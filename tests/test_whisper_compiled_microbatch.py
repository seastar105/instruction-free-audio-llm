from __future__ import annotations

import torch

from audio_lfm.model.frontends.base import AudioFrontend
from audio_lfm.model.frontends.whisper import WhisperFrontend


def test_compiled_whisper_pads_only_the_block_batch_dimension() -> None:
    frontend = WhisperFrontend.__new__(WhisperFrontend)
    AudioFrontend.__init__(frontend)
    frontend.device = torch.device("cpu")
    frontend.max_samples = 480_000
    frontend.encoder_microbatch_max_padded_samples = 8 * 480_000
    calls: list[tuple[int, ...]] = []

    def compiled(features: torch.Tensor) -> torch.Tensor:
        calls.append(tuple(features.shape))
        # Mimic Whisper's 2x temporal convolution reduction.
        return features[:, :1, ::2].transpose(1, 2)

    frontend._compiled_encoder_forward = compiled
    first = torch.randn(1, 80, 3000)
    second = torch.randn(9, 80, 3000)
    outputs = frontend.encode_precomputed(
        [first, second],
        [(238,), (1500,) * 8 + (750,)],
    )

    assert calls == [(8, 80, 3000), (8, 80, 3000)]
    assert [tuple(value.shape) for value in outputs] == [(238, 1), (12_750, 1)]
