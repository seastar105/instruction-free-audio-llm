from __future__ import annotations

import pytest
import torch


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_whisper_full_length_and_padding_isolation() -> None:
    from transformers import WhisperModel

    from audio_lfm.model.frontends.whisper_encoder import (
        VariableLengthWhisperEncoder,
    )

    official = WhisperModel.from_pretrained("openai/whisper-small").encoder
    official.eval().requires_grad_(False).to("cuda", dtype=torch.bfloat16)
    variable = VariableLengthWhisperEncoder.from_encoder(official).eval()
    full = torch.randn(1, 80, 3000, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = official(full).last_hidden_state
        actual = variable(full, torch.tensor([3000], device="cuda"))[0]
    torch.testing.assert_close(actual, expected[0], atol=3e-2, rtol=3e-2)

    short_a = full[..., :211].clone()
    short_b = full[..., :319].clone()
    padded_a = torch.nn.functional.pad(short_a, (0, 319 - 211))
    batch = torch.cat([padded_a, short_b])
    lengths = torch.tensor([211, 319], device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        together = variable(batch, lengths)
        alone_a = variable(short_a, lengths[:1])[0]
        alone_b = variable(short_b, lengths[1:])[0]
        perturbed = batch.clone()
        perturbed[0, :, 211:] = 100
        invariant = variable(perturbed, lengths)[0]
    torch.testing.assert_close(together[0], alone_a, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(together[1], alone_b, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(invariant, together[0], atol=3e-2, rtol=3e-2)
