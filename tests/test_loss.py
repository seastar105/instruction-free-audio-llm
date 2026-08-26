from __future__ import annotations

import torch
import torch.nn.functional as F

from audio_lfm.training.loss import selective_causal_lm_loss


def test_selective_logits_match_full_reference() -> None:
    torch.manual_seed(4)
    hidden = torch.randn(1, 7, 5)
    head = torch.nn.Linear(5, 11, bias=False)
    labels = torch.tensor([[-100, -100, 2, 3, -100, 5, 6]])
    supervised_token_indices = torch.nonzero(
        labels[:, 1:].reshape(-1).ne(-100), as_tuple=False
    ).flatten()
    selected, count = selective_causal_lm_loss(
        hidden_states=hidden,
        labels=labels,
        supervised_token_indices=supervised_token_indices,
        lm_head=head,
    )
    logits = head(hidden[:, :-1])
    reference = F.cross_entropy(
        logits.reshape(-1, 11),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    assert count == 4
    torch.testing.assert_close(selected, reference)
