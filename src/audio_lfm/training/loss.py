from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossOutput:
    loss_sum: torch.Tensor
    supervised_tokens: int
    input_tokens: int

    @property
    def mean_loss(self) -> torch.Tensor:
        return self.loss_sum / self.supervised_tokens


def selective_causal_lm_loss(
    *,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    supervised_token_indices: torch.Tensor,
    lm_head: torch.nn.Module,
) -> tuple[torch.Tensor, int]:
    hidden = hidden_states[:, :-1, :].reshape(-1, hidden_states.shape[-1])
    shifted_labels = labels[:, 1:].reshape(-1)
    selected_hidden = hidden.index_select(0, supervised_token_indices)
    selected_labels = shifted_labels.index_select(0, supervised_token_indices)
    if selected_labels.numel() == 0:
        raise RuntimeError("Packed batch has no supervised tokens after shifting")
    logits = lm_head(selected_hidden)
    return (
        F.cross_entropy(logits.float(), selected_labels, reduction="sum"),
        int(selected_labels.numel()),
    )
