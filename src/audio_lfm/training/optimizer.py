from __future__ import annotations

import torch


def trainable_projector_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    named = [
        (name, value) for name, value in model.named_parameters() if value.requires_grad
    ]
    invalid = [name for name, _ in named if not name.startswith("projector.")]
    if invalid:
        raise RuntimeError(f"Only projector parameters may be optimized: {invalid}")
    if not named:
        raise RuntimeError("No projector parameters are trainable")
    return [value for _, value in named]


def create_optimizer(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    fused: bool,
) -> torch.optim.AdamW:
    parameters = trainable_projector_parameters(model)

    def ordinary() -> torch.optim.AdamW:
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )

    if fused:
        try:
            return torch.optim.AdamW(
                parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps,
                fused=True,
            )
        except (RuntimeError, TypeError):
            return ordinary()
    return ordinary()
