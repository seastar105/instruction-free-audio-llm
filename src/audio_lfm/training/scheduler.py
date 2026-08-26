from __future__ import annotations

import math

import torch


def cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_updates: int,
    max_updates: int,
    min_learning_rate: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    base = max(group["lr"] for group in optimizer.param_groups)
    minimum_ratio = min_learning_rate / base

    def scale(update: int) -> float:
        if warmup_updates and update < warmup_updates:
            return (update + 1) / warmup_updates
        progress = (update - warmup_updates) / max(1, max_updates - warmup_updates)
        progress = min(max(progress, 0.0), 1.0)
        return float(
            minimum_ratio
            + (1 - minimum_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
