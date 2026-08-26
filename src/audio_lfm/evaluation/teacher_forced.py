from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

import torch

from audio_lfm.data.types import PrecomputedAudioExample
from audio_lfm.model.preparation import prepare_example


@dataclass(frozen=True)
class ValidationMetrics:
    target_weighted_mean_nll: float
    audio_weighted_mean_nll: float
    target_count: int
    token_count: int
    reference_count_distribution: dict[int, int]


def evaluate_all_references(
    model: Any, examples: Iterable[PrecomputedAudioExample], *, device: torch.device
) -> ValidationMetrics:
    model.eval()
    target_nlls: list[float] = []
    by_audio: defaultdict[str, list[float]] = defaultdict(list)
    token_count = 0
    distribution: defaultdict[int, int] = defaultdict(int)
    with torch.no_grad():
        for raw in examples:
            distribution[len(raw.style_captions)] += 1
            for target in raw.style_captions:
                target_raw = replace(raw, selected_target=target)
                prepared = prepare_example(
                    target_raw,
                    prompt_compiler=model.prompt_compiler,
                    projector=model.projector,
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    batch = model.prepare_batch([prepared], device=device)
                    output = model.forward_packed(batch)
                nll = float(output.loss_sum.item() / output.supervised_tokens)
                target_nlls.append(nll)
                by_audio[raw.audio_id].append(nll)
                token_count += output.supervised_tokens
    if not target_nlls:
        raise RuntimeError("Evaluation received no targets")
    audio_means = [sum(values) / len(values) for values in by_audio.values()]
    return ValidationMetrics(
        target_weighted_mean_nll=sum(target_nlls) / len(target_nlls),
        audio_weighted_mean_nll=sum(audio_means) / len(audio_means),
        target_count=len(target_nlls),
        token_count=token_count,
        reference_count_distribution=dict(distribution),
    )
