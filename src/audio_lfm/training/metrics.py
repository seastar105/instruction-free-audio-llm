from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from audio_lfm.utils.logging import append_jsonl


@dataclass
class UpdateMetrics:
    update: int
    epoch: int
    nll: float
    input_tokens: int
    supervised_tokens: int
    logical_examples: int
    packs: int
    pack_utilization: float
    learning_rate: float
    gradient_norm: float
    elapsed_seconds: float
    input_tokens_per_second: float
    oversized_examples_skipped: int
    decode_failures_skipped: int
    data_wait_seconds: float
    h2d_seconds: float
    whisper_seconds: float
    projector_lfm_forward_seconds: float
    backward_seconds: float
    optimizer_seconds: float
    end_to_end_input_tokens_per_second: float

    @property
    def perplexity(self) -> float:
        return math.exp(min(self.nll, 20.0))


class MetricsLogger:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.output_dir / "metrics.jsonl"
        self.tensorboard = SummaryWriter(self.output_dir / "tensorboard")

    def log(self, metrics: UpdateMetrics) -> None:
        record = {**asdict(metrics), "perplexity": metrics.perplexity}
        if torch.cuda.is_available():
            record.update(
                cuda_allocated=torch.cuda.memory_allocated(),
                cuda_reserved=torch.cuda.memory_reserved(),
                cuda_peak=torch.cuda.max_memory_allocated(),
            )
        # This schema is constructed entirely from numeric trainer metrics;
        # names such as input_tokens are measurements, not credentials.
        append_jsonl(self.jsonl, record, redact_secrets=False)
        for key, value in record.items():
            if isinstance(value, (float, int)):
                self.tensorboard.add_scalar(key, value, metrics.update)

    def log_validation(self, *, update: int, nll: float, best_nll: float) -> None:
        """Log sparse teacher-forced validation metrics at their train update."""
        self.tensorboard.add_scalar("validation/nll", nll, update)
        self.tensorboard.add_scalar(
            "validation/perplexity", math.exp(min(nll, 20.0)), update
        )
        self.tensorboard.add_scalar("validation/best_nll", best_nll, update)
        self.tensorboard.flush()

    def close(self) -> None:
        self.tensorboard.close()


def monotonic_time() -> float:
    return time.perf_counter()
