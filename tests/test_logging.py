from __future__ import annotations

import json
from pathlib import Path

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from audio_lfm.training.metrics import MetricsLogger
from audio_lfm.utils.logging import REDACTED, append_jsonl


def test_jsonl_redacts_secrets_by_default_but_metrics_can_opt_out(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "records.jsonl"
    record = {"HF_TOKEN": "secret", "input_tokens": 123}
    append_jsonl(destination, record)
    append_jsonl(destination, record, redact_secrets=False)
    first, second = [json.loads(line) for line in destination.read_text().splitlines()]
    assert first == {"HF_TOKEN": REDACTED, "input_tokens": REDACTED}
    assert second == record


def test_validation_metrics_are_written_to_tensorboard(tmp_path: Path) -> None:
    logger = MetricsLogger(tmp_path)
    logger.log_validation(update=1000, nll=1.25, best_nll=1.2)
    logger.close()

    events = EventAccumulator(str(tmp_path / "tensorboard"))
    events.Reload()
    assert events.Scalars("validation/nll")[0].step == 1000
    assert events.Scalars("validation/nll")[0].value == 1.25
    assert events.Scalars("validation/best_nll")[0].value == pytest.approx(1.2)
    assert events.Scalars("validation/perplexity")[0].value == pytest.approx(3.4903429)
