#!/usr/bin/env python3
"""Backfill and follow validation JSON files into a live TensorBoard log."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def _update_from_path(path: Path) -> int:
    return int(path.stem.removeprefix("validation-"))


def sync(run_dir: Path, writer: SummaryWriter, last_update: int) -> int:
    files = sorted(run_dir.glob("validation-[0-9]*.json"))
    values = [
        (
            _update_from_path(path),
            float(
                json.loads(path.read_text(encoding="utf-8"))["audio_weighted_mean_nll"]
            ),
        )
        for path in files
    ]
    best = math.inf
    for update, nll in values:
        best = min(best, nll)
        if update <= last_update:
            continue
        writer.add_scalar("validation/nll", nll, update)
        writer.add_scalar("validation/perplexity", math.exp(min(nll, 20.0)), update)
        writer.add_scalar("validation/best_nll", best, update)
        last_update = update
    writer.flush()
    return last_update


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    state_path = run_dir / ".validation_tensorboard_sync.json"
    last_update = -1
    if state_path.exists():
        last_update = int(
            json.loads(state_path.read_text(encoding="utf-8"))["last_update"]
        )

    with SummaryWriter(run_dir / "tensorboard") as writer:
        while True:
            last_update = sync(run_dir, writer, last_update)
            state_path.write_text(
                json.dumps({"last_update": last_update}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if args.once:
                break
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
