from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any, cast


def _generate(item: dict[str, Any]) -> dict[str, Any]:
    from api_judge import generate

    return cast(dict[str, Any], generate(item))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", required=True, type=Path)
    args = parser.parse_args()
    workers = int(os.environ.get("AUDIO_LFM_VOICEBENCH_JUDGE_WORKERS", "16"))
    if workers <= 0:
        raise ValueError("AUDIO_LFM_VOICEBENCH_JUDGE_WORKERS must be positive")

    with args.src_file.open(encoding="utf-8") as handle:
        data = [json.loads(line) for line in handle if line.strip()]
    with multiprocessing.Pool(workers) as pool:
        scores = list(pool.imap(_generate, data))

    target = args.src_file.with_name(f"result-{args.src_file.name}")
    with target.open("w", encoding="utf-8") as handle:
        for record in scores:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
