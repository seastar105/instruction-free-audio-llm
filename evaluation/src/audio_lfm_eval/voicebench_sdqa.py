from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def score_gpt_judgments(input_path: str | Path) -> dict[str, Any]:
    correct = 0
    records_by_id: dict[str, dict[str, Any]] = {}
    with Path(input_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = str(record.get("id") or record.get("key") or "")
            if not record_id:
                record_id = json.dumps(
                    [record.get("prompt"), record.get("reference")],
                    ensure_ascii=False,
                )
            if record_id == "[null, null]":
                raise ValueError(
                    "VoiceBench SD-QA judge record has no stable identity fields"
                )
            records_by_id[record_id] = record
    for record in records_by_id.values():
        judgments = [str(value).strip().casefold() for value in record.get("score", [])]
        yes = sum(value == "yes" for value in judgments)
        no = sum(value == "no" for value in judgments)
        correct += int(yes > no)
    total = len(records_by_id)
    if total == 0:
        raise ValueError("No VoiceBench SD-QA judge records were found")
    return {
        "gpt": correct / total * 100.0,
        "panda": None,
        "num_samples": total,
        "complete_official_sdqa": False,
        "omitted_official_metrics": ["panda"],
        "reason": (
            "qa_metrics.PEDANT downloads mutable pickle files and deserializes "
            "them with joblib; only the official GPT majority-vote metric was run"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely score VoiceBench SD-QA GPT judgments without PANDA pickles"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = score_gpt_judgments(args.input)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
