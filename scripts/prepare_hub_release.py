#!/usr/bin/env python3
"""Stage a publication-safe projector export for Hugging Face Hub upload."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

EXPORT_FILES = (
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "projector_manifest.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def portable(value: Any) -> Any:
    """Redact host-specific absolute paths while preserving useful filenames."""
    if isinstance(value, dict):
        return {str(key): portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return f"<local-path>/{Path(value).name}"
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_release(
    *,
    export_dir: Path,
    output_dir: Path,
    model_card: Path,
    evaluation_results: list[Path],
    trainer_state: Path,
    lfm_license: Path,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Release directory already exists: {output_dir}")
    for required in (*EXPORT_FILES, "export_manifest.json"):
        if not (export_dir / required).is_file():
            raise FileNotFoundError(export_dir / required)
    state = json.loads(trainer_state.read_text())
    public_state = {
        key: state[key]
        for key in (
            "audio_seconds_processed",
            "best_validation_metric",
            "checkpoint_format_version",
            "current_epoch",
            "global_update",
            "input_tokens_processed",
            "supervised_tokens_processed",
        )
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        for name in EXPORT_FILES:
            shutil.copy2(export_dir / name, temporary / name)
        manifest = portable(
            json.loads((export_dir / "export_manifest.json").read_text())
        )
        (temporary / "export_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        (temporary / "training_summary.json").write_text(
            json.dumps(public_state, indent=2, sort_keys=True) + "\n"
        )
        shutil.copy2(model_card, temporary / "README.md")
        shutil.copy2(lfm_license, temporary / "LICENSE")
        for result in evaluation_results:
            shutil.copy2(result, temporary / result.name)
        release_manifest = {
            "format_version": 1,
            "files": {
                path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
                for path in sorted(temporary.iterdir())
                if path.is_file()
            },
        }
        (temporary / "release_manifest.json").write_text(
            json.dumps(release_manifest, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-card", required=True, type=Path)
    parser.add_argument("--evaluation-results", required=True, nargs="+", type=Path)
    parser.add_argument("--trainer-state", required=True, type=Path)
    parser.add_argument("--lfm-license", required=True, type=Path)
    args = parser.parse_args()
    print(
        stage_release(
            export_dir=args.export_dir,
            output_dir=args.output_dir,
            model_card=args.model_card,
            evaluation_results=args.evaluation_results,
            trainer_state=args.trainer_state,
            lfm_license=args.lfm_license,
        )
    )


if __name__ == "__main__":
    main()
