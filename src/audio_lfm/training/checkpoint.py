from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import load_file, save_file

from audio_lfm.data.resume_state import DataResumeState
from audio_lfm.utils.rng import capture_rng_state, restore_rng_state

CHECKPOINT_FORMAT_VERSION = 1
PACKING_SEMANTICS_VERSION = 1


@dataclass
class TrainerState:
    global_update: int = 0
    current_epoch: int = 0
    input_tokens_processed: int = 0
    supervised_tokens_processed: int = 0
    audio_seconds_processed: float = 0.0
    best_validation_metric: float | None = None
    checkpoint_format_version: int = CHECKPOINT_FORMAT_VERSION


def _projector_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
        if name.startswith("projector.")
    }
    if not state:
        raise RuntimeError("No projector tensors found for checkpoint")
    return state


def save_checkpoint(
    destination: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    trainer_state: TrainerState,
    data_state: DataResumeState,
    resolved_config: dict[str, Any],
    run_manifest: dict[str, Any],
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        save_file(_projector_state(model), temporary / "projector.safetensors")
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
        torch.save(capture_rng_state(), temporary / "rng_state.pt")
        (temporary / "trainer_state.json").write_text(
            json.dumps(asdict(trainer_state), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (temporary / "data_state.json").write_text(
            json.dumps(
                {
                    "epoch": data_state.epoch,
                    "packing_semantics_version": PACKING_SEMANTICS_VERSION,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        data_state.save_ids(temporary / "committed_audio_ids.txt.zst")
        (temporary / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved_config, sort_keys=True), encoding="utf-8"
        )
        (temporary / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        for path in temporary.iterdir():
            if path.is_file():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if destination.exists():
            raise FileExistsError(f"Checkpoint already exists: {destination}")
        os.replace(temporary, destination)
        return destination
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validate_manifest(
    saved: dict[str, Any], current: dict[str, Any], *, allow_nonreproducible: bool
) -> None:
    keys = (
        "llm_revision",
        "frontend_revision",
        "chat_template_sha256",
        "prompt_sha256",
        "catalog_fingerprint",
        "projector_architecture_sha256",
        "data_semantics_sha256",
        "packing_semantics_sha256",
    )
    mismatches = {
        key: (saved.get(key), current.get(key))
        for key in keys
        if saved.get(key) != current.get(key)
    }
    if mismatches and not allow_nonreproducible:
        raise RuntimeError(f"Reproducibility manifest mismatch: {mismatches}")


def load_checkpoint(
    source: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    current_manifest: dict[str, Any],
    allow_nonreproducible: bool = False,
) -> tuple[TrainerState, DataResumeState]:
    source = Path(source)
    saved_manifest = json.loads((source / "run_manifest.json").read_text())
    _validate_manifest(
        saved_manifest, current_manifest, allow_nonreproducible=allow_nonreproducible
    )
    incompatible = model.load_state_dict(
        load_file(source / "projector.safetensors"), strict=False
    )
    unexpected = [
        key for key in incompatible.unexpected_keys if key.startswith("projector.")
    ]
    missing = [key for key in incompatible.missing_keys if key.startswith("projector.")]
    if unexpected or missing:
        raise RuntimeError(
            f"Incomplete projector restore; missing={missing}, unexpected={unexpected}"
        )
    optimizer.load_state_dict(torch.load(source / "optimizer.pt", weights_only=False))
    scheduler.load_state_dict(torch.load(source / "scheduler.pt", weights_only=False))
    restore_rng_state(torch.load(source / "rng_state.pt", weights_only=False))
    trainer_state = TrainerState(
        **json.loads((source / "trainer_state.json").read_text())
    )
    data_payload = json.loads((source / "data_state.json").read_text())
    if data_payload["packing_semantics_version"] != PACKING_SEMANTICS_VERSION:
        raise RuntimeError("Packing semantics version changed")
    data_state = DataResumeState(epoch=int(data_payload["epoch"]))
    data_state.load_ids(source / "committed_audio_ids.txt.zst")
    trainable = [
        name for name, value in model.named_parameters() if value.requires_grad
    ]
    if any(not name.startswith("projector.") for name in trainable):
        raise RuntimeError(
            "Checkpoint restore made a non-projector parameter trainable"
        )
    return trainer_state, data_state
