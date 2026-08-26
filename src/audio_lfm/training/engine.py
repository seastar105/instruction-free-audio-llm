from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch
import yaml

from audio_lfm.config import AppConfig
from audio_lfm.data.resume_state import DataResumeState
from audio_lfm.data.types import PackedHostItem
from audio_lfm.training.checkpoint import TrainerState, save_checkpoint
from audio_lfm.training.metrics import MetricsLogger, UpdateMetrics


class TrainingEngine:
    def __init__(
        self,
        *,
        config: AppConfig,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        run_manifest: dict[str, Any],
        trainer_state: TrainerState | None = None,
        data_state: DataResumeState | None = None,
        evaluation_callback: Callable[[int], float] | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.run_manifest = run_manifest
        self.trainer_state = trainer_state or TrainerState()
        self.data_state = data_state or DataResumeState()
        self.evaluation_callback = evaluation_callback
        self._best_updated_at: int | None = None
        self.logger = MetricsLogger(config.run.output_dir)
        self._oversized_example_count = 0
        self._decode_failure_count = 0
        self.device = next(model.projector.parameters()).device
        self.model.assert_only_projector_trainable()
        self._pending_loss_sum = 0.0
        self._pending_supervised = 0
        self._pending_input = 0
        self._pending_microbatches = 0
        self._pending_audio_ids: list[str] = []
        self._pending_examples = 0
        self._pending_audio_seconds = 0.0
        self._pending_started = 0.0
        self._pending_data_wait = 0.0
        self._pending_phase_events: list[tuple[torch.cuda.Event, ...]] = []
        output_dir = Path(config.run.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        (output_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(config.redacted_dict(), sort_keys=True), encoding="utf-8"
        )

    def train(
        self,
        epoch_stream: Callable[[int, frozenset[str]], Iterable[PackedHostItem]],
    ) -> TrainerState:
        torch.set_float32_matmul_precision("high")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.use_deterministic_algorithms(self.config.run.deterministic_algorithms)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        while self.trainer_state.global_update < self.config.optimization.max_updates:
            stream = epoch_stream(
                self.data_state.epoch,
                frozenset(self.data_state.committed_audio_ids),
            )
            emitted = False
            batch_iterator = iter(stream)
            try:
                while True:
                    wait_started = time.perf_counter()
                    try:
                        item = next(batch_iterator)
                    except StopIteration:
                        break
                    data_wait_seconds = time.perf_counter() - wait_started
                    emitted = True
                    self._consume_batch(item, data_wait_seconds=data_wait_seconds)
                    if (
                        self.trainer_state.global_update
                        >= self.config.optimization.max_updates
                    ):
                        break
            finally:
                close = getattr(batch_iterator, "close", None)
                if callable(close):
                    close()
            if self._pending_microbatches:
                self._optimizer_step()
            if self.trainer_state.global_update >= self.config.optimization.max_updates:
                break
            if not emitted:
                if self.data_state.committed_audio_ids:
                    self.data_state.advance_epoch()
                    self.trainer_state.current_epoch = self.data_state.epoch
                    continue
                raise RuntimeError("Training stream yielded no usable examples")
            self.data_state.advance_epoch()
            self.trainer_state.current_epoch = self.data_state.epoch
        self.logger.close()
        return self.trainer_state

    def _consume_batch(self, item: PackedHostItem, *, data_wait_seconds: float) -> None:
        host_batch = item.batch
        self._oversized_example_count += item.oversized_examples_skipped
        self._decode_failure_count += item.decode_failures_skipped
        if not self._pending_microbatches:
            self._pending_started = time.perf_counter()
        try:
            phase_events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(5))
            phase_events[0].record()
            cuda_batch = host_batch.to(self.device)
            phase_events[1].record()
            encoded_audio = self.model.encode_audio_blocks(cuda_batch)
            phase_events[2].record()
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                batch = self.model.prepare_vectorized_batch(cuda_batch, encoded_audio)
                output = self.model.forward_packed(batch)
            phase_events[3].record()
            output.loss_sum.backward()
            phase_events[4].record()
        except torch.cuda.OutOfMemoryError:
            self.optimizer.zero_grad(set_to_none=True)
            diagnostic = {
                "audio_ids": host_batch.layout.audio_ids,
                "logical_lengths": host_batch.layout.logical_lengths,
                "audio_seconds": host_batch.audio_seconds,
                "recommendation": (
                    "Lower packing.max_lfm_tokens or "
                    "frontend.encoder_microbatch_max_padded_samples"
                ),
            }
            output_path = Path(self.config.run.output_dir) / "oom-diagnostic.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
            raise
        self._pending_loss_sum += float(output.loss_sum.detach().item())
        self._pending_supervised += output.supervised_tokens
        self._pending_input += output.input_tokens
        self._pending_microbatches += 1
        self._pending_data_wait += data_wait_seconds
        self._pending_phase_events.append(phase_events)
        self._pending_audio_ids.extend(batch.audio_ids)
        self._pending_examples += len(batch.audio_ids)
        self._pending_audio_seconds += host_batch.audio_seconds
        threshold = (
            self._pending_input
            >= self.config.optimization.target_input_tokens_per_update
            or self._pending_microbatches
            >= self.config.optimization.max_microbatches_per_update
        )
        if threshold:
            self._optimizer_step()

    def _optimizer_step(self) -> None:
        if self._pending_supervised <= 0:
            raise RuntimeError("Cannot step without supervised tokens")
        optimizer_start = torch.cuda.Event(enable_timing=True)
        optimizer_end = torch.cuda.Event(enable_timing=True)
        optimizer_start.record()
        for parameter in self.model.projector.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(self._pending_supervised)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.projector.parameters(), self.config.optimization.max_grad_norm
        )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        optimizer_end.record()
        optimizer_end.synchronize()
        for name, parameter in self.model.named_parameters():
            if not name.startswith("projector.") and parameter.grad is not None:
                raise RuntimeError(f"Frozen parameter acquired a gradient: {name}")
        self.data_state.commit(self._pending_audio_ids)
        self.trainer_state.global_update += 1
        self.trainer_state.input_tokens_processed += self._pending_input
        self.trainer_state.supervised_tokens_processed += self._pending_supervised
        self.trainer_state.audio_seconds_processed += self._pending_audio_seconds
        elapsed = time.perf_counter() - self._pending_started
        h2d_seconds = sum(
            started.elapsed_time(transferred) / 1000
            for started, transferred, _, _, _ in self._pending_phase_events
        )
        whisper_seconds = sum(
            transferred.elapsed_time(encoded) / 1000
            for _, transferred, encoded, _, _ in self._pending_phase_events
        )
        projector_lfm_forward_seconds = sum(
            encoded.elapsed_time(forwarded) / 1000
            for _, _, encoded, forwarded, _ in self._pending_phase_events
        )
        backward_seconds = sum(
            forwarded.elapsed_time(backward) / 1000
            for _, _, _, forwarded, backward in self._pending_phase_events
        )
        metrics = UpdateMetrics(
            update=self.trainer_state.global_update,
            epoch=self.data_state.epoch,
            nll=self._pending_loss_sum / self._pending_supervised,
            input_tokens=self._pending_input,
            supervised_tokens=self._pending_supervised,
            logical_examples=self._pending_examples,
            packs=self._pending_microbatches,
            pack_utilization=(
                self._pending_input
                / (self.config.packing.max_lfm_tokens * self._pending_microbatches)
            ),
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            gradient_norm=float(gradient_norm),
            elapsed_seconds=elapsed,
            input_tokens_per_second=self._pending_input / elapsed,
            oversized_examples_skipped=self._oversized_example_count,
            decode_failures_skipped=self._decode_failure_count,
            data_wait_seconds=self._pending_data_wait,
            h2d_seconds=h2d_seconds,
            whisper_seconds=whisper_seconds,
            projector_lfm_forward_seconds=projector_lfm_forward_seconds,
            backward_seconds=backward_seconds,
            optimizer_seconds=optimizer_start.elapsed_time(optimizer_end) / 1000,
            end_to_end_input_tokens_per_second=(
                self._pending_input / (elapsed + self._pending_data_wait)
            ),
        )
        self.logger.log(metrics)
        self._reset_pending()
        if (
            self.evaluation_callback is not None
            and self.trainer_state.global_update % self.config.run.eval_every_updates
            == 0
        ):
            validation_metric = self.evaluation_callback(
                self.trainer_state.global_update
            )
            best = self.trainer_state.best_validation_metric
            if best is None or validation_metric < best:
                self.trainer_state.best_validation_metric = validation_metric
                self._best_updated_at = self.trainer_state.global_update
            self.logger.log_validation(
                update=self.trainer_state.global_update,
                nll=validation_metric,
                best_nll=self.trainer_state.best_validation_metric,
            )
            self.model.train()
        if (
            self.trainer_state.global_update % self.config.run.checkpoint_every_updates
            == 0
        ):
            checkpoint = (
                Path(self.config.run.output_dir)
                / f"checkpoint-{self.trainer_state.global_update:08d}"
            )
            save_checkpoint(
                checkpoint,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                trainer_state=self.trainer_state,
                data_state=self.data_state,
                resolved_config=self.config.redacted_dict(),
                run_manifest=self.run_manifest,
            )
            checkpoints = sorted(
                path
                for path in Path(self.config.run.output_dir).glob("checkpoint-[0-9]*")
                if path.is_dir()
            )
            excess = checkpoints[: -self.config.run.keep_last_checkpoints]
            best_link = Path(self.config.run.output_dir) / "checkpoint-best"
            best_target = best_link.readlink().name if best_link.is_symlink() else None
            for old_checkpoint in excess:
                if old_checkpoint.name != best_target:
                    shutil.rmtree(old_checkpoint)
            if self._best_updated_at == self.trainer_state.global_update:
                temporary_link = best_link.with_name(".checkpoint-best.tmp")
                temporary_link.unlink(missing_ok=True)
                temporary_link.symlink_to(checkpoint.name, target_is_directory=True)
                temporary_link.replace(best_link)

    def _reset_pending(self) -> None:
        self._pending_loss_sum = 0.0
        self._pending_supervised = 0
        self._pending_input = 0
        self._pending_microbatches = 0
        self._pending_audio_ids = []
        self._pending_examples = 0
        self._pending_audio_seconds = 0.0
        self._pending_data_wait = 0.0
        self._pending_phase_events = []
