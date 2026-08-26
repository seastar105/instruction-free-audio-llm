from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from audio_lfm.data.types import HostAudioBatch, PackedBatch, PreparedExample
from audio_lfm.model.packed_batch import (
    build_host_audio_batch,
    build_packed_batch,
    build_vectorized_packed_batch,
)
from audio_lfm.training.loss import LossOutput, selective_causal_lm_loss


class AudioLfmModel(nn.Module):
    def __init__(
        self,
        *,
        frontend: nn.Module,
        projector: nn.Module,
        llm: nn.Module,
        tokenizer: Any,
        prompt_compiler: Any,
        max_lfm_tokens: int,
    ) -> None:
        super().__init__()
        self.frontend = frontend
        self.projector = projector
        self.llm = llm
        self.tokenizer = tokenizer
        self.prompt_compiler = prompt_compiler
        self.max_lfm_tokens = max_lfm_tokens
        self._project_frames: Callable[[torch.Tensor], torch.Tensor] = (
            self.projector.project_frames
        )
        self._backbone_forward: Callable[..., torch.Tensor] = (
            self._eager_backbone_forward
        )
        self.frontend.eval().requires_grad_(False)
        self.llm.requires_grad_(False)
        self.assert_only_projector_trainable()

    def enable_torch_compile(
        self,
        *,
        backend: str,
        mode: str,
        dynamic: bool,
        compile_whisper_encoder: bool,
        compile_projector: bool,
        compile_lfm_backbone: bool,
    ) -> dict[str, bool | str]:
        if compile_whisper_encoder:
            enable_frontend = getattr(self.frontend, "enable_torch_compile", None)
            if enable_frontend is None:
                raise ValueError("The selected frontend cannot be compiled")
            enable_frontend(backend=backend, mode=mode, dynamic=dynamic)
        if compile_projector:
            self._project_frames = torch.compile(
                self.projector.project_frames,
                backend=backend,
                mode=mode,
                dynamic=dynamic,
                fullgraph=False,
            )
        if compile_lfm_backbone:
            self._backbone_forward = torch.compile(
                self._eager_backbone_forward,
                backend=backend,
                mode=mode,
                dynamic=dynamic,
                fullgraph=False,
            )
        return {
            "enabled": True,
            "backend": backend,
            "mode": mode,
            "dynamic": dynamic,
            "whisper_encoder": compile_whisper_encoder,
            "projector": compile_projector,
            "lfm_backbone": compile_lfm_backbone,
        }

    def assert_only_projector_trainable(self) -> None:
        trainable = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]
        invalid = [name for name in trainable if not name.startswith("projector.")]
        if invalid:
            raise RuntimeError(f"Non-projector parameters are trainable: {invalid}")
        if not trainable:
            raise RuntimeError("Projector has no trainable parameters")

    def prepare_batch(
        self, examples: list[PreparedExample], *, device: torch.device
    ) -> PackedBatch:
        input_features = self.transfer_audio_features(examples, device=device)
        encoded_audio = self.encode_audio(examples, input_features)
        return self.prepare_batch_from_encoded(examples, encoded_audio, device=device)

    def transfer_audio_features(
        self, examples: list[PreparedExample], *, device: torch.device
    ) -> list[torch.Tensor]:
        return [example.raw.input_features.detach().to(device) for example in examples]

    def prepare_host_audio_batch(
        self, examples: list[PreparedExample]
    ) -> HostAudioBatch:
        return build_host_audio_batch(
            examples=examples,
            vocabulary_size=int(self.llm.config.vocab_size),
            max_lfm_tokens=self.max_lfm_tokens,
        )

    def encode_audio_blocks(self, batch: HostAudioBatch) -> torch.Tensor:
        self.frontend.eval()
        return self.frontend.encode_blocks(batch.input_features)

    def prepare_vectorized_batch(
        self, batch: HostAudioBatch, encoded_blocks: torch.Tensor
    ) -> PackedBatch:
        projected = self.projector.project_blocks(
            encoded_blocks,
            batch.encoder_frame_mask,
            batch.projected_frame_mask,
            batch.projected_frame_indices,
        )
        return build_vectorized_packed_batch(
            layout=batch.layout,
            projected_audio=projected,
            embed_tokens=self.llm.get_input_embeddings(),
            audio_start=self.projector.audio_start,
            audio_end=self.projector.audio_end,
        )

    def encode_audio(
        self,
        examples: list[PreparedExample],
        input_features: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        self.frontend.eval()
        return self.frontend.encode_precomputed(
            input_features,
            [example.raw.effective_encoder_lengths for example in examples],
        )

    def prepare_batch_from_encoded(
        self,
        examples: list[PreparedExample],
        encoded_audio: list[torch.Tensor],
        *,
        device: torch.device,
    ) -> PackedBatch:
        projected = [self._project_frames(value) for value in encoded_audio]
        embeddings = self.llm.get_input_embeddings()
        return build_packed_batch(
            examples=examples,
            projected_audio=projected,
            embed_tokens=embeddings,
            audio_start=self.projector.audio_start,
            audio_end=self.projector.audio_end,
            device=device,
            vocabulary_size=int(self.llm.config.vocab_size),
            max_lfm_tokens=self.max_lfm_tokens,
        )

    def forward_packed(self, batch: PackedBatch) -> LossOutput:
        hidden_states = self._backbone_forward(
            inputs_embeds=batch.inputs_embeds,
            position_ids=batch.position_ids,
            seq_idx=batch.seq_idx,
            cu_seq_lens_q=batch.cu_seq_lens_q,
            cu_seq_lens_k=batch.cu_seq_lens_k,
            max_length_q=batch.max_length_q,
            max_length_k=batch.max_length_k,
        )
        loss_sum, supervised = selective_causal_lm_loss(
            hidden_states=hidden_states,
            labels=batch.labels,
            supervised_token_indices=batch.supervised_token_indices,
            lm_head=self.llm.lm_head,
        )
        return LossOutput(
            loss_sum=loss_sum,
            supervised_tokens=supervised,
            input_tokens=batch.input_token_count,
        )

    def _eager_backbone_forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        seq_idx: torch.Tensor,
        cu_seq_lens_q: torch.Tensor,
        cu_seq_lens_k: torch.Tensor,
        max_length_q: int,
        max_length_k: int,
    ) -> torch.Tensor:
        outputs = self.llm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            position_ids=position_ids,
            use_cache=False,
            seq_idx=seq_idx,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            max_length_q=max_length_q,
            max_length_k=max_length_k,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def forward_unpacked(self, batch: PackedBatch) -> LossOutput:
        """One-example correctness oracle through the same frozen backbone."""
        if len(batch.logical_lengths) != 1:
            raise ValueError("Unpacked reference forward accepts one logical sequence")
        return self.forward_packed(batch)

    def train(self, mode: bool = True) -> AudioLfmModel:
        super().train(mode)
        self.frontend.eval()
        self.projector.train(mode)
        # LFM must be in training mode for gradient checkpointing; dropout is gated.
        self.llm.train(mode)
        return self


def assert_frozen_llm_dropout(llm: nn.Module, *, allow: bool) -> None:
    nonzero: list[str] = []
    for name, module in llm.named_modules():
        if isinstance(module, nn.Dropout) and module.p:
            nonzero.append(f"{name}={module.p}")
    for name, value in vars(llm.config).items():
        if "dropout" in name.lower() and isinstance(value, (float, int)) and value:
            nonzero.append(f"config.{name}={value}")
    if nonzero and not allow:
        raise RuntimeError("Frozen LLM has nonzero dropout: " + ", ".join(nonzero))
