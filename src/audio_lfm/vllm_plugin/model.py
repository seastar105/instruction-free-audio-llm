from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from audio_lfm.model.frontends.whisper_encoder import VariableLengthWhisperEncoder
from audio_lfm.model.projector import FrameStackMLPProjector
from audio_lfm.vllm_plugin.weight_mapping import map_weight_names


def _vllm_symbols() -> dict[str, Any]:
    from transformers import WhisperConfig
    from vllm.model_executor.model_loader import DefaultModelLoader
    from vllm.model_executor.models.interfaces import (
        IsHybrid,
        SupportsMultiModal,
        SupportsPP,
    )
    from vllm.model_executor.models.lfm2 import Lfm2ForCausalLM
    from vllm.model_executor.models.module_mapping import MultiModelKeys
    from vllm.model_executor.models.utils import (
        AutoWeightsLoader,
        init_vllm_registered_model,
        maybe_prefix,
    )
    from vllm.multimodal import MULTIMODAL_REGISTRY

    return {
        "WhisperConfig": WhisperConfig,
        "DefaultModelLoader": DefaultModelLoader,
        "AutoWeightsLoader": AutoWeightsLoader,
        "IsHybrid": IsHybrid,
        "SupportsMultiModal": SupportsMultiModal,
        "SupportsPP": SupportsPP,
        "Lfm2ForCausalLM": Lfm2ForCausalLM,
        "MultiModelKeys": MultiModelKeys,
        "init_vllm_registered_model": init_vllm_registered_model,
        "maybe_prefix": maybe_prefix,
        "MULTIMODAL_REGISTRY": MULTIMODAL_REGISTRY,
    }


# vLLM imports this heavy module only after lazy registry resolution.
symbols = _vllm_symbols()
NativeLfm2ForCausalLM = symbols["Lfm2ForCausalLM"]
SupportsMultiModal = symbols["SupportsMultiModal"]
SupportsPP = symbols["SupportsPP"]
IsHybrid = symbols["IsHybrid"]


class AudioLfm2ForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, IsHybrid
):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "audio":
            return "<|audio|>"
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1:
            raise NotImplementedError(
                "AudioLFM2 v1 supports tensor_parallel_size=1 only"
            )
        if parallel.pipeline_parallel_size != 1:
            raise NotImplementedError(
                "AudioLFM2 v1 supports pipeline_parallel_size=1 only"
            )
        self.config = config
        self.vllm_config = vllm_config
        self.configure_mm_token_handling(config.vocab_size, [config.audio_token_index])
        Source = symbols["DefaultModelLoader"].Source
        self.secondary_weights = [
            Source(
                model_or_path=config.text_model_id,
                revision=config.text_model_revision,
                prefix="language_model.",
            ),
            Source(
                model_or_path=config.audio_model_id,
                revision=config.audio_model_revision,
                prefix="audio_tower.",
            ),
        ]
        whisper_config = symbols["WhisperConfig"].from_dict(config.audio_config)
        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_tower = VariableLengthWhisperEncoder(whisper_config)
            self.multi_modal_projector = FrameStackMLPProjector.from_config(
                config.projector_config
            )
        with self._mark_language_model(vllm_config):
            self.language_model = symbols["init_vllm_registered_model"](
                vllm_config=vllm_config,
                hf_config=config,
                prefix=symbols["maybe_prefix"](prefix, "language_model"),
                architectures=["Lfm2ForCausalLM"],
            )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: Any) -> Any:
        return NativeLfm2ForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: Any) -> Any:
        return NativeLfm2ForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls) -> Any:
        return NativeLfm2ForCausalLM.get_mamba_state_copy_func()

    def get_mm_mapping(self) -> Any:
        return symbols["MultiModelKeys"].from_string_field(
            language_model="language_model.",
            connector="multi_modal_projector.",
            tower_model="audio_tower.",
        )

    def embed_multimodal(self, **kwargs: object) -> tuple[torch.Tensor, ...]:
        if "audio_embeds" in kwargs:
            embeds = kwargs["audio_embeds"]
            if isinstance(embeds, torch.Tensor) and embeds.ndim == 3:
                values = list(embeds.unbind(0))
            else:
                values = embeds if isinstance(embeds, (list, tuple)) else [embeds]
            final_embeds = tuple(torch.as_tensor(value) for value in values)
            for value in final_embeds:
                if value.ndim != 2 or value.shape[1] != self.config.hidden_size:
                    raise ValueError("Final audio embeddings have invalid shape")
                if not torch.isfinite(value).all():
                    raise ValueError("Final audio embeddings are non-finite")
            return final_embeds
        features = kwargs.get("audio_features")
        feature_lengths = kwargs.get("audio_feature_length")
        chunk_lengths = kwargs.get("audio_chunk_length")
        token_lengths = kwargs.get("audio_token_length")
        if features is None or feature_lengths is None or token_lengths is None:
            raise ValueError("Incomplete audio feature arguments")
        feature_values = self._field_items(features, item_ndim=2)
        length_values = torch.as_tensor(feature_lengths).flatten().tolist()
        token_values = torch.as_tensor(token_lengths).flatten().tolist()
        if len(feature_values) != len(length_values) or len(feature_values) != len(
            token_values
        ):
            raise ValueError("Audio feature fields contain different item counts")
        if chunk_lengths is None:
            chunk_length_values = [
                torch.tensor([length], dtype=torch.long) for length in length_values
            ]
        else:
            chunk_length_values = self._field_items(chunk_lengths, item_ndim=1)
        if len(chunk_length_values) != len(feature_values):
            raise ValueError("Audio chunk lengths contain a different item count")

        chunks: list[tuple[int, torch.Tensor, int]] = []
        for item_index, (value, total_length, item_chunk_lengths) in enumerate(
            zip(
                feature_values,
                length_values,
                chunk_length_values,
                strict=True,
            )
        ):
            feature = torch.as_tensor(value)
            lengths = torch.as_tensor(item_chunk_lengths).flatten().tolist()
            if (
                feature.ndim != 2
                or feature.shape[0] != 80
                or any(int(length) <= 0 for length in lengths)
                or sum(int(length) for length in lengths) != int(total_length)
                or int(total_length) != feature.shape[-1]
            ):
                raise ValueError("Invalid packed chunked audio features")
            offset = 0
            for length in lengths:
                stop = offset + int(length)
                chunks.append((item_index, feature[..., offset:stop], int(length)))
                offset = stop

        encoded_by_item: list[list[torch.Tensor]] = [
            [] for _ in range(len(feature_values))
        ]
        configured_microbatch = getattr(self.config, "audio_encoder_microbatch_size", 4)
        microbatch = int(
            os.environ.get(
                "AUDIO_LFM_AUDIO_ENCODER_MICROBATCH_SIZE",
                str(configured_microbatch),
            )
        )
        if microbatch <= 0:
            raise ValueError("Audio encoder microbatch size must be positive")
        for start in range(0, len(chunks), microbatch):
            selected = chunks[start : start + microbatch]
            batch_values = [value for _, value, _ in selected]
            batch_lengths = torch.tensor(
                [length for _, _, length in selected],
                device=batch_values[0].device,
            )
            max_frames = max(value.shape[-1] for value in batch_values)
            padded = torch.stack(
                [
                    torch.nn.functional.pad(value, (0, max_frames - value.shape[-1]))
                    for value in batch_values
                ]
            )
            padded = padded.to(dtype=self.audio_tower.conv1.weight.dtype)
            encoded = self.audio_tower(padded, batch_lengths)
            for (item_index, _, _), hidden in zip(selected, encoded, strict=True):
                encoded_by_item[item_index].append(hidden)

        embed_results: list[torch.Tensor] = []
        for values, expected_length in zip(encoded_by_item, token_values, strict=True):
            if not values:
                raise RuntimeError("Audio item produced no encoded chunks")
            projected = self.multi_modal_projector.project_frames(torch.cat(values))
            output = torch.cat(
                [
                    self.multi_modal_projector.audio_start[None],
                    projected,
                    self.multi_modal_projector.audio_end[None],
                ]
            )
            if output.shape[0] != int(expected_length):
                raise RuntimeError("Processor/model audio-token length mismatch")
            embed_results.append(output)
        return tuple(embed_results)

    @staticmethod
    def _field_items(value: object, *, item_ndim: int) -> list[Any]:
        if isinstance(value, torch.Tensor):
            if value.ndim == item_ndim:
                return [value]
            if value.ndim == item_ndim + 1:
                return list(value.unbind(0))
            raise ValueError("Audio field has an invalid tensor rank")
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        *,
        is_multimodal: Any = None,
    ) -> torch.Tensor:
        if multimodal_embeddings is None or is_multimodal is None:
            return super().embed_input_ids(input_ids)
        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> Any:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = symbols["AutoWeightsLoader"](
            self,
            skip_prefixes=[
                "audio_tower.model.decoder.",
                "audio_tower.proj_out.",
            ],
        )
        return loader.load_weights(map_weight_names(weights))


info_class, dummy_class, processor_class = __import__(
    "audio_lfm.vllm_plugin.processing", fromlist=["build_processing_classes"]
).build_processing_classes()
AudioLfm2ForConditionalGeneration = symbols["MULTIMODAL_REGISTRY"].register_processor(
    processor_class, info=info_class, dummy_inputs=dummy_class
)(AudioLfm2ForConditionalGeneration)
