from __future__ import annotations

import importlib.metadata
from typing import Any

import torch
from huggingface_hub import HfApi

from audio_lfm.config import AppConfig
from audio_lfm.model.audio_lfm import AudioLfmModel, assert_frozen_llm_dropout
from audio_lfm.model.frontends.dmel import DmelFrontend
from audio_lfm.model.frontends.whisper import WhisperFrontend
from audio_lfm.model.projector import DmelProjector, FrameStackMLPProjector
from audio_lfm.model.prompt_compiler import PromptCompiler
from audio_lfm.utils.tensors import tensor_rms


def resolve_revision(model_id: str, revision: str) -> str:
    info = HfApi().model_info(model_id, revision=revision)
    if not info.sha:
        raise RuntimeError(f"Hub did not return an immutable revision for {model_id}")
    return info.sha


def load_audio_lfm(
    config: AppConfig,
    *,
    device: torch.device | None = None,
    resolved_llm_revision: str | None = None,
    resolved_frontend_revision: str | None = None,
) -> tuple[AudioLfmModel, dict[str, Any]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or torch.device("cuda")
    llm_revision = resolved_llm_revision or resolve_revision(
        config.llm.model_id, config.llm.revision
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.llm.model_id,
        revision=llm_revision,
        trust_remote_code=False,
    )
    llm = AutoModelForCausalLM.from_pretrained(
        config.llm.model_id,
        revision=llm_revision,
        dtype=torch.bfloat16,
        attn_implementation=config.llm.attention_implementation,
        trust_remote_code=False,
    )
    llm.to(device)
    llm.requires_grad_(False)
    llm.config.use_cache = False
    if not hasattr(llm, "model") or not hasattr(llm, "lm_head"):
        raise TypeError("Expected an LFM2 causal LM with model and lm_head")
    if "Lfm2" not in type(llm).__name__:
        raise TypeError(f"Expected an LFM2 causal LM, received {type(llm).__name__}")
    context_length = int(llm.config.max_position_embeddings)
    if context_length < config.packing.max_lfm_tokens:
        raise ValueError(
            f"LFM context {context_length} is below pack limit "
            f"{config.packing.max_lfm_tokens}"
        )
    implementation = getattr(llm.config, "_attn_implementation", None)
    if implementation != config.llm.attention_implementation:
        raise RuntimeError(
            f"LFM attention implementation is {implementation!r}, expected "
            f"{config.llm.attention_implementation!r}"
        )
    if (
        llm.config.tie_word_embeddings
        and llm.get_input_embeddings().weight.data_ptr()
        != llm.lm_head.weight.data_ptr()
    ):
        raise RuntimeError("LFM checkpoint declares tied embeddings but weights differ")
    assert_frozen_llm_dropout(llm, allow=config.llm.allow_frozen_llm_dropout)
    if config.llm.gradient_checkpointing:
        llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": config.llm.gradient_checkpointing_use_reentrant
            }
        )
    embedding_rms = float(tensor_rms(llm.get_input_embeddings().weight).item())
    hidden_size = int(llm.config.hidden_size)
    if config.frontend.kind == "whisper":
        frontend_model_id = config.frontend.model_id
        frontend_revision_name = config.frontend.revision
        if frontend_model_id is None or frontend_revision_name is None:
            raise ValueError("Whisper requires model_id and revision")
        frontend_revision = resolved_frontend_revision or resolve_revision(
            frontend_model_id, frontend_revision_name
        )
        frontend = WhisperFrontend(
            model_id=frontend_model_id,
            revision=frontend_revision,
            device=device,
            max_seconds=config.frontend.max_seconds,
            mode=config.frontend.mode,
            chunk_long_audio=config.frontend.chunk_long_audio,
            encoder_microbatch_max_padded_samples=(
                config.frontend.encoder_microbatch_max_padded_samples
            ),
        )
        projector: torch.nn.Module = FrameStackMLPProjector(
            input_dim=frontend.output_dim,
            output_dim=hidden_size,
            stack_factor=config.projector.stack_factor,
            hidden_dim=config.projector.hidden_dim,
            dropout=config.projector.dropout,
            use_input_layer_norm=config.projector.use_input_layer_norm,
            use_output_rms_norm=config.projector.use_output_rms_norm,
            target_embedding_rms=embedding_rms,
        )
    else:
        frontend = DmelFrontend(sample_rate=config.frontend.sample_rate)
        frontend_revision = "dmel-" + importlib.metadata.version("dmel")
        projector = DmelProjector(
            num_bins=frontend.num_bins,
            num_channels=frontend.num_channels,
            output_dim=hidden_size,
            bin_embedding_dim=config.projector.dmel_bin_embedding_dim,
            temporal_patch_size=config.projector.temporal_patch_size,
            hidden_dim=config.projector.hidden_dim,
            target_embedding_rms=embedding_rms,
        )
    projector = projector.float().to(device)
    compiler = PromptCompiler(
        tokenizer,
        prompt_file=config.prompt.prompt_file,
        audio_sentinel=config.prompt.audio_sentinel,
        mode=config.prompt.mode,
        system_message=config.prompt.system_message,
        supervise_assistant_termination=config.prompt.supervise_assistant_termination,
    )
    model = AudioLfmModel(
        frontend=frontend,
        projector=projector,
        llm=llm,
        tokenizer=tokenizer,
        prompt_compiler=compiler,
        max_lfm_tokens=config.packing.max_lfm_tokens,
    )
    metadata = {
        "llm_model_id": config.llm.model_id,
        "llm_revision": llm_revision,
        "frontend_model_id": (
            config.frontend.model_id if config.frontend.kind == "whisper" else "dmel"
        ),
        "frontend_revision": frontend_revision,
        "tokenizer_repository": config.llm.model_id,
        "tokenizer_revision": llm_revision,
        "chat_template_sha256": compiler.chat_template_sha256,
        "prompt_sha256": compiler.prompt_sha256,
        "projector_embedding_target_rms": embedding_rms,
        "rendered_prompt_example": compiler.render(
            "<target-redacted>"
        ).redacted_example,
    }
    return model, metadata
