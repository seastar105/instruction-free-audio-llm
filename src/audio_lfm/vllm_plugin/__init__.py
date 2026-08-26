from __future__ import annotations

ARCHITECTURE = "AudioLfm2ForConditionalGeneration"


def register() -> None:
    """Register without importing Torch or initializing CUDA."""
    from vllm import ModelRegistry

    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            ARCHITECTURE,
            "audio_lfm.vllm_plugin.model:AudioLfm2ForConditionalGeneration",
        )


__all__ = ["ARCHITECTURE", "register"]
