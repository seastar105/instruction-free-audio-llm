from __future__ import annotations

from typing import Any

import torch


def generate_one(
    model: Any,
    raw: Any,
    *,
    device: torch.device,
    max_new_tokens: int,
) -> tuple[str, list[int]]:
    prepared = model.prompt_compiler.compile(raw.selected_target)
    with torch.no_grad():
        input_features = [raw.input_features.detach().to(device)]
        features = model.frontend.encode_precomputed(
            input_features, [raw.effective_encoder_lengths]
        )[0]
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        projected = model.projector([features.to(device)])[0]
    embedding = model.llm.get_input_embeddings()
    before = torch.tensor(prepared.before_audio_ids, device=device)
    after = torch.tensor(prepared.after_audio_prompt_ids, device=device)
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ),
    ):
        prompt_embeds = torch.cat(
            [
                embedding(before),
                model.projector.audio_start[None].to(projected.dtype),
                projected,
                model.projector.audio_end[None].to(projected.dtype),
                embedding(after),
            ]
        ).unsqueeze(0)
        output = model.llm(
            inputs_embeds=prompt_embeds, use_cache=True, return_dict=True
        )
        next_token = output.logits[:, -1].argmax(dim=-1)
        past = output.past_key_values
        generated: list[int] = []
        eos_ids = model.llm.generation_config.eos_token_id
        eos = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
        for _ in range(max_new_tokens):
            token = int(next_token.item())
            generated.append(token)
            if token in eos:
                break
            output = model.llm(
                input_ids=next_token[:, None],
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = output.past_key_values
            next_token = output.logits[:, -1].argmax(dim=-1)
    return model.tokenizer.decode(generated, skip_special_tokens=True), generated
