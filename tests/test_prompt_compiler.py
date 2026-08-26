from __future__ import annotations

from types import SimpleNamespace

from audio_lfm.data.types import TargetRecord
from audio_lfm.model.prompt_compiler import PromptCompiler
from audio_lfm.prompts import CAPTION_EXPANSION_SYSTEM_PROMPT


class CharacterTokenizer:
    chat_template = "character-template-v1"
    eos_token_id = ord("§")

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in conversation
        )
        if add_generation_prompt:
            return rendered + "<assistant>"
        return rendered + "§"

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool = False,
    ) -> SimpleNamespace:
        assert not add_special_tokens
        result = {"input_ids": [ord(character) for character in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return SimpleNamespace(**result)


def test_prompt_target_begins_after_generation_prefix(tmp_path) -> None:
    sentinel = "<<AUDIO>>"
    prompt = tmp_path / "prompt.txt"
    prompt.write_text(f"Listen.\n{sentinel}\n")
    compiler = PromptCompiler(
        CharacterTokenizer(),
        prompt_file=prompt,
        audio_sentinel=sentinel,
        supervise_assistant_termination=True,
    )
    target = TargetRecord(
        audio_id="a",
        target_id="t",
        target_type="style_caption",
        text="calm",
        split="train_base",
        source="official",
        review_status="accepted",
    )
    prepared = compiler.compile(target)
    suffix = "".join(chr(value) for value in prepared.target_suffix_ids)
    after = "".join(chr(value) for value in prepared.after_audio_prompt_ids)
    assert after.endswith("<assistant>")
    assert suffix == "calm</assistant>§"
    assert sentinel not in after


def test_caption_expansion_training_uses_audio_only_without_system() -> None:
    sentinel = "<<AUDIO>>"
    compiler = PromptCompiler(
        CharacterTokenizer(),
        prompt_file=None,
        audio_sentinel=sentinel,
        mode="caption_expansion_alignment",
        system_message=None,
    )
    rendered = compiler.render("A speaker greets the listener.")
    assert rendered.prompt_only == f"<user>{sentinel}</user><assistant>"
    assert "<system>" not in rendered.prompt_only
    assert "<system>" not in rendered.full
    assert CAPTION_EXPANSION_SYSTEM_PROMPT not in rendered.full
    assert rendered.full.count(sentinel) == 1
