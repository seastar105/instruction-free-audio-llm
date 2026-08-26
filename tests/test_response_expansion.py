from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from audio_lfm.overlays import response_expansion


class FakeTokenizer:
    chat_template = "fake-chat-template"

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        suffix = "<assistant>" if add_generation_prompt else ""
        return (
            "".join(
                f"<{message['role']}>{message['content']}</{message['role']}>"
                for message in conversation
            )
            + suffix
        )


class FakeLlm:
    def generate(
        self, prompts: list[str], *, sampling_params: object, use_tqdm: bool
    ) -> list[SimpleNamespace]:
        del sampling_params, use_tqdm
        reasons = ["length", "stop"]
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text=f"response {index}",
                        finish_reason=reasons[index],
                        stop_reason=None,
                        token_ids=[1, 2, 3],
                    )
                ]
            )
            for index, _ in enumerate(prompts)
        ]


def test_expansion_writes_boolean_truncation_and_resumes(
    tmp_path: Path, monkeypatch
) -> None:
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "audio_id": f"audio-{index}",
                    "target_id": f"target-{index}",
                    "target_type": "transcription",
                    "text": f"spoken words {index}",
                    "split": "train_base",
                }
                for index in range(2)
            ]
        ),
        catalog / "part-00000.parquet",
    )
    monkeypatch.setattr(
        response_expansion.importlib.metadata, "version", lambda _: "0.27.1"
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(SamplingParams=lambda **kwargs: kwargs),
    )
    output = tmp_path / "output"
    arguments = {
        "catalog_dir": catalog,
        "output_dir": output,
        "dataset_name": "ParaSpeechCaps-Base",
        "source_target_types": ("transcription",),
        "model_path": tmp_path,
        "model_id": "LiquidAI/LFM2.5-1.2B-Instruct",
        "model_revision": "a" * 40,
        "tokenizer": FakeTokenizer(),
        "llm": FakeLlm(),
    }
    first = response_expansion.expand_catalog_with_vllm(**arguments)
    assert first["recipe"]["generation"]["sampling_params"] == {
        "temperature": 0.1,
        "top_k": 50,
        "repetition_penalty": 1.05,
        "max_tokens": 1024,
    }
    assert first["recipe"]["generation"]["do_sample"] is True
    table = pq.read_table(output / "part-00000.parquet")
    assert table.schema.field("truncated").type == pa.bool_()
    assert table["truncated"].to_pylist() == [True, False]
    assert first["truncated_target_count"] == 1
    second = response_expansion.expand_catalog_with_vllm(**arguments)
    assert second["complete"] is True
    assert second["completed_target_count"] == 2
    assert second["truncated_target_count"] == 1
    assert len(list(output.glob("part-*.parquet"))) == 1


def test_paraspeech_sources_are_joined_per_style_caption_row() -> None:
    rows = [
        {
            "audio_id": "audio-1",
            "target_id": "style-1",
            "target_type": "style_caption",
            "text": "A calm female speaker.",
            "split": "train_base",
        },
        {
            "audio_id": "audio-1",
            "target_id": "style-2",
            "target_type": "style_caption",
            "text": "The voice sounds close and clear.",
            "split": "train_base",
        },
        {
            "audio_id": "audio-1",
            "target_id": "transcript-1",
            "target_type": "transcription",
            "text": "Hello there.",
            "split": "train_base",
        },
    ]
    combined = response_expansion._combine_paraspeech_sources(rows)
    assert len(combined) == 2
    assert {row["target_id"] for row in combined} == {"style-1", "style-2"}
    assert all(
        row["official_transcription_target_id"] == "transcript-1" for row in combined
    )
    assert {row["text"] for row in combined} == {
        "Style caption: A calm female speaker.\nTranscription: Hello there.",
        "Style caption: The voice sounds close and clear.\nTranscription: Hello there.",
    }
