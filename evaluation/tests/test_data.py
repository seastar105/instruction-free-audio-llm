import pyarrow as pa
import pyarrow.parquet as pq

from audio_lfm_eval.data import iter_rows, normalize_row


def test_normalize_text_question_and_choices() -> None:
    sample = normalize_row(
        {
            "id": "sample-1",
            "audio": {"bytes": b"audio", "path": None},
            "question": "What happened?",
            "choices": ["Rain", "Wind"],
            "answer": "Rain",
        }
    )
    assert sample.sample_id == "sample-1"
    assert sample.question == "What happened?"
    assert sample.choices == ("Rain", "Wind")
    assert sample.source_record["audio"]["bytes"] == "<embedded-audio-bytes>"


def test_normalize_choice_columns() -> None:
    sample = normalize_row(
        {
            "key": "x",
            "audio_path": "audio.wav",
            "choice_a": "one",
            "choice_b": "two",
        }
    )
    assert sample.choices == ("one", "two")


def test_normalize_mmau_context_audio() -> None:
    sample = normalize_row(
        {
            "context": {"bytes": b"wav", "path": "sample.wav"},
            "instruction": "Identify the speaker.",
            "choices": ["Man", "Woman"],
        }
    )
    assert sample.audio_values == ({"bytes": b"wav", "path": "sample.wav"},)
    assert sample.question == "Identify the speaker."


def test_normalize_mmau_pro_singular_field_with_multiple_paths() -> None:
    sample = normalize_row(
        {
            "id": "multi",
            "audio_path": ["first.wav", "second.wav", "third.wav"],
            "question": "Compare the clips.",
        }
    )
    assert sample.audio_values == ("first.wav", "second.wav", "third.wav")


def test_default_subset_discovers_nested_hub_parquet_shards(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pq.write_table(pa.Table.from_pylist([{"id": "nested"}]), data_dir / "part.parquet")

    assert list(iter_rows(tmp_path, "default")) == [{"id": "nested"}]
