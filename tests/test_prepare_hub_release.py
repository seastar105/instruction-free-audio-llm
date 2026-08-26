from __future__ import annotations

from scripts.prepare_hub_release import portable


def test_portable_redacts_absolute_paths_recursively() -> None:
    assert portable(
        {
            "root": "/private/data/CaptionStew",
            "items": ["relative/file.json", "/secret/decoder_lock.json"],
        }
    ) == {
        "root": "<local-path>/CaptionStew",
        "items": ["relative/file.json", "<local-path>/decoder_lock.json"],
    }
