from __future__ import annotations

from audio_lfm.data.resume_state import DataResumeState


def test_committed_ids_roundtrip_and_epoch_clear(tmp_path) -> None:
    state = DataResumeState(epoch=2)
    state.commit(["b", "a"])
    path = tmp_path / "ids.zst"
    state.save_ids(path)
    restored = DataResumeState(epoch=2)
    restored.load_ids(path)
    assert restored.committed_audio_ids == {"a", "b"}
    restored.advance_epoch()
    assert restored.epoch == 3
    assert restored.committed_audio_ids == set()
