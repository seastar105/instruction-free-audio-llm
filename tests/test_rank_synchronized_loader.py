from __future__ import annotations

from collections.abc import Iterator

import torch

from audio_lfm.data.loader import iter_rank_synchronized


def test_rank_synchronized_iterator_is_passthrough_without_distributed() -> None:
    assert list(iter_rank_synchronized([1, 2, 3])) == [1, 2, 3]


def test_remote_exhaustion_stops_before_unmatched_yield(monkeypatch) -> None:
    calls = 0

    monkeypatch.setattr("audio_lfm.data.loader.dist.is_available", lambda: True)
    monkeypatch.setattr("audio_lfm.data.loader.dist.is_initialized", lambda: True)
    monkeypatch.setattr("audio_lfm.data.loader.dist.get_world_size", lambda: 2)
    monkeypatch.setattr("audio_lfm.data.loader.dist.get_backend", lambda: "gloo")

    def all_reduce(value: torch.Tensor, *, op: object) -> None:
        del op
        nonlocal calls
        calls += 1
        if calls == 2:
            value.zero_()

    monkeypatch.setattr("audio_lfm.data.loader.dist.all_reduce", all_reduce)
    assert list(iter_rank_synchronized([1, 2, 3])) == [1]
    assert calls == 2


def test_local_exhaustion_still_performs_final_handshake(monkeypatch) -> None:
    observed: list[int] = []

    monkeypatch.setattr("audio_lfm.data.loader.dist.is_available", lambda: True)
    monkeypatch.setattr("audio_lfm.data.loader.dist.is_initialized", lambda: True)
    monkeypatch.setattr("audio_lfm.data.loader.dist.get_world_size", lambda: 2)
    monkeypatch.setattr("audio_lfm.data.loader.dist.get_backend", lambda: "gloo")

    def all_reduce(value: torch.Tensor, *, op: object) -> None:
        del op
        observed.append(int(value.item()))

    def values() -> Iterator[int]:
        yield 1
        yield 2

    monkeypatch.setattr("audio_lfm.data.loader.dist.all_reduce", all_reduce)
    assert list(iter_rank_synchronized(values())) == [1, 2]
    assert observed == [1, 1, 0]
