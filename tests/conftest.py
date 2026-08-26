from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.build_tiny_captionstew import build_tiny_captionstew


@pytest.fixture
def tiny_captionstew(tmp_path: Path) -> dict[str, object]:
    return build_tiny_captionstew(tmp_path)
