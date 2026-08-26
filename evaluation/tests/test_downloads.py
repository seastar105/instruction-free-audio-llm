from __future__ import annotations

import subprocess

from audio_lfm_eval import downloads


def test_scorer_patch_is_applied_once(tmp_path, monkeypatch) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    source = checkout / "value.txt"
    source.write_text("old\n")
    subprocess.run(["git", "-C", str(checkout), "add", "value.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Audio LFM test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "base",
        ],
        check=True,
    )
    patch_root = tmp_path / "patches"
    patch_root.mkdir()
    (patch_root / "test.patch").write_text(
        "diff --git a/value.txt b/value.txt\n"
        "index 3367afd..3e75765 100644\n"
        "--- a/value.txt\n"
        "+++ b/value.txt\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    monkeypatch.setitem(downloads._SCORER_PATCHES, "test", "test.patch")

    downloads._apply_scorer_patch(checkout, "test", patch_root=patch_root)
    downloads._apply_scorer_patch(checkout, "test", patch_root=patch_root)

    assert source.read_text() == "new\n"
