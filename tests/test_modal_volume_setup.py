from __future__ import annotations

import json

from painting_inpaint_backend.modal.volume_utils import (
    READY_MARKER_NAME,
    snapshot_summary,
    write_ready_marker,
)


def test_partial_volume_directory_is_not_treated_as_ready(tmp_path):
    target = tmp_path / "model"
    target.mkdir()
    (target / ".gitattributes").write_text("partial", encoding="utf-8")

    partial = snapshot_summary(target)
    write_ready_marker(target, repo_id="example/model")
    complete = snapshot_summary(target)

    assert partial["exists"] is True
    assert partial["ready"] is False
    assert complete["ready"] is True
    assert (target / READY_MARKER_NAME).exists()


def test_ready_marker_records_pinned_revision(tmp_path):
    target = tmp_path / "model"
    target.mkdir()

    write_ready_marker(target, repo_id="example/model", revision="a" * 40)

    payload = json.loads((target / READY_MARKER_NAME).read_text(encoding="utf-8"))
    assert payload["repo_id"] == "example/model"
    assert payload["revision"] == "a" * 40
