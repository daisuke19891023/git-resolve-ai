from __future__ import annotations

import json
import os
from pathlib import Path

from goapgit.plugins.json_merge import MergeInputs, merge_structured_documents, main


class _ChangeCwd:
    """Context manager to temporarily switch working directory."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._previous: Path | None = None

    def __enter__(self) -> None:
        self._previous = Path.cwd()
        os.chdir(self._path)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> bool:
        if self._previous is not None:
            os.chdir(self._previous)
        return False


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write formatted JSON content to disk."""
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_merge_structured_documents_merges_complementary_changes(tmp_path: Path) -> None:
    """Merge complementary updates without leaving conflicts."""
    with _ChangeCwd(tmp_path):
        base = tmp_path / "base.json"
        current = tmp_path / "current.json"
        other = tmp_path / "other.json"

        _write_json(base, {"name": "example", "config": {"timeout": 10}})
        _write_json(current, {"name": "example", "config": {"timeout": 20}})
        _write_json(other, {"name": "example", "config": {"timeout": 10, "retries": 3}})

        inputs = MergeInputs(base=base, current=current, other=other)
        assert merge_structured_documents(inputs) is True

        merged = json.loads(current.read_text(encoding="utf-8"))
        assert merged["config"] == {"timeout": 20, "retries": 3}


def test_merge_structured_documents_detects_conflicts(tmp_path: Path) -> None:
    """Report a merge failure when both sides modify the same value."""
    with _ChangeCwd(tmp_path):
        base = tmp_path / "base.json"
        current = tmp_path / "current.json"
        other = tmp_path / "other.json"

        _write_json(base, {"value": 1})
        _write_json(current, {"value": 2})
        _write_json(other, {"value": 3})

        original = current.read_text(encoding="utf-8")
        inputs = MergeInputs(base=base, current=current, other=other)
        assert merge_structured_documents(inputs) is False
        assert current.read_text(encoding="utf-8") == original


def test_main_returns_non_zero_on_conflict(tmp_path: Path) -> None:
    """Return a non-zero exit code when automatic merge fails."""
    with _ChangeCwd(tmp_path):
        base = tmp_path / "base.json"
        current = tmp_path / "current.json"
        other = tmp_path / "other.json"

        _write_json(base, {"value": 1})
        _write_json(current, {"value": 2})
        _write_json(other, {"value": 3})

        exit_code = main([str(base), str(current), str(other)])
        assert exit_code == 1

        _write_json(other, {"value": 1, "extra": True})
        exit_code = main([str(base), str(current), str(other)])
        assert exit_code == 0
        merged = json.loads(current.read_text(encoding="utf-8"))
        assert merged == {"value": 2, "extra": True}


def test_merge_structured_documents_rejects_symlink_inputs(tmp_path: Path) -> None:
    """Refuse to merge when any merge input is a symlink."""
    with _ChangeCwd(tmp_path):
        base = tmp_path / "base.json"
        current_target = tmp_path / "real-current.json"
        current = tmp_path / "current.json"
        other = tmp_path / "other.json"

        _write_json(base, {"value": 1})
        _write_json(current_target, {"value": 2})
        current.symlink_to(current_target)
        _write_json(other, {"value": 1})

        original = current_target.read_text(encoding="utf-8")
        inputs = MergeInputs(base=base, current=current, other=other)
        assert merge_structured_documents(inputs) is False
        assert current_target.read_text(encoding="utf-8") == original


def test_merge_structured_documents_allows_missing_base(tmp_path: Path) -> None:
    """Allow merges to proceed when the base input is absent."""
    with _ChangeCwd(tmp_path):
        base = tmp_path / "base.json"
        current = tmp_path / "current.json"
        other = tmp_path / "other.json"

        _write_json(current, {"value": 1})
        _write_json(other, {"value": 1, "extra": True})

        inputs = MergeInputs(base=base, current=current, other=other)
        assert merge_structured_documents(inputs) is True

        merged = json.loads(current.read_text(encoding="utf-8"))
        assert merged == {"value": 1, "extra": True}


def test_repository_declares_json_merge_driver() -> None:
    """Ensure the repository configures the JSON merge driver."""
    attributes = Path(".gitattributes").read_text(encoding="utf-8")
    assert "merge=json" in attributes
    driver = Path("merge.json.driver").read_text(encoding="utf-8")
    assert "goapgit.plugins.json_merge" in driver
