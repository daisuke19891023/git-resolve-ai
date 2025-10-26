from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pytest

from goapgit.actions.quality import explain_range_diff
from goapgit.git.facade import GitFacade
from goapgit.io.logging import StructuredLogger

if TYPE_CHECKING:
    from pathlib import Path


def _make_facade(path: Path, name: str) -> tuple[GitFacade, StructuredLogger]:
    """Create a GitFacade with a dedicated structured logger."""
    logger = StructuredLogger(name=name, stream=io.StringIO())
    return GitFacade(repo_path=path, logger=logger), logger


@pytest.mark.integration
def test_explain_range_diff_logs_summary(tmp_path: Path) -> None:
    """Compute and persist a range-diff summary for a rebased branch."""
    repo = tmp_path / "repo"
    repo.mkdir()
    facade, logger = _make_facade(repo, "git-quality")
    facade.run(["git", "init", "--initial-branch=main"])
    facade.run(["git", "config", "user.email", "quality@example.com"])
    facade.run(["git", "config", "user.name", "Quality User"])

    (repo / "file.txt").write_text("base\n", encoding="utf-8")
    facade.run(["git", "add", "file.txt"])
    facade.run(["git", "commit", "-m", "initial"])
    initial = facade.run(["git", "rev-parse", "HEAD"]).stdout.strip()

    facade.run(["git", "checkout", "-b", "feature"])
    (repo / "file.txt").write_text("feature change\n", encoding="utf-8")
    facade.run(["git", "commit", "-am", "feature change"])
    feature_before = facade.run(["git", "rev-parse", "HEAD"]).stdout.strip()

    facade.run(["git", "checkout", "main"])
    (repo / "upstream.txt").write_text("upstream\n", encoding="utf-8")
    facade.run(["git", "add", "upstream.txt"])
    facade.run(["git", "commit", "-m", "upstream change"])
    upstream_after = facade.run(["git", "rev-parse", "HEAD"]).stdout.strip()

    facade.run(["git", "checkout", "feature"])
    facade.run(["git", "rebase", "main"])
    feature_after = facade.run(["git", "rev-parse", "HEAD"]).stdout.strip()

    before_range = f"{initial}..{feature_before}"
    after_range = f"{upstream_after}..{feature_after}"
    output_file = tmp_path / "range-diff.txt"

    summary = explain_range_diff(
        facade,
        logger,
        before_range,
        after_range,
        output_path=output_file,
    )

    assert "feature change" in summary
    assert output_file.read_text(encoding="utf-8").strip().startswith("1:")
