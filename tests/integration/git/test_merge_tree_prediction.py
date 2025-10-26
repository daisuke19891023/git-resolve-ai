from __future__ import annotations

import io
import os
import shutil
from pathlib import Path

import pytest

from goapgit.git.facade import GitFacade
from goapgit.git.parse import predict_merge_conflicts
from goapgit.io.logging import StructuredLogger


@pytest.mark.integration
def test_merge_tree_prediction_matches_actual_merge(tmp_path: Path) -> None:
    """Ensure merge-tree predicted conflicts match a real merge attempt."""
    if shutil.which("git") is None:
        pytest.skip("git executable not available")

    repo_root = Path(tmp_path)
    env = {**os.environ, "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@example.com"}
    env["GIT_COMMITTER_NAME"] = env["GIT_AUTHOR_NAME"]
    env["GIT_COMMITTER_EMAIL"] = env["GIT_AUTHOR_EMAIL"]

    logger = StructuredLogger(name="merge-tree-test", stream=io.StringIO())
    facade = GitFacade(repo_root, logger, env={**env, "PATH": os.environ.get("PATH", "")})

    facade.run(["git", "init", "-q"])
    facade.run(["git", "checkout", "-q", "-b", "main"])
    (repo_root / "file.txt").write_text("base\n", encoding="utf-8")
    facade.run(["git", "add", "file.txt"])
    facade.run(["git", "commit", "-q", "-m", "base"])

    facade.run(["git", "checkout", "-q", "-b", "feature"])
    (repo_root / "file.txt").write_text("feature\n", encoding="utf-8")
    facade.run(["git", "commit", "-am", "feature"])

    facade.run(["git", "checkout", "-q", "main"])
    (repo_root / "file.txt").write_text("master\n", encoding="utf-8")
    facade.run(["git", "commit", "-am", "master"])

    predicted = predict_merge_conflicts(facade, "HEAD", "feature")
    assert predicted == {"file.txt"}

    merge_result = facade.run(["git", "merge", "feature"], check=False)
    assert merge_result.returncode != 0

    status_output = facade.run(
        ["git", "status", "--porcelain=v2", "--branch", "--show-stash"],
        capture_output=True,
    ).stdout.splitlines()
    actual_conflicts: set[str] = {
        line.split()[-1] for line in status_output if line.startswith("u ")
    }
    assert predicted == actual_conflicts

    facade.run(["git", "merge", "--abort"], check=False)
