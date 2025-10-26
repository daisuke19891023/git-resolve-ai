from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pytest

from goapgit.actions.sync import push_with_lease
from goapgit.git.facade import GitCommandError, GitFacade
from goapgit.io.logging import StructuredLogger

if TYPE_CHECKING:
    from pathlib import Path


def _make_facade(path: Path, name: str) -> tuple[GitFacade, StructuredLogger]:
    """Create a GitFacade with a dedicated structured logger."""
    logger = StructuredLogger(name=name, stream=io.StringIO())
    return GitFacade(repo_path=path, logger=logger), logger


@pytest.mark.integration
def test_push_with_lease_refuses_when_remote_has_moved(tmp_path: Path) -> None:
    """Ensure push with lease fails when the remote advanced unexpectedly."""
    remote = tmp_path / "remote.git"
    remote.mkdir()
    remote_facade, _ = _make_facade(remote, "git-remote")
    remote_facade.run(["git", "init", "--bare", "."])
    remote_facade.run(["git", "symbolic-ref", "HEAD", "refs/heads/main"])

    local = tmp_path / "local"
    local.mkdir()
    local_facade, local_logger = _make_facade(local, "git-local")
    local_facade.run(["git", "init", "--initial-branch=main"])
    local_facade.run(["git", "config", "user.email", "local@example.com"])
    local_facade.run(["git", "config", "user.name", "Local User"])

    (local / "file.txt").write_text("initial\n", encoding="utf-8")
    local_facade.run(["git", "add", "file.txt"])
    local_facade.run(["git", "commit", "-m", "initial"])
    local_facade.run(["git", "remote", "add", "origin", str(remote)])
    local_facade.run(["git", "push", "-u", "origin", "main"])

    root_facade, _ = _make_facade(tmp_path, "git-root")
    other = tmp_path / "other"
    root_facade.run(["git", "clone", str(remote), str(other)])
    other_facade, _ = _make_facade(other, "git-other")
    other_facade.run(["git", "config", "user.email", "other@example.com"])
    other_facade.run(["git", "config", "user.name", "Other User"])
    (other / "file.txt").write_text("remote change\n", encoding="utf-8")
    other_facade.run(["git", "add", "file.txt"])
    other_facade.run(["git", "commit", "-m", "remote change"])
    other_facade.run(["git", "push", "origin", "main"])

    (local / "file.txt").write_text("local rewrite\n", encoding="utf-8")
    local_facade.run(["git", "commit", "-am", "local rewrite"])

    with pytest.raises(GitCommandError):
        push_with_lease(local_facade, local_logger)
