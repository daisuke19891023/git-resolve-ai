from __future__ import annotations

from typing import TYPE_CHECKING

from goapgit.actions.safety import create_backup_ref, ensure_clean_or_stash
from goapgit.git.facade import GitFacade
from goapgit.io.logging import StructuredLogger

if TYPE_CHECKING:
    from pathlib import Path


_DEF_USER = ("git", "config", "user.email", "test@example.com")
_DEF_NAME = ("git", "config", "user.name", "Test User")


def _setup_repo(tmp_path: Path) -> tuple[Path, GitFacade, StructuredLogger]:
    repo = tmp_path / "repo"
    repo.mkdir()
    logger = StructuredLogger(name="test")
    facade = GitFacade(repo_path=repo, logger=logger)
    facade.run(["git", "init"])
    facade.run(list(_DEF_USER))
    facade.run(list(_DEF_NAME))
    (repo / "README.md").write_text("initial\n", encoding="utf-8")
    facade.run(["git", "add", "README.md"])
    facade.run(["git", "commit", "-m", "init"])
    return repo, facade, logger


def test_create_backup_ref_records_head_sha(tmp_path: Path) -> None:
    """Backup refs should capture the HEAD commit SHA."""
    _repo, facade, logger = _setup_repo(tmp_path)

    ref_name = create_backup_ref(facade, logger)

    show_ref = facade.run(["git", "show-ref", ref_name])
    head = facade.run(["git", "rev-parse", "HEAD"])
    assert show_ref.stdout.split()[0] == head.stdout.strip()


def test_ensure_clean_or_stash_creates_stash_when_dirty(tmp_path: Path) -> None:
    """A dirty worktree should result in a stash entry being created."""
    repo, facade, logger = _setup_repo(tmp_path)
    (repo / "README.md").write_text("changed\n", encoding="utf-8")

    created = ensure_clean_or_stash(facade, logger)

    assert created is True
    stash_list = facade.run(["git", "stash", "list"])
    assert "goap/" in stash_list.stdout


def test_ensure_clean_or_stash_noop_when_clean(tmp_path: Path) -> None:
    """A clean worktree should not create a stash entry."""
    _, facade, logger = _setup_repo(tmp_path)

    created = ensure_clean_or_stash(facade, logger)

    assert created is False
    stash_list = facade.run(["git", "stash", "list"])
    assert stash_list.stdout.strip() == ""
