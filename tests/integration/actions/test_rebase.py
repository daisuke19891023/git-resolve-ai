from __future__ import annotations

from typing import TYPE_CHECKING

from goapgit.actions.rebase import rebase_onto_upstream
from goapgit.git.facade import GitFacade
from goapgit.io.logging import StructuredLogger

if TYPE_CHECKING:
    from pathlib import Path


_DEF_USER = ("git", "config", "user.email", "test@example.com")
_DEF_NAME = ("git", "config", "user.name", "Test User")


def _append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line)


def _setup_repo(tmp_path: Path) -> tuple[Path, GitFacade, StructuredLogger]:
    repo = tmp_path / "repo"
    repo.mkdir()
    logger = StructuredLogger(name="test")
    facade = GitFacade(repo_path=repo, logger=logger)
    facade.run(["git", "init"])
    facade.run(list(_DEF_USER))
    facade.run(list(_DEF_NAME))
    base_file = repo / "base.txt"
    base_file.write_text("base\n", encoding="utf-8")
    facade.run(["git", "add", "base.txt"])
    facade.run(["git", "commit", "-m", "base"])
    facade.run(["git", "branch", "-M", "main"])
    return repo, facade, logger


def test_rebase_updates_stack_references(tmp_path: Path) -> None:
    """Rebasing with update-refs should rewrite stacked branch references."""
    repo, facade, logger = _setup_repo(tmp_path)

    base_file = repo / "base.txt"
    _append_line(base_file, "main-1\n")
    facade.run(["git", "commit", "-am", "main update"])

    facade.run(["git", "checkout", "-b", "feature"])
    feature_file = repo / "feature.txt"
    feature_file.write_text("feature\n", encoding="utf-8")
    facade.run(["git", "add", "feature.txt"])
    facade.run(["git", "commit", "-m", "feature commit"])
    feature_sha_before = facade.run(["git", "rev-parse", "HEAD"]).stdout.strip()

    child_branch = "feature-stack"
    facade.run(["git", "checkout", "-b", child_branch])
    child_file = repo / "child.txt"
    child_file.write_text("child\n", encoding="utf-8")
    facade.run(["git", "add", "child.txt"])
    facade.run(["git", "commit", "-m", "child commit"])
    child_sha_before = facade.run(["git", "rev-parse", "HEAD"]).stdout.strip()

    facade.run(["git", "checkout", "main"])
    _append_line(base_file, "main-2\n")
    facade.run(["git", "commit", "-am", "main second"])

    facade.run(["git", "checkout", "feature"])
    rebase_onto_upstream(facade, logger, "main", update_refs=True)

    feature_sha_after = facade.run(["git", "rev-parse", "feature"]).stdout.strip()
    child_sha_after = facade.run(["git", "rev-parse", child_branch]).stdout.strip()

    assert feature_sha_after != feature_sha_before
    assert child_sha_after != child_sha_before
    parents = facade.run(["git", "log", "-1", "--pretty=%P", child_branch]).stdout.strip().split()
    assert feature_sha_after in parents
