from __future__ import annotations

from typing import TYPE_CHECKING

from goapgit.actions.conflict import apply_path_strategy, auto_trivial_resolve
from goapgit.core.models import ConflictDetail, StrategyRule
from goapgit.git.facade import GitFacade
from goapgit.io.logging import StructuredLogger

if TYPE_CHECKING:
    from pathlib import Path


_DEF_USER = ("git", "config", "user.email", "test@example.com")
_DEF_NAME = ("git", "config", "user.name", "Test User")


def _init_repo(tmp_path: Path) -> tuple[Path, GitFacade, StructuredLogger]:
    repo = tmp_path / "repo"
    repo.mkdir()
    logger = StructuredLogger(name="test")
    facade = GitFacade(repo_path=repo, logger=logger)
    facade.run(["git", "init"])
    facade.run(list(_DEF_USER))
    facade.run(list(_DEF_NAME))
    return repo, facade, logger


def _seed_conflict_history(repo: Path, facade: GitFacade) -> None:
    (repo / "conflict.txt").write_text("base\n", encoding="utf-8")
    facade.run(["git", "add", "conflict.txt"])
    facade.run(["git", "commit", "-m", "base"])
    facade.run(["git", "branch", "-M", "main"])
    facade.run(["git", "config", "rerere.enabled", "true"])
    facade.run(["git", "config", "rerere.autoupdate", "false"])

    facade.run(["git", "checkout", "-b", "feature"])
    (repo / "conflict.txt").write_text("feature\n", encoding="utf-8")
    facade.run(["git", "commit", "-am", "feature change"])

    facade.run(["git", "checkout", "main"])
    (repo / "conflict.txt").write_text("main\n", encoding="utf-8")
    facade.run(["git", "commit", "-am", "main change"])

    merge = facade.run(["git", "merge", "feature"], check=False)
    assert merge.returncode != 0
    (repo / "conflict.txt").write_text("resolved\n", encoding="utf-8")
    facade.run(["git", "add", "conflict.txt"])
    facade.run(["git", "commit", "-m", "resolve"])

    facade.run(["git", "reset", "--hard", "HEAD^"])
    merge_again = facade.run(["git", "merge", "feature"], check=False)
    assert merge_again.returncode != 0


def test_auto_trivial_resolve_applies_rerere(tmp_path: Path) -> None:
    """Recorded rerere resolutions should be reapplied automatically."""
    repo, facade, logger = _init_repo(tmp_path)
    _seed_conflict_history(repo, facade)

    applied = auto_trivial_resolve(facade, logger)

    assert applied is True
    diff = facade.run(["git", "diff"])
    assert diff.stdout.strip() == ""


def test_apply_path_strategy_uses_theirs_for_lock(tmp_path: Path) -> None:
    """Lock files should accept the incoming changes automatically."""
    repo, facade, logger = _init_repo(tmp_path)
    (repo / "dep.lock").write_text("base\n", encoding="utf-8")
    facade.run(["git", "add", "dep.lock"])
    facade.run(["git", "commit", "-m", "base"])
    facade.run(["git", "branch", "-M", "main"])

    facade.run(["git", "checkout", "-b", "feature"])
    (repo / "dep.lock").write_text("feature\n", encoding="utf-8")
    facade.run(["git", "commit", "-am", "feature"])

    facade.run(["git", "checkout", "main"])
    (repo / "dep.lock").write_text("main\n", encoding="utf-8")
    facade.run(["git", "commit", "-am", "main"])

    merge = facade.run(["git", "merge", "feature"], check=False)
    assert merge.returncode != 0

    rules = [StrategyRule(pattern="**/*.lock", resolution="theirs")]
    conflicts = [ConflictDetail(path="dep.lock")]

    resolved = apply_path_strategy(facade, logger, conflicts, rules)

    assert resolved == ["dep.lock"]
    contents = (repo / "dep.lock").read_text(encoding="utf-8")
    assert contents == "feature\n"
    status = facade.run(["git", "status", "--porcelain"])
    assert "U" not in status.stdout


def test_apply_path_strategy_prefers_ours_for_whitespace_markdown(tmp_path: Path) -> None:
    """Whitespace-only markdown conflicts should keep the local content."""
    repo, facade, logger = _init_repo(tmp_path)
    (repo / "README.md").write_text("value=1\n", encoding="utf-8")
    facade.run(["git", "add", "README.md"])
    facade.run(["git", "commit", "-m", "base"])
    facade.run(["git", "branch", "-M", "main"])

    facade.run(["git", "checkout", "-b", "feature"])
    (repo / "README.md").write_text("value= 1\n", encoding="utf-8")
    facade.run(["git", "commit", "-am", "feature"])

    facade.run(["git", "checkout", "main"])
    (repo / "README.md").write_text("value =1\n", encoding="utf-8")
    facade.run(["git", "commit", "-am", "main"])

    merge = facade.run(["git", "merge", "feature"], check=False)
    assert merge.returncode != 0

    rules = [
        StrategyRule(pattern="**/*.lock", resolution="theirs"),
        StrategyRule(pattern="*.md", resolution="ours", when="whitespace_only"),
    ]
    conflicts = [ConflictDetail(path="README.md")]

    resolved = apply_path_strategy(facade, logger, conflicts, rules)

    assert resolved == ["README.md"]
    contents = (repo / "README.md").read_text(encoding="utf-8")
    assert contents == "value =1\n"
    status = facade.run(["git", "status", "--porcelain"])
    assert "U" not in status.stdout
