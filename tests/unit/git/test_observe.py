from __future__ import annotations

import io
import subprocess
import textwrap
from pathlib import Path

import pytest

from goapgit.core.models import ConflictDetail, ConflictType, RepoState, RiskLevel
from goapgit.git.facade import GitFacade
from goapgit.git.observe import RepoObserver
from goapgit.io.logging import StructuredLogger


@pytest.fixture
def logger() -> StructuredLogger:
    """Provide a structured logger instance for tests."""
    return StructuredLogger(name="observer-test", stream=io.StringIO())


def test_observe_clean_repository(
    tmp_path: Path, logger: StructuredLogger, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create a clean RepoState from porcelain output."""
    output = textwrap.dedent(
        """
        # branch.oid deadbeefdeadbeefdeadbeefdeadbeefdeadbeef
        # branch.head main
        # branch.upstream origin/main
        # branch.ab +0 -0
        # stash 2
        """,
    ).strip()
    workspace = Path(tmp_path)
    facade = GitFacade(workspace, logger)

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        assert args == [
            "git",
            "status",
            "--porcelain=v2",
            "--branch",
            "--show-stash",
        ]
        return subprocess.CompletedProcess(tuple(args), 0, stdout=output, stderr="")

    monkeypatch.setattr(facade, "run", fake_run)
    observer = RepoObserver(facade)

    state = observer.observe()

    assert isinstance(state, RepoState)
    assert state.ref.branch == "main"
    assert state.ref.tracking == "origin/main"
    assert state.ref.sha == "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    assert state.working_tree_clean is True
    assert state.staged_changes is False
    assert state.stash_entries == 2
    assert state.risk_level is RiskLevel.low
    assert not state.conflicts


def test_observe_with_local_changes(
    tmp_path: Path, logger: StructuredLogger, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detect staged changes and divergence markers."""
    output = textwrap.dedent(
        """
        # branch.oid cafe0000cafe0000cafe0000cafe0000cafe0000
        # branch.head feature
        # branch.upstream origin/feature
        # branch.ab +2 -1
        1 M. N... 100644 100644 100644 100644 dead dead tracked.txt
        ? newfile.txt
        """,
    ).strip()
    workspace = Path(tmp_path)
    facade = GitFacade(workspace, logger)

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        assert args[0] == "git"
        return subprocess.CompletedProcess(tuple(args), 0, stdout=output, stderr="")

    monkeypatch.setattr(facade, "run", fake_run)
    observer = RepoObserver(facade)

    state = observer.observe()

    assert state.working_tree_clean is False
    assert state.staged_changes is True
    assert state.diverged_local == 2
    assert state.diverged_remote == 1
    assert state.has_unpushed_commits is True
    assert state.risk_level is RiskLevel.med
    assert not state.conflicts


def test_observe_detects_conflicts(
    tmp_path: Path, logger: StructuredLogger, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identify conflicts and classify them via the parser hook."""
    output = textwrap.dedent(
        """
        # branch.oid feed0000feed0000feed0000feed0000feed0000
        # branch.head feature
        # branch.upstream origin/feature
        # branch.ab +0 -3
        # rebase-merge
        u UU N... 100644 100644 100644 100644 sha1 sha2 sha3 conflict.json
        """,
    ).strip()
    recorded_paths: list[str] = []

    workspace = Path(tmp_path)
    facade = GitFacade(workspace, logger)

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(tuple(args), 0, stdout=output, stderr="")

    def fake_parser(repo_path: Path, path: str) -> ConflictDetail:
        recorded_paths.append(str(repo_path / path))
        return ConflictDetail(path=path, hunk_count=2, ctype=ConflictType.json)

    monkeypatch.setattr(facade, "run", fake_run)
    observer = RepoObserver(facade, conflict_parser=fake_parser)

    state = observer.observe()

    assert state.ongoing_rebase is True
    assert state.ongoing_merge is False
    assert state.risk_level is RiskLevel.high
    assert state.conflicts[0].ctype is ConflictType.json
    assert state.conflicts[0].hunk_count == 2
    assert recorded_paths == [str(tmp_path / "conflict.json")]
