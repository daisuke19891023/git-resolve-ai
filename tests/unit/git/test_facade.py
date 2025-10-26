from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pytest

from goapgit.git.facade import GitCommandError, GitFacade
from goapgit.io.logging import StructuredLogger


@pytest.fixture
def logger() -> StructuredLogger:
    """Provide a structured logger backed by an in-memory stream."""
    return StructuredLogger(name="test", stream=io.StringIO())


@pytest.fixture
def facade(tmp_path: Path, logger: StructuredLogger) -> GitFacade:
    """Create a GitFacade bound to a temporary directory."""
    workspace = Path(tmp_path)
    return GitFacade(workspace, logger)


def test_run_invokes_subprocess_and_logs(
    monkeypatch: pytest.MonkeyPatch, facade: GitFacade,
) -> None:
    """Ensure run() delegates to subprocess and records history."""
    captured: dict[str, object] = {}

    def fake_run(command: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(facade, "_subprocess_run", fake_run)
    result = facade.run(["git", "status"])

    assert result.stdout == "ok"
    assert captured["command"] == ("git", "status")
    history = facade.command_history
    assert len(history) == 1
    assert history[0]["returncode"] == 0


def test_run_raises_on_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch, facade: GitFacade,
) -> None:
    """Raise GitCommandError when the underlying command fails."""

    def fake_run(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="boom")

    monkeypatch.setattr(facade, "_subprocess_run", fake_run)
    with pytest.raises(GitCommandError) as exc:
        facade.run(["git", "status"])
    assert exc.value.returncode == 1
    assert facade.command_history[0]["returncode"] == 1


def test_run_dry_run_records_without_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    logger: StructuredLogger,
) -> None:
    """Dry-run mode should not invoke subprocess."""
    workspace = Path(tmp_path)
    facade = GitFacade(workspace, logger, dry_run=True)

    def fake_run(*_: object, **__: object) -> None:  # pragma: no cover - guard
        raise AssertionError("subprocess should not be executed in dry-run mode")

    monkeypatch.setattr(facade, "_subprocess_run", fake_run)
    result = facade.run(["git", "status"])
    assert result.returncode == 0
    assert facade.command_history[0]["dry_run"] is True


def test_fetch_invokes_git_fetch(
    monkeypatch: pytest.MonkeyPatch, facade: GitFacade,
) -> None:
    """Ensure fetch() constructs the expected command."""
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((list(args), dict(kwargs)))
        return subprocess.CompletedProcess(tuple(args), 0, stdout="", stderr="")

    monkeypatch.setattr(facade, "run", fake_run)
    facade.fetch("origin", extra_args=["--atomic"])
    assert calls[0][0] == ["git", "fetch", "--prune", "--tags", "--atomic", "origin"]


def test_rebase_with_options(
    monkeypatch: pytest.MonkeyPatch, facade: GitFacade,
) -> None:
    """rebase() should honour additional options and --onto."""
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        return subprocess.CompletedProcess(tuple(args), 0, stdout="", stderr="")

    monkeypatch.setattr(facade, "run", fake_run)
    facade.rebase("feature", onto="main", opts=["--update-refs"])
    assert calls[0] == ["git", "rebase", "--update-refs", "--onto", "main", "feature"]


def test_rebase_continue_and_abort(
    monkeypatch: pytest.MonkeyPatch, facade: GitFacade,
) -> None:
    """Ensure rebase_continue() and rebase_abort() issue the right commands."""
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        return subprocess.CompletedProcess(tuple(args), 0, stdout="", stderr="")

    monkeypatch.setattr(facade, "run", fake_run)
    facade.rebase_continue()
    facade.rebase_abort()
    assert calls == [["git", "rebase", "--continue"], ["git", "rebase", "--abort"]]


def test_push_with_lease(
    monkeypatch: pytest.MonkeyPatch, facade: GitFacade,
) -> None:
    """push_with_lease() should pass lease and refspec arguments."""
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        return subprocess.CompletedProcess(tuple(args), 0, stdout="", stderr="")

    monkeypatch.setattr(facade, "run", fake_run)
    facade.push_with_lease("origin", ["HEAD:main"], extra_args=["--atomic"], force=True)
    assert calls[0] == [
        "git",
        "push",
        "--force-with-lease",
        "--force",
        "--atomic",
        "origin",
        "HEAD:main",
    ]
