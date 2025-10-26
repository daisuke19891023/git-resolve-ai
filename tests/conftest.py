"""Shared fixtures for the goapgit test suite."""
from __future__ import annotations
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

import pytest
from goapgit.git.facade import GitCommandError

@dataclass(frozen=True)
class GitResponse:
    """Represents a scripted response for a git command."""

    stdout: str = ""
    stderr: str = ""
    returncode: int = 0

class ScriptQueue:
    """Queue managing scripted git responses for :class:`FakeGitFacade`."""

    def __init__(self) -> None:
        """Initialise an empty script queue."""
        self._scripts: deque[dict[tuple[str, ...], deque[GitResponse]]] = deque()
    def push(self, script: dict[tuple[str, ...], list[GitResponse] | GitResponse]) -> None:
        """Append a new script that will be consumed by the next facade instance."""
        prepared: dict[tuple[str, ...], deque[GitResponse]] = {}
        for command, responses in script.items():
            if isinstance(responses, GitResponse):
                prepared[command] = deque([responses])
            else:
                prepared[command] = deque(responses)
        self._scripts.append(prepared)
    def pop(self) -> dict[tuple[str, ...], deque[GitResponse]]:
        """Return the next script or an empty script when none are queued."""
        if not self._scripts:
            return {}
        return self._scripts.popleft()
    def clear(self) -> None:
        """Remove all queued scripts."""
        self._scripts.clear()
class FakeGitFacade:
    """Test double for :class:`goapgit.git.facade.GitFacade`."""

    script_queue: ScriptQueue | None = None
    def __init__(
        self,
        *,
        repo_path: Path,
        logger: Any,
        dry_run: bool = False,
        env: dict[str, str] | None = None,
    ) -> None:
        """Initialise the facade with scripted responses."""
        self._repo_path = Path(repo_path)
        self._logger = logger
        self._dry_run = dry_run
        self._env = dict(env or {})
        self._command_history: list[dict[str, Any]] = []
        script_source = self.script_queue.pop() if self.script_queue is not None else {}
        self._script = script_source
    @property
    def repo_path(self) -> Path:
        """Return the repository root associated with the facade."""
        return self._repo_path
    @property
    def dry_run(self) -> bool:
        """Return whether the facade operates in dry-run mode."""
        return self._dry_run
    @property
    def command_history(self) -> tuple[dict[str, Any], ...]:
        """Return the recorded command history."""
        return tuple(self._command_history)
    def run(
        self,
        args: Any,
        *,
        cwd: Path | None = None,
        timeout: float | None = None,
        check: bool = True,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Execute a git command using the scripted responses."""
        del timeout, capture_output  # unused but kept for parity with real facade
        command = tuple(str(part) for part in args)
        working_dir = Path(cwd) if cwd is not None else self._repo_path
        if self._dry_run:
            completed = subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            self._command_history.append(
                {
                    "command": list(command),
                    "cwd": str(working_dir),
                    "returncode": 0,
                    "dry_run": True,
                },
            )
            return completed
        response = self._resolve_response(command)
        completed = subprocess.CompletedProcess(
            command,
            response.returncode,
            stdout=response.stdout,
            stderr=response.stderr,
        )
        self._command_history.append(
            {
                "command": list(command),
                "cwd": str(working_dir),
                "returncode": completed.returncode,
                "dry_run": False,
            },
        )
        if check and completed.returncode != 0:
            raise GitCommandError(command, completed.returncode, completed.stdout or "", completed.stderr or "")
        return completed
    def fetch(
        self,
        remote: str = "origin",
        *,
        prune: bool = True,
        tags: bool = True,
        extra_args: Any | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Simulate `git fetch` for the provided remote."""
        command = ["git", "fetch"]
        if prune:
            command.append("--prune")
        if tags:
            command.append("--tags")
        if extra_args:
            command.extend(str(arg) for arg in extra_args)
        command.append(remote)
        return self.run(command)
    def rebase(
        self,
        branch: str,
        *,
        onto: str | None = None,
        opts: Any | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Simulate `git rebase` commands."""
        command = ["git", "rebase"]
        if opts:
            command.extend(str(option) for option in opts)
        if onto is not None:
            command.extend(["--onto", onto])
        if branch:
            command.append(branch)
        return self.run(command)
    def rebase_continue(self) -> subprocess.CompletedProcess[str]:
        """Simulate `git rebase --continue`."""
        return self.run(["git", "rebase", "--continue"])
    def rebase_abort(self) -> subprocess.CompletedProcess[str]:
        """Simulate `git rebase --abort`."""
        return self.run(["git", "rebase", "--abort"])
    def push_with_lease(
        self,
        remote: str = "origin",
        refspecs: Any | None = None,
        *,
        force: bool = False,
        extra_args: Any | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Simulate `git push --force-with-lease`."""
        command = ["git", "push", "--force-with-lease"]
        if force:
            command.append("--force")
        if extra_args:
            command.extend(str(arg) for arg in extra_args)
        command.append(remote)
        if refspecs:
            command.extend(str(spec) for spec in refspecs)
        return self.run(command)
    def _resolve_response(self, command: tuple[str, ...]) -> GitResponse:
        """Retrieve the scripted response for ``command``."""
        if command not in self._script:
            message = f"Unexpected git command: {command}"
            raise AssertionError(message)
        responses = self._script[command]
        return responses.popleft() if len(responses) > 1 else responses[0]
@pytest.fixture
def configure_fake_git_facade(monkeypatch: pytest.MonkeyPatch) -> Iterator[ScriptQueue]:
    """Patch :class:`GitFacade` with a scripted fake for tests."""
    queue = ScriptQueue()
    FakeGitFacade.script_queue = queue
    monkeypatch.setattr("goapgit.cli.runtime.GitFacade", FakeGitFacade)
    yield queue
    queue.clear()
    FakeGitFacade.script_queue = None
__all__ = ["FakeGitFacade", "GitResponse", "ScriptQueue"]
