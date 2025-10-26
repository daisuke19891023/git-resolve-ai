"""Git command execution facade."""

from __future__ import annotations

import inspect
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, MutableSequence, Sequence
    from goapgit.io.logging import StructuredLogger



class GitCommandError(RuntimeError):
    """Raised when a git command exits with a non-zero status."""

    def __init__(
        self,
        command: Sequence[str],
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        """Initialise the error with details from a git command invocation."""
        self.command = tuple(command)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        message = (
            "git command failed",
            f"command={self.command}",
            f"returncode={returncode}",
        )
        super().__init__("; ".join(message))


class GitFacade:
    """Provide a safe wrapper around subprocess-based git invocations."""

    def __init__(
        self,
        repo_path: Path,
        logger: StructuredLogger,
        *,
        dry_run: bool = False,
        env: Mapping[str, str] | None = None,
    ) -> None:
        """Create a facade bound to a repository root and logger."""
        self._repo_path = Path(repo_path)
        self._logger = logger
        self._dry_run = dry_run
        self._env = dict(env or {})
        self._command_history: MutableSequence[dict[str, object]] = []
        self._subprocess_run: Callable[
            ..., subprocess.CompletedProcess[str],
        ] = subprocess.run

    @property
    def repo_path(self) -> Path:
        """Return the repository root for the facade."""
        return self._repo_path

    @property
    def dry_run(self) -> bool:
        """Return whether the facade operates in dry-run mode."""
        return self._dry_run

    @property
    def command_history(self) -> Sequence[dict[str, object]]:
        """Return an immutable view of recorded commands."""
        return tuple(self._command_history)

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        timeout: float | None = None,
        check: bool = True,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Execute a git command while handling dry-run and logging."""
        command = tuple(str(part) for part in args)
        working_dir = Path(cwd) if cwd is not None else self._repo_path
        self._logger.info(
            "executing git command",
            command=list(command),
            cwd=str(working_dir),
            dry_run=self._dry_run,
        )
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

        kwargs: dict[str, object] = {
            "cwd": str(working_dir),
            "capture_output": capture_output,
            "text": True,
            "timeout": timeout,
            "check": False,
            "env": self._env or None,
        }
        filtered_kwargs = _filter_runner_kwargs(self._subprocess_run, kwargs)
        completed = self._subprocess_run(command, **filtered_kwargs)
        self._command_history.append(
            {
                "command": list(command),
                "cwd": str(working_dir),
                "returncode": completed.returncode,
                "dry_run": False,
            },
        )
        if completed.stdout:
            self._logger.debug("git stdout", stdout=completed.stdout)
        if completed.stderr:
            self._logger.debug("git stderr", stderr=completed.stderr)
        if check and completed.returncode != 0:
            raise GitCommandError(command, completed.returncode, completed.stdout, completed.stderr)
        return completed

    def fetch(
        self,
        remote: str = "origin",
        *,
        prune: bool = True,
        tags: bool = True,
        extra_args: Sequence[str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Fetch from the remote with safe defaults."""
        command: list[str] = ["git", "fetch"]
        if prune:
            command.append("--prune")
        if tags:
            command.append("--tags")
        if extra_args:
            command.extend(extra_args)
        command.append(remote)
        return self.run(command)

    def rebase(
        self,
        branch: str,
        *,
        onto: str | None = None,
        opts: Sequence[str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run git rebase with optional --onto and extra options."""
        command: list[str] = ["git", "rebase"]
        if opts:
            command.extend(opts)
        if onto is not None:
            command.extend(["--onto", onto])
        if branch:
            command.append(branch)
        return self.run(command)

    def rebase_continue(self) -> subprocess.CompletedProcess[str]:
        """Continue an in-progress rebase."""
        return self.run(["git", "rebase", "--continue"])

    def rebase_abort(self) -> subprocess.CompletedProcess[str]:
        """Abort an in-progress rebase."""
        return self.run(["git", "rebase", "--abort"])

    def push_with_lease(
        self,
        remote: str = "origin",
        refspecs: Sequence[str] | None = None,
        *,
        force: bool = False,
        extra_args: Sequence[str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Push with lease guarantees to the remote."""
        command: list[str] = ["git", "push", "--force-with-lease"]
        if force:
            command.append("--force")
        if extra_args:
            command.extend(extra_args)
        command.append(remote)
        if refspecs:
            command.extend(refspecs)
        return self.run(command)


__all__ = ["GitCommandError", "GitFacade"]


def _filter_runner_kwargs(
    runner: Callable[..., subprocess.CompletedProcess[str]],
    kwargs: dict[str, object],
) -> dict[str, object]:
    """Limit keyword arguments to those supported by the runner callable."""
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        return kwargs
    parameters = tuple(signature.parameters.values())
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return kwargs
    accepted = {
        parameter.name
        for parameter in parameters
        if parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    return {name: value for name, value in kwargs.items() if name in accepted}
