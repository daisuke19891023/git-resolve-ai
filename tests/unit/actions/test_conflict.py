from __future__ import annotations

import io
import subprocess
from dataclasses import dataclass
from unittest.mock import call

import pytest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

from goapgit.actions.conflict import apply_path_strategy, auto_trivial_resolve
from goapgit.git.facade import GitFacade
from goapgit.io.logging import StructuredLogger


@pytest.fixture
def logger() -> StructuredLogger:
    """Return a structured logger backed by an in-memory stream."""
    return StructuredLogger(name="test", stream=io.StringIO())


def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    """Create a CompletedProcess instance mirroring git output."""
    return subprocess.CompletedProcess(["git"], returncode, stdout, "")


@dataclass
class Conflict:
    """Simple conflict stub used to expose path handling."""

    path: str


@dataclass
class StrategyRule:
    """Simple rule stub defining resolution behaviour for tests."""

    pattern: str
    resolution: str
    when: str | None = None


def test_auto_trivial_resolve_adds_separator_for_hyphen_prefixed_paths(
    mocker: MockerFixture,
    logger: StructuredLogger,
) -> None:
    """Ensure git add is invoked with ``--`` before hyphenated paths."""
    facade = mocker.create_autospec(GitFacade, instance=True)

    def run_side_effect(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        command = tuple(args)
        if command == ("git", "config", "--bool", "rerere.enabled"):
            return _completed("true\n")
        if command == ("git", "rerere"):
            return _completed()
        if command == ("git", "status", "--porcelain"):
            return _completed("UU -leading\n")
        if command == ("git", "add", "--", "-leading"):
            return _completed()
        message = f"Unexpected git command: {command}"
        raise AssertionError(message)

    facade.run.side_effect = run_side_effect

    result = auto_trivial_resolve(facade, logger)

    assert result is True
    assert call(["git", "add", "--", "-leading"]) in facade.run.call_args_list


@pytest.mark.parametrize(
    ("resolution", "checkout_flag"),
    [("theirs", "--theirs"), ("ours", "--ours")],
)
def test_apply_path_strategy_prefixes_separator_before_conflict_path(
    mocker: MockerFixture,
    logger: StructuredLogger,
    resolution: str,
    checkout_flag: str,
) -> None:
    """Ensure checkout and add commands pass ``--`` before conflicted paths."""
    facade = mocker.create_autospec(GitFacade, instance=True)
    facade.run.return_value = _completed()
    conflict = Conflict(path="-leading")
    rule = StrategyRule(pattern="*", resolution=resolution)

    resolved = apply_path_strategy(facade, logger, [conflict], [rule])

    assert resolved == ["-leading"]
    assert call(["git", "checkout", checkout_flag, "--", "-leading"]) in facade.run.call_args_list
    assert call(["git", "add", "--", "-leading"]) in facade.run.call_args_list
