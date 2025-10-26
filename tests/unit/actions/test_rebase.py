from __future__ import annotations

import io
import subprocess
from typing import TYPE_CHECKING

import pytest

from goapgit.actions.rebase import rebase_continue_or_abort
from goapgit.git.facade import GitCommandError, GitFacade
from goapgit.io.logging import StructuredLogger


if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture
def logger() -> StructuredLogger:
    """Return a structured logger backed by an in-memory buffer."""
    return StructuredLogger(name="test", stream=io.StringIO())


def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    """Create a CompletedProcess instance with canned stdout."""
    return subprocess.CompletedProcess(["git"], returncode, stdout, "")


def test_rebase_continue_or_abort_succeeds_when_clean(
    mocker: MockerFixture,
    logger: StructuredLogger,
) -> None:
    """Rebase should continue successfully once conflicts are resolved."""
    facade = mocker.create_autospec(GitFacade, instance=True)
    facade.run.return_value = _completed()
    facade.rebase_continue.return_value = _completed()

    result = rebase_continue_or_abort(facade, logger)

    assert result is True
    facade.rebase_continue.assert_called_once_with()
    facade.rebase_abort.assert_not_called()


def test_rebase_continue_or_abort_detects_unresolved_conflicts(
    mocker: MockerFixture,
    logger: StructuredLogger,
) -> None:
    """Return False when conflicted paths remain before continuing."""
    facade = mocker.create_autospec(GitFacade, instance=True)
    facade.run.return_value = _completed("UU conflicted.json\n")

    result = rebase_continue_or_abort(facade, logger)

    assert result is False
    facade.rebase_continue.assert_not_called()
    facade.rebase_abort.assert_not_called()


def test_rebase_continue_or_abort_aborts_on_git_error(
    mocker: MockerFixture,
    logger: StructuredLogger,
) -> None:
    """Abort the rebase and restore from backup when continue fails."""
    facade = mocker.create_autospec(GitFacade, instance=True)
    facade.run.side_effect = [_completed(), _completed(returncode=0)]
    facade.rebase_continue.side_effect = GitCommandError(("git", "rebase", "--continue"), 1, "", "conflict")
    facade.rebase_abort.return_value = _completed()

    result = rebase_continue_or_abort(facade, logger, backup_ref="refs/backup/goap/123")

    assert result is False
    facade.rebase_continue.assert_called_once_with()
    facade.rebase_abort.assert_called_once_with()
    facade.run.assert_called_with(["git", "reset", "--hard", "refs/backup/goap/123"])
