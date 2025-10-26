from __future__ import annotations

import io
from typing import TYPE_CHECKING

from goapgit.actions.sync import fetch_all
from goapgit.git.facade import GitFacade
from goapgit.io.logging import StructuredLogger

if TYPE_CHECKING:
    from pathlib import Path


def test_fetch_all_logs_command(tmp_path: Path) -> None:
    """Fetching should log the expected git command."""
    buffer = io.StringIO()
    logger = StructuredLogger(name="test", stream=buffer)
    facade = GitFacade(repo_path=tmp_path, logger=logger, dry_run=True)

    fetch_all(facade, logger)

    log_output = buffer.getvalue()
    assert "git" in log_output
    assert "fetch" in log_output
    assert "--prune" in log_output
    assert "--tags" in log_output
    history = facade.command_history
    assert history
    assert history[0]["command"] == ["git", "fetch", "--prune", "--tags", "origin"]
