"""Synchronisation related git actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from goapgit.git.facade import GitFacade
    from goapgit.io.logging import StructuredLogger


_DEFAULT_REMOTE = "origin"


def fetch_all(
    facade: GitFacade,
    logger: StructuredLogger,
    *,
    remote: str = _DEFAULT_REMOTE,
    extra_args: Sequence[str] | None = None,
) -> None:
    """Fetch all refs from ``remote`` while logging the command."""
    command = ["git", "fetch", "--prune", "--tags", remote]
    if extra_args:
        command[2:2] = list(extra_args)
    logger.info("fetching all remotes", command=command, remote=remote)
    facade.fetch(remote=remote, prune=True, tags=True, extra_args=extra_args)


def push_with_lease(
    facade: GitFacade,
    logger: StructuredLogger,
    *,
    remote: str = _DEFAULT_REMOTE,
    refspecs: Sequence[str] | None = None,
    force: bool = False,
    extra_args: Sequence[str] | None = None,
) -> None:
    """Push the current branch using force-with-lease semantics."""
    logger.info(
        "pushing with lease",
        remote=remote,
        refspecs=list(refspecs) if refspecs else None,
        force=force,
        extra_args=list(extra_args) if extra_args else None,
    )
    facade.push_with_lease(
        remote=remote,
        refspecs=refspecs,
        force=force,
        extra_args=extra_args,
    )
