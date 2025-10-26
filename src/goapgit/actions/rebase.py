"""Rebase related actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from goapgit.git.facade import GitCommandError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from goapgit.git.facade import GitFacade
    from goapgit.io.logging import StructuredLogger


def rebase_onto_upstream(
    facade: GitFacade,
    logger: StructuredLogger,
    upstream: str,
    *,
    update_refs: bool = False,
    onto: str | None = None,
    extra_args: Sequence[str] | None = None,
) -> None:
    """Rebase the current branch onto ``upstream`` optionally updating refs."""
    opts: list[str] = []
    dependent_branches: list[str] = []
    current_branch = _current_branch(facade)
    original_head: str | None = None

    if update_refs:
        opts.append("--update-refs")
        original_head = _rev_parse_head(facade)
        dependent_branches = _branches_containing_commit(
            facade,
            original_head,
            exclude={current_branch} if current_branch else set(),
        )
        facade.run(["git", "config", "--local", "rebase.updateRefs", "true"])

    if extra_args:
        opts.extend(extra_args)

    logger.info(
        "rebasing onto upstream",
        upstream=upstream,
        update_refs=update_refs,
        onto=onto,
        opts=opts,
        dependent_branches=dependent_branches,
    )
    facade.rebase(upstream, onto=onto, opts=opts if opts else None)

    if not update_refs or not dependent_branches or original_head is None:
        return

    new_head = _rev_parse_head(facade)
    updated: list[str] = []
    for branch in dependent_branches:
        facade.run(["git", "rebase", "--onto", new_head, original_head, branch])
        updated.append(branch)
    if current_branch:
        facade.run(["git", "checkout", current_branch])
    logger.info(
        "updated dependent branches after rebase",
        branches=updated,
        new_base=new_head,
        previous_base=original_head,
    )



def rebase_continue_or_abort(
    facade: GitFacade,
    logger: StructuredLogger,
    *,
    backup_ref: str | None = None,
) -> bool:
    """Continue an in-progress rebase, aborting on failure."""
    status = facade.run(["git", "status", "--porcelain"], check=True)
    conflicts = _extract_conflicted_paths(status.stdout)
    if conflicts:
        logger.error(
            "cannot continue rebase; conflicts remain",
            conflicted_paths=conflicts,
        )
        return False

    try:
        facade.rebase_continue()
    except GitCommandError as error:
        logger.error(
            "rebase --continue failed",
            returncode=error.returncode,
            stderr=error.stderr,
        )
        facade.rebase_abort()
        if backup_ref:
            facade.run(["git", "reset", "--hard", backup_ref])
            logger.warning("restored head from backup", backup_ref=backup_ref)
        return False

    logger.info("rebase continued successfully")
    return True


def _extract_conflicted_paths(status: str) -> list[str]:
    conflicts: list[str] = []
    for line in status.splitlines():
        if not line:
            continue
        code = line[:2]
        if "U" in code:
            conflicts.append(line[3:].strip())
    return conflicts


def _current_branch(facade: GitFacade) -> str | None:
    result = facade.run(["git", "branch", "--show-current"], check=False)
    branch = result.stdout.strip()
    return branch or None


def _rev_parse_head(facade: GitFacade) -> str:
    return facade.run(["git", "rev-parse", "HEAD"]).stdout.strip()


def _branches_containing_commit(
    facade: GitFacade,
    commit: str,
    *,
    exclude: set[str],
) -> list[str]:
    result = facade.run(
        [
            "git",
            "for-each-ref",
            "--format=%(refname:short)",
            "--contains",
            commit,
            "refs/heads",
        ],
        check=False,
    )
    if result.returncode != 0:
        return []
    branches = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return [branch for branch in branches if branch not in exclude]
