"""Safety-focused git actions like creating backups and stashing work."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from goapgit.git.facade import GitFacade
    from goapgit.io.logging import StructuredLogger


_BACKUP_PREFIX = "refs/backup/goap"
_STASH_PREFIX = "goap"


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def create_backup_ref(facade: GitFacade, logger: StructuredLogger) -> str:
    """Create a backup ref pointing at ``HEAD`` and return its full name."""
    head_result = facade.run(["git", "rev-parse", "HEAD"])
    head_sha = head_result.stdout.strip()
    ref_name = f"{_BACKUP_PREFIX}/{_timestamp()}"
    facade.run(["git", "update-ref", ref_name, head_sha])
    logger.info("created backup ref", ref=ref_name, sha=head_sha)
    return ref_name


def ensure_clean_or_stash(facade: GitFacade, logger: StructuredLogger) -> bool:
    """Create a stash with a timestamped label when the worktree is dirty."""
    status = facade.run(["git", "status", "--porcelain"], check=True)
    if not status.stdout.strip():
        logger.info("working tree already clean; no stash required")
        return False

    label = f"{_STASH_PREFIX}/{_timestamp()}"
    facade.run(["git", "stash", "push", "--include-untracked", "-m", label])
    logger.info("created stash for dirty worktree", label=label)
    return True
