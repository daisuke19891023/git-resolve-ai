"""Parsing utilities for git output and conflict markers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from goapgit.core.models import ConflictDetail, ConflictType

if TYPE_CHECKING:
    from goapgit.git.facade import GitFacade


LOGGER = logging.getLogger(__name__)


def parse_conflict_markers(repo_path: Path, path: str) -> ConflictDetail:
    """Inspect a conflicted file and estimate hunk count and conflict type."""
    root = Path(repo_path).resolve()
    requested_path = root / path
    conflict_type = _detect_conflict_type(path)

    try:
        resolved_path = requested_path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        LOGGER.warning(
            "Skipping conflict marker parsing for path %s due to resolution error: %s",
            path,
            exc,
        )
        return ConflictDetail(path=path, hunk_count=0, ctype=conflict_type)

    if not _is_path_within_repository(root, resolved_path):
        LOGGER.warning(
            "Skipping conflict marker parsing for path outside repository: %s",
            path,
        )
        return ConflictDetail(path=path, hunk_count=0, ctype=conflict_type)

    if _path_contains_symlink(requested_path, root):
        LOGGER.warning(
            "Skipping conflict marker parsing for symlinked path: %s",
            path,
        )
        return ConflictDetail(path=path, hunk_count=0, ctype=conflict_type)

    hunk_count = 0
    try:
        with resolved_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("<<<<<<<"):
                    hunk_count += 1
    except OSError as exc:
        LOGGER.warning(
            "Failed to read conflict markers from %s: %s",
            path,
            exc,
        )
        hunk_count = 0

    return ConflictDetail(path=path, hunk_count=hunk_count, ctype=conflict_type)


def _is_path_within_repository(root: Path, resolved_path: Path) -> bool:
    """Return True when the resolved path stays under the repository root."""
    try:
        return resolved_path.is_relative_to(root)
    except ValueError:  # pragma: no cover - defensive, not expected on Python >=3.9
        return False


def _path_contains_symlink(target: Path, root: Path) -> bool:
    """Return True if any component from target up to root is a symlink."""
    current = target
    while True:
        if current.is_symlink():
            return True
        if current == root:
            return False
        if not current.is_relative_to(root):
            return False
        parent = current.parent
        if parent == current:
            return False
        current = parent


def _detect_conflict_type(path: str) -> ConflictType:
    lowered = path.lower()
    if lowered.endswith(".json"):
        return ConflictType.json
    if lowered.endswith((".yaml", ".yml")):
        return ConflictType.yaml
    if lowered.endswith(".lock"):
        return ConflictType.lock
    return ConflictType.text


def parse_merge_tree_conflicts(output: str) -> set[str]:
    """Parse `git merge-tree --write-tree` output into a conflict set."""
    conflicts: set[str] = set()
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("CONFLICT") and " in " in stripped:
            _, path = stripped.rsplit(" in ", 1)
            conflicts.add(path.strip())
    return conflicts


def predict_merge_conflicts(
    facade: GitFacade, ours: str, theirs: str,
) -> set[str]:
    """Run git merge-tree and predict the conflicted paths."""
    result = facade.run(
        ["git", "merge-tree", "--write-tree", ours, theirs],
        capture_output=True,
        check=False,
    )
    stdout = result.stdout or ""
    return parse_merge_tree_conflicts(stdout)


__all__ = [
    "parse_conflict_markers",
    "parse_merge_tree_conflicts",
    "predict_merge_conflicts",
]
