"""Conflict resolution helpers leveraging git rerere and strategy rules."""

from __future__ import annotations

import fnmatch
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from goapgit.git.facade import GitFacade
    from goapgit.io.logging import StructuredLogger


class ConflictLike(Protocol):
    """Protocol describing the minimal conflict information required."""

    path: str


class StrategyRuleLike(Protocol):
    """Protocol describing resolution rules for conflicted paths."""

    pattern: str
    resolution: str
    when: str | None


def auto_trivial_resolve(facade: GitFacade, logger: StructuredLogger) -> bool:
    """Run ``git rerere`` when the feature is enabled to reuse known merges."""
    enabled_result = facade.run(
        ["git", "config", "--bool", "rerere.enabled"],
        check=False,
    )
    enabled = enabled_result.returncode == 0 and enabled_result.stdout.strip().lower() == "true"
    if not enabled:
        logger.info("rerere disabled; skipping auto resolution", returncode=enabled_result.returncode)
        return False

    facade.run(["git", "rerere"])
    staged: list[str] = []
    status = facade.run(["git", "status", "--porcelain"])
    for line in status.stdout.splitlines():
        if not line:
            continue
        code = line[:2]
        path = line[3:].strip()
        if "U" in code and path:
            facade.run(["git", "add", "--", path])
            staged.append(path)
    logger.info("applied rerere resolutions", staged_paths=staged)
    return True


def apply_path_strategy(
    facade: GitFacade,
    logger: StructuredLogger,
    conflicts: Iterable[ConflictLike],
    rules: Sequence[StrategyRuleLike],
) -> list[str]:
    """Apply configured path strategies for conflicts and return resolved paths."""
    resolved: list[str] = []
    for conflict in conflicts:
        rule = _select_rule(conflict.path, rules, facade)
        if rule is None:
            continue
        if rule.resolution == "theirs":
            facade.run(["git", "checkout", "--theirs", "--", conflict.path])
        elif rule.resolution == "ours":
            facade.run(["git", "checkout", "--ours", "--", conflict.path])
        else:
            logger.warning(
                "unsupported resolution strategy",
                rule={"pattern": rule.pattern, "resolution": rule.resolution, "when": rule.when},
            )
            continue
        facade.run(["git", "add", "--", conflict.path])
        logger.info(
            "applied path strategy",
            path=conflict.path,
            resolution=rule.resolution,
            when=rule.when,
        )
        resolved.append(conflict.path)
    return resolved


def _select_rule(
    path: str,
    rules: Sequence[StrategyRuleLike],
    facade: GitFacade,
) -> StrategyRuleLike | None:
    normalized = PurePosixPath(path).as_posix()
    for rule in rules:
        if not _matches_pattern(normalized, rule.pattern):
            continue
        if rule.when == "whitespace_only" and not _is_whitespace_only(facade, path):
            continue
        return rule
    return None


def _matches_pattern(path: str, pattern: str) -> bool:
    candidates = [pattern]
    if pattern.startswith("**/"):
        candidates.append(pattern[3:])
    return any(fnmatch.fnmatchcase(path, candidate) for candidate in candidates)


def _is_whitespace_only(facade: GitFacade, path: str) -> bool:
    ours = facade.run(["git", "show", f":2:{path}"], check=False)
    theirs = facade.run(["git", "show", f":3:{path}"], check=False)
    if ours.returncode != 0 or theirs.returncode != 0:
        full_diff = facade.run(["git", "diff", "--", path])
        if not full_diff.stdout.strip():
            return False
        whitespace_diff = facade.run(["git", "diff", "-w", "--", path])
        return not whitespace_diff.stdout.strip()
    return _strip_whitespace(ours.stdout) == _strip_whitespace(theirs.stdout)


def _strip_whitespace(text: str) -> str:
    return "".join(text.split())
