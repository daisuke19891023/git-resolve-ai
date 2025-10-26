"""Diagnostics for repository and git configuration health."""

from __future__ import annotations

import io
import json
import os
from typing import TYPE_CHECKING

from goapgit.git.facade import GitCommandError, GitFacade
from goapgit.io.logging import StructuredLogger
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

__all__ = [
    "DiagnoseError",
    "DiagnoseReport",
    "GitConfigCheck",
    "LargeRepoGuidance",
    "RepoStats",
    "generate_diagnosis",
    "report_to_json",
]

# Thresholds chosen to indicate when the working tree or history may become unwieldy.
TRACKED_FILE_THRESHOLD = 100_000
SIZE_PACK_THRESHOLD_KIB = 1_000_000  # ~= 1 GiB
COMMIT_COUNT_THRESHOLD = 50_000


class DiagnoseError(RuntimeError):
    """Raised when diagnosis cannot be completed."""


class GitConfigCheck(BaseModel):
    """Represents the state of a git configuration key."""

    key: str
    recommended: str
    detected: str | None = None
    matches_recommendation: bool

    model_config = ConfigDict(extra="forbid", frozen=True)


class RepoStats(BaseModel):
    """Aggregated repository statistics used for guidance."""

    tracked_files: int | None = None
    size_pack_kib: int | None = None
    size_loose_kib: int | None = None
    commit_count: int | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class LargeRepoGuidance(BaseModel):
    """Advice for handling large repositories."""

    triggered: bool
    reasons: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", frozen=True)


class DiagnoseReport(BaseModel):
    """Full diagnosis report."""

    git_config: list[GitConfigCheck]
    repo_stats: RepoStats | None
    large_repo_guidance: LargeRepoGuidance

    model_config = ConfigDict(extra="forbid", frozen=True)


_RECOMMENDED_SETTINGS: tuple[tuple[str, str], ...] = (
    ("merge.conflictStyle", "zdiff3"),
    ("rerere.enabled", "true"),
    ("pull.rebase", "true"),
)


def generate_diagnosis(repo_path: Path, *, env: Mapping[str, str] | None = None) -> DiagnoseReport:
    """Collect git configuration status and repository statistics."""
    env_vars = _prepare_env(env)
    facade = _create_facade(repo_path, env_vars)
    config_checks = [_check_setting(key, expected, facade=facade) for key, expected in _RECOMMENDED_SETTINGS]
    repo_stats = _gather_repo_stats(facade)
    guidance = _build_guidance(repo_stats)
    return DiagnoseReport(git_config=config_checks, repo_stats=repo_stats, large_repo_guidance=guidance)


def report_to_json(report: DiagnoseReport, *, pretty: bool = False) -> str:
    """Serialise the report to JSON."""
    payload = report.model_dump(mode="json")
    if pretty:
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    return json.dumps(payload, ensure_ascii=False)


def _check_setting(key: str, expected: str, *, facade: GitFacade) -> GitConfigCheck:
    """Check a git configuration key against the expected value."""
    detected = _run_git_config(key, facade=facade)
    matches = detected is not None and detected.lower() == expected.lower()
    return GitConfigCheck(key=key, recommended=expected, detected=detected, matches_recommendation=matches)


def _gather_repo_stats(facade: GitFacade) -> RepoStats | None:
    """Collect repository statistics for large repo guidance."""
    try:
        count_output = facade.run(("git", "count-objects", "-v"))
    except GitCommandError:
        return None
    stats = _parse_count_objects(count_output.stdout)
    tracked_files = _count_tracked_files(facade)
    commit_count = _count_commits(facade)
    size_pack = stats.get("size-pack")
    size_loose = stats.get("size")
    return RepoStats(
        tracked_files=tracked_files,
        size_pack_kib=size_pack,
        size_loose_kib=size_loose,
        commit_count=commit_count,
    )


def _build_guidance(stats: RepoStats | None) -> LargeRepoGuidance:
    """Build guidance for handling large repositories."""
    if stats is None:
        return LargeRepoGuidance(triggered=False)
    reasons: list[str] = []
    if stats.tracked_files is not None and stats.tracked_files >= TRACKED_FILE_THRESHOLD:
        reasons.append(
            f"tracked_files {stats.tracked_files} exceeds threshold {TRACKED_FILE_THRESHOLD}",
        )
    if stats.size_pack_kib is not None and stats.size_pack_kib >= SIZE_PACK_THRESHOLD_KIB:
        reasons.append(
            f"size_pack_kib {stats.size_pack_kib} exceeds threshold {SIZE_PACK_THRESHOLD_KIB}",
        )
    if stats.commit_count is not None and stats.commit_count >= COMMIT_COUNT_THRESHOLD:
        reasons.append(
            f"commit_count {stats.commit_count} exceeds threshold {COMMIT_COUNT_THRESHOLD}",
        )
    triggered = bool(reasons)
    suggestions: list[str] = []
    if triggered:
        suggestions.append(
            "Repository is large; consider using 'git sparse-checkout' to focus on required paths.",
        )
        suggestions.append(
            "Leverage 'git worktree add' to create focused working directories without duplicating the full clone.",
        )
    return LargeRepoGuidance(triggered=triggered, reasons=reasons, suggestions=suggestions)


def _count_tracked_files(facade: GitFacade) -> int | None:
    """Return the number of tracked files in the repository."""
    try:
        completed = facade.run(("git", "ls-files", "-z"))
    except GitCommandError:
        return None
    if not completed.stdout:
        return 0
    return completed.stdout.count("\0")


def _count_commits(facade: GitFacade) -> int | None:
    """Return the number of commits reachable from HEAD."""
    try:
        completed = facade.run(("git", "rev-list", "--count", "HEAD"))
    except GitCommandError:
        return None
    output = completed.stdout.strip()
    if not output:
        return None
    try:
        return int(output)
    except ValueError:
        return None


def _parse_count_objects(output: str) -> dict[str, int]:
    """Parse the `git count-objects -v` output into integers."""
    stats: dict[str, int] = {}
    for line in output.splitlines():
        key, _, value = line.partition(":")
        if not value:
            continue
        key = key.strip()
        value = value.strip()
        try:
            stats[key] = int(value)
        except ValueError:
            continue
    return stats


def _run_git_config(key: str, *, facade: GitFacade) -> str | None:
    """Read a git configuration value from the global scope."""
    try:
        completed = facade.run(("git", "config", "--global", "--get", key))
    except GitCommandError:
        return None
    value = completed.stdout.strip()
    return value or None


def _create_facade(repo_path: Path, env: Mapping[str, str]) -> GitFacade:
    """Construct a git facade that silences diagnostic logging."""
    logger = StructuredLogger(name="goapgit.diagnose", json_mode=False, stream=io.StringIO())
    return GitFacade(repo_path=repo_path, logger=logger, env=env)


def _prepare_env(env: Mapping[str, str] | None) -> dict[str, str]:
    """Prepare environment variables for subprocess execution."""
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return merged
