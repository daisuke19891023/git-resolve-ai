from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from goapgit.core.models import (
    ConflictDetail,
    RepoRef,
    RepoState,
    RiskLevel,
)
from goapgit.git.parse import parse_conflict_markers

if TYPE_CHECKING:
    from goapgit.git.facade import GitFacade

_STATUS_COMMAND = [
    "git",
    "status",
    "--porcelain=v2",
    "--branch",
    "--show-stash",
]

_MIN_BRANCH_AB_TOKENS = 3
_CONFLICT_FLAG = "U"


class ConflictParser(Protocol):
    """Protocol describing a callable that parses conflicted files."""

    def __call__(self, repo_path: Path, path: str) -> ConflictDetail: ...


def _empty_conflicts() -> list[ConflictDetail]:
    """Return a new list for storing conflict details."""
    return []


@dataclass(frozen=True)
class StatusSummary:
    """Summarised porcelain information used to build RepoState."""

    branch: str
    tracking: str | None
    sha: str | None
    ahead: int
    behind: int
    staged_changes: bool
    working_tree_dirty: bool
    untracked_present: bool
    ongoing_rebase: bool
    ongoing_merge: bool
    stash_entries: int
    conflicts: tuple[ConflictDetail, ...]


class RepoObserver:
    """Observe git repository state using porcelain v2 output."""

    def __init__(
        self,
        facade: GitFacade,
        *,
        conflict_parser: ConflictParser | None = None,
    ) -> None:
        """Initialise the observer with a facade and conflict parser."""
        self._facade = facade
        self._repo_path = Path(facade.repo_path)
        self._conflict_parser = conflict_parser or parse_conflict_markers

    def observe(self) -> RepoState:
        """Return a RepoState reflecting the repository status."""
        result = self._facade.run(_STATUS_COMMAND, capture_output=True)
        stdout = result.stdout or ""
        summary = _parse_porcelain(self._repo_path, stdout.splitlines(), self._conflict_parser)
        conflicts = summary.conflicts
        working_tree_clean = not (
            summary.staged_changes or summary.working_tree_dirty or summary.untracked_present
        )
        has_unpushed = summary.ahead > 0
        if conflicts:
            risk = RiskLevel.high
        elif summary.staged_changes or summary.working_tree_dirty:
            risk = RiskLevel.med
        else:
            risk = RiskLevel.low
        return RepoState(
            repo_path=self._repo_path,
            ref=RepoRef(branch=summary.branch, tracking=summary.tracking, sha=summary.sha),
            diverged_local=summary.ahead,
            diverged_remote=summary.behind,
            working_tree_clean=working_tree_clean,
            staged_changes=summary.staged_changes,
            ongoing_rebase=summary.ongoing_rebase,
            ongoing_merge=summary.ongoing_merge,
            stash_entries=summary.stash_entries,
            conflicts=conflicts,
            conflict_difficulty=float(sum(detail.hunk_count for detail in conflicts)),
            tests_last_result=None,
            has_unpushed_commits=has_unpushed,
            staleness_score=float(summary.behind),
            risk_level=risk,
        )


@dataclass
class _PorcelainAccumulator:
    branch: str = "HEAD"
    tracking: str | None = None
    sha: str | None = None
    ahead: int = 0
    behind: int = 0
    staged_changes: bool = False
    working_tree_dirty: bool = False
    untracked_present: bool = False
    ongoing_rebase: bool = False
    ongoing_merge: bool = False
    stash_entries: int = 0
    conflicts: list[ConflictDetail] = field(default_factory=_empty_conflicts)

    def handle_header(self, header: str) -> None:
        if header.startswith("branch.head "):
            self.branch = header.split(" ", 1)[1]
        elif header.startswith("branch.upstream "):
            self.tracking = header.split(" ", 1)[1]
        elif header.startswith("branch.oid "):
            sha_value = header.split(" ", 1)[1]
            self.sha = None if sha_value == "(initial)" else sha_value
        elif header.startswith("branch.ab "):
            tokens = header.split()
            if len(tokens) >= _MIN_BRANCH_AB_TOKENS:
                self.ahead = int(tokens[1].lstrip("+"))
                self.behind = int(tokens[2].lstrip("-"))
        elif header.startswith("stash "):
            try:
                self.stash_entries = int(header.split(" ", 1)[1])
            except ValueError:
                self.stash_entries = 0
        elif header.startswith("rebase"):
            self.ongoing_rebase = True
        elif header.startswith("merge"):
            self.ongoing_merge = True

    def handle_entry(
        self,
        repo_path: Path,
        line: str,
        conflict_parser: ConflictParser,
    ) -> None:
        prefix = line[0]
        if prefix in {"1", "2"}:
            self._handle_tracked_entry(repo_path, line, conflict_parser)
        elif prefix == "u":
            self._handle_unmerged_entry(repo_path, line, conflict_parser)
        elif prefix == "?":
            self.untracked_present = True
        elif prefix == "!":
            return
        else:
            self.working_tree_dirty = True

    def _handle_tracked_entry(
        self,
        repo_path: Path,
        line: str,
        conflict_parser: ConflictParser,
    ) -> None:
        meta, _, remainder = line.partition("\t")
        meta_parts = meta.split()
        status = meta_parts[1] if len(meta_parts) > 1 else ""
        staged_code = status[0] if status else "."
        worktree_code = status[1] if len(status) > 1 else "."
        if staged_code != ".":
            self.staged_changes = True
        if worktree_code != ".":
            self.working_tree_dirty = True
        path = remainder.split("\0", 1)[0] if remainder else meta_parts[-1]
        if _CONFLICT_FLAG in {staged_code, worktree_code} or status[:2] in {"DD", "AA"}:
            self.conflicts.append(conflict_parser(repo_path, path))

    def _handle_unmerged_entry(
        self,
        repo_path: Path,
        line: str,
        conflict_parser: ConflictParser,
    ) -> None:
        self.working_tree_dirty = True
        _, _, remainder = line.partition("\t")
        path = remainder.split("\0", 1)[0] if remainder else line.split(" ")[-1]
        self.conflicts.append(conflict_parser(repo_path, path))

    def to_summary(self) -> StatusSummary:
        return StatusSummary(
            branch=self.branch,
            tracking=self.tracking,
            sha=self.sha,
            ahead=self.ahead,
            behind=self.behind,
            staged_changes=self.staged_changes,
            working_tree_dirty=self.working_tree_dirty,
            untracked_present=self.untracked_present,
            ongoing_rebase=self.ongoing_rebase,
            ongoing_merge=self.ongoing_merge,
            stash_entries=self.stash_entries,
            conflicts=tuple(self.conflicts),
        )


def _parse_porcelain(
    repo_path: Path,
    lines: list[str],
    conflict_parser: ConflictParser,
) -> StatusSummary:
    accumulator = _PorcelainAccumulator()
    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if not line:
            continue
        if line.startswith("# "):
            accumulator.handle_header(line[2:])
        else:
            accumulator.handle_entry(repo_path, line, conflict_parser)
    return accumulator.to_summary()


__all__ = ["RepoObserver", "StatusSummary"]
