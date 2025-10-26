"""Core data models for goapgit."""

from __future__ import annotations

from enum import Enum
import pathlib
import typing

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RiskLevel(str, Enum):
    """Represents the assessed risk for a repository state."""

    low = "low"
    med = "med"
    high = "high"


class ConflictType(str, Enum):
    """Known conflict categories used to tune resolution strategies."""

    text = "text"
    json = "json"
    yaml = "yaml"
    lock = "lock"
    binary = "binary"


class GoalMode(str, Enum):
    """Supported goal modes that influence planning."""

    resolve_only = "resolve_only"
    rebase_to_upstream = "rebase_to_upstream"
    push_with_lease = "push_with_lease"


class RepoRef(BaseModel):
    """Reference metadata for the active repository head."""

    branch: str
    tracking: str | None = None
    sha: str | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")


class ConflictDetail(BaseModel):
    """Describes a single conflicted path detected in the repository."""

    path: str
    hunk_count: int = 0
    ctype: ConflictType = ConflictType.text
    trivial_ratio: float = 0.0
    preferred_strategy: str | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")


class RepoState(BaseModel):
    """Snapshot of observable repository attributes relevant for planning."""

    repo_path: pathlib.Path
    ref: RepoRef
    diverged_local: int = 0
    diverged_remote: int = 0
    working_tree_clean: bool = True
    staged_changes: bool = False
    ongoing_rebase: bool = False
    ongoing_merge: bool = False
    stash_entries: int = 0
    conflicts: tuple[ConflictDetail, ...] = Field(default_factory=tuple)
    conflict_difficulty: float = 0.0
    tests_last_result: bool | None = None
    has_unpushed_commits: bool = False
    staleness_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.low

    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("repo_path")
    @classmethod
    def _coerce_repo_path(cls, value: pathlib.Path) -> pathlib.Path:
        """Ensure repo_path is materialised as a pathlib.Path instance."""
        return pathlib.Path(value)


class GoalSpec(BaseModel):
    """User provided goal configuration for the planner."""

    mode: GoalMode = GoalMode.rebase_to_upstream
    tests_must_pass: bool = False
    push_with_lease: bool = False

    model_config = ConfigDict(extra="forbid")


class ActionSpec(BaseModel):
    """Planned action that can be executed against the repository."""

    name: str
    params: typing.Mapping[str, str] | None = None
    cost: float
    rationale: str | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")


class Plan(BaseModel):
    """A full plan produced by the planner containing multiple actions."""

    actions: list[ActionSpec]
    estimated_cost: float
    notes: list[str] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True, extra="forbid")


class StrategyRule(BaseModel):
    """A file matching rule that hints how conflicts should be resolved."""

    pattern: str
    resolution: str
    when: str | None = None

    model_config = ConfigDict(extra="forbid")


def _empty_strategy_rules() -> list[StrategyRule]:
    return []


class Config(BaseModel):
    """Top level configuration schema validated from TOML files."""

    goal: GoalSpec
    strategy_rules: list[StrategyRule] = Field(default_factory=_empty_strategy_rules)
    enable_rerere: bool = True
    conflict_style: str = "zdiff3"
    allow_force_push: bool = False
    dry_run: bool = True
    max_test_runtime_sec: int = 600

    model_config = ConfigDict(extra="forbid")


__all__ = [
    "ActionSpec",
    "Config",
    "ConflictDetail",
    "ConflictType",
    "GoalMode",
    "GoalSpec",
    "Plan",
    "RepoRef",
    "RepoState",
    "RiskLevel",
    "StrategyRule",
]
