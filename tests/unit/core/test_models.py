from __future__ import annotations

import pathlib

import pytest

from pydantic import ValidationError

from goapgit.core.models import (
    ActionSpec,
    ConflictDetail,
    ConflictType,
    GoalMode,
    GoalSpec,
    Plan,
    RepoRef,
    RepoState,
    RiskLevel,
)


@pytest.fixture
def sample_repo_state() -> RepoState:
    """Return a representative repository state for testing."""
    conflict = ConflictDetail(
        path="src/app.py",
        hunk_count=2,
        ctype=ConflictType.text,
        trivial_ratio=0.5,
        preferred_strategy="theirs",
    )
    return RepoState(
        repo_path=pathlib.Path("/opt/mock-repo"),
        ref=RepoRef(branch="feature", tracking="origin/feature", sha="deadbeef"),
        diverged_local=1,
        diverged_remote=2,
        working_tree_clean=False,
        staged_changes=True,
        ongoing_rebase=False,
        ongoing_merge=True,
        stash_entries=1,
        conflicts=(conflict,),
        conflict_difficulty=1.25,
        tests_last_result=True,
        has_unpushed_commits=True,
        staleness_score=0.2,
        risk_level=RiskLevel.med,
    )


@pytest.fixture
def sample_plan() -> Plan:
    """Return a sample plan populated with two actions."""
    return Plan(
        actions=[
            ActionSpec(name="Safety:CreateBackupRef", cost=1.0, rationale="Protect current state"),
            ActionSpec(name="Sync:FetchAll", params={"prune": "true"}, cost=0.5),
        ],
        estimated_cost=1.5,
        notes=["Plan generated for test"],
    )


def test_repo_state_json_round_trip(sample_repo_state: RepoState) -> None:
    """RepoState should serialize and deserialize without loss."""
    dumped = sample_repo_state.model_dump_json()
    reloaded = RepoState.model_validate_json(dumped)
    assert reloaded == sample_repo_state


def test_plan_json_round_trip(sample_plan: Plan) -> None:
    """Plan should maintain equality across JSON round trips."""
    dumped = sample_plan.model_dump_json()
    reloaded = Plan.model_validate_json(dumped)
    assert reloaded == sample_plan


def test_models_are_immutable(sample_repo_state: RepoState, sample_plan: Plan) -> None:
    """Frozen models should raise when assignments are attempted."""
    with pytest.raises((TypeError, ValidationError)):
        sample_repo_state.diverged_local = 5
    with pytest.raises((TypeError, ValidationError)):
        sample_plan.estimated_cost = 3.0


def test_goal_defaults() -> None:
    """Default goal configuration should match specification."""
    spec = GoalSpec()
    assert spec.mode is GoalMode.rebase_to_upstream
    assert spec.tests_must_pass is False
    assert spec.push_with_lease is False
