"""Tests for the planner and heuristic utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from goapgit.core import (
    ActionSpec,
    GoalMode,
    GoalSpec,
    HeuristicWeights,
    RepoRef,
    RepoState,
    RiskLevel,
    SimplePlanner,
    heuristic_score,
)
from goapgit.core.models import ConflictDetail


@pytest.fixture
def base_state() -> RepoState:
    """Return a clean baseline repository state for planner tests."""
    return RepoState(
        repo_path=Path("/repo"),
        ref=RepoRef(branch="main"),
        conflicts=(),
        conflict_difficulty=0.0,
        diverged_local=0,
        diverged_remote=0,
        staleness_score=0.0,
        risk_level=RiskLevel.low,
    )


def test_heuristic_increases_with_conflicts_and_divergence(base_state: RepoState) -> None:
    """Ensure the heuristic grows monotonically with increasing risk factors."""
    weights = HeuristicWeights(alpha=1.0, beta=1.0, gamma=1.0, delta=1.0)

    state_with_conflict = base_state.model_copy(
        update={
            "conflicts": (
                ConflictDetail(path="file.txt", hunk_count=2, trivial_ratio=0.1),
            ),
            "conflict_difficulty": 3.0,
        },
    )
    state_with_divergence = state_with_conflict.model_copy(
        update={
            "diverged_local": 2,
            "diverged_remote": 1,
        },
    )
    state_high_risk = state_with_divergence.model_copy(update={"risk_level": RiskLevel.high})

    baseline = heuristic_score(base_state, weights)
    conflict_score = heuristic_score(state_with_conflict, weights)
    divergence_score = heuristic_score(state_with_divergence, weights)
    high_risk_score = heuristic_score(state_high_risk, weights)

    assert baseline < conflict_score < divergence_score < high_risk_score


def test_planner_selects_expected_actions(base_state: RepoState) -> None:
    """Validate that the planner selects the cheapest actions and sums their costs."""
    weights = HeuristicWeights(alpha=0.5, beta=0.5, gamma=0.5, delta=0.5)
    planner = SimplePlanner(weights=weights)
    goal = GoalSpec(mode=GoalMode.resolve_only)
    actions = [
        ActionSpec(name="action_1", cost=5.0),
        ActionSpec(name="action_2", cost=1.0),
        ActionSpec(name="action_3", cost=3.0),
        ActionSpec(name="action_4", cost=2.0),
    ]

    plan = planner.plan(base_state, goal, actions)

    assert 3 <= len(plan.actions) <= 5
    # Costs should be sorted ascending, so the first action has the smallest cost
    costs = [action.cost for action in plan.actions]
    assert costs == sorted(costs)

    expected_sum = sum(costs)
    heuristic = heuristic_score(base_state, weights)
    assert abs(plan.estimated_cost - (expected_sum + heuristic)) < 1e-6
    assert any(note.startswith("heuristic=") for note in plan.notes)


def test_planner_requires_minimum_actions(base_state: RepoState) -> None:
    """Raise an error when insufficient actions are provided."""
    planner = SimplePlanner()
    goal = GoalSpec(mode=GoalMode.resolve_only)
    actions = [ActionSpec(name="action_only", cost=2.0)]

    with pytest.raises(ValueError, match="at least three candidate actions"):
        planner.plan(base_state, goal, actions)
