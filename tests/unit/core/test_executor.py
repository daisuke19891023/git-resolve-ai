"""Tests for the executor replanning loop."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from goapgit.core import (
    ActionSpec,
    ExecutionResult,
    Executor,
    GoalMode,
    GoalSpec,
    HeuristicWeights,
    RepoRef,
    RepoState,
    RiskLevel,
    SimplePlanner,
)
from goapgit.core.models import ConflictDetail

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from collections.abc import Callable, Iterator


@pytest.fixture
def sample_actions() -> list[ActionSpec]:
    """Provide a catalogue of representative actions."""
    return [
        ActionSpec(name="fetch", cost=1.0),
        ActionSpec(name="rebase", cost=2.0),
        ActionSpec(name="resolve", cost=3.0),
    ]


@pytest.fixture
def base_state() -> RepoState:
    """Return a clean initial repository state."""
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


@pytest.fixture
def goal() -> GoalSpec:
    """Return the default goal used for executor tests."""
    return GoalSpec(mode=GoalMode.rebase_to_upstream)


def _observer_from_states(states: list[RepoState]) -> Callable[[], RepoState]:
    """Create an observer that yields repository states sequentially."""
    iterator: Iterator[RepoState] = iter(states)

    def _observer() -> RepoState:
        try:
            return next(iterator)
        except StopIteration:  # pragma: no cover - defensive guard
            return states[-1]

    return _observer


def test_executor_completes_without_replan(
    base_state: RepoState,
    goal: GoalSpec,
    sample_actions: list[ActionSpec],
) -> None:
    """Executor finishes when actions succeed and observations remain steady."""
    planner = SimplePlanner(weights=HeuristicWeights())
    observer_states = [base_state] * len(sample_actions)

    def runner_success(action: ActionSpec) -> bool:
        del action
        return True

    executor = Executor(
        planner=planner,
        observer=_observer_from_states(observer_states),
        runner=runner_success,
        available_actions=sample_actions,
        goal=goal,
    )

    plan = planner.plan(base_state, goal, sample_actions)
    result = executor.execute(base_state, plan)

    assert isinstance(result, ExecutionResult)
    assert not result.replanned
    assert len(result.executed_actions) == len(plan.actions)


def test_executor_replans_on_action_failure(
    base_state: RepoState,
    goal: GoalSpec,
    sample_actions: list[ActionSpec],
) -> None:
    """The executor triggers replanning when an action fails."""
    planner = SimplePlanner(weights=HeuristicWeights())
    failed_state = base_state.model_copy(
        update={
            "conflicts": (
                ConflictDetail(path="file.txt", hunk_count=1, trivial_ratio=0.0),
            ),
            "conflict_difficulty": 2.0,
        },
    )

    observer_states = [failed_state]

    def runner(action: ActionSpec) -> bool:
        del action
        return False

    executor = Executor(
        planner=planner,
        observer=_observer_from_states(observer_states),
        runner=runner,
        available_actions=sample_actions,
        goal=goal,
    )

    initial_plan = planner.plan(base_state, goal, sample_actions)
    result = executor.execute(base_state, initial_plan)

    assert result.replanned
    assert len(result.executed_actions) == 0
    assert result.final_plan != initial_plan


def test_executor_replans_when_observation_diverges(
    base_state: RepoState,
    goal: GoalSpec,
    sample_actions: list[ActionSpec],
) -> None:
    """The executor requests a new plan when observations diverge."""
    planner = SimplePlanner(weights=HeuristicWeights())
    diverged_state = base_state.model_copy(update={"diverged_local": 2})
    observer_states = [diverged_state]

    def runner_success(action: ActionSpec) -> bool:
        del action
        return True

    executor = Executor(
        planner=planner,
        observer=_observer_from_states(observer_states),
        runner=runner_success,
        available_actions=sample_actions,
        goal=goal,
    )

    initial_plan = planner.plan(base_state, goal, sample_actions)
    result = executor.execute(base_state, initial_plan)

    assert result.replanned
    assert len(result.executed_actions) == 0
