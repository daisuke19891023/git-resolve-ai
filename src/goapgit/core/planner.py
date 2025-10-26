"""Planner and heuristic utilities for goapgit."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from .models import ActionSpec, GoalSpec, Plan, RepoState

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from collections.abc import Iterable, Sequence


class HeuristicWeights(BaseModel):
    """Weights used to compute the heuristic score for a repository state."""

    alpha: float = Field(default=1.0, ge=0.0)
    beta: float = Field(default=1.2, ge=0.0)
    gamma: float = Field(default=0.5, ge=0.0)
    delta: float = Field(default=0.3, ge=0.0)

    model_config = {
        "frozen": True,
        "extra": "forbid",
    }


def heuristic_score(state: RepoState, weights: HeuristicWeights | None = None) -> float:
    """Calculate the heuristic score for a repository state.

    The score monotonically increases as the repository accumulates conflicts or diverges
    from its tracked branch. Each term is non-negative and scaled by the configured
    weights (alpha, beta, gamma, delta):

    - alpha * conflict_count
    - beta * conflict_difficulty
    - gamma * total_divergence (ahead + behind)
    - delta * (staleness_score + risk_bias)

    Args:
        state: The repository state to evaluate.
        weights: Optional custom weights. Defaults to :class:`HeuristicWeights`.

    Returns:
        The heuristic cost value for ``state``.

    """
    weights = weights or HeuristicWeights()

    conflict_count = float(len(state.conflicts))
    conflict_difficulty = max(state.conflict_difficulty, 0.0)
    total_divergence = float(max(state.diverged_local, 0) + max(state.diverged_remote, 0))
    staleness = max(state.staleness_score, 0.0)
    risk_bias_map = {"low": 0.0, "med": 1.0, "high": 2.0}
    risk_bias = risk_bias_map.get(state.risk_level.value, 0.0)

    return (
        weights.alpha * conflict_count
        + weights.beta * conflict_difficulty
        + weights.gamma * total_divergence
        + weights.delta * (staleness + risk_bias)
    )


class SimplePlanner:
    """Lightweight A* inspired planner that assembles a plan from injected actions."""

    _MIN_ACTIONS = 3
    _MAX_ACTIONS = 5

    def __init__(self, *, weights: HeuristicWeights | None = None) -> None:
        """Create a planner with the provided heuristic weights."""
        self._weights = weights or HeuristicWeights()

    def plan(
        self,
        start_state: RepoState,
        goal: GoalSpec,
        actions: Sequence[ActionSpec],
    ) -> Plan:
        """Create a plan from the provided action catalogue.

        The planner currently selects a bounded slice (3-5 actions) of the injected action
        catalogue and estimates the total cost as the sum of action costs plus the current
        heuristic evaluation.
        """
        if len(actions) < self._MIN_ACTIONS:
            msg = "planner requires at least three candidate actions"
            raise ValueError(msg)

        # Ensure deterministic selection: prefer the cheapest actions while capping the size.
        sorted_actions = sorted(actions, key=lambda action: action.cost)
        slice_size = min(self._MAX_ACTIONS, max(self._MIN_ACTIONS, len(sorted_actions)))
        selected = list(sorted_actions[:slice_size])

        heuristic = heuristic_score(start_state, self._weights)
        estimated_cost = heuristic + sum(action.cost for action in selected)

        notes: list[str] = [
            "heuristic_alpha_beta_gamma_delta",  # explicit marker for inspection/tests
            f"heuristic={heuristic:.2f}",
            f"goal_mode={goal.mode.value}",
        ]

        return Plan(actions=selected, estimated_cost=estimated_cost, notes=notes)

    def expand_actions(self, base_actions: Iterable[ActionSpec]) -> list[ActionSpec]:
        """Materialise actions from an iterable for deterministic planning."""
        return list(base_actions)


__all__ = [
    "HeuristicWeights",
    "SimplePlanner",
    "heuristic_score",
]
