"""Execution loop for running plans with observation and replanning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from collections.abc import Sequence

    from .planner import SimplePlanner
    from .models import ActionSpec, GoalSpec, Plan, RepoState


class ActionRunner(Protocol):
    """Callable protocol used to execute a single action."""

    def __call__(self, action: ActionSpec) -> bool:
        """Run ``action`` and return ``True`` when it succeeded."""
        ...


class StateObserver(Protocol):
    """Callable protocol returning the latest repository state."""

    def __call__(self) -> RepoState:
        """Collect the most recent :class:`RepoState`."""
        ...


@dataclass(slots=True, frozen=True)
class ExecutionResult:
    """Result information for a single executor run."""

    final_plan: Plan
    executed_actions: list[ActionSpec]
    replanned: bool = False


class Executor:
    """Execute plans action-by-action while observing and replanning when required."""

    def __init__(
        self,
        *,
        planner: SimplePlanner,
        observer: StateObserver,
        runner: ActionRunner,
        available_actions: Sequence[ActionSpec],
        goal: GoalSpec,
    ) -> None:
        """Initialise the executor with the injected collaborators."""
        self._planner = planner
        self._observer = observer
        self._runner = runner
        self._available_actions = list(available_actions)
        self._goal = goal

    def execute(
        self,
        initial_state: RepoState,
        plan: Plan | None = None,
    ) -> ExecutionResult:
        """Execute ``plan`` and replan when the observed state diverges."""
        current_plan = plan or self._planner.plan(initial_state, self._goal, self._available_actions)
        executed: list[ActionSpec] = []
        previous_state = initial_state

        for action in current_plan.actions:
            success = self._runner(action)
            observed_state = self._observer()

            if not success or self._needs_replan(previous_state, observed_state):
                new_plan = self._planner.plan(observed_state, self._goal, self._available_actions)
                return ExecutionResult(final_plan=new_plan, executed_actions=executed, replanned=True)

            executed.append(action)
            previous_state = observed_state

        return ExecutionResult(final_plan=current_plan, executed_actions=executed, replanned=False)

    def _needs_replan(self, previous: RepoState, observed: RepoState) -> bool:
        """Return ``True`` when a new plan should be generated."""
        return any(
            (
                len(previous.conflicts) != len(observed.conflicts),
                previous.diverged_local != observed.diverged_local,
                previous.diverged_remote != observed.diverged_remote,
                previous.risk_level != observed.risk_level,
            ),
        )


__all__ = ["ActionRunner", "ExecutionResult", "Executor", "StateObserver"]
