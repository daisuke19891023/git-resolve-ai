"""Utilities for explaining action plans to users."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ActionSpec, Plan


@dataclass(frozen=True, slots=True)
class ActionContext:
    """Additional metadata describing why an action was selected."""

    reason: str
    alternatives: tuple[str, ...] = ()
    cost_override: float | None = None


@dataclass(frozen=True, slots=True)
class ActionExplanation:
    """Human readable explanation for an action within a plan."""

    action: ActionSpec
    reason: str
    alternatives: tuple[str, ...]
    cost: float


def explain_plan(
    plan: Plan,
    *,
    contexts: dict[str, ActionContext] | None = None,
) -> list[ActionExplanation]:
    """Generate explanations for each action within ``plan``.

    Args:
        plan: The plan to explain.
        contexts: Optional mapping providing additional metadata per action name.

    Returns:
        A list of :class:`ActionExplanation` entries mirroring the order of
        actions within ``plan``.

    """
    context_map = contexts or {}
    explanations: list[ActionExplanation] = []

    for action in plan.actions:
        context = context_map.get(action.name)
        reason = context.reason if context else action.rationale or "No rationale provided."
        alternatives = (
            tuple(context.alternatives)
            if context and context.alternatives
            else ()
        )
        cost = context.cost_override if context and context.cost_override is not None else action.cost
        explanations.append(
            ActionExplanation(
                action=action,
                reason=reason,
                alternatives=alternatives,
                cost=cost,
            ),
        )

    return explanations


__all__ = ["ActionContext", "ActionExplanation", "explain_plan"]

