"""Plan hint helpers using the Responses API."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING

from .instructions import planner_instructions
from .responses import CompleteJsonResult, ResponsesClient, complete_json
from .schema import PlanHint, sanitize_model_schema

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from collections.abc import Iterable
    from goapgit.core.models import ActionSpec, Plan, RepoState
    from goapgit.io.logging import StructuredLogger

__all__ = [
    "PlanHintResult",
    "apply_plan_hint",
    "build_plan_prompt",
    "clamp_cost_adjustment",
    "request_plan_hint",
]


@dataclass(frozen=True, slots=True)
class PlanHintResult:
    """Structured output returned from the planner hint endpoint."""

    hint: PlanHint
    response_id: str
    output_text: str


def build_plan_prompt(state: RepoState, plan: Plan) -> str:
    """Create the prompt describing the current plan for evaluation."""
    action_lines = _format_action_lines(plan.actions)
    conflict_count = len(state.conflicts)
    divergence = state.diverged_local + state.diverged_remote

    return dedent(
        f"""
        Evaluate the following GOAP plan for repository: {state.repo_path}

        Branch: {state.ref.branch}
        Conflicts detected: {conflict_count}
        Total divergence (ahead+behind): {divergence}
        Current estimated cost: {plan.estimated_cost:.2f}

        Planned actions:
        {action_lines}

        Suggest an alternative action sequence when beneficial and provide
        a cost adjustment percentage in the range [-20%, +20%]. Explain any
        risks or prerequisites in the optional note field.
        """,
    ).strip()


def request_plan_hint(
    client: ResponsesClient,
    *,
    model: str,
    state: RepoState,
    plan: Plan,
    previous_response_id: str | None = None,
    extra_rules: Iterable[str] | None = None,
) -> PlanHintResult:
    """Request a plan hint and cost adjustment from the Responses API."""
    instructions = planner_instructions(extra_rules=tuple(extra_rules or ()))
    schema = sanitize_model_schema(PlanHint)
    prompt = build_plan_prompt(state, plan)

    result: CompleteJsonResult = complete_json(
        client,
        model=model,
        instructions=instructions,
        schema=schema,
        prompt=prompt,
        previous_response_id=previous_response_id,
        schema_name="goapgit_plan_hint",
    )

    hint = PlanHint.model_validate(result.payload)
    return PlanHintResult(hint=hint, response_id=result.response_id, output_text=result.output_text)


def clamp_cost_adjustment(value: float, *, minimum: float = -0.2, maximum: float = 0.2) -> float:
    """Clamp ``value`` between the configured minimum and maximum bounds."""
    if minimum > maximum:
        msg = "minimum must not exceed maximum"
        raise ValueError(msg)

    return min(max(value, minimum), maximum)


def apply_plan_hint(plan: Plan, hint: PlanHint, *, logger: StructuredLogger) -> Plan:
    """Return a new plan whose estimated cost reflects the provided hint."""
    clamped_pct = clamp_cost_adjustment(hint.cost_adjustment_pct)
    adjusted_cost = plan.estimated_cost * (1.0 + clamped_pct)

    logger.info(
        "plan hint applied",
        base_cost=plan.estimated_cost,
        cost_adjustment_pct=clamped_pct,
        adjusted_cost=adjusted_cost,
        hint_action=hint.action,
        hint_note=hint.note,
    )

    notes: list[str] = list(plan.notes)
    notes.append(f"plan_hint_action={hint.action}")
    notes.append(f"cost_adjustment_pct={clamped_pct:.3f}")
    if hint.note:
        notes.append(f"plan_hint_note={hint.note}")

    return plan.model_copy(update={"estimated_cost": adjusted_cost, "notes": notes})


def _format_action_lines(actions: Iterable[ActionSpec]) -> str:
    lines: list[str] = []
    for action in actions:
        description = action.rationale or "No rationale provided"
        lines.append(f"- {action.name} (cost={action.cost:.2f}): {description}")
    return "\n".join(lines) if lines else "- No actions selected"
