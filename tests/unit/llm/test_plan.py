from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from goapgit.core.models import ActionSpec, GoalSpec, Plan, RepoRef, RepoState
from goapgit.core.planner import SimplePlanner
from goapgit.io.logging import StructuredLogger
from goapgit.llm.plan import (
    PlanHintResult,
    apply_plan_hint,
    build_plan_prompt,
    clamp_cost_adjustment,
    request_plan_hint,
)
from goapgit.llm.responses import ResponsesAPI
from goapgit.llm.schema import PlanHint


@dataclass
class _StubResponse:
    id: str
    output_text: str


class _StubResponses(ResponsesAPI):
    def __init__(self, *responses: _StubResponse) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **payload: Any) -> _StubResponse:
        self.calls.append(payload)
        if not self._responses:  # pragma: no cover - defensive guard
            raise RuntimeError("No stub responses remaining")
        return self._responses.pop(0)


class _StubClient:
    responses: ResponsesAPI

    def __init__(self, *responses: _StubResponse) -> None:
        self.responses = _StubResponses(*responses)


def _make_state(repo_path: Path) -> RepoState:
    return RepoState(
        repo_path=repo_path,
        ref=RepoRef(branch="main"),
        diverged_local=1,
        diverged_remote=2,
        conflicts=(),
    )


def _make_plan() -> Plan:
    planner = SimplePlanner()
    state = _make_state(Path("/repo"))
    goal = GoalSpec()
    actions = [
        ActionSpec(name="Safety:CreateBackupRef", cost=0.4, rationale="Ensure recoverability."),
        ActionSpec(name="Safety:EnsureCleanOrStash", cost=0.6, rationale="Prepare working tree."),
        ActionSpec(name="Conflict:AutoTrivialResolve", cost=0.8, rationale="Apply rerere knowledge."),
        ActionSpec(name="Quality:RunTests", cost=1.2, rationale="Verify automated tests."),
    ]
    return planner.plan(state, goal, actions)


def test_build_plan_prompt_lists_actions(tmp_path: Path) -> None:
    """Prompt should include repo path, branch, divergence, and actions."""
    state = _make_state(tmp_path)
    plan = _make_plan()

    prompt = build_plan_prompt(state, plan)

    assert str(tmp_path) in prompt
    assert "Branch: main" in prompt
    for action in plan.actions:
        assert action.name in prompt
    assert "[-20%, +20%]" in prompt


def _make_payload(action: str, pct: float, note: str | None = None) -> str:
    payload = {"action": action, "cost_adjustment_pct": pct, "note": note}
    return json.dumps(payload)


def test_request_plan_hint_calls_responses_with_schema(tmp_path: Path) -> None:
    """Plan hint request should forward schema, instructions, and prompt."""
    state = _make_state(tmp_path)
    plan = _make_plan()
    stub_response = _StubResponse(id="resp_plan", output_text=_make_payload("alt sequence", 0.15, "risky"))
    client = _StubClient(stub_response)

    result = request_plan_hint(
        client,
        model="responses-plan",
        state=state,
        plan=plan,
    )

    assert isinstance(result, PlanHintResult)
    assert result.hint.action == "alt sequence"
    assert result.hint.cost_adjustment_pct == 0.15
    assert result.hint.note == "risky"
    assert result.response_id == "resp_plan"

    stub_responses = client.responses
    assert isinstance(stub_responses, _StubResponses)
    assert len(stub_responses.calls) == 1
    payload = stub_responses.calls[0]
    assert payload["model"] == "responses-plan"
    instructions = payload["instructions"]
    assert "Adjust costs within ±20%" in instructions
    assert payload["response_format"]["json_schema"]["name"] == "goapgit_plan_hint"
    prompt = payload["input"][0]["content"][0]["text"]
    assert "Conflicts detected" in prompt
    assert "Planned actions" in prompt


def test_clamp_cost_adjustment_enforces_bounds() -> None:
    """Cost adjustment helper should clamp to the configured bounds."""
    assert clamp_cost_adjustment(0.1) == 0.1
    assert clamp_cost_adjustment(0.5) == 0.2
    assert clamp_cost_adjustment(-0.5) == -0.2


def test_apply_plan_hint_adjusts_cost_and_logs() -> None:
    """Applying a plan hint should adjust cost, append notes, and log details."""
    plan = _make_plan()
    hint = PlanHint(action="Use manual merge", cost_adjustment_pct=0.45, note="High conflict density")

    stream = io.StringIO()
    logger = StructuredLogger(name="test", stream=stream, json_mode=False)

    adjusted = apply_plan_hint(plan, hint, logger=logger)

    assert adjusted.estimated_cost > plan.estimated_cost
    expected_cost = plan.estimated_cost * 1.2  # clamped to +20%
    assert abs(adjusted.estimated_cost - expected_cost) < 1e-6
    assert any(note.startswith("plan_hint_action=") for note in adjusted.notes)
    assert "plan_hint_note=High conflict density" in adjusted.notes

    log_output = stream.getvalue()
    assert "adjusted_cost" in log_output
    assert "plan hint applied" in log_output

