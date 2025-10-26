"""Unit tests for the strategy advice helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, cast

import pytest

from goapgit.llm.advice import (
    StrategyAdviceResult,
    advise_strategy,
    build_strategy_prompt,
)
from goapgit.llm.responses import ResponsesAPI


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


def _make_payload(*, resolution: str, reason: str = "ok", confidence: str = "med") -> str:
    payload = {
        "resolution": resolution,
        "reason": reason,
        "confidence": confidence,
    }
    return json.dumps(payload)


def test_build_strategy_prompt_includes_merge_tree_summary() -> None:
    """Ensure the prompt surfaces the conflicted path and merge-tree summary."""
    summary = dedent(
        """
        CONFLICT (content): Merge conflict in yarn.lock
        CONFLICT (content): Merge conflict in package.json
        """,
    ).strip()

    prompt = build_strategy_prompt(path="packages/yarn.lock", merge_tree_summary=summary)

    assert "packages/yarn.lock" in prompt
    assert "lockfile" in prompt
    assert "Merge-tree summary" in prompt
    assert "CONFLICT" in prompt
    assert "Reference patterns" in prompt


def test_advise_strategy_calls_responses_with_rules() -> None:
    """advise_strategy should forward schema, prompt, and instruction hints."""
    summary = "CONFLICT (content): Merge conflict in api/schema.json"
    stub_response = _StubResponse(
        id="resp_123",
        output_text=_make_payload(resolution="merge-driver", reason="structured"),
    )
    client = _StubClient(stub_response)

    result = advise_strategy(
        client,
        model="responses-test",
        path="api/schema.json",
        merge_tree_summary=summary,
    )

    assert isinstance(result, StrategyAdviceResult)
    assert result.advice.resolution == "merge-driver"
    assert result.advice.reason == "structured"
    assert result.response_id == "resp_123"

    stub_responses = cast("_StubResponses", client.responses)
    assert len(stub_responses.calls) == 1
    payload = stub_responses.calls[0]
    assert payload["model"] == "responses-test"
    instructions = payload["instructions"]
    assert "lockfile" in instructions
    assert "merge-driver" in instructions
    schema_name = payload["response_format"]["json_schema"]["name"]
    assert schema_name == "goapgit_strategy_advice"
    prompt = payload["input"][0]["content"][0]["text"]
    assert "api/schema.json" in prompt
    assert "JSON" in prompt


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("deps/yarn.lock", "theirs"),
        ("api/schema.json", "merge-driver"),
        ("docs/guide.md", "manual"),
    ],
)
def test_advise_strategy_enforces_suffix_overrides(path: str, expected: str) -> None:
    """Suffix heuristics must override mismatched model recommendations."""
    client = _StubClient(
        _StubResponse(
            id="resp_456",
            output_text=_make_payload(resolution="ours", reason="model guess"),
        ),
    )

    result = advise_strategy(
        client,
        model="responses-test",
        path=path,
        merge_tree_summary="CONFLICT",
    )

    assert result.advice.resolution == expected
    assert "override" in result.advice.reason.lower()
    assert result.advice.confidence == "high"
