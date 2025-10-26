"""Tests for the Responses API wrapper helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from goapgit.llm.responses import CompleteJsonResult, ResponsesClient, complete_json
from goapgit.llm.safety import BudgetExceededError, BudgetTracker, Redactor
from goapgit.llm.telemetry import TelemetryLogger


@dataclass
class _StubUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None


@dataclass
class _StubResponse:
    """Minimal stand-in for the OpenAI Responses object."""

    id: str
    output_text: str
    usage: _StubUsage | None = None


class _StubResponses:
    """Collects calls to ``responses.create`` and returns stubbed responses."""

    def __init__(self, *responses: _StubResponse) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **payload: Any) -> _StubResponse:
        self.calls.append(payload)
        if not self._responses:  # pragma: no cover - defensive guard
            raise RuntimeError("No stub responses remaining")
        return self._responses.pop(0)


class _StubClient:
    """Client stub exposing the ``responses.create`` surface."""

    responses: _StubResponses

    def __init__(self, *responses: _StubResponse) -> None:
        self.responses = _StubResponses(*responses)


SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"message": {"type": "string"}},
    "required": ["message"],
}


def test_complete_json_submits_instructions_and_schema() -> None:
    """The wrapper should send strict schema requests and parse the response."""
    stub_client = _StubClient(_StubResponse(id="resp_1", output_text='{"message": "hi"}'))

    result = complete_json(
        cast("ResponsesClient", stub_client),
        model="test-model",
        instructions="Always respond in JSON",
        schema=SCHEMA,
        prompt="Say hi",
    )

    assert isinstance(result, CompleteJsonResult)
    assert result.payload == {"message": "hi"}
    assert result.response_id == "resp_1"
    assert result.output_text == '{"message": "hi"}'
    assert result.redaction is None
    assert result.usage is None

    assert len(stub_client.responses.calls) == 1
    request_payload = stub_client.responses.calls[0]
    assert request_payload["instructions"] == "Always respond in JSON"
    assert request_payload["model"] == "test-model"
    assert request_payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "goapgit_payload",
            "schema": SCHEMA,
            "strict": True,
        },
    }
    assert request_payload["input"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Say hi",
                },
            ],
        },
    ]
    assert "previous_response_id" not in request_payload


def test_complete_json_links_previous_response_id() -> None:
    """Chained calls must forward the prior response id when provided."""
    stub_client = _StubClient(
        _StubResponse(id="resp_1", output_text='{"message": "first"}'),
        _StubResponse(id="resp_2", output_text='{"message": "second"}'),
    )

    first = complete_json(
        cast("ResponsesClient", stub_client),
        model="test-model",
        instructions="Always respond in JSON",
        schema=SCHEMA,
        prompt="First turn",
    )

    second = complete_json(
        cast("ResponsesClient", stub_client),
        model="test-model",
        instructions="Always respond in JSON",
        schema=SCHEMA,
        prompt="Second turn",
        previous_response_id=first.response_id,
    )

    assert first.payload == {"message": "first"}
    assert second.payload == {"message": "second"}
    assert second.response_id == "resp_2"
    assert first.redaction is None
    assert second.redaction is None

    assert len(stub_client.responses.calls) == 2
    assert "previous_response_id" not in stub_client.responses.calls[0]
    assert stub_client.responses.calls[1]["previous_response_id"] == "resp_1"

    # The same instructions are sent on every call to maintain guard rails.
    assert stub_client.responses.calls[0]["instructions"] == "Always respond in JSON"
    assert stub_client.responses.calls[1]["instructions"] == "Always respond in JSON"


def test_complete_json_rejects_blank_instructions() -> None:
    """Empty instructions would violate the guard rail policy."""
    stub_client = _StubClient(_StubResponse(id="resp_1", output_text='{"message": "hi"}'))

    with pytest.raises(ValueError, match="Instructions must be a non-empty string"):
        complete_json(
            cast("ResponsesClient", stub_client),
            model="test-model",
            instructions="   ",
            schema=SCHEMA,
            prompt="Say hi",
        )


def test_complete_json_returns_redaction_when_enabled() -> None:
    """Providing a redactor should expose the sanitized prompt."""
    stub_client = _StubClient(
        _StubResponse(id="resp_1", output_text='{"message": "hi"}'),
    )

    prompt = "api key sk-secretvalue123456789012345"
    result = complete_json(
        cast("ResponsesClient", stub_client),
        model="test-model",
        instructions="Always respond in JSON",
        schema=SCHEMA,
        prompt=prompt,
        redactor=Redactor(),
    )

    assert result.redaction is not None
    assert result.redaction.has_matches
    assert "sk-***" in result.redaction.text
    assert stub_client.responses.calls[0]["input"][0]["content"][0]["text"] == prompt


def test_complete_json_logs_usage_and_telemetry(tmp_path: Path) -> None:
    """Usage metrics should be surfaced, budget applied, and telemetry recorded."""
    stub_client = _StubClient(
        _StubResponse(
            id="resp_1",
            output_text='{"message": "hi"}',
            usage=_StubUsage(input_tokens=5, output_tokens=3, total_tokens=8, cost=0.4),
        ),
    )
    budget = BudgetTracker(max_tokens=20, max_cost=5.0)
    telemetry_path = tmp_path / "telemetry.jsonl"
    telemetry = TelemetryLogger(telemetry_path)

    result = complete_json(
        cast("ResponsesClient", stub_client),
        model="test-model",
        instructions="Always respond in JSON",
        schema=SCHEMA,
        prompt="Say hi",
        budget=budget,
        telemetry_logger=telemetry,
        mode="resolver",
    )

    assert result.usage is not None
    assert result.usage.total_tokens == 8
    assert abs(budget.consumed_cost - 0.4) < 1e-6
    assert budget.consumed_tokens == 8

    payload = json.loads(telemetry_path.read_text(encoding="utf-8").strip())
    assert payload["result"] == "success"
    assert payload["usage"]["total_tokens"] == 8
    assert payload["mode"] == "resolver"


def test_complete_json_budget_exceeded_logs_and_raises(tmp_path: Path) -> None:
    """Budget exhaustion should log telemetry and raise an error."""
    stub_client = _StubClient(
        _StubResponse(
            id="resp_1",
            output_text='{"message": "first"}',
            usage=_StubUsage(total_tokens=9, cost=0.5),
        ),
        _StubResponse(
            id="resp_2",
            output_text='{"message": "second"}',
            usage=_StubUsage(total_tokens=5, cost=0.6),
        ),
    )
    budget = BudgetTracker(max_tokens=10, max_cost=1.0)
    telemetry_path = tmp_path / "telemetry.jsonl"
    telemetry = TelemetryLogger(telemetry_path)

    first = complete_json(
        cast("ResponsesClient", stub_client),
        model="test-model",
        instructions="Always respond in JSON",
        schema=SCHEMA,
        prompt="First",
        budget=budget,
        telemetry_logger=telemetry,
        mode="resolver",
    )
    assert first.payload == {"message": "first"}

    with pytest.raises(BudgetExceededError):
        complete_json(
            cast("ResponsesClient", stub_client),
            model="test-model",
            instructions="Always respond in JSON",
            schema=SCHEMA,
            prompt="Second",
            previous_response_id=first.response_id,
            budget=budget,
            telemetry_logger=telemetry,
            mode="resolver",
        )

    lines = telemetry_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first_payload = json.loads(lines[0])
    second_payload = json.loads(lines[1])

    assert first_payload["result"] == "success"
    assert second_payload["result"] == "budget_exceeded"
    assert second_payload["previous_response_id"] == first.response_id
