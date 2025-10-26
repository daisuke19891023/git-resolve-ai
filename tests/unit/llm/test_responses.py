"""Tests for the Responses API wrapper helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest

from goapgit.llm.responses import CompleteJsonResult, ResponsesClient, complete_json


@dataclass
class _StubResponse:
    """Minimal stand-in for the OpenAI Responses object."""

    id: str
    output_text: str


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
