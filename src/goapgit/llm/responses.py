"""Structured responses helpers for the Responses API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, cast


@dataclass(frozen=True, slots=True)
class CompleteJsonResult:
    """Payload returned by :func:`complete_json`."""

    payload: dict[str, Any]
    response_id: str
    output_text: str


class ResponsesAPI(Protocol):
    """Protocol for the Responses namespace on OpenAI clients."""

    def create(self, **payload: Any) -> Any:  # pragma: no cover - protocol definition
        """Create a completion response."""


class ResponsesClient(Protocol):
    """Protocol describing the minimal Responses API client surface."""

    responses: ResponsesAPI


def complete_json(
    client: ResponsesClient,
    *,
    model: str,
    instructions: str,
    schema: dict[str, Any],
    prompt: str,
    previous_response_id: str | None = None,
    schema_name: str = "goapgit_payload",
) -> CompleteJsonResult:
    """Call the Responses API and return strict JSON content.

    Parameters
    ----------
    client:
        Instantiated OpenAI client (standard or Azure).
    model:
        Model or deployment identifier compatible with the Responses API.
    instructions:
        Prompt prefix sent on every call to enforce guard rails.
    schema:
        Strict JSON Schema expected from the model.
    prompt:
        User content delivered as a single text block.
    previous_response_id:
        Optional identifier of the previous response to maintain minimal history.
    schema_name:
        Name passed to the Structured Outputs contract. Defaults to
        ``"goapgit_payload"``.

    Returns
    -------
    CompleteJsonResult
        Parsed JSON payload, the originating ``response.id``, and the raw
        ``response.output_text`` for logging/telemetry.

    """
    if not instructions or not instructions.strip():
        message = "Instructions must be a non-empty string for every Responses call."
        raise ValueError(message)

    schema_payload = dict(schema)

    request_payload: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema_payload,
                "strict": True,
            },
        },
    }
    if previous_response_id is not None:
        request_payload["previous_response_id"] = previous_response_id

    response = client.responses.create(**request_payload)

    output_text = cast("str | None", getattr(response, "output_text", None))
    if not output_text:
        message = "Responses API returned empty output_text for JSON completion."
        raise ValueError(message)

    try:
        parsed_payload = cast("dict[str, Any]", json.loads(output_text))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        message = "Responses API did not return valid JSON matching the schema."
        raise ValueError(message) from exc

    return CompleteJsonResult(
        payload=parsed_payload,
        response_id=cast("str", response.id),
        output_text=output_text,
    )
