"""Structured responses helpers for the Responses API."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast

from .safety import BudgetExceededError, BudgetTracker, RedactionResult, Redactor, UsageMetrics
from .telemetry import TelemetryLogger, TelemetryRecord


@dataclass(frozen=True, slots=True)
class CompleteJsonResult:
    """Payload returned by :func:`complete_json`."""

    payload: dict[str, Any]
    response_id: str
    output_text: str
    redaction: RedactionResult | None = None
    usage: UsageMetrics | None = None


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
    redactor: Redactor | None = None,
    budget: BudgetTracker | None = None,
    telemetry_logger: TelemetryLogger | None = None,
    mode: str | None = None,
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
    redactor:
        Optional :class:`Redactor` used to sanitise the prompt for logging.
    budget:
        Optional :class:`BudgetTracker` enforcing token and cost limits.
    telemetry_logger:
        Optional :class:`TelemetryLogger` to capture usage statistics.
    mode:
        Logical mode of the call (resolver/plan/etc.) recorded in telemetry.

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

    redaction: RedactionResult | None = redactor.redact(prompt) if redactor else None

    if budget is not None:
        budget.ensure_can_continue()

    request_payload = _build_request_payload(
        model=model,
        instructions=instructions,
        schema_payload=schema_payload,
        prompt=prompt,
        schema_name=schema_name,
        previous_response_id=previous_response_id,
    )

    try:
        response = client.responses.create(**request_payload)
    except Exception:
        _emit_telemetry(
            telemetry_logger,
            mode=mode,
            result="error",
            model=model,
            previous_response_id=previous_response_id,
        )
        raise

    output_text = cast("str | None", getattr(response, "output_text", None))
    if not output_text:
        message = "Responses API returned empty output_text for JSON completion."
        raise ValueError(message)

    try:
        parsed_payload = cast("dict[str, Any]", json.loads(output_text))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        _emit_telemetry(
            telemetry_logger,
            mode=mode,
            result="decode_error",
            model=model,
            previous_response_id=previous_response_id,
        )
        message = "Responses API did not return valid JSON matching the schema."
        raise ValueError(message) from exc

    usage = _extract_usage_metrics(getattr(response, "usage", None))
    budget_error = _apply_budget(budget, usage)

    result = CompleteJsonResult(
        payload=parsed_payload,
        response_id=cast("str", response.id),
        output_text=output_text,
        redaction=redaction,
        usage=usage,
    )

    _emit_telemetry(
        telemetry_logger,
        mode=mode,
        result="budget_exceeded" if budget_error else "success",
        model=model,
        previous_response_id=previous_response_id,
        response_id=result.response_id,
        usage=usage,
    )

    if budget_error is not None:
        raise budget_error

    return result


def _build_request_payload(
    *,
    model: str,
    instructions: str,
    schema_payload: dict[str, Any],
    prompt: str,
    schema_name: str,
    previous_response_id: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
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
        payload["previous_response_id"] = previous_response_id
    return payload


def _apply_budget(
    budget: BudgetTracker | None,
    usage: UsageMetrics | None,
) -> BudgetExceededError | None:
    if budget is None or usage is None:
        return None
    try:
        budget.register(usage)
    except BudgetExceededError as exc:
        return exc
    return None


def _emit_telemetry(
    telemetry_logger: TelemetryLogger | None,
    *,
    mode: str | None,
    result: str,
    model: str,
    previous_response_id: str | None,
    response_id: str | None = None,
    usage: UsageMetrics | None = None,
) -> None:
    if telemetry_logger is None:
        return
    usage_payload = usage.model_dump(mode="json") if usage is not None else None
    record = TelemetryRecord(
        mode=mode or "unspecified",
        result=result,
        response_id=response_id,
        previous_response_id=previous_response_id,
        model=model,
        usage=usage_payload,
    )
    telemetry_logger.log(record)


def _extract_usage_metrics(raw_usage: Any) -> UsageMetrics | None:
    """Normalise Responses usage payloads into :class:`UsageMetrics`."""
    if raw_usage is None:
        return None

    mapping: Mapping[str, Any] | None = None
    if isinstance(raw_usage, Mapping):
        mapping = cast("Mapping[str, Any]", raw_usage)
    target = cast("object", raw_usage)

    def _get_value(*names: str) -> Any:
        for name in names:
            if mapping is not None and name in mapping:
                return mapping[name]
            value = getattr(target, name, None)
            if value is not None:
                return value
        return None

    prompt = _get_value("prompt_tokens", "input_tokens")
    completion = _get_value("completion_tokens", "output_tokens")
    total = _get_value("total_tokens")
    cost = _get_value("cost", "total_cost", "estimated_cost", "total_cost_usd")

    if prompt is None and completion is None and total is None and cost is None:
        return None

    return UsageMetrics(
        prompt_tokens=int(prompt) if prompt is not None else None,
        completion_tokens=int(completion) if completion is not None else None,
        total_tokens=int(total) if total is not None else None,
        cost=float(cost) if cost is not None else None,
    )
