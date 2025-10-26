"""Diagnostic helpers for validating LLM configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from pydantic import BaseModel

from goapgit.llm.client import LLMSettings, make_client_from_env
from goapgit.llm.instructions import InstructionRole, compose_instructions
from goapgit.llm.responses import CompleteJsonResult, ResponsesClient, complete_json
from goapgit.llm.schema import sanitize_model_schema


class _DoctorPayload(BaseModel):
    """Minimal structured output payload for the doctor check."""

    status: Literal["ok"]
    detail: str


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    """Represents the outcome of an individual diagnostic check."""

    name: str
    status: Literal["ok", "error"]
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """Aggregated diagnostic outcome returned by :func:`run_doctor`."""

    provider: str | None
    checks: tuple[DoctorCheck, ...]
    mocked: bool

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON serialisable payload."""
        payload: dict[str, Any] = {
            "provider": self.provider,
            "mock": self.mocked,
            "checks": [
                {"name": check.name, "status": check.status, "detail": check.detail}
                for check in self.checks
            ],
        }
        return payload


@dataclass(slots=True)
class _MockResponse:
    """Simple structure mimicking Responses API payloads."""

    id: str
    output_text: str


class _MockResponses:
    """In-memory stub collecting create calls for inspection."""

    def __init__(self, responses: tuple[_MockResponse, ...]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **payload: Any) -> _MockResponse:
        self.calls.append(payload)
        if not self._responses:
            message = "No mock responses configured"
            raise RuntimeError(message)
        return self._responses.pop(0)


@dataclass(slots=True)
class _MockClient:
    """Client exposing a ``responses`` attribute compatible with OpenAI."""

    responses: _MockResponses


def _build_instructions() -> str:
    return compose_instructions(
        InstructionRole.PLANNER,
        extra_rules=("Return the diagnostic payload exactly as specified.",),
    )


def _build_schema() -> dict[str, Any]:
    return sanitize_model_schema(_DoctorPayload)


def _perform_structured_output_check(
    client: ResponsesClient,
    *,
    model: str,
    previous_response_id: str | None = None,
) -> CompleteJsonResult:
    instructions = _build_instructions()
    schema = _build_schema()
    prompt = "Report successful doctor diagnosis using the structured schema."
    return complete_json(
        client,
        model=model,
        instructions=instructions,
        schema=schema,
        prompt=prompt,
        schema_name="goapgit_llm_doctor",
        previous_response_id=previous_response_id,
    )


def run_doctor(*, model: str, mock: bool = False) -> DoctorReport:
    """Execute the LLM doctor checks and return their outcomes."""
    checks: list[DoctorCheck] = []
    settings: LLMSettings | None = None
    try:
        settings = LLMSettings()
    except Exception as exc:
        checks.append(DoctorCheck(name="environment", status="error", detail=str(exc)))
        provider = None
        return DoctorReport(provider=provider, checks=tuple(checks), mocked=mock)

    checks.append(DoctorCheck(name="environment", status="ok"))

    mock_responses: _MockResponses | None = None
    if mock:
        mock_responses = _MockResponses(
            (
                _MockResponse(id="mock-1", output_text='{"status": "ok", "detail": "first"}'),
                _MockResponse(id="mock-2", output_text='{"status": "ok", "detail": "second"}'),
            ),
        )
        mock_client = _MockClient(mock_responses)
        client: ResponsesClient = cast("ResponsesClient", mock_client)
    else:
        real_client = make_client_from_env(settings=settings)
        client = cast("ResponsesClient", real_client)

    connection_detail: str | None = None
    try:
        first = _perform_structured_output_check(client, model=model)
        payload = _DoctorPayload.model_validate(first.payload)
        connection_status = DoctorCheck(name="structured_output", status="ok", detail=payload.detail)
        previous_id = first.response_id
    except Exception as exc:  # pragma: no cover - exercised in integration failures
        checks.append(DoctorCheck(name="connection", status="error", detail=str(exc)))
        return DoctorReport(provider=settings.provider.value, checks=tuple(checks), mocked=mock)

    checks.append(DoctorCheck(name="connection", status="ok", detail=connection_detail))
    checks.append(connection_status)

    chain_status = DoctorCheck(name="chain", status="ok")

    try:
        second = _perform_structured_output_check(
            client,
            model=model,
            previous_response_id=previous_id,
        )
        _DoctorPayload.model_validate(second.payload)
        if mock_responses is not None:
            previous_param: str | None = mock_responses.calls[-1].get("previous_response_id")
            if previous_param != previous_id:
                chain_status = DoctorCheck(
                    name="chain",
                    status="error",
                    detail="previous_response_id was not forwarded correctly",
                )
    except Exception as exc:
        chain_status = DoctorCheck(name="chain", status="error", detail=str(exc))

    checks.append(chain_status)
    return DoctorReport(
        provider=settings.provider.value,
        checks=tuple(checks),
        mocked=mock,
    )


__all__ = ["DoctorCheck", "DoctorReport", "run_doctor"]
