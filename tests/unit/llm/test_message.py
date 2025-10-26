from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from goapgit.llm.message import (
    MessageDraftResult,
    REQUIRED_SECTIONS,
    TITLE_MAX_LENGTH,
    build_message_prompt,
    request_message_draft,
    validate_message_draft,
)
from goapgit.llm.responses import ResponsesAPI
from goapgit.llm.schema import MessageDraft


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


def _build_body(sections: dict[str, str]) -> str:
    return "\n\n".join(
        f"{heading}\n{sections[heading]}" for heading in REQUIRED_SECTIONS
    )


def test_build_message_prompt_includes_summary_and_requirements() -> None:
    """Prompt should mention summary, testing, follow-ups, and headings."""
    prompt = build_message_prompt(
        "* 1:1 -> 1:2 (abc123..def456)",
        message_kind="pull_request",
        tests_summary="pytest -q",
        follow_up_items=("add docs", "monitor metrics"),
    )

    assert "pull request" in prompt
    assert "* 1:1 -> 1:2" in prompt
    for heading in REQUIRED_SECTIONS:
        assert heading in prompt
    assert "pytest -q" in prompt
    assert "- add docs" in prompt
    assert "- monitor metrics" in prompt


def test_request_message_draft_calls_responses_with_schema() -> None:
    """Message draft request should forward schema, instructions, and prompt."""
    body = _build_body(
        {
            "## 目的": "Explain why the change exists.",
            "## 変更": "List the concrete modifications.",
            "## 影響": "Note downstream impact and stakeholders.",
            "## ロールバック": "Describe the rollback procedure.",
        },
    )
    stub_response = _StubResponse(
        id="resp_msg",
        output_text=json.dumps({"title": "Summarize updates", "body": body}),
    )
    client = _StubClient(stub_response)

    result = request_message_draft(
        client,
        model="responses-messenger",
        range_diff_summary="* 1:1 -> 1:2 (abc..def)\n  - feat: new api",
        message_kind="commit",
        tests_summary="pytest -q",
        follow_up_items=["document behaviour"],
    )

    assert isinstance(result, MessageDraftResult)
    assert result.draft.title == "Summarize updates"
    assert result.draft.body == body
    assert result.response_id == "resp_msg"
    assert result.output_text == stub_response.output_text

    stub_responses = client.responses
    assert isinstance(stub_responses, _StubResponses)
    assert len(stub_responses.calls) == 1
    payload = stub_responses.calls[0]
    assert payload["model"] == "responses-messenger"
    instructions = payload["instructions"]
    assert "Keep titles under 72" in instructions
    schema_name = payload["response_format"]["json_schema"]["name"]
    assert schema_name == "goapgit_message_draft"
    prompt = payload["input"][0]["content"][0]["text"]
    assert "feat: new api" in prompt
    assert "document behaviour" in prompt


@pytest.mark.parametrize(
    "title",
    [
        "",  # empty title
        "x" * (TITLE_MAX_LENGTH + 1),  # too long
    ],
)
def test_validate_message_draft_enforces_title_rules(title: str) -> None:
    """Validation should reject empty or overly long titles."""
    body = _build_body(
        {
            "## 目的": "Purpose",
            "## 変更": "Changes",
            "## 影響": "Impact",
            "## ロールバック": "Rollback",
        },
    )
    draft = MessageDraft(title=title or "  ", body=body)
    expected = "must not be empty" if not title else "72 characters or fewer"

    with pytest.raises(ValueError, match=expected):
        validate_message_draft(draft)


def test_validate_message_draft_requires_sections() -> None:
    """Validation should require all sections with content."""
    incomplete_body = "## 目的\nWhy\n\n## 変更\nWhat"
    draft = MessageDraft(title="Valid", body=incomplete_body)

    with pytest.raises(ValueError, match="required headings"):
        validate_message_draft(draft)
