"""Commit and pull request message drafting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from collections.abc import Iterable, Sequence

from .instructions import messenger_instructions
from .responses import CompleteJsonResult, ResponsesClient, complete_json
from .schema import MessageDraft, sanitize_model_schema

__all__ = [
    "REQUIRED_SECTIONS",
    "MessageDraftResult",
    "build_message_prompt",
    "request_message_draft",
    "validate_message_draft",
]


REQUIRED_SECTIONS: tuple[str, ...] = ("## 目的", "## 変更", "## 影響", "## ロールバック")
TITLE_MAX_LENGTH = 72


@dataclass(frozen=True, slots=True)
class MessageDraftResult:
    """Structured output returned from the messenger endpoint."""

    draft: MessageDraft
    response_id: str
    output_text: str


def build_message_prompt(
    range_diff_summary: str,
    *,
    message_kind: Literal["commit", "pull_request"] = "commit",
    tests_summary: str | None = None,
    follow_up_items: Iterable[str] | None = None,
) -> str:
    """Create the prompt describing how to summarize a range-diff."""
    summary = range_diff_summary.strip()
    if not summary:
        msg = "range_diff_summary must not be empty"
        raise ValueError(msg)

    if message_kind not in {"commit", "pull_request"}:
        msg = f"Unsupported message_kind: {message_kind!r}"
        raise ValueError(msg)

    tests_line = (tests_summary or "Tests status unknown; state explicitly in the draft.").strip()

    follow_up_lines = ""
    if follow_up_items:
        sanitized_items = [item.strip() for item in follow_up_items if item and item.strip()]
        if sanitized_items:
            joined_items = "\n".join(f"- {item}" for item in sanitized_items)
            follow_up_lines = dedent(
                f"""

                Follow-up tasks to mention when relevant:
                {joined_items}
                """,
            ).rstrip()

    audience_label = "commit" if message_kind == "commit" else "pull request"
    sections_text = "\n".join(REQUIRED_SECTIONS)

    return dedent(
        f"""
        Draft a {audience_label} message using the provided git range-diff summary.

        Range-diff summary:
        {summary}

        Requirements:
        - Title must be {TITLE_MAX_LENGTH} characters or fewer once trimmed.
        - Provide concise, action-oriented language suitable for reviewers.
        - Body must be Markdown with these headings in this exact order and with
          concrete details under each heading:
          {sections_text}
        - Reference the testing status and any remaining risks.

        Testing status to reference:
        {tests_line}{follow_up_lines}
        """,
    ).strip()


def request_message_draft(
    client: ResponsesClient,
    *,
    model: str,
    range_diff_summary: str,
    message_kind: Literal["commit", "pull_request"] = "commit",
    tests_summary: str | None = None,
    follow_up_items: Sequence[str] | None = None,
    previous_response_id: str | None = None,
    extra_rules: Iterable[str] | None = None,
) -> MessageDraftResult:
    """Request a commit or PR message draft from the Responses API."""
    instructions = messenger_instructions(extra_rules=tuple(extra_rules or ()))
    schema = sanitize_model_schema(MessageDraft)
    prompt = build_message_prompt(
        range_diff_summary,
        message_kind=message_kind,
        tests_summary=tests_summary,
        follow_up_items=follow_up_items,
    )

    result: CompleteJsonResult = complete_json(
        client,
        model=model,
        instructions=instructions,
        schema=schema,
        prompt=prompt,
        previous_response_id=previous_response_id,
        schema_name="goapgit_message_draft",
    )

    draft = MessageDraft.model_validate(result.payload)
    validated = validate_message_draft(draft)
    return MessageDraftResult(draft=validated, response_id=result.response_id, output_text=result.output_text)


def validate_message_draft(draft: MessageDraft) -> MessageDraft:
    """Ensure the draft obeys the 72 character and section structure rules."""
    title = draft.title.strip()
    if not title:
        msg = "Message title must not be empty"
        raise ValueError(msg)
    if len(title) > TITLE_MAX_LENGTH:
        msg = "Message title must be 72 characters or fewer"
        raise ValueError(msg)

    _ensure_section_structure(draft.body)

    if title == draft.title:
        return draft
    return draft.model_copy(update={"title": title})


def _ensure_section_structure(body: str) -> None:
    normalized = body.strip()
    if not normalized:
        msg = "Message body must include required sections"
        raise ValueError(msg)

    sections: list[tuple[str, list[str]]] = []
    current_heading: str | None = None
    current_lines: list[str] = []
    for line in normalized.splitlines():
        stripped = line.rstrip()
        if stripped.startswith("## "):
            if current_heading is not None:
                sections.append((current_heading, current_lines))
            current_heading = stripped
            current_lines = []
            continue
        if current_heading is None:
            msg = "Message body must start with a Markdown heading"
            raise ValueError(msg)
        current_lines.append(stripped)

    if current_heading is not None:
        sections.append((current_heading, current_lines))

    headings = [heading for heading, _ in sections]
    if headings != list(REQUIRED_SECTIONS):
        msg = "Message body must contain required headings in order"
        raise ValueError(msg)

    for heading, lines in sections:
        content = "\n".join(lines).strip()
        if not content:
            msg = f"Section {heading!r} must include descriptive content"
            raise ValueError(msg)
