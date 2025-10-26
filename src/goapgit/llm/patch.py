"""Patch proposal helpers using the Responses API."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent, indent
from typing import TYPE_CHECKING

from goapgit.core.models import ConflictType

from .instructions import resolver_instructions
from .responses import CompleteJsonResult, ResponsesClient, complete_json
from .schema import PatchSet, sanitize_model_schema

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

__all__ = [
    "ConflictExcerpt",
    "PatchProposalResult",
    "build_initial_prompt",
    "build_retry_prompt",
    "propose_patch",
]


_DEFAULT_EXTRA_RULES: tuple[str, ...] = (
    "Return only unified diff patches required to resolve the shown hunks.",
    "Do not reformat untouched sections or introduce new files.",
    "Ensure every patch applies cleanly with `git apply --check`.",
)


@dataclass(frozen=True, slots=True)
class ConflictExcerpt:
    """Minimal excerpt describing a conflict hunk."""

    path: str
    snippet: str
    ctype: ConflictType = ConflictType.text
    hunk_header: str | None = None

    def __post_init__(self) -> None:
        """Validate that the excerpt includes path and snippet content."""
        if not self.path:
            msg = "ConflictExcerpt.path must be a non-empty string"
            raise ValueError(msg)
        if not self.snippet.strip():
            msg = "ConflictExcerpt.snippet must contain non-whitespace characters"
            raise ValueError(msg)

    def format_for_prompt(self) -> str:
        """Render the excerpt for inclusion in the LLM prompt."""
        header_line = f"Path: {self.path}"
        if self.hunk_header:
            header_line = f"{header_line} @ {self.hunk_header}"

        kind_label = _describe_conflict_type(self.ctype)
        formatted_snippet = indent(self.snippet.strip("\n"), "    ")

        return dedent(
            f"""
            {header_line}
            File type: {kind_label}
            Minimal conflict hunk:
            {formatted_snippet}
            """,
        ).strip()


@dataclass(frozen=True, slots=True)
class PatchProposalResult:
    """Parsed PatchSet returned from the Responses API."""

    patch_set: PatchSet
    response_id: str
    output_text: str


def build_initial_prompt(conflict: ConflictExcerpt, *, repo_path: Path) -> str:
    """Create the initial prompt describing the conflict."""
    conflict_section = conflict.format_for_prompt()
    return dedent(
        f"""
        You are resolving a Git merge conflict inside repository: {repo_path}

        {conflict_section}

        Provide only the patches necessary to resolve the conflict while preserving
        unrelated content. Keep edits minimal and deterministic so they satisfy
        `git apply --check`.
        """,
    ).strip()


def build_retry_prompt(feedback: str) -> str:
    """Create the retry prompt that only forwards failure feedback."""
    feedback_block = indent(feedback.strip(), "    ")
    return dedent(
        f"""
        The previous patch proposal failed to apply. Use the feedback below to
        issue a corrected PatchSet.

        Failure feedback:
        {feedback_block}
        """,
    ).strip()


def propose_patch(
    client: ResponsesClient,
    *,
    model: str,
    repo_path: Path,
    conflict: ConflictExcerpt,
    previous_response_id: str | None = None,
    failure_feedback: str | None = None,
    extra_rules: Iterable[str] | None = None,
) -> PatchProposalResult:
    """Request a PatchSet for the given conflict excerpt."""
    if failure_feedback and not previous_response_id:
        msg = "previous_response_id is required when providing failure_feedback"
        raise ValueError(msg)

    rules: tuple[str, ...] = (
        tuple(_DEFAULT_EXTRA_RULES) + tuple(extra_rules)
        if extra_rules
        else _DEFAULT_EXTRA_RULES
    )

    instructions = resolver_instructions(extra_rules=rules)
    schema = sanitize_model_schema(PatchSet)

    prompt: str
    if failure_feedback:
        prompt = build_retry_prompt(failure_feedback)
    else:
        prompt = build_initial_prompt(conflict, repo_path=repo_path)

    result: CompleteJsonResult = complete_json(
        client,
        model=model,
        instructions=instructions,
        schema=schema,
        prompt=prompt,
        previous_response_id=previous_response_id,
        schema_name="goapgit_patch_set",
    )

    patch_set = PatchSet.model_validate(result.payload)
    return PatchProposalResult(
        patch_set=patch_set,
        response_id=result.response_id,
        output_text=result.output_text,
    )


def _describe_conflict_type(conflict_type: ConflictType) -> str:
    mapping = {
        ConflictType.json: "JSON document",
        ConflictType.yaml: "YAML document",
        ConflictType.lock: "lockfile",
        ConflictType.binary: "binary blob",
        ConflictType.text: "text file",
    }
    return mapping.get(conflict_type, conflict_type.value)
