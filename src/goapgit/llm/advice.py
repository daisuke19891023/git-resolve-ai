"""Strategy recommendation helpers leveraging the Responses API."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent, indent
from typing import TYPE_CHECKING

from goapgit.core.models import ConflictType

from .instructions import resolver_instructions
from .responses import CompleteJsonResult, ResponsesClient, complete_json
from .schema import StrategyAdvice, sanitize_model_schema

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "StrategyAdviceResult",
    "advise_strategy",
    "build_strategy_prompt",
]


@dataclass(frozen=True, slots=True)
class StrategyAdviceResult:
    """Parsed :class:`StrategyAdvice` returned from the Responses API."""

    advice: StrategyAdvice
    response_id: str
    output_text: str


_DEFAULT_EXTRA_RULES: tuple[str, ...] = (
    "When a lockfile (*.lock) is conflicted, prefer `theirs` to keep the incoming resolved graph.",
    "For structured data like *.json, recommend `merge-driver` so a specialised merge can run before manual edits.",
    "Markdown (*.md) conflicts usually need human intent review—recommend `manual`.",
)

_SUFFIX_RULES: tuple[tuple[str, str, str], ...] = (
    (
        ".lock",
        "theirs",
        "Lockfiles are generated artifacts; keep the incoming version to avoid dependency drift.",
    ),
    (
        ".json",
        "merge-driver",
        "JSON conflicts should invoke the structured merge driver before manual edits.",
    ),
    (
        ".md",
        "manual",
        "Documentation changes require human review to reconcile intent.",
    ),
)

_FEW_SHOT_EXAMPLES = dedent(
    """
    Example 1
    Path: yarn.lock
    Merge-tree summary:
        CONFLICT (content): Merge conflict in yarn.lock
    Recommended JSON:
    {
        "resolution": "theirs",
        "reason": "Lockfiles are deterministic outputs; prefer the incoming lockfile.",
        "confidence": "high"
    }

    Example 2
    Path: api/schema.json
    Merge-tree summary:
        CONFLICT (content): Merge conflict in api/schema.json
    Recommended JSON:
    {
        "resolution": "merge-driver",
        "reason": "Invoke the JSON merge driver first; it handles structured keys safely.",
        "confidence": "med"
    }

    Example 3
    Path: docs/handbook.md
    Merge-tree summary:
        CONFLICT (content): Merge conflict in docs/handbook.md
    Recommended JSON:
    {
        "resolution": "manual",
        "reason": "Wording changes benefit from human review to align tone and intent.",
        "confidence": "med"
    }
    """,
).strip()


def build_strategy_prompt(*, path: str, merge_tree_summary: str) -> str:
    """Create the prompt describing the conflicted path and merge-tree output."""
    if not path:
        message = "Conflicted path must be a non-empty string"
        raise ValueError(message)

    summary = merge_tree_summary.strip()
    summary_block = indent(summary, "    ") if summary else "    (merge-tree summary omitted)"

    conflict_type = _detect_conflict_type(path)
    kind_label = _describe_conflict_type(conflict_type)
    example_block = indent(_FEW_SHOT_EXAMPLES, "    ")

    return dedent(
        f"""
        You are reviewing git merge-tree output to recommend a conflict resolution strategy.

        Conflicted path: {path}
        File type: {kind_label}

        Merge-tree summary:
        {summary_block}

        Recommend one of: ours, theirs, manual, merge-driver. Provide a concise reason and confidence.
        Reference patterns:
        {example_block}
        """,
    ).strip()


def advise_strategy(
    client: ResponsesClient,
    *,
    model: str,
    path: str,
    merge_tree_summary: str,
    previous_response_id: str | None = None,
    extra_rules: Iterable[str] | None = None,
) -> StrategyAdviceResult:
    """Request strategy advice for a conflicted path."""
    rules = tuple(_DEFAULT_EXTRA_RULES) + tuple(extra_rules or ())
    instructions = resolver_instructions(extra_rules=rules)
    schema = sanitize_model_schema(StrategyAdvice)
    prompt = build_strategy_prompt(path=path, merge_tree_summary=merge_tree_summary)

    result: CompleteJsonResult = complete_json(
        client,
        model=model,
        instructions=instructions,
        schema=schema,
        prompt=prompt,
        previous_response_id=previous_response_id,
        schema_name="goapgit_strategy_advice",
    )

    advice = StrategyAdvice.model_validate(result.payload)
    advice = _apply_suffix_overrides(path=path, advice=advice)

    return StrategyAdviceResult(
        advice=advice,
        response_id=result.response_id,
        output_text=result.output_text,
    )


def _apply_suffix_overrides(*, path: str, advice: StrategyAdvice) -> StrategyAdvice:
    lowered = path.lower()
    for suffix, expected_resolution, canonical_reason in _SUFFIX_RULES:
        if not lowered.endswith(suffix):
            continue

        current_reason = advice.reason.strip() if advice.reason else ""
        if advice.resolution != expected_resolution:
            enforced_reason = (
                f"{canonical_reason} Forced override to '{expected_resolution}' "
                f"instead of model suggestion '{advice.resolution}'."
            )
            if current_reason:
                enforced_reason = f"{enforced_reason} Original reason: {current_reason}"
            return advice.model_copy(
                update={
                    "resolution": expected_resolution,
                    "reason": enforced_reason.strip(),
                    "confidence": "high",
                },
            )

        if not current_reason:
            return advice.model_copy(update={"reason": canonical_reason})

        return advice

    return advice


def _detect_conflict_type(path: str) -> ConflictType:
    lowered = path.lower()
    if lowered.endswith(".json"):
        return ConflictType.json
    if lowered.endswith((".yaml", ".yml")):
        return ConflictType.yaml
    if lowered.endswith(".lock"):
        return ConflictType.lock
    return ConflictType.text


def _describe_conflict_type(conflict_type: ConflictType) -> str:
    mapping = {
        ConflictType.json: "JSON document",
        ConflictType.yaml: "YAML document",
        ConflictType.lock: "lockfile",
        ConflictType.binary: "binary blob",
        ConflictType.text: "text file",
    }
    return mapping.get(conflict_type, conflict_type.value)
