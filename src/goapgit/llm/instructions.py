"""Role-specific instruction templates for Responses API calls."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from textwrap import dedent


class InstructionRole(str, Enum):
    """Supported instruction personas for GOAPGit."""

    RESOLVER = "resolver"
    MESSENGER = "messenger"
    PLANNER = "planner"


@dataclass(frozen=True)
class InstructionTemplate:
    """Container describing how to render a role template."""

    headline: str
    responsibilities: tuple[str, ...]


_ROLE_TEMPLATES: dict[InstructionRole, InstructionTemplate] = {
    InstructionRole.RESOLVER: InstructionTemplate(
        headline="Resolve Git conflicts with minimal context and deterministic patches.",
        responsibilities=(
            "Generate strictly applicable patches that would satisfy `git apply --check`.",
            "Prefer minimal edits that keep the author intent intact and never invent new functionality.",
            "Explain when more context is required before proposing a patch.",
        ),
    ),
    InstructionRole.MESSENGER: InstructionTemplate(
        headline="Draft commit or pull request messages for human review.",
        responsibilities=(
            "Keep titles under 72 characters and provide concise, action-oriented summaries.",
            "Structure the body using Markdown headings that match the requested outline.",
            "Call out testing status and follow-up tasks explicitly.",
        ),
    ),
    InstructionRole.PLANNER: InstructionTemplate(
        headline="Suggest alternative plan steps and cost adjustments for the GOAP planner.",
        responsibilities=(
            "Return only actions that are reachable from the provided repository state.",
            "Adjust costs within ±20% and clamp the result to this range if necessary.",
            "Highlight blocking risks or prerequisites the operator must address first.",
        ),
    ),
}


_COMMON_RULES: tuple[str, ...] = (
    "Always respond with JSON that matches the provided strict schema.",
    (
        "Treat each request as stateless: instructions are re-sent every turn "
        "and past replies are not implicitly remembered."
    ),
    "Request the minimum missing details when the provided snippets are insufficient to act safely.",
    "Avoid placeholders like 'TODO' or speculative code; prefer explicit guidance or refusal.",
)


def _format_bullets(items: tuple[str, ...] | list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def compose_instructions(
    role: InstructionRole,
    *,
    extra_rules: tuple[str, ...] | list[str] | None = None,
) -> str:
    """Build the instruction text for a given role."""
    try:
        template = _ROLE_TEMPLATES[role]
    except KeyError as exc:  # pragma: no cover - defensive guard
        message = f"Unsupported instruction role: {role!r}"
        raise ValueError(message) from exc

    rules: list[str] = list(_COMMON_RULES)
    if extra_rules:
        rules.extend(list(extra_rules))

    return dedent(
        f"""
        You are GOAPGit's {role.value} specialist. {template.headline}

        Core rules:
        {_format_bullets(rules)}

        Focus for this request:
        {_format_bullets(template.responsibilities)}
        """,
    ).strip()


def resolver_instructions(*, extra_rules: tuple[str, ...] | list[str] | None = None) -> str:
    """Shortcut for :data:`InstructionRole.RESOLVER`."""
    return compose_instructions(InstructionRole.RESOLVER, extra_rules=extra_rules)


def messenger_instructions(*, extra_rules: tuple[str, ...] | list[str] | None = None) -> str:
    """Shortcut for :data:`InstructionRole.MESSENGER`."""
    return compose_instructions(InstructionRole.MESSENGER, extra_rules=extra_rules)


def planner_instructions(*, extra_rules: tuple[str, ...] | list[str] | None = None) -> str:
    """Shortcut for :data:`InstructionRole.PLANNER`."""
    return compose_instructions(InstructionRole.PLANNER, extra_rules=extra_rules)


__all__ = [
    "InstructionRole",
    "compose_instructions",
    "messenger_instructions",
    "planner_instructions",
    "resolver_instructions",
]

