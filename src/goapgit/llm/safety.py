"""Safety helpers for LLM interactions.

This module provides two core features required for LLM operations:

* **Redaction** - mask sensitive fragments such as API keys before prompts are
  logged or dispatched to the model.
* **Budget enforcement** - keep cumulative token and monetary usage within the
  bounds configured by CLI flags (``--llm-max-tokens`` / ``--llm-max-cost``).

Both concerns are implemented using small utility classes so that higher level
components (CLI flows, planners, etc.) can compose them without inheriting
external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import re
from collections.abc import Callable, Iterable

__all__ = [
    "BudgetExceededError",
    "BudgetTracker",
    "RedactionResult",
    "RedactionRule",
    "Redactor",
]


def _mask_value(value: str, *, visible_prefix: int = 4, visible_suffix: int = 2) -> str:
    """Return a masked representation of ``value`` for logging purposes."""
    if len(value) <= visible_prefix + visible_suffix:
        return "***"
    prefix = value[:visible_prefix]
    suffix = value[-visible_suffix:]
    return f"{prefix}***{suffix}"


RedactionReplacer = Callable[[re.Match[str]], str]


@dataclass(frozen=True)
class RedactionRule:
    """A compiled regex rule that replaces sensitive fragments."""

    pattern: re.Pattern[str]
    replacement: str | RedactionReplacer

    @classmethod
    def from_pattern(
        cls,
        expression: str,
        *,
        flags: int = 0,
        replacement: str | RedactionReplacer,
    ) -> RedactionRule:
        """Build a :class:`RedactionRule` from a pattern string."""
        return cls(pattern=re.compile(expression, flags), replacement=replacement)


@dataclass(frozen=True)
class RedactionResult:
    """Represents the outcome of redacting a text block."""

    original: str
    redacted: str

    @property
    def has_redactions(self) -> bool:
        """Return ``True`` when the redacted text differs from the original."""
        return self.original != self.redacted


_DEFAULT_RULES: tuple[RedactionRule, ...] = (
    # OpenAI style keys (sk-, rk-, pk-, etc.).
    RedactionRule.from_pattern(
        r"\b(?P<prefix>[spr]k-[a-z0-9_-]{3,}?-?)(?P<value>[a-z0-9]{6,})\b",
        flags=re.IGNORECASE,
        replacement=lambda match: f"{match.group('prefix')}{_mask_value(match.group('value'))}",
    ),
    # Generic KEY=VALUE assignments.
    RedactionRule.from_pattern(
        r"(?P<label>(?:api[_-]?key|token|secret|password)\s*[:=]\s*)(?P<value>[A-Za-z0-9_\-]{8,})",
        flags=re.IGNORECASE,
        replacement=lambda match: f"{match.group('label')}{_mask_value(match.group('value'))}",
    ),
    # Bearer tokens.
    RedactionRule.from_pattern(
        r"(?P<prefix>Bearer\s+)(?P<value>[A-Za-z0-9._-]{16,})",
        replacement=lambda match: f"{match.group('prefix')}{_mask_value(match.group('value'))}",
    ),
    # URL embedded credentials (user:password@host).
    RedactionRule.from_pattern(
        r"(?P<scheme>https?://[^:/\s]+:)(?P<secret>[^@\s]+)(?P<suffix>@)",
        replacement=lambda match: f"{match.group('scheme')}***{match.group('suffix')}",
    ),
)


class Redactor:
    """Apply redaction rules to arbitrary text content."""

    def __init__(self, *, rules: Iterable[RedactionRule] | None = None) -> None:
        """Initialise the redactor with default or custom rules."""
        self._rules: tuple[RedactionRule, ...] = (
            tuple(rules) if rules is not None else _DEFAULT_RULES
        )

    def redact(self, text: str) -> RedactionResult:
        """Return the redacted variant of ``text``."""
        redacted = text
        for rule in self._rules:
            redacted = rule.pattern.sub(rule.replacement, redacted)
        return RedactionResult(original=text, redacted=redacted)


class BudgetExceededError(RuntimeError):
    """Raised when an LLM budget would be exceeded."""

    def __init__(self, *, limit: str, attempted: str) -> None:
        """Compose a descriptive budget error message."""
        message = f"LLM budget exceeded (limit={limit}, attempted={attempted})"
        super().__init__(message)
        self.limit = limit
        self.attempted = attempted


def _to_decimal(value: Decimal | float | None) -> Decimal:
    if value is None:
        return Decimal(0)
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


@dataclass
class BudgetTracker:
    """Track cumulative token and cost usage for LLM interactions."""

    max_tokens: int | None = None
    max_cost: Decimal | float | int | None = None
    used_tokens: int = 0
    used_cost: Decimal = Decimal(0)

    def remaining_tokens(self) -> int | None:
        """Return the number of tokens still available, if a limit exists."""
        if self.max_tokens is None:
            return None
        return max(self.max_tokens - self.used_tokens, 0)

    def remaining_cost(self) -> Decimal | None:
        """Return the remaining monetary budget, if configured."""
        if self.max_cost is None:
            return None
        limit = _to_decimal(self.max_cost)
        remaining = limit - self.used_cost
        return max(remaining, Decimal(0))

    def _project_tokens(self, tokens: int | None) -> int:
        additional = tokens or 0
        return self.used_tokens + additional

    def _project_cost(self, cost: Decimal | float | None) -> Decimal:
        additional = _to_decimal(cost)
        return self.used_cost + additional

    def ensure_within_budget(
        self,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        estimated_cost: Decimal | float | None = None,
    ) -> None:
        """Raise if the planned request would exceed configured budgets."""
        projected_tokens = self._project_tokens((prompt_tokens or 0) + (completion_tokens or 0))
        if self.max_tokens is not None and projected_tokens > self.max_tokens:
            raise BudgetExceededError(
                limit=str(self.max_tokens), attempted=str(projected_tokens),
            )

        projected_cost = self._project_cost(estimated_cost)
        if self.max_cost is not None and projected_cost > _to_decimal(self.max_cost):
            raise BudgetExceededError(
                limit=str(_to_decimal(self.max_cost)), attempted=str(projected_cost),
            )

    def register_usage(
        self,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        cost: Decimal | float | None = None,
    ) -> None:
        """Record consumed tokens and cost, enforcing limits strictly."""
        tokens_consumed = (prompt_tokens or 0) + (completion_tokens or 0)
        cost_consumed = _to_decimal(cost)

        self.used_tokens += tokens_consumed
        self.used_cost += cost_consumed

        if self.max_tokens is not None and self.used_tokens > self.max_tokens:
            raise BudgetExceededError(
                limit=str(self.max_tokens), attempted=str(self.used_tokens),
            )
        if self.max_cost is not None and self.used_cost > _to_decimal(self.max_cost):
            raise BudgetExceededError(
                limit=str(_to_decimal(self.max_cost)), attempted=str(self.used_cost),
            )

