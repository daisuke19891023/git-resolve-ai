"""Safety helpers for LLM interactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
import math
import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Iterable


class RedactionRule(BaseModel):
    """Declarative rule describing a redaction pattern."""

    name: str
    pattern: str
    replacement: str = Field(default="***", min_length=1)
    flags: int = re.IGNORECASE

    model_config = ConfigDict(frozen=True, extra="forbid")

    @cached_property
    def compiled(self) -> re.Pattern[str]:
        """Return the compiled regular expression for this rule."""
        return re.compile(self.pattern, self.flags)


class RedactionMatch(BaseModel):
    """Information about a redacted fragment."""

    rule: str
    start: int
    end: int

    model_config = ConfigDict(frozen=True, extra="forbid")


class RedactionResult(BaseModel):
    """Redacted text and summary of applied rules."""

    text: str
    matches: tuple[RedactionMatch, ...] = Field(default_factory=tuple)

    model_config = ConfigDict(frozen=True, extra="forbid")

    @property
    def has_matches(self) -> bool:
        """Return whether any redactions were applied."""
        return bool(self.matches)


class Redactor:
    """Apply a series of regex-based redaction rules to text."""

    _rules: tuple[RedactionRule, ...]

    def __init__(self, *, rules: Iterable[RedactionRule] | None = None) -> None:
        """Initialise the redactor with the provided rules."""
        if rules is None:
            rules = DEFAULT_RULES
        self._rules = tuple(rules)

    def redact(self, text: str) -> RedactionResult:
        """Mask fragments matching any configured rule."""
        working = text
        matches: list[RedactionMatch] = []

        for rule in self._rules:
            def _replacement(match: re.Match[str], *, _rule: RedactionRule = rule) -> str:
                matches.append(
                    RedactionMatch(
                        rule=_rule.name,
                        start=match.start(),
                        end=match.end(),
                    ),
                )
                return _rule.replacement

            working = rule.compiled.sub(_replacement, working)

        return RedactionResult(text=working, matches=tuple(matches))


DEFAULT_RULES: tuple[RedactionRule, ...] = (
    RedactionRule(name="openai_key", pattern=r"sk-[A-Za-z0-9]{20,}", replacement="sk-***"),
    RedactionRule(name="aws_key", pattern=r"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])", replacement="AKIA***"),
    RedactionRule(
        name="secret_assignment",
        pattern=r"(?i)(secret|api|token|key)[_-]?(id|key|token)?\s*[:=]\s*([A-Za-z0-9-_]{8,})",
        replacement="\1\2=***",
    ),
    RedactionRule(
        name="bearer_token",
        pattern=r"Bearer\s+[A-Za-z0-9\-_=]{8,}",
        replacement="Bearer ***",
    ),
)


class BudgetExceededError(RuntimeError):
    """Raised when the configured LLM budget has been exhausted."""


class UsageMetrics(BaseModel):
    """Normalised usage statistics reported by LLM calls."""

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cost: float | None = Field(default=None, ge=0.0)

    model_config = ConfigDict(extra="forbid")

    @property
    def total(self) -> int:
        """Return the best available total token count."""
        if self.total_tokens is not None:
            return self.total_tokens
        prompt = self.prompt_tokens or 0
        completion = self.completion_tokens or 0
        return prompt + completion


@dataclass(slots=True)
class BudgetTracker:
    """Track cumulative usage against configured budgets."""

    max_tokens: int | None = None
    max_cost: float | None = None
    _consumed_tokens: int = field(init=False, default=0)
    _consumed_cost: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        """Validate configured limits and initialise counters."""
        if self.max_tokens is not None and self.max_tokens < 1:
            msg = "max_tokens must be positive when provided"
            raise ValueError(msg)
        if self.max_cost is not None and self.max_cost <= 0:
            msg = "max_cost must be positive when provided"
            raise ValueError(msg)

    def ensure_can_continue(self) -> None:
        """Raise :class:`BudgetExceededError` if the budget is exhausted."""
        if self.max_tokens is not None and self._consumed_tokens >= self.max_tokens:
            msg = (
                "LLM token budget exhausted: "
                f"consumed={self._consumed_tokens} limit={self.max_tokens}"
            )
            raise BudgetExceededError(msg)
        if self.max_cost is not None and self._consumed_cost >= self.max_cost:
            msg = (
                "LLM cost budget exhausted: "
                f"consumed=${self._consumed_cost:.6f} limit=${self.max_cost:.6f}"
            )
            raise BudgetExceededError(msg)

    def register(self, usage: UsageMetrics) -> None:
        """Record a usage event and enforce limits."""
        self.ensure_can_continue()
        total_tokens = usage.total
        self._consumed_tokens += total_tokens
        if usage.cost is not None:
            self._consumed_cost += usage.cost
        self._check_limits()

    def _check_limits(self) -> None:
        if self.max_tokens is not None and self._consumed_tokens > self.max_tokens:
            msg = (
                "LLM token budget exceeded: "
                f"consumed={self._consumed_tokens} limit={self.max_tokens}"
            )
            raise BudgetExceededError(msg)
        if self.max_cost is not None and self._consumed_cost > self.max_cost and not math.isclose(
            self._consumed_cost, self.max_cost,
        ):
            msg = (
                "LLM cost budget exceeded: "
                f"consumed=${self._consumed_cost:.6f} limit=${self.max_cost:.6f}"
            )
            raise BudgetExceededError(msg)

    @property
    def consumed_tokens(self) -> int:
        """Return the cumulative consumed tokens."""
        return self._consumed_tokens

    @property
    def consumed_cost(self) -> float:
        """Return the cumulative consumed cost."""
        return self._consumed_cost

    @property
    def remaining_tokens(self) -> int | None:
        """Return remaining token budget, if available."""
        if self.max_tokens is None:
            return None
        return max(self.max_tokens - self._consumed_tokens, 0)

    @property
    def remaining_cost(self) -> float | None:
        """Return remaining monetary budget, if available."""
        if self.max_cost is None:
            return None
        return max(self.max_cost - self._consumed_cost, 0.0)


__all__ = [
    "DEFAULT_RULES",
    "BudgetExceededError",
    "BudgetTracker",
    "RedactionMatch",
    "RedactionResult",
    "RedactionRule",
    "Redactor",
    "UsageMetrics",
]
