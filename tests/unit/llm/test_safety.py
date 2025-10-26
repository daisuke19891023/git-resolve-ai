"""Tests for LLM safety helpers."""

from __future__ import annotations

import pytest

from goapgit.llm.safety import (
    BudgetExceededError,
    BudgetTracker,
    Redactor,
    UsageMetrics,
)


def test_redactor_masks_pseudo_keys() -> None:
    """Redactor should mask suspicious key-like fragments."""
    redactor = Redactor()

    result = redactor.redact("api token sk-testsecretkey1234567890")

    assert result.has_matches
    assert "sk-***" in result.text
    assert all(match.rule for match in result.matches)


def test_redactor_leaves_safe_text_untouched() -> None:
    """Text without secrets should remain unchanged."""
    redactor = Redactor()

    result = redactor.redact("plain status message")

    assert result.text == "plain status message"
    assert not result.has_matches


def test_budget_tracker_enforces_token_limit() -> None:
    """Registering usage beyond the token limit raises an error."""
    tracker = BudgetTracker(max_tokens=10)

    tracker.register(UsageMetrics(prompt_tokens=4, completion_tokens=3))
    tracker.register(UsageMetrics(total_tokens=3))

    with pytest.raises(BudgetExceededError, match="token budget"):
        tracker.register(UsageMetrics(total_tokens=1))


def test_budget_tracker_enforces_cost_limit() -> None:
    """Cost ceilings should block further registrations when exceeded."""
    tracker = BudgetTracker(max_cost=1.0)

    tracker.register(UsageMetrics(total_tokens=1, cost=0.7))

    with pytest.raises(BudgetExceededError, match="cost budget"):
        tracker.register(UsageMetrics(total_tokens=1, cost=0.4))

