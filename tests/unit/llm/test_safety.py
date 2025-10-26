"""Unit tests for LLM safety helpers."""

from __future__ import annotations

from decimal import Decimal

import pytest

from goapgit.llm.safety import BudgetExceededError, BudgetTracker, Redactor


def test_redactor_masks_pseudo_keys() -> None:
    """Secrets embedded in prompts should be masked before logging."""
    redactor = Redactor()
    prompt = "OpenAI key sk-test-ABCD1234EFGH5678 and Bearer sk-live-secret"

    result = redactor.redact(prompt)

    assert result.has_redactions is True
    assert "ABCD1234EFGH5678" not in result.redacted
    assert "sk-test-ABCD" in result.redacted
    assert "sk-live-secret" not in result.redacted


def test_budget_tracker_prevents_token_overrun() -> None:
    """Projected token usage above the limit should trigger a guard."""
    tracker = BudgetTracker(max_tokens=100)
    tracker.ensure_within_budget(prompt_tokens=40, completion_tokens=20)
    tracker.register_usage(prompt_tokens=40, completion_tokens=20)

    with pytest.raises(BudgetExceededError, match="LLM budget exceeded"):
        tracker.ensure_within_budget(prompt_tokens=50, completion_tokens=30)


def test_budget_tracker_prevents_cost_overrun() -> None:
    """Both estimated and actual cost overages must raise an error."""
    tracker = BudgetTracker(max_cost=Decimal("5.00"))
    tracker.ensure_within_budget(estimated_cost=Decimal("2.50"))
    tracker.register_usage(cost=Decimal("2.50"))

    with pytest.raises(BudgetExceededError):
        tracker.ensure_within_budget(estimated_cost=Decimal("2.75"))

    with pytest.raises(BudgetExceededError):
        tracker.register_usage(cost=Decimal("3.00"))

