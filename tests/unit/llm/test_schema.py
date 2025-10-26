"""Tests for the schema sanitizer utilities."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from goapgit.llm.schema import (
    MessageDraft,
    PatchSet,
    PlanHint,
    StrategyAdvice,
    sanitize_model_schema,
)


def test_patchset_schema_is_strict_and_required() -> None:
    """PatchSet schema must be strict with all fields required."""
    schema = sanitize_model_schema(PatchSet)

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["confidence", "patches", "rationale"]

    patches_schema = schema["properties"]["patches"]
    assert patches_schema["type"] == "array"
    assert patches_schema["items"]["type"] == "string"

    confidence_schema = schema["properties"]["confidence"]
    assert confidence_schema["enum"] == ["low", "med", "high"]


def test_plan_hint_optional_field_is_required_with_nullability() -> None:
    """Optional fields should be marked required and accept explicit nulls."""
    schema = sanitize_model_schema(PlanHint)

    assert schema["required"] == ["action", "cost_adjustment_pct", "note"]

    note_schema = schema["properties"]["note"]
    assert note_schema["type"] == ["null", "string"]


def test_strategy_advice_schema_enforces_enums() -> None:
    """Structured enums must be preserved during sanitization."""
    schema = sanitize_model_schema(StrategyAdvice)

    resolution_schema = schema["properties"]["resolution"]
    assert resolution_schema["enum"] == ["ours", "theirs", "manual", "merge-driver"]

    confidence_schema = schema["properties"]["confidence"]
    assert confidence_schema["enum"] == ["low", "med", "high"]


def test_message_draft_schema_has_required_fields() -> None:
    """MessageDraft should provide a simple strict object schema."""
    schema = sanitize_model_schema(MessageDraft)

    assert schema == {
        "type": "object",
        "properties": {
            "body": {"type": "string"},
            "title": {"type": "string"},
        },
        "required": ["body", "title"],
        "additionalProperties": False,
    }


def test_sanitizer_rejects_excessive_depth() -> None:
    """A ValueError should be raised when schemas exceed the allowed depth."""

    class _LevelThree(BaseModel):
        value: str
        model_config = ConfigDict(extra="forbid")

    class _LevelTwo(BaseModel):
        nested: _LevelThree
        model_config = ConfigDict(extra="forbid")

    class _LevelOne(BaseModel):
        nested: _LevelTwo
        model_config = ConfigDict(extra="forbid")

    with pytest.raises(ValueError, match="maximum depth"):
        sanitize_model_schema(_LevelOne, max_depth=2)
