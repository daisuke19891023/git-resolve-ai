"""Utilities for building strict JSON schemas for Structured Outputs."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict

__all__ = [
    "ConfidenceLevel",
    "MessageDraft",
    "PatchSet",
    "PlanHint",
    "ResolutionStrategy",
    "StrategyAdvice",
    "sanitize_model_schema",
]

ConfidenceLevel = Literal["low", "med", "high"]
ResolutionStrategy = Literal["ours", "theirs", "manual", "merge-driver"]


class PatchSet(BaseModel):
    """Structured output describing a set of patches to apply."""

    patches: tuple[str, ...]
    confidence: ConfidenceLevel
    rationale: str
    model_config = ConfigDict(extra="forbid")


class StrategyAdvice(BaseModel):
    """Strategy recommendation for resolving a conflict."""

    resolution: ResolutionStrategy
    reason: str
    confidence: ConfidenceLevel
    model_config = ConfigDict(extra="forbid")


class PlanHint(BaseModel):
    """Hint with an adjusted action cost."""

    action: str
    cost_adjustment_pct: float
    note: str | None = None
    model_config = ConfigDict(extra="forbid")


class MessageDraft(BaseModel):
    """Draft for commit/PR messaging."""

    title: str
    body: str
    model_config = ConfigDict(extra="forbid")


def sanitize_model_schema(
    model: type[BaseModel], *, max_depth: int = 4,
) -> dict[str, Any]:
    """Return a strict JSON schema accepted by Structured Outputs.

    Parameters
    ----------
    model:
        Pydantic model representing the Structured Output contract.
    max_depth:
        Maximum allowed nesting depth. Structured Outputs reject schemas with
        overly deep object trees, so the sanitizer raises ``ValueError`` when the
        generated schema would exceed this threshold.

    """
    if max_depth < 1:
        msg = "max_depth must be >= 1"
        raise ValueError(msg)

    raw_schema = model.model_json_schema()
    definitions = cast("dict[str, Any]", raw_schema.pop("$defs", {}))
    inlined_schema = _inline_refs(raw_schema, definitions, set[str]())
    return _sanitize_schema_node(inlined_schema, depth=0, max_depth=max_depth)


_DROPPED_KEYS = {"title", "description", "examples", "default"}


def _sanitize_schema_node(schema: Any, *, depth: int, max_depth: int) -> Any:
    if depth > max_depth:
        msg = "Schema exceeds maximum depth for Structured Outputs"
        raise ValueError(msg)

    if not isinstance(schema, Mapping):
        return schema

    mapping_schema = cast("Mapping[str, Any]", schema)
    working = _strip_metadata(mapping_schema)

    nullable_schema = _extract_nullable_schema(working)
    if nullable_schema is not None:
        sanitized_non_null = _sanitize_schema_node(
            nullable_schema, depth=depth, max_depth=max_depth,
        )
        sanitized_non_null = cast("dict[str, Any]", sanitized_non_null)
        return _ensure_nullable_type(sanitized_non_null)

    schema_type = working.get("type")

    if schema_type == "object" or (schema_type is None and "properties" in working):
        return _sanitize_object_schema(working, depth=depth, max_depth=max_depth)

    if schema_type == "array":
        return _sanitize_array_schema(working, depth=depth, max_depth=max_depth)

    return dict(working)


def _sanitize_object_schema(
    schema: Mapping[str, Any], *, depth: int, max_depth: int,
) -> dict[str, Any]:
    next_depth = depth + 1
    properties = schema.get("properties", {})
    sanitized_properties: dict[str, Any] = {}
    for name, subschema in properties.items():
        sanitized_properties[name] = _sanitize_schema_node(
            subschema, depth=next_depth, max_depth=max_depth,
        )

    required = sorted(sanitized_properties)

    sanitized: dict[str, Any] = {
        key: value
        for key, value in schema.items()
        if key not in {"properties", "required", "additionalProperties"}
    }
    sanitized["type"] = "object"
    sanitized["properties"] = sanitized_properties
    sanitized["required"] = required
    sanitized["additionalProperties"] = False
    return sanitized


def _sanitize_array_schema(
    schema: Mapping[str, Any], *, depth: int, max_depth: int,
) -> dict[str, Any]:
    next_depth = depth + 1
    sanitized: dict[str, Any] = {
        key: value for key, value in schema.items() if key not in {"items"}
    }
    items = schema.get("items")
    if items is None:
        sanitized["items"] = {}
    else:
        sanitized["items"] = _sanitize_schema_node(
            items, depth=next_depth, max_depth=max_depth,
        )
    sanitized["type"] = "array"
    return sanitized


def _inline_refs(
    node: Any, definitions: Mapping[str, Any], seen: set[str],
) -> Any:
    if isinstance(node, Mapping):
        mapping_node = cast("Mapping[str, Any]", node)
        ref_value = mapping_node.get("$ref")
        if isinstance(ref_value, str):
            ref_name = _extract_ref_name(ref_value)
            if ref_name in seen:
                msg = f"Cyclic schema reference detected: {ref_name}"
                raise ValueError(msg)
            try:
                target_schema = definitions[ref_name]
            except KeyError as exc:  # pragma: no cover - defensive guard
                msg = f"Unknown schema reference: {ref_value}"
                raise ValueError(msg) from exc
            resolved = _inline_refs(
                deepcopy(target_schema), definitions, seen | {ref_name},
            )
            extras: dict[str, Any] = {
                key: _inline_refs(value, definitions, seen)
                for key, value in mapping_node.items()
                if key != "$ref"
            }
            if isinstance(resolved, Mapping):
                merged: dict[str, Any] = dict(cast("Mapping[str, Any]", resolved))
                merged.update(extras)
                return merged
            if extras:
                msg = "Non-object schema definitions cannot be merged with overrides"
                raise ValueError(msg)
            return resolved
        return {
            key: _inline_refs(value, definitions, seen)
            for key, value in mapping_node.items()
            if key != "$defs"
        }
    if isinstance(node, list):
        return [
            _inline_refs(item, definitions, seen)
            for item in cast("list[Any]", node)
        ]
    return node


def _extract_ref_name(reference: str) -> str:
    prefix = "#/$defs/"
    if reference.startswith(prefix):
        return reference[len(prefix) :]
    return reference.rsplit("/", 1)[-1]


def _strip_metadata(schema: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in schema.items() if key not in _DROPPED_KEYS}


def _extract_nullable_schema(schema: dict[str, Any]) -> Mapping[str, Any] | None:
    any_of = schema.get("anyOf")
    if not isinstance(any_of, list):
        return None

    null_option: Mapping[str, Any] | None = None
    non_null_option: Mapping[str, Any] | None = None
    for raw_candidate in cast("list[object]", any_of):
        if not isinstance(raw_candidate, Mapping):
            continue
        candidate_mapping = cast("Mapping[str, Any]", raw_candidate)
        candidate_type = candidate_mapping.get("type")
        if candidate_type == "null":
            null_option = candidate_mapping
        else:
            non_null_option = candidate_mapping

    if null_option is None or non_null_option is None:
        return None

    schema.pop("anyOf", None)
    return non_null_option


def _ensure_nullable_type(schema: dict[str, Any]) -> dict[str, Any]:
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        allowed_types: set[str] = {"null"}
        for raw_value in cast("list[object]", schema_type):
            if isinstance(raw_value, str):
                allowed_types.add(raw_value)
        schema["type"] = sorted(allowed_types)
    elif isinstance(schema_type, str):
        schema["type"] = sorted({schema_type, "null"})
    else:
        schema["type"] = ["null"]
    return schema
