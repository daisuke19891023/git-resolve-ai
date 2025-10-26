"""LLM integration helpers."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "CompleteJsonResult",
    "ConfidenceLevel",
    "LLMProvider",
    "LLMSettings",
    "MessageDraft",
    "PatchSet",
    "PlanHint",
    "ResolutionStrategy",
    "StrategyAdvice",
    "complete_json",
    "make_client_from_env",
    "sanitize_model_schema",
]


if TYPE_CHECKING:
    from .client import LLMProvider, LLMSettings, make_client_from_env
    from .responses import CompleteJsonResult, complete_json
    from .schema import (
        ConfidenceLevel,
        MessageDraft,
        PatchSet,
        PlanHint,
        ResolutionStrategy,
        StrategyAdvice,
        sanitize_model_schema,
    )


_MODULE_EXPORTS: dict[str, tuple[str, str]] = {
    "CompleteJsonResult": ("goapgit.llm.responses", "CompleteJsonResult"),
    "complete_json": ("goapgit.llm.responses", "complete_json"),
    "ConfidenceLevel": ("goapgit.llm.schema", "ConfidenceLevel"),
    "MessageDraft": ("goapgit.llm.schema", "MessageDraft"),
    "PatchSet": ("goapgit.llm.schema", "PatchSet"),
    "PlanHint": ("goapgit.llm.schema", "PlanHint"),
    "ResolutionStrategy": ("goapgit.llm.schema", "ResolutionStrategy"),
    "StrategyAdvice": ("goapgit.llm.schema", "StrategyAdvice"),
    "sanitize_model_schema": ("goapgit.llm.schema", "sanitize_model_schema"),
    "LLMProvider": ("goapgit.llm.client", "LLMProvider"),
    "LLMSettings": ("goapgit.llm.client", "LLMSettings"),
    "make_client_from_env": ("goapgit.llm.client", "make_client_from_env"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _MODULE_EXPORTS[name]
    except KeyError as exc:  # pragma: no cover - defensive guard
        message = f"module 'goapgit.llm' has no attribute {name!r}"
        raise AttributeError(message) from exc

    module = import_module(module_name)
    value = getattr(module, attribute)
    globals()[name] = value
    return value
