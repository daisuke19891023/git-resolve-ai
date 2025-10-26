"""LLM integration helpers."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "BudgetExceededError",
    "BudgetTracker",
    "CompleteJsonResult",
    "ConfidenceLevel",
    "InstructionRole",
    "LLMProvider",
    "LLMSettings",
    "MessageDraft",
    "PatchSet",
    "PlanHint",
    "RedactionResult",
    "RedactionRule",
    "Redactor",
    "ResolutionStrategy",
    "StrategyAdvice",
    "complete_json",
    "compose_instructions",
    "make_client_from_env",
    "messenger_instructions",
    "planner_instructions",
    "resolver_instructions",
    "sanitize_model_schema",
]


if TYPE_CHECKING:
    from .client import LLMProvider, LLMSettings, make_client_from_env
    from .responses import CompleteJsonResult, complete_json
    from .instructions import (
        InstructionRole,
        compose_instructions,
        messenger_instructions,
        planner_instructions,
        resolver_instructions,
    )
    from .safety import (
        BudgetExceededError,
        BudgetTracker,
        RedactionResult,
        RedactionRule,
        Redactor,
    )
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
    "InstructionRole": ("goapgit.llm.instructions", "InstructionRole"),
    "ConfidenceLevel": ("goapgit.llm.schema", "ConfidenceLevel"),
    "MessageDraft": ("goapgit.llm.schema", "MessageDraft"),
    "PatchSet": ("goapgit.llm.schema", "PatchSet"),
    "PlanHint": ("goapgit.llm.schema", "PlanHint"),
    "ResolutionStrategy": ("goapgit.llm.schema", "ResolutionStrategy"),
    "StrategyAdvice": ("goapgit.llm.schema", "StrategyAdvice"),
    "sanitize_model_schema": ("goapgit.llm.schema", "sanitize_model_schema"),
    "compose_instructions": ("goapgit.llm.instructions", "compose_instructions"),
    "messenger_instructions": ("goapgit.llm.instructions", "messenger_instructions"),
    "planner_instructions": ("goapgit.llm.instructions", "planner_instructions"),
    "resolver_instructions": ("goapgit.llm.instructions", "resolver_instructions"),
    "LLMProvider": ("goapgit.llm.client", "LLMProvider"),
    "LLMSettings": ("goapgit.llm.client", "LLMSettings"),
    "make_client_from_env": ("goapgit.llm.client", "make_client_from_env"),
    "BudgetExceededError": ("goapgit.llm.safety", "BudgetExceededError"),
    "BudgetTracker": ("goapgit.llm.safety", "BudgetTracker"),
    "RedactionResult": ("goapgit.llm.safety", "RedactionResult"),
    "RedactionRule": ("goapgit.llm.safety", "RedactionRule"),
    "Redactor": ("goapgit.llm.safety", "Redactor"),
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
