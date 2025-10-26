"""LLM integration helpers."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "CompleteJsonResult",
    "ConfidenceLevel",
    "InstructionRole",
    "LLMProvider",
    "LLMSettings",
    "MessageDraft",
    "MessageDraftResult",
    "PatchProposalResult",
    "PatchSet",
    "PlanHint",
    "PlanHintResult",
    "ResolutionStrategy",
    "StrategyAdvice",
    "StrategyAdviceResult",
    "advise_strategy",
    "apply_plan_hint",
    "build_message_prompt",
    "build_plan_prompt",
    "build_strategy_prompt",
    "clamp_cost_adjustment",
    "complete_json",
    "compose_instructions",
    "make_client_from_env",
    "messenger_instructions",
    "planner_instructions",
    "request_message_draft",
    "request_plan_hint",
    "resolver_instructions",
    "sanitize_model_schema",
    "validate_message_draft",
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
    from .schema import (
        ConfidenceLevel,
        MessageDraft,
        PatchSet,
        PlanHint,
        ResolutionStrategy,
        StrategyAdvice,
        sanitize_model_schema,
    )
    from .patch import PatchProposalResult
    from .message import (
        MessageDraftResult,
        build_message_prompt,
        request_message_draft,
        validate_message_draft,
    )
    from .advice import StrategyAdviceResult, advise_strategy, build_strategy_prompt
    from .plan import (
        PlanHintResult,
        apply_plan_hint,
        build_plan_prompt,
        clamp_cost_adjustment,
        request_plan_hint,
    )


_MODULE_EXPORTS: dict[str, tuple[str, str]] = {
    "CompleteJsonResult": ("goapgit.llm.responses", "CompleteJsonResult"),
    "complete_json": ("goapgit.llm.responses", "complete_json"),
    "InstructionRole": ("goapgit.llm.instructions", "InstructionRole"),
    "ConfidenceLevel": ("goapgit.llm.schema", "ConfidenceLevel"),
    "MessageDraft": ("goapgit.llm.schema", "MessageDraft"),
    "MessageDraftResult": ("goapgit.llm.message", "MessageDraftResult"),
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
    "PatchProposalResult": ("goapgit.llm.patch", "PatchProposalResult"),
    "StrategyAdviceResult": ("goapgit.llm.advice", "StrategyAdviceResult"),
    "advise_strategy": ("goapgit.llm.advice", "advise_strategy"),
    "build_strategy_prompt": ("goapgit.llm.advice", "build_strategy_prompt"),
    "PlanHintResult": ("goapgit.llm.plan", "PlanHintResult"),
    "request_plan_hint": ("goapgit.llm.plan", "request_plan_hint"),
    "build_message_prompt": ("goapgit.llm.message", "build_message_prompt"),
    "build_plan_prompt": ("goapgit.llm.plan", "build_plan_prompt"),
    "apply_plan_hint": ("goapgit.llm.plan", "apply_plan_hint"),
    "request_message_draft": ("goapgit.llm.message", "request_message_draft"),
    "clamp_cost_adjustment": ("goapgit.llm.plan", "clamp_cost_adjustment"),
    "validate_message_draft": ("goapgit.llm.message", "validate_message_draft"),
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
