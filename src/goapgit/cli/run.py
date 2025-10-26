"""LLM-aware helpers for the ``goapgit run`` command."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import tempfile
from typing import Any, TYPE_CHECKING, Protocol

from goapgit.llm.client import make_client_from_env
from goapgit.llm.patch import ConflictExcerpt, PatchProposalResult, propose_patch
from goapgit.llm.plan import PlanHintResult, request_plan_hint

if TYPE_CHECKING:
    from goapgit.core.models import ConflictDetail, Plan, RepoState
    from goapgit.llm.schema import PatchSet


class WorkflowContextLike(Protocol):
    repo_path: Path
    config: Any
    logger: Any
    action_facade: Any


DEFAULT_MODEL = "gpt-4o-mini"


class LLMRunMode(StrEnum):
    """Supported run modes for the CLI."""

    OFF = "off"
    EXPLAIN = "explain"
    SUGGEST = "suggest"
    AUTO = "auto"


class LLMSafetyLevel(StrEnum):
    """Safety thresholds used when auto-applying patches."""

    CAUTIOUS = "cautious"
    BALANCED = "balanced"
    EXPERIMENTAL = "experimental"


@dataclass(frozen=True, slots=True)
class LLMRunOptions:
    """User supplied configuration for LLM assistance."""

    mode: LLMRunMode = LLMRunMode.OFF
    safety: LLMSafetyLevel = LLMSafetyLevel.BALANCED
    model: str | None = None
    max_tokens: int | None = None
    max_cost: float | None = None
    mock: bool = False


@dataclass(frozen=True, slots=True)
class LLMSuggestion:
    """Suggestion returned by the LLM workflow."""

    path: str
    confidence: str
    rationale: str
    patches: tuple[str, ...]
    applied: bool
    response_id: str | None
    error: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": self.path,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "patch_count": len(self.patches),
            "applied": self.applied,
            "response_id": self.response_id,
        }
        if self.patches:
            payload["patches"] = list(self.patches)
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True, slots=True)
class LLMRunSummary:
    """High level summary reported back to the CLI."""

    mode: LLMRunMode
    safety: LLMSafetyLevel
    model: str | None
    plan_hint: dict[str, Any] | None
    suggestions: tuple[LLMSuggestion, ...]
    errors: tuple[str, ...]
    mock: bool

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mode": self.mode.value,
            "safety": self.safety.value,
        }
        if self.model:
            payload["model"] = self.model
        if self.plan_hint is not None:
            payload["plan_hint"] = self.plan_hint
        if self.suggestions:
            payload["suggestions"] = [s.to_payload() for s in self.suggestions]
        if self.errors:
            payload["errors"] = list(self.errors)
        if self.mock:
            payload["mock"] = True
        return payload


def perform_llm_assistance(
    *,
    context: WorkflowContextLike,
    state: RepoState,
    plan: Plan,
    options: LLMRunOptions,
) -> dict[str, Any]:
    """Execute optional LLM assistance and return a serialisable payload."""
    if options.mode is LLMRunMode.OFF:
        return {}

    if options.mock:
        context.logger.info("llm assistance running in mock mode", mode=options.mode.value)
        summary = LLMRunSummary(
            mode=options.mode,
            safety=options.safety,
            model=options.model,
            plan_hint=None,
            suggestions=(),
            errors=("LLM interactions skipped (mock mode)",),
            mock=True,
        )
        return summary.to_payload()

    model_name = options.model or DEFAULT_MODEL

    try:
        client = make_client_from_env()
    except Exception as exc:  # pragma: no cover - depends on environment
        context.logger.error("failed to initialise llm client", error=str(exc))
        summary = LLMRunSummary(
            mode=options.mode,
            safety=options.safety,
            model=model_name,
            plan_hint=None,
            suggestions=(),
            errors=(str(exc),),
            mock=False,
        )
        return summary.to_payload()

    plan_hint_payload: dict[str, Any] | None = None
    suggestions: list[LLMSuggestion] = []
    errors: list[str] = []
    previous_response_id: str | None = None

    if options.mode is LLMRunMode.EXPLAIN:
        plan_hint_payload, previous_response_id = _request_plan_hint(
            client,
            context=context,
            state=state,
            plan=plan,
            model=model_name,
            errors=errors,
        )

    if options.mode in {LLMRunMode.SUGGEST, LLMRunMode.AUTO}:
        suggestions = _gather_patch_suggestions(
            client,
            context=context,
            state=state,
            model=model_name,
            previous_response_id=previous_response_id,
            mode=options.mode,
            safety=options.safety,
            errors=errors,
        )

    summary = LLMRunSummary(
        mode=options.mode,
        safety=options.safety,
        model=model_name,
        plan_hint=plan_hint_payload,
        suggestions=tuple(suggestions),
        errors=tuple(errors),
        mock=False,
    )
    return summary.to_payload()


def _request_plan_hint(
    client: Any,
    *,
    context: WorkflowContextLike,
    state: RepoState,
    plan: Plan,
    model: str,
    errors: list[str],
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        result: PlanHintResult = request_plan_hint(
            client,
            model=model,
            state=state,
            plan=plan,
        )
    except Exception as exc:
        context.logger.error("plan hint request failed", error=str(exc))
        errors.append(str(exc))
        return None, None

    hint_payload = result.hint.model_dump(mode="json")
    context.logger.info("received plan hint", response_id=result.response_id, hint=hint_payload)
    return hint_payload, result.response_id


def _gather_patch_suggestions(
    client: Any,
    *,
    context: WorkflowContextLike,
    state: RepoState,
    model: str,
    previous_response_id: str | None,
    mode: LLMRunMode,
    safety: LLMSafetyLevel,
    errors: list[str],
) -> list[LLMSuggestion]:
    suggestions: list[LLMSuggestion] = []
    for conflict in state.conflicts:
        excerpt = _build_conflict_excerpt(context.repo_path, conflict)
        if excerpt is None:
            message = f"Unable to load conflict excerpt for {conflict.path}"
            context.logger.warning("skipping conflict", path=conflict.path, reason=message)
            suggestions.append(
                LLMSuggestion(
                    path=conflict.path,
                    confidence="low",
                    rationale="conflict excerpt unavailable",
                    patches=(),
                    applied=False,
                    response_id=None,
                    error=message,
                ),
            )
            continue

        try:
            result: PatchProposalResult = propose_patch(
                client,
                model=model,
                repo_path=context.repo_path,
                conflict=excerpt,
                previous_response_id=previous_response_id,
            )
        except Exception as exc:
            context.logger.error("patch proposal failed", path=conflict.path, error=str(exc))
            errors.append(f"{conflict.path}: {exc}")
            suggestions.append(
                LLMSuggestion(
                    path=conflict.path,
                    confidence="low",
                    rationale="patch proposal failed",
                    patches=(),
                    applied=False,
                    response_id=None,
                    error=str(exc),
                ),
            )
            continue

        previous_response_id = result.response_id
        patch_set = result.patch_set
        applied = False
        apply_error: str | None = None

        if mode is LLMRunMode.AUTO and _should_auto_apply(patch_set, safety):
            applied, apply_error = _apply_patch_set(context, patch_set)
        elif mode is LLMRunMode.AUTO:
            context.logger.info(
                "auto mode skipped patch",
                path=conflict.path,
                confidence=patch_set.confidence,
                patch_count=len(patch_set.patches),
            )

        suggestions.append(
            LLMSuggestion(
                path=conflict.path,
                confidence=patch_set.confidence,
                rationale=patch_set.rationale,
                patches=patch_set.patches,
                applied=applied,
                response_id=result.response_id,
                error=apply_error,
            ),
        )

    return suggestions


def _build_conflict_excerpt(repo_path: Path, conflict: ConflictDetail) -> ConflictExcerpt | None:
    candidate = Path(repo_path) / conflict.path
    try:
        text = candidate.read_text(encoding="utf-8")
    except OSError:
        text = ""

    snippet = text.strip()
    if not snippet:
        snippet = "Conflict markers could not be read.".strip()

    snippet = snippet[:2000]

    try:
        return ConflictExcerpt(path=conflict.path, snippet=snippet, ctype=conflict.ctype)
    except ValueError:
        return None


AUTO_APPLY_PATCH_LIMIT = 3


def _should_auto_apply(patch_set: PatchSet, safety: LLMSafetyLevel) -> bool:
    if not patch_set.patches:
        return False

    patch_count = len(patch_set.patches)
    confidence = patch_set.confidence

    if safety is LLMSafetyLevel.CAUTIOUS:
        return confidence == "high" and patch_count == 1
    if safety is LLMSafetyLevel.BALANCED:
        return confidence in {"high", "med"} and patch_count <= AUTO_APPLY_PATCH_LIMIT
    return confidence in {"high", "med"}


def _apply_patch_set(context: WorkflowContextLike, patch_set: PatchSet) -> tuple[bool, str | None]:
    facade = context.action_facade
    for patch in patch_set.patches:
        if not patch.strip():
            continue
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write(patch)
            handle.flush()
            temp_path = Path(handle.name)
        try:
            facade.run(["git", "apply", "--check", str(temp_path)])
            facade.run(["git", "apply", str(temp_path)])
        except Exception as exc:
            context.logger.error("git apply failed", error=str(exc))
            temp_path.unlink(missing_ok=True)
            return False, str(exc)
        temp_path.unlink(missing_ok=True)
    context.logger.info(
        "auto-applied patch set",
        confidence=patch_set.confidence,
        patch_count=len(patch_set.patches),
    )
    return True, None


__all__ = [
    "DEFAULT_MODEL",
    "LLMRunMode",
    "LLMRunOptions",
    "LLMSafetyLevel",
    "perform_llm_assistance",
]
