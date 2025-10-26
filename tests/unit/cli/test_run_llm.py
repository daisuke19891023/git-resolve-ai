"""Unit tests covering LLM assistance for the run command."""

from __future__ import annotations

from dataclasses import dataclass
import io
from typing import TYPE_CHECKING, Any, cast

import pytest

from goapgit.cli.run import (
    LLMRunMode,
    LLMRunOptions,
    LLMSafetyLevel,
    perform_llm_assistance,
)
from goapgit.core.models import ConflictDetail, ConflictType, Plan, RepoRef, RepoState
from goapgit.io.logging import StructuredLogger
from goapgit.llm.patch import PatchProposalResult
from goapgit.llm.plan import PlanHint, PlanHintResult
from goapgit.llm.schema import PatchSet
from goapgit.cli import run as run_module
from goapgit.cli.runtime import default_config
from goapgit.git.facade import GitFacade

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class _StubContext:
    repo_path: Path
    config: object
    logger: StructuredLogger
    action_facade: GitFacade


@pytest.fixture
def workspace(tmp_path: Path) -> _StubContext:
    """Create a stub context with a dry-run git facade."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    conflict_file = repo_path / "conflict.txt"
    conflict_file.write_text(
        """<<<<<<< ours\n- old\n=======\n- new\n>>>>>>> theirs\n""",
        encoding="utf-8",
    )

    logger = StructuredLogger(name="test", json_mode=True, stream=io.StringIO())
    facade = GitFacade(repo_path=repo_path, logger=logger, dry_run=True)
    return _StubContext(
        repo_path=repo_path,
        config=default_config(),
        logger=logger,
        action_facade=facade,
    )


def _make_state(repo_path: Path) -> RepoState:
    return RepoState(
        repo_path=repo_path,
        ref=RepoRef(branch="main"),
        conflicts=(
            ConflictDetail(path="conflict.txt", hunk_count=1, ctype=ConflictType.text),
        ),
    )


def _make_plan() -> Plan:
    return Plan(actions=[], estimated_cost=1.0)


def test_perform_llm_assistance_returns_plan_hint(monkeypatch: pytest.MonkeyPatch, workspace: _StubContext) -> None:
    """Explain mode should include plan hint payload."""

    def fake_make_client() -> object:  # pragma: no cover - simple stub
        return object()

    hint = PlanHint(action="Adjust", cost_adjustment_pct=0.1, note="ok")
    hint_result = PlanHintResult(hint=hint, response_id="resp-1", output_text="{}")
    monkeypatch.setattr(run_module, "make_client_from_env", fake_make_client)

    def fake_request_plan_hint(*_: Any, **__: Any) -> PlanHintResult:
        return hint_result

    monkeypatch.setattr(run_module, "request_plan_hint", fake_request_plan_hint)

    payload = perform_llm_assistance(
        context=workspace,
        state=_make_state(workspace.repo_path),
        plan=_make_plan(),
        options=LLMRunOptions(mode=LLMRunMode.EXPLAIN, safety=LLMSafetyLevel.BALANCED, model="test-model"),
    )

    assert payload["mode"] == "explain"
    assert payload["plan_hint"]["action"] == "Adjust"
    assert abs(payload["plan_hint"]["cost_adjustment_pct"] - 0.1) < 1e-6


def test_perform_llm_assistance_auto_applies_when_safe(
    monkeypatch: pytest.MonkeyPatch, workspace: _StubContext,
) -> None:
    """Auto mode should apply patches when the safety threshold is satisfied."""

    def fake_make_client() -> object:  # pragma: no cover - simple stub
        return object()

    patch = (
        "diff --git a/conflict.txt b/conflict.txt\n"
        "@@ -1,3 +1 @@\n"
        "-<<<<<<< ours\n"
        "- old\n"
        "-=======\n"
        "- new\n"
        "->>>>>>> theirs\n"
        "+resolved\n"
    )
    patch_set = PatchSet(patches=(patch,), confidence="high", rationale="safe")
    proposal = PatchProposalResult(patch_set=patch_set, response_id="resp-2", output_text=patch)

    monkeypatch.setattr(run_module, "make_client_from_env", fake_make_client)

    def fake_propose_patch(*_: Any, **__: Any) -> PatchProposalResult:
        return proposal

    monkeypatch.setattr(run_module, "propose_patch", fake_propose_patch)

    payload = perform_llm_assistance(
        context=workspace,
        state=_make_state(workspace.repo_path),
        plan=_make_plan(),
        options=LLMRunOptions(mode=LLMRunMode.AUTO, safety=LLMSafetyLevel.BALANCED, model="test-model"),
    )

    suggestions = payload["suggestions"]
    assert suggestions, "LLM suggestions should be present"
    assert suggestions[0]["applied"] is True
    commands = cast("list[dict[str, Any]]", list(workspace.action_facade.command_history))
    applied = any(
        isinstance(entry.get("command"), list) and entry["command"][0:2] == ["git", "apply"]
        for entry in commands
    )
    assert applied


def test_perform_llm_assistance_respects_safety(
    monkeypatch: pytest.MonkeyPatch, workspace: _StubContext,
) -> None:
    """Auto mode should avoid applying patches when confidence is low."""

    def fake_make_client() -> object:  # pragma: no cover - simple stub
        return object()

    patch = (
        "diff --git a/conflict.txt b/conflict.txt\n"
        "@@ -1,3 +1 @@\n"
        "-<<<<<<< ours\n"
        "- old\n"
        "-=======\n"
        "- new\n"
        "->>>>>>> theirs\n"
        "+resolved\n"
    )
    patch_set = PatchSet(patches=(patch,), confidence="low", rationale="uncertain")
    proposal = PatchProposalResult(patch_set=patch_set, response_id="resp-3", output_text=patch)

    monkeypatch.setattr(run_module, "make_client_from_env", fake_make_client)

    def fake_propose_patch_low(*_: Any, **__: Any) -> PatchProposalResult:
        return proposal

    monkeypatch.setattr(run_module, "propose_patch", fake_propose_patch_low)

    payload = perform_llm_assistance(
        context=workspace,
        state=_make_state(workspace.repo_path),
        plan=_make_plan(),
        options=LLMRunOptions(mode=LLMRunMode.AUTO, safety=LLMSafetyLevel.CAUTIOUS, model="test-model"),
    )

    suggestions = payload["suggestions"]
    assert suggestions, "LLM suggestions should be present"
    assert suggestions[0]["applied"] is False
    commands = workspace.action_facade.command_history
    assert commands == ()
