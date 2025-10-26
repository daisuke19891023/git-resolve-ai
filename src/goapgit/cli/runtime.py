"""Helpers shared across CLI commands for planning and execution."""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from goapgit.actions import (
    apply_path_strategy,
    auto_trivial_resolve,
    create_backup_ref,
    ensure_clean_or_stash,
    rebase_continue_or_abort,
)
from goapgit.core.explain import ActionContext
from goapgit.core.models import ActionSpec, Config, GoalSpec, RepoState, StrategyRule
from goapgit.core.planner import SimplePlanner
from goapgit.git.facade import GitCommandError, GitFacade
from goapgit.git.observe import RepoObserver
from goapgit.io import StructuredLogger, load_config

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path
    from goapgit.core.executor import ActionRunner


@dataclass(slots=True)
class WorkflowContext:
    """Container bundling CLI dependencies for planning and execution."""

    repo_path: Path
    config: Config
    logger: StructuredLogger
    action_facade: GitFacade
    observer_facade: GitFacade
    observer: RepoObserver
    planner: SimplePlanner

    def build_action_runner(
        self,
    ) -> ActionRunner:
        """Return an executor-compatible action runner."""

        def runner(action: ActionSpec) -> bool:
            handler: ActionHandler | None = ACTION_HANDLERS.get(action.name)
            if handler is None:
                self.logger.warning("unknown action", action=action.name)
                return False

            try:
                result = handler.run(self, action)
            except GitCommandError as error:
                self.logger.error(
                    "action failed",
                    action=action.name,
                    returncode=error.returncode,
                    stderr=error.stderr,
                )
                return False
            except Exception as error:  # pragma: no cover - defensive logging
                self.logger.error("unexpected action failure", action=action.name, error=str(error))
                return False

            return bool(result)

        return runner


def default_config() -> Config:
    """Return the default configuration used when no config file is provided."""
    return Config(
        goal=GoalSpec(),
        strategy_rules=[],
        enable_rerere=True,
        conflict_style="zdiff3",
        allow_force_push=False,
        dry_run=True,
        max_test_runtime_sec=600,
    )


def load_cli_config(config_path: Path | None) -> Config:
    """Load configuration from ``config_path`` or fall back to defaults."""
    if config_path is None:
        return default_config()
    return load_config(path=config_path)


def build_workflow_context(
    repo_path: Path,
    config: Config,
    *,
    json_logs: bool,
    dry_run_actions: bool,
    silence_logs: bool,
) -> WorkflowContext:
    """Assemble the context required by CLI commands."""
    stream = io.StringIO() if silence_logs else sys.stderr
    logger = StructuredLogger(name="goapgit.cli", json_mode=json_logs, stream=stream)
    observer_facade = GitFacade(repo_path=repo_path, logger=logger, dry_run=False)
    action_facade = GitFacade(repo_path=repo_path, logger=logger, dry_run=dry_run_actions)
    observer = RepoObserver(observer_facade)
    planner = SimplePlanner()
    return WorkflowContext(
        repo_path=repo_path,
        config=config,
        logger=logger,
        action_facade=action_facade,
        observer_facade=observer_facade,
        observer=observer,
        planner=planner,
    )


@dataclass(frozen=True, slots=True)
class ActionHandler:
    """Bundle the data required to describe and execute an action."""

    name: str
    build_spec: Callable[[RepoState, Config], ActionSpec | None]
    build_context: Callable[[Config], ActionContext | None]
    run: Callable[[WorkflowContext, ActionSpec], bool]


def _build_create_backup_spec(_: RepoState, __: Config) -> ActionSpec:
    return ActionSpec(
        name="Safety:CreateBackupRef",
        cost=0.4,
        rationale="Create a recoverable snapshot before making changes.",
    )


def _build_create_backup_context(_: Config) -> ActionContext:
    return ActionContext(
        reason="Create a timestamped backup ref so HEAD can be restored if later steps fail.",
        alternatives=(
            "Skip the backup and rely on reflog entries for recovery.",
            "Create a lightweight branch instead of an update-ref entry.",
        ),
        cost_override=1.0,
    )


def _run_create_backup(context: WorkflowContext, _: ActionSpec) -> bool:
    create_backup_ref(context.action_facade, context.logger)
    return True


def _build_ensure_clean_spec(_: RepoState, __: Config) -> ActionSpec:
    return ActionSpec(
        name="Safety:EnsureCleanOrStash",
        cost=0.6,
        rationale="Ensure the working tree is clean or safely stashed.",
    )


def _build_ensure_clean_context(_: Config) -> ActionContext:
    return ActionContext(
        reason="Guarantee a clean working tree before automated operations continue.",
        alternatives=(
            "Abort the workflow and ask the operator to clean up manually.",
            "Create a temporary worktree rather than stashing changes.",
        ),
        cost_override=0.6,
    )


def _run_ensure_clean(context: WorkflowContext, _: ActionSpec) -> bool:
    ensure_clean_or_stash(context.action_facade, context.logger)
    return True


def _build_auto_trivial_spec(_: RepoState, __: Config) -> ActionSpec:
    return ActionSpec(
        name="Conflict:AutoTrivialResolve",
        cost=0.8,
        rationale="Reuse rerere knowledge to resolve trivial conflicts.",
    )


def _build_auto_trivial_context(_: Config) -> ActionContext:
    return ActionContext(
        reason="Reuse git rerere to automatically apply previously recorded resolutions.",
        alternatives=(
            "Resolve conflicts manually to confirm each change.",
            "Run a domain specific merge driver for known file types.",
        ),
        cost_override=0.8,
    )


def _run_auto_trivial(context: WorkflowContext, _: ActionSpec) -> bool:
    auto_trivial_resolve(context.action_facade, context.logger)
    return True


def _build_apply_strategy_spec(_: RepoState, config: Config) -> ActionSpec | None:
    if not config.strategy_rules:
        return None
    return ActionSpec(
        name="Conflict:ApplyPathStrategy",
        cost=1.2,
        rationale="Apply configured conflict resolution strategies to matching paths.",
    )


def _build_apply_strategy_context(config: Config) -> ActionContext | None:
    if not config.strategy_rules:
        return None
    return ActionContext(
        reason="Use configured strategy rules to prefer ours/theirs on matching paths.",
        alternatives=(
            "Escalate to manual resolution in an editor.",
            "Invoke a custom merge driver tuned for the file type.",
        ),
        cost_override=1.2,
    )


def _run_apply_strategy(context: WorkflowContext, _: ActionSpec) -> bool:
    state = context.observer.observe()
    apply_path_strategy(
        context.action_facade,
        context.logger,
        state.conflicts,
        context.config.strategy_rules,
    )
    return True


def _build_rebase_spec(state: RepoState, _: Config) -> ActionSpec | None:
    if not state.ongoing_rebase:
        return None
    return ActionSpec(
        name="Rebase:ContinueOrAbort",
        cost=1.5,
        rationale="Complete or abort the ongoing rebase safely.",
    )


def _build_rebase_context(_: Config) -> ActionContext:
    return ActionContext(
        reason="Continue the rebase if conflicts are cleared, otherwise abort to restore HEAD.",
        alternatives=(
            "Abort immediately without attempting to continue.",
            "Skip rebase continuation and return control to the operator.",
        ),
        cost_override=1.5,
    )


def _run_rebase(context: WorkflowContext, action: ActionSpec) -> bool:
    backup_ref = action.params.get("backup_ref") if action.params else None
    return rebase_continue_or_abort(
        context.action_facade,
        context.logger,
        backup_ref=backup_ref,
    )


ACTION_HANDLER_SEQUENCE: tuple[ActionHandler, ...] = (
    ActionHandler(
        name="Safety:CreateBackupRef",
        build_spec=_build_create_backup_spec,
        build_context=_build_create_backup_context,
        run=_run_create_backup,
    ),
    ActionHandler(
        name="Safety:EnsureCleanOrStash",
        build_spec=_build_ensure_clean_spec,
        build_context=_build_ensure_clean_context,
        run=_run_ensure_clean,
    ),
    ActionHandler(
        name="Conflict:AutoTrivialResolve",
        build_spec=_build_auto_trivial_spec,
        build_context=_build_auto_trivial_context,
        run=_run_auto_trivial,
    ),
    ActionHandler(
        name="Conflict:ApplyPathStrategy",
        build_spec=_build_apply_strategy_spec,
        build_context=_build_apply_strategy_context,
        run=_run_apply_strategy,
    ),
    ActionHandler(
        name="Rebase:ContinueOrAbort",
        build_spec=_build_rebase_spec,
        build_context=_build_rebase_context,
        run=_run_rebase,
    ),
)


ACTION_HANDLERS: dict[str, ActionHandler] = {
    handler.name: handler for handler in ACTION_HANDLER_SEQUENCE
}


def build_action_specs(state: RepoState, config: Config) -> list[ActionSpec]:
    """Return the default action catalogue for the current ``state``."""
    actions: list[ActionSpec] = []
    for handler in ACTION_HANDLER_SEQUENCE:
        spec = handler.build_spec(state, config)
        if spec is not None:
            actions.append(spec)
    return actions


def build_action_contexts(config: Config) -> dict[str, ActionContext]:
    """Create explanation metadata for known actions."""
    contexts: dict[str, ActionContext] = {}
    for handler in ACTION_HANDLER_SEQUENCE:
        context_value = handler.build_context(config)
        if context_value is not None:
            contexts[handler.name] = context_value
    return contexts


def strategy_rules_to_params(rules: Sequence[StrategyRule]) -> list[dict[str, str | None]]:
    """Convert strategy rules to serialisable dictionaries for display."""
    return [
        {
            "pattern": rule.pattern,
            "resolution": rule.resolution,
            "when": rule.when,
        }
        for rule in rules
    ]


__all__ = [
    "ACTION_HANDLERS",
    "ACTION_HANDLER_SEQUENCE",
    "ActionHandler",
    "WorkflowContext",
    "build_action_contexts",
    "build_action_specs",
    "build_workflow_context",
    "default_config",
    "load_cli_config",
    "strategy_rules_to_params",
]

