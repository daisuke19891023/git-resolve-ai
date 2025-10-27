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
    explain_range_diff,
    fetch_all,
    preview_merge_conflicts,
    push_with_lease,
    rebase_continue_or_abort,
    rebase_onto_upstream,
    run_tests,
)
from goapgit.core.explain import ActionContext
from goapgit.core.models import (
    ActionSpec,
    Config,
    GoalMode,
    GoalSpec,
    RepoState,
    StrategyRule,
)
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


def _split_tracking(tracking: str) -> tuple[str, str]:
    """Return the remote and branch components extracted from ``tracking``."""
    if "/" in tracking:
        remote, branch = tracking.split("/", 1)
        remote = remote or "origin"
        branch = branch or tracking
    else:
        remote, branch = "origin", tracking
    return remote, branch


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


def _build_preview_conflicts_spec(state: RepoState, _: Config) -> ActionSpec | None:
    tracking = state.ref.tracking
    if (
        tracking is None
        or state.diverged_remote <= 0
        or state.ongoing_rebase
        or state.ongoing_merge
        or state.conflicts
    ):
        return None
    return ActionSpec(
        name="Conflict:PreviewMergeConflicts",
        cost=0.9,
        rationale="Simulate the upstream merge to anticipate conflicts before rebasing.",
        params={"ours": "HEAD", "theirs": tracking},
    )


def _build_preview_conflicts_context(_: Config) -> ActionContext:
    return ActionContext(
        reason="Use git merge-tree to predict upcoming conflicts before applying a rebase.",
        alternatives=(
            "Skip the prediction step and resolve conflicts if they arise during the rebase.",
            "Create a throwaway worktree to experiment with a real merge instead of a preview.",
        ),
        cost_override=0.9,
    )


def _run_preview_conflicts(context: WorkflowContext, action: ActionSpec) -> bool:
    params = action.params or {}
    theirs = params.get("theirs")
    if not theirs:
        context.logger.warning("preview merge conflicts skipped; missing target branch")
        return False
    ours = params.get("ours", "HEAD")
    preview_merge_conflicts(context.observer_facade, context.logger, ours, theirs)
    return True


def _build_fetch_all_spec(state: RepoState, _: Config) -> ActionSpec | None:
    tracking = state.ref.tracking
    if tracking is None:
        return None
    if state.diverged_remote <= 0 or state.ongoing_rebase or state.ongoing_merge or state.conflicts:
        return None
    remote, _remote_branch = _split_tracking(tracking)
    return ActionSpec(
        name="Sync:FetchAll",
        cost=1.1,
        rationale="Fetch the latest remote refs before attempting to rebase onto upstream.",
        params={"remote": remote},
    )


def _build_fetch_all_context(_: Config) -> ActionContext:
    return ActionContext(
        reason="Refresh local knowledge of the upstream branch to avoid rebasing on stale commits.",
        alternatives=(
            "Skip the fetch and rely on locally cached refs, risking conflicts with unseen commits.",
            "Fetch only the tracked branch instead of the whole remote.",
        ),
        cost_override=1.1,
    )


def _run_fetch_all(context: WorkflowContext, action: ActionSpec) -> bool:
    remote = (action.params or {}).get("remote") or "origin"
    fetch_all(context.action_facade, context.logger, remote=remote)
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


def _build_rebase_onto_spec(state: RepoState, config: Config) -> ActionSpec | None:
    tracking = state.ref.tracking
    if (
        tracking is None
        or state.diverged_remote <= 0
        or state.ongoing_rebase
        or state.ongoing_merge
        or state.conflicts
    ):
        return None
    update_refs = config.goal.mode is not GoalMode.resolve_only
    return ActionSpec(
        name="Rebase:OntoUpstream",
        cost=1.0,
        rationale="Replay local commits on top of the latest upstream revision.",
        params={
            "upstream": tracking,
            "update_refs": "true" if update_refs else "false",
        },
    )


def _build_rebase_onto_context(_: Config) -> ActionContext:
    return ActionContext(
        reason="Align the branch with its tracked upstream so subsequent pushes fast-forward cleanly.",
        alternatives=(
            "Merge the upstream branch instead of rebasing, accepting a merge commit.",
            "Abort automated recovery and request manual intervention.",
        ),
        cost_override=1.0,
    )


def _run_rebase_onto(context: WorkflowContext, action: ActionSpec) -> bool:
    params = action.params or {}
    upstream = params.get("upstream")
    if not upstream:
        context.logger.warning("rebase onto upstream skipped; missing upstream reference")
        return False
    update_refs = params.get("update_refs", "false").lower() == "true"
    onto = params.get("onto")
    rebase_onto_upstream(
        context.action_facade,
        context.logger,
        upstream,
        update_refs=update_refs,
        onto=onto,
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


def _build_push_with_lease_spec(state: RepoState, config: Config) -> ActionSpec | None:
    tracking = state.ref.tracking
    if tracking is None:
        return None
    if not (config.goal.push_with_lease or config.goal.mode is GoalMode.push_with_lease):
        return None
    if state.ongoing_rebase or state.ongoing_merge:
        return None
    if not (state.has_unpushed_commits or state.diverged_local > 0):
        return None
    remote, remote_branch = _split_tracking(tracking)
    return ActionSpec(
        name="Sync:PushWithLease",
        cost=1.6,
        rationale="Publish local commits using --force-with-lease once the branch is clean.",
        params={
            "remote": remote,
            "remote_branch": remote_branch,
            "local_branch": state.ref.branch,
        },
    )


def _build_push_with_lease_context(config: Config) -> ActionContext | None:
    if not (config.goal.push_with_lease or config.goal.mode is GoalMode.push_with_lease):
        return None
    return ActionContext(
        reason="Update the remote branch with local commits using force-with-lease safeguards.",
        alternatives=(
            "Push without --force-with-lease and risk overwriting remote updates.",
            "Pause automation and request a human to review the outgoing commits.",
        ),
        cost_override=1.6,
    )


def _run_push_with_lease(context: WorkflowContext, action: ActionSpec) -> bool:
    params = action.params or {}
    remote = params.get("remote") or "origin"
    remote_branch = params.get("remote_branch")
    local_branch = params.get("local_branch")
    refspecs: list[str] | None = None
    if local_branch and remote_branch:
        refspecs = [f"{local_branch}:{remote_branch}"]
    push_with_lease(
        context.action_facade,
        context.logger,
        remote=remote,
        refspecs=refspecs,
        force=context.config.allow_force_push,
    )
    return True


def _build_run_tests_spec(state: RepoState, config: Config) -> ActionSpec | None:
    if not config.goal.tests_must_pass:
        return None
    if state.conflicts or not state.working_tree_clean:
        return None
    if state.ongoing_rebase or state.ongoing_merge:
        return None
    return ActionSpec(
        name="Quality:RunTests",
        cost=1.2,
        rationale="Execute the configured test command to validate the branch before pushing.",
        params={"timeout_sec": str(config.max_test_runtime_sec)},
    )


def _build_run_tests_context(config: Config) -> ActionContext | None:
    if not config.goal.tests_must_pass:
        return None
    return ActionContext(
        reason="Ensure automated checks pass before publishing branch updates.",
        alternatives=(
            "Skip the automated suite and rely solely on manual review.",
            "Run a lighter smoke-test suite instead of the full command.",
        ),
        cost_override=1.2,
    )


def _run_run_tests(context: WorkflowContext, _: ActionSpec) -> bool:
    timeout = float(context.config.max_test_runtime_sec) if context.config.max_test_runtime_sec > 0 else None
    run_tests(
        context.action_facade,
        context.logger,
        context.config.goal.tests_command,
        timeout=timeout,
    )
    return True


def _build_range_diff_spec(state: RepoState, _: Config) -> ActionSpec | None:
    tracking = state.ref.tracking
    if (
        tracking is None
        or (state.diverged_local <= 0 and not state.has_unpushed_commits)
        or state.ongoing_rebase
        or state.ongoing_merge
        or state.conflicts
    ):
        return None
    return ActionSpec(
        name="Quality:ExplainRangeDiff",
        cost=1.3,
        rationale="Summarise how local commits differ from the tracked upstream branch.",
        params={"tracking": tracking},
    )


def _build_range_diff_context(_: Config) -> ActionContext:
    return ActionContext(
        reason="Use git range-diff to describe the delta between local commits and upstream history.",
        alternatives=(
            "Review the commits manually with git log or git show.",
            "Generate a patch series and inspect it with git format-patch.",
        ),
        cost_override=1.3,
    )


def _run_range_diff(context: WorkflowContext, action: ActionSpec) -> bool:
    tracking = (action.params or {}).get("tracking")
    if not tracking:
        context.logger.warning("range-diff skipped; missing tracking branch")
        return False
    merge_base = context.observer_facade.run(["git", "merge-base", "HEAD", tracking])
    base = merge_base.stdout.strip()
    if not base:
        context.logger.warning("range-diff skipped; merge-base lookup returned empty result", tracking=tracking)
        return False
    before_range = f"{base}..{tracking}"
    after_range = f"{base}..HEAD"
    summary = explain_range_diff(context.observer_facade, context.logger, before_range, after_range)
    context.logger.info(
        "range-diff summary generated",
        before=before_range,
        after=after_range,
        has_output=bool(summary.strip()),
    )
    return True


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
        name="Conflict:PreviewMergeConflicts",
        build_spec=_build_preview_conflicts_spec,
        build_context=_build_preview_conflicts_context,
        run=_run_preview_conflicts,
    ),
    ActionHandler(
        name="Conflict:ApplyPathStrategy",
        build_spec=_build_apply_strategy_spec,
        build_context=_build_apply_strategy_context,
        run=_run_apply_strategy,
    ),
    ActionHandler(
        name="Sync:FetchAll",
        build_spec=_build_fetch_all_spec,
        build_context=_build_fetch_all_context,
        run=_run_fetch_all,
    ),
    ActionHandler(
        name="Rebase:OntoUpstream",
        build_spec=_build_rebase_onto_spec,
        build_context=_build_rebase_onto_context,
        run=_run_rebase_onto,
    ),
    ActionHandler(
        name="Rebase:ContinueOrAbort",
        build_spec=_build_rebase_spec,
        build_context=_build_rebase_context,
        run=_run_rebase,
    ),
    ActionHandler(
        name="Sync:PushWithLease",
        build_spec=_build_push_with_lease_spec,
        build_context=_build_push_with_lease_context,
        run=_run_push_with_lease,
    ),
    ActionHandler(
        name="Quality:RunTests",
        build_spec=_build_run_tests_spec,
        build_context=_build_run_tests_context,
        run=_run_run_tests,
    ),
    ActionHandler(
        name="Quality:ExplainRangeDiff",
        build_spec=_build_range_diff_spec,
        build_context=_build_range_diff_context,
        run=_run_range_diff,
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

