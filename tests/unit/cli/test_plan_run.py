"""Tests covering plan/run/explain/dry-run CLI commands."""

from __future__ import annotations

import importlib
import io
import json
import subprocess
from typing import TYPE_CHECKING, Any, cast

import pytest
from typer.testing import CliRunner

from goapgit.cli.runtime import (
    ACTION_HANDLERS,
    WorkflowContext,
    build_action_specs,
    default_config,
)
from goapgit.core.models import ActionSpec, Config, RepoRef, RepoState
from goapgit.core.planner import SimplePlanner
from goapgit.git.facade import GitFacade
from goapgit.io.logging import StructuredLogger


cli_main = importlib.import_module("goapgit.cli.main")

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from goapgit.cli.main import PlanComputation
    from pytest_mock import MockerFixture

    WorkflowContextFactory = Callable[..., WorkflowContext]
    PlanPayloadBuilder = Callable[[WorkflowContext], PlanComputation]
else:
    WorkflowContextFactory = object
    PlanPayloadBuilder = object


@pytest.fixture
def git_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Prepare git identity and configuration for isolated repositories."""
    config_file = tmp_path / "gitconfig"
    config_file.write_text(
        """
[user]
    name = Test User
    email = test@example.com
[merge]
    conflictStyle = zdiff3
[pull]
    rebase = true
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    env = {
        "GIT_CONFIG_GLOBAL": str(config_file),
        "GIT_AUTHOR_NAME": "Test User",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test User",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return env


@pytest.fixture
def init_repo(tmp_path: Path, git_env: dict[str, str]) -> Path:
    """Create a git repository with an initial commit."""
    _ = git_env
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(("git", "init"), cwd=repo, check=True)
    (repo / "README.md").write_text("seed", encoding="utf-8")
    subprocess.run(("git", "add", "README.md"), cwd=repo, check=True)
    subprocess.run(("git", "commit", "-m", "initial"), cwd=repo, check=True)
    return repo


def test_plan_command_outputs_json(init_repo: Path) -> None:
    """Ensure the plan command provides structured JSON output."""
    runner = CliRunner()
    result = runner.invoke(cli_main.app, ["plan", "--repo", str(init_repo), "--json"])
    assert result.exit_code == 0, result.stderr

    payload = json.loads(result.stdout)
    assert payload["plan"]["actions"], "plan should contain at least one action"
    assert payload["state"]["ref"]["branch"], "branch name must be reported"


def test_plan_command_reports_validation_error(tmp_path: Path) -> None:
    """Invalid configuration files should surface a friendly validation error."""
    config_path = tmp_path / "invalid.toml"
    config_path.write_text("""
[goal]
mode = "invalid"
""".strip(), encoding="utf-8")

    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(cli_main.app, ["plan", "--config", str(config_path)])

    assert result.exit_code == 2
    assert "Invalid configuration" in result.stderr
    assert "goal.mode" in result.stderr
    assert "Traceback" not in result.stderr


def test_plan_command_reports_value_error_for_bad_path(tmp_path: Path) -> None:
    """Invalid config paths should surface ValueError messages cleanly."""
    config_dir = tmp_path / "config_dir"
    config_dir.mkdir()

    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(cli_main.app, ["plan", "--config", str(config_dir)])

    assert result.exit_code == 2
    assert "not a file" in result.stderr
    assert str(config_dir) in result.stderr


def test_run_command_without_confirm_is_dry(init_repo: Path) -> None:
    """The run command without --confirm must not create backup refs."""
    runner = CliRunner()
    result = runner.invoke(cli_main.app, ["run", "--repo", str(init_repo)])
    assert result.exit_code == 0, result.stderr

    refs = subprocess.run(
        ("git", "show-ref"),
        cwd=init_repo,
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    assert "refs/backup/goap" not in refs


def test_run_command_with_confirm_creates_backup(init_repo: Path) -> None:
    """When --confirm is provided a backup ref should be created."""
    runner = CliRunner()
    result = runner.invoke(cli_main.app, ["run", "--repo", str(init_repo), "--confirm"])
    assert result.exit_code == 0, result.stderr

    refs = subprocess.run(
        ("git", "show-ref"),
        cwd=init_repo,
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    assert "refs/backup/goap" in refs


def test_explain_command_lists_reasons(init_repo: Path) -> None:
    """Explain command should provide rationale for each action."""
    runner = CliRunner()
    result = runner.invoke(cli_main.app, ["explain", "--repo", str(init_repo), "--json"])
    assert result.exit_code == 0, result.stderr

    payload = json.loads(result.stdout)
    explanations = payload["explanations"]
    assert explanations, "explanations should not be empty"
    assert all(entry["reason"] for entry in explanations)


def test_dry_run_command_reports_history(init_repo: Path) -> None:
    """Dry-run command must report the simulated git command history."""
    runner = CliRunner()
    result = runner.invoke(cli_main.app, ["dry-run", "--repo", str(init_repo), "--json"])
    assert result.exit_code == 0, result.stderr

    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["command_history"], "dry-run should record command history"


def test_dry_run_command_escapes_control_sequences(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure control characters in recorded commands are escaped before display."""
    payload: dict[str, object] = {
        "dry_run": True,
        "executed_actions": [],
        "command_history": [
            {
                "command": ["git", "commit", "\x1b[31mred\x1b[0m"],
                "returncode": 0,
                "dry_run": True,
            },
        ],
    }

    sentinel_context = object()

    def fake_prepare_context(*_: object, **__: object) -> object:
        return sentinel_context

    def fake_execute_workflow(_: object, **__: object) -> dict[str, object]:
        return payload

    monkeypatch.setattr(cli_main, "_prepare_context", fake_prepare_context)
    monkeypatch.setattr(cli_main, "_execute_workflow", fake_execute_workflow)

    runner = CliRunner()
    result = runner.invoke(cli_main.app, ["dry-run"])

    assert result.exit_code == 0, result.stderr
    assert "\\x1b" in result.stdout
    assert "\x1b" not in result.stdout


def test_plan_includes_remote_sync_actions_when_behind(tmp_path: Path) -> None:
    """Remote divergence should schedule preview, fetch, and rebase actions."""
    state = RepoState(
        repo_path=tmp_path,
        ref=RepoRef(branch="feature", tracking="origin/feature", sha="deadbeef"),
        diverged_remote=3,
        diverged_local=1,
        has_unpushed_commits=True,
        working_tree_clean=True,
    )
    config = default_config()

    actions = build_action_specs(state, config)
    plan = SimplePlanner().plan(state, config.goal, actions)

    names = {action.name for action in plan.actions}
    assert {"Conflict:PreviewMergeConflicts", "Sync:FetchAll", "Rebase:OntoUpstream"} <= names


def test_plan_includes_run_tests_when_required(tmp_path: Path) -> None:
    """Enabling tests_must_pass should add the Quality:RunTests action."""
    state = RepoState(
        repo_path=tmp_path,
        ref=RepoRef(branch="feature", tracking="origin/feature", sha="deadbeef"),
        working_tree_clean=True,
    )
    config = default_config()
    config.goal.tests_must_pass = True

    actions = build_action_specs(state, config)
    plan = SimplePlanner().plan(state, config.goal, actions)

    names = {action.name for action in plan.actions}
    assert "Quality:RunTests" in names


def test_plan_includes_range_diff_when_ahead(tmp_path: Path) -> None:
    """Ahead-of-tracking branches should request a range-diff summary."""
    state = RepoState(
        repo_path=tmp_path,
        ref=RepoRef(branch="feature", tracking="origin/feature", sha="deadbeef"),
        diverged_local=2,
        has_unpushed_commits=True,
        working_tree_clean=True,
    )
    config = default_config()

    actions = build_action_specs(state, config)
    plan = SimplePlanner().plan(state, config.goal, actions)

    names = {action.name for action in plan.actions}
    assert "Quality:ExplainRangeDiff" in names


def test_plan_includes_push_with_lease_when_goal_enabled(tmp_path: Path) -> None:
    """Push goal should surface the Sync:PushWithLease action."""
    state = RepoState(
        repo_path=tmp_path,
        ref=RepoRef(branch="feature", tracking="origin/feature", sha="deadbeef"),
        diverged_local=1,
        has_unpushed_commits=True,
        working_tree_clean=True,
    )
    config = default_config()
    config.goal.push_with_lease = True

    actions = build_action_specs(state, config)
    plan = SimplePlanner().plan(state, config.goal, actions)

    names = {action.name for action in plan.actions}
    assert "Sync:PushWithLease" in names


def _make_action_context(
    tmp_path: Path,
    mocker: MockerFixture,
    *,
    config: Config | None = None,
) -> WorkflowContext:
    cfg = config or default_config()
    logger = StructuredLogger(name="test-runtime", stream=io.StringIO())
    action_facade = cast("Any", mocker.create_autospec(GitFacade, instance=True))
    observer_facade = cast("Any", mocker.create_autospec(GitFacade, instance=True))
    observer = cast("Any", mocker.Mock())
    observer.observe.return_value = RepoState(
        repo_path=tmp_path,
        ref=RepoRef(branch="feature", tracking="origin/feature", sha="deadbeef"),
    )
    return WorkflowContext(
        repo_path=tmp_path,
        config=cfg,
        logger=logger,
        action_facade=action_facade,
        observer_facade=observer_facade,
        observer=observer,
        planner=SimplePlanner(),
    )


def test_fetch_handler_invokes_helper(tmp_path: Path, mocker: MockerFixture) -> None:
    """The fetch action handler should call the sync helper."""
    context = _make_action_context(tmp_path, mocker)
    fetch_mock = mocker.patch("goapgit.cli.runtime.fetch_all")

    action = ActionSpec(name="Sync:FetchAll", params={"remote": "origin"}, cost=1.1)

    handler = ACTION_HANDLERS["Sync:FetchAll"]
    assert handler.run(context, action) is True
    fetch_mock.assert_called_once_with(context.action_facade, context.logger, remote="origin")


def test_preview_handler_invokes_merge_tree(tmp_path: Path, mocker: MockerFixture) -> None:
    """Preview action should delegate to the merge-tree predictor using the observer facade."""
    context = _make_action_context(tmp_path, mocker)
    preview_mock = mocker.patch("goapgit.cli.runtime.preview_merge_conflicts", return_value=set())

    action = ActionSpec(
        name="Conflict:PreviewMergeConflicts",
        params={"ours": "HEAD", "theirs": "origin/main"},
        cost=0.9,
    )

    handler = ACTION_HANDLERS["Conflict:PreviewMergeConflicts"]
    assert handler.run(context, action) is True
    preview_mock.assert_called_once_with(context.observer_facade, context.logger, "HEAD", "origin/main")


def test_rebase_onto_handler_invokes_helper(tmp_path: Path, mocker: MockerFixture) -> None:
    """Rebase handler should call rebase_onto_upstream with parsed params."""
    context = _make_action_context(tmp_path, mocker)
    rebase_mock = mocker.patch("goapgit.cli.runtime.rebase_onto_upstream")

    action = ActionSpec(
        name="Rebase:OntoUpstream",
        params={"upstream": "origin/main", "update_refs": "true"},
        cost=1.0,
    )

    handler = ACTION_HANDLERS["Rebase:OntoUpstream"]
    assert handler.run(context, action) is True
    rebase_mock.assert_called_once_with(
        context.action_facade,
        context.logger,
        "origin/main",
        update_refs=True,
        onto=None,
    )


def test_push_handler_builds_refspec(tmp_path: Path, mocker: MockerFixture) -> None:
    """Push handler should construct the refspec before calling git push."""
    context = _make_action_context(tmp_path, mocker)
    push_mock = mocker.patch("goapgit.cli.runtime.push_with_lease")

    action = ActionSpec(
        name="Sync:PushWithLease",
        params={"remote": "origin", "remote_branch": "main", "local_branch": "feature"},
        cost=1.6,
    )

    handler = ACTION_HANDLERS["Sync:PushWithLease"]
    assert handler.run(context, action) is True
    push_mock.assert_called_once_with(
        context.action_facade,
        context.logger,
        remote="origin",
        refspecs=["feature:main"],
        force=context.config.allow_force_push,
    )


def test_run_tests_handler_invokes_helper(tmp_path: Path, mocker: MockerFixture) -> None:
    """Run tests handler should execute the configured command via the helper."""
    config = default_config()
    config.goal.tests_command = ("pytest",)
    config.max_test_runtime_sec = 30
    context = _make_action_context(tmp_path, mocker, config=config)
    run_tests_mock = mocker.patch("goapgit.cli.runtime.run_tests")

    action = ActionSpec(name="Quality:RunTests", params={"timeout_sec": "30"}, cost=1.2)

    handler = ACTION_HANDLERS["Quality:RunTests"]
    assert handler.run(context, action) is True
    run_tests_mock.assert_called_once_with(
        context.action_facade,
        context.logger,
        ("pytest",),
        timeout=30.0,
    )


def test_range_diff_handler_invokes_merge_base(tmp_path: Path, mocker: MockerFixture) -> None:
    """Range-diff handler should compute the merge base and invoke explain_range_diff."""
    context = _make_action_context(tmp_path, mocker)
    cast("Any", context.observer_facade.run).return_value = subprocess.CompletedProcess(
        ["git"],
        0,
        stdout="base\n",
        stderr="",
    )
    explain_mock = mocker.patch("goapgit.cli.runtime.explain_range_diff", return_value="summary")

    action = ActionSpec(name="Quality:ExplainRangeDiff", params={"tracking": "origin/main"}, cost=1.3)

    handler = ACTION_HANDLERS["Quality:ExplainRangeDiff"]
    assert handler.run(context, action) is True
    cast("Any", context.observer_facade.run).assert_called_once_with(["git", "merge-base", "HEAD", "origin/main"])
    explain_mock.assert_called_once_with(
        context.observer_facade,
        context.logger,
        "base..origin/main",
        "base..HEAD",
    )
def test_build_plan_payload_returns_expected_sections(init_repo: Path) -> None:
    """Ensure the shared helper reports state, actions, and plans."""
    prepare_context = cast("WorkflowContextFactory",
        object.__getattribute__(cli_main, "_prepare_context"),
    )
    build_plan_payload = cast("PlanPayloadBuilder",
        object.__getattribute__(cli_main, "_build_plan_payload"),
    )

    context = prepare_context(
        repo=init_repo,
        config_path=None,
        json_logs=True,
        dry_run_actions=True,
        silence_logs=True,
    )
    computation = build_plan_payload(context)

    assert computation.state.ref.branch
    assert computation.actions, "actions catalogue should not be empty"
    assert computation.plan.actions, "plan should contain actions"
