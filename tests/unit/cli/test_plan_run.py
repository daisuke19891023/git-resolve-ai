"""Tests covering plan/run/explain/dry-run CLI commands."""

from __future__ import annotations

import importlib
import json
import subprocess
from typing import TYPE_CHECKING, cast

import pytest
from typer.testing import CliRunner


cli_main = importlib.import_module("goapgit.cli.main")

if TYPE_CHECKING:
    from pathlib import Path
    from goapgit.cli.main import PlanComputation
    from goapgit.cli.runtime import WorkflowContext
    from collections.abc import Callable

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

    def fake_execute_workflow(_: object) -> dict[str, object]:
        return payload

    monkeypatch.setattr(cli_main, "_prepare_context", fake_prepare_context)
    monkeypatch.setattr(cli_main, "_execute_workflow", fake_execute_workflow)

    runner = CliRunner()
    result = runner.invoke(cli_main.app, ["dry-run"])

    assert result.exit_code == 0, result.stderr
    assert "\\x1b" in result.stdout
    assert "\x1b" not in result.stdout


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
