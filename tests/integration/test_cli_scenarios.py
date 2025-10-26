"""End-to-end scenario coverage for goapgit CLI commands."""

from __future__ import annotations

import json
import textwrap
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from goapgit.cli.main import app
from goapgit.core.models import ConflictType

from tests.conftest import GitResponse, ScriptQueue

if TYPE_CHECKING:
    from pathlib import Path
    import pytest


runner = CliRunner()
STATUS_COMMAND = ("git", "status", "--porcelain=v2", "--branch", "--show-stash")
PORCELAIN_COMMAND = ("git", "status", "--porcelain")


def _response(*, stdout: str = "", returncode: int = 0, stderr: str = "") -> GitResponse:
    return GitResponse(stdout=stdout, returncode=returncode, stderr=stderr)


def test_plan_simple_text_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configure_fake_git_facade: ScriptQueue,
) -> None:
    """A dry-run plan should list actions and heuristic cost in text mode."""
    monkeypatch.chdir(tmp_path)
    status_output = textwrap.dedent(
        """
        # branch.oid deadbeefdeadbeefdeadbeefdeadbeefdeadbeef
        # branch.head feature/awesome
        # branch.upstream origin/feature/awesome
        # branch.ab +2 -3
        """,
    ).strip()
    status_script: dict[tuple[str, ...], list[GitResponse] | GitResponse] = {
        STATUS_COMMAND: [_response(stdout=status_output)],
    }
    configure_fake_git_facade.push(status_script)
    configure_fake_git_facade.push({})

    result = runner.invoke(app, ["plan"])

    assert result.exit_code == 0
    assert f"Repository: {tmp_path}" in result.stdout
    assert "Estimated cost:" in result.stdout
    assert "Actions:" in result.stdout

    prefixes = tuple(f"{index}." for index in range(1, 6))
    action_lines = [line.strip() for line in result.stdout.splitlines() if line.strip().startswith(prefixes)]
    assert action_lines, "expected numbered actions to be listed"
    assert any("Safety:CreateBackupRef" in line for line in action_lines)
    assert any("Safety:EnsureCleanOrStash" in line for line in action_lines)
    assert any("Conflict:AutoTrivialResolve" in line for line in action_lines)


def test_plan_json_output_includes_conflict_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configure_fake_git_facade: ScriptQueue,
) -> None:
    """JSON mode should surface conflict metadata and strategy rules."""
    monkeypatch.chdir(tmp_path)
    conflict_file = tmp_path / "dashboard.json"
    conflict_file.write_text("<<<<<<< HEAD\n1\n=======\n2\n>>>>>>> theirs\n", encoding="utf-8")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        textwrap.dedent(
            """
            [goal]
            mode = "rebase_to_upstream"
            tests_must_pass = true

            [strategy]
            enable_rerere = true
            conflict_style = "zdiff3"
            rules = [
              { pattern = "**/*.lock", resolution = "theirs" },
              { pattern = "**/*.json", resolution = "merge-driver:json", when = "whitespace_only" }
            ]

            [safety]
            dry_run = true
            allow_force_push = false
            """,
        ).strip(),
        encoding="utf-8",
    )
    conflict_line = (
        "1 UU N... 100644 100644 100644 100644 "
        "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef "
        "cafebabecafebabecafebabecafebabecafebabe "
        "abcdefabcdefabcdefabcdefabcdefabcdefabcd dashboard.json"
    )
    status_output = textwrap.dedent(
        f"""
        # branch.oid cafebabecafebabecafebabecafebabecafebabe
        # branch.head feature/json
        # branch.upstream origin/feature/json
        # branch.ab +1 -1
        {conflict_line}
        """,
    ).strip()
    status_script: dict[tuple[str, ...], list[GitResponse] | GitResponse] = {
        STATUS_COMMAND: [_response(stdout=status_output)],
    }
    configure_fake_git_facade.push(status_script)
    configure_fake_git_facade.push({})

    result = runner.invoke(app, ["plan", "--json", "--config", str(config_path)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["repository"] == str(tmp_path)
    assert payload["state"]["conflicts"][0]["path"] == "dashboard.json"
    assert payload["state"]["conflicts"][0]["ctype"] == ConflictType.json.value
    assert payload["plan"]["notes"]
    action_names = [action["name"] for action in payload["plan"]["actions"]]
    assert action_names[:3] == [
        "Safety:CreateBackupRef",
        "Safety:EnsureCleanOrStash",
        "Conflict:AutoTrivialResolve",
    ]
    assert "Conflict:ApplyPathStrategy" in action_names
    assert payload["strategy_rules"][1]["when"] == "whitespace_only"


def test_plan_reports_git_command_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configure_fake_git_facade: ScriptQueue,
) -> None:
    """A failing git status should surface a concise CLI error and exit code."""
    monkeypatch.chdir(tmp_path)
    error_message = "fatal: not a git repository (or any of the parent directories): .git"
    status_script: dict[tuple[str, ...], list[GitResponse] | GitResponse] = {
        STATUS_COMMAND: [
            _response(stdout="", stderr=error_message, returncode=128),
        ],
    }
    configure_fake_git_facade.push(status_script)
    configure_fake_git_facade.push({})

    local_runner = CliRunner(mix_stderr=False)
    result = local_runner.invoke(app, ["plan"])

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert result.stdout == ""
    assert error_message in result.stderr
    assert "Traceback" not in result.stderr


def test_run_confirm_applies_lock_strategy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configure_fake_git_facade: ScriptQueue,
) -> None:
    """A confirmed run should apply lockfile path strategy via git commands."""
    monkeypatch.chdir(tmp_path)
    timestamp = "20240101T000000Z"
    monkeypatch.setattr("goapgit.actions.safety._timestamp", lambda: timestamp)

    conflict_file = tmp_path / "yarn.lock"
    conflict_file.write_text("<<<<<<< HEAD\nlock\n=======\nlock\n>>>>>>> theirs\n", encoding="utf-8")
    config_path = tmp_path / "lock-config.toml"
    config_path.write_text(
        textwrap.dedent(
            """
            [goal]
            mode = "rebase_to_upstream"

            [strategy]
            enable_rerere = true
            conflict_style = "zdiff3"
            rules = [
              { pattern = "**/*.lock", resolution = "theirs" }
            ]

            [safety]
            dry_run = false
            allow_force_push = false
            """,
        ).strip(),
        encoding="utf-8",
    )
    conflict_line = (
        "u UU N... 100644 100644 100644 100644 "
        "1111111111111111111111111111111111111111 "
        "2222222222222222222222222222222222222222 "
        "3333333333333333333333333333333333333333 yarn.lock"
    )
    status_output = textwrap.dedent(
        f"""
        # branch.oid 0123456789abcdef0123456789abcdef01234567
        # branch.head feature/locks
        # branch.upstream origin/feature/locks
        # branch.ab +3 -2
        {conflict_line}
        """,
    ).strip()
    status_script: dict[tuple[str, ...], list[GitResponse] | GitResponse] = {
        STATUS_COMMAND: [_response(stdout=status_output) for _ in range(5)],
    }
    configure_fake_git_facade.push(status_script)
    backup_ref = f"refs/backup/goap/{timestamp}"
    action_script: dict[tuple[str, ...], list[GitResponse] | GitResponse] = {
        ("git", "rev-parse", "HEAD"): [_response(stdout="abc123\n")],
        ("git", "update-ref", backup_ref, "abc123"): [_response()],
        PORCELAIN_COMMAND: [
            _response(stdout=" M README.md\n?? stray.txt\n"),
            _response(stdout="UU yarn.lock\n"),
        ],
        ("git", "stash", "push", "--include-untracked", "-m", f"goap/{timestamp}"): [_response(stdout="Saved\n")],
        ("git", "config", "--bool", "rerere.enabled"): [_response(stdout="true\n")],
        ("git", "rerere"): [_response()],
        ("git", "add", "--", "yarn.lock"): [_response(), _response()],
        ("git", "checkout", "--theirs", "--", "yarn.lock"): [_response()],
    }
    configure_fake_git_facade.push(action_script)

    result = runner.invoke(app, ["run", "--confirm", "--json", "--config", str(config_path)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is False
    assert payload["initial_state"]["conflicts"][0]["ctype"] == ConflictType.lock.value
    executed_names = [action["name"] for action in payload["executed_actions"]]
    assert executed_names == [
        "Safety:CreateBackupRef",
        "Safety:EnsureCleanOrStash",
        "Conflict:AutoTrivialResolve",
        "Conflict:ApplyPathStrategy",
    ]
    history_commands = [entry["command"] for entry in payload["command_history"]]
    assert ["git", "rev-parse", "HEAD"] in history_commands
    assert ["git", "update-ref", backup_ref, "abc123"] in history_commands
    assert ["git", "stash", "push", "--include-untracked", "-m", f"goap/{timestamp}"] in history_commands
    assert ["git", "config", "--bool", "rerere.enabled"] in history_commands
    assert ["git", "rerere"] in history_commands
    assert ["git", "checkout", "--theirs", "--", "yarn.lock"] in history_commands
    assert ["git", "add", "--", "yarn.lock"] in history_commands
