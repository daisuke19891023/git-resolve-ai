from __future__ import annotations

import json
import importlib
import subprocess

from pathlib import Path

import pytest
from typer.testing import CliRunner

from goapgit.cli import diagnose

cli_main = importlib.import_module("goapgit.cli.main")


@pytest.fixture
def git_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Prepare an isolated git configuration and author identity."""
    config_file = tmp_path / "gitconfig"
    config_file.write_text(
        """
[merge]
conflictStyle = diff3
[pull]
rebase = false
""".strip()
        + "\n",
        encoding="utf-8",
    )
    env: dict[str, str] = {
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
    """Initialise a temporary git repository for testing."""
    _ = git_env
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(("git", "init"), cwd=repo, check=True)
    return repo


def test_generate_diagnosis_reports_recommendations(
    init_repo: Path,
    git_env: dict[str, str],
) -> None:
    """Ensure the diagnosis reports mismatched configuration values."""
    report = diagnose.generate_diagnosis(init_repo, env=git_env)

    settings = {entry.key: entry for entry in report.git_config}
    assert settings["merge.conflictStyle"].detected == "diff3"
    assert not settings["merge.conflictStyle"].matches_recommendation
    assert settings["rerere.enabled"].detected is None
    assert not settings["rerere.enabled"].matches_recommendation
    assert settings["pull.rebase"].detected == "false"
    assert not settings["pull.rebase"].matches_recommendation

    assert report.repo_stats is not None
    assert report.repo_stats.tracked_files == 0
    assert report.large_repo_guidance.triggered is False


def test_large_repo_guidance_triggers(
    init_repo: Path,
    git_env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirm guidance suggests sparse-checkout/worktree for large repos."""
    (init_repo / "file.txt").write_text("data", encoding="utf-8")
    subprocess.run(("git", "add", "file.txt"), cwd=init_repo, check=True)
    subprocess.run(("git", "commit", "-m", "initial"), cwd=init_repo, check=True)

    monkeypatch.setattr(diagnose, "TRACKED_FILE_THRESHOLD", 0)
    monkeypatch.setattr(diagnose, "SIZE_PACK_THRESHOLD_KIB", 0)
    monkeypatch.setattr(diagnose, "COMMIT_COUNT_THRESHOLD", 0)

    report = diagnose.generate_diagnosis(init_repo, env=git_env)
    guidance = report.large_repo_guidance
    assert guidance.triggered is True
    assert guidance.reasons
    assert any("sparse-checkout" in suggestion for suggestion in guidance.suggestions)
    assert any("worktree" in suggestion for suggestion in guidance.suggestions)


def test_cli_main_outputs_json(
    init_repo: Path,
    git_env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the CLI and ensure JSON output is emitted."""
    config_file = Path(git_env["GIT_CONFIG_GLOBAL"])
    config_file.write_text(
        """
[merge]
conflictStyle = zdiff3
[rerere]
enabled = true
[pull]
rebase = true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (init_repo / "tracked.txt").write_text("payload", encoding="utf-8")
    subprocess.run(("git", "add", "tracked.txt"), cwd=init_repo, check=True)
    subprocess.run(("git", "commit", "-m", "tracked"), cwd=init_repo, check=True)

    monkeypatch.chdir(init_repo)

    runner = CliRunner()
    result = runner.invoke(cli_main.app, ["diagnose"])
    assert result.exit_code == 0

    payload = json.loads(result.stdout)
    assert payload["git_config"][0]["recommended"] == "zdiff3"
    assert payload["large_repo_guidance"]["triggered"] is False

    assert cli_main.main(["diagnose"]) == 0
