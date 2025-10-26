"""Tests for the ``goapgit llm doctor`` command."""

from __future__ import annotations

import importlib
import json

import pytest
from typer.testing import CliRunner


cli_main = importlib.import_module("goapgit.cli.main")


@pytest.fixture(autouse=True)
def clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure LLM-related environment variables do not leak between tests."""
    variables = [
        "GOAPGIT_LLM_PROVIDER",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "OPENAI_API_VERSION",
        "OPENAI_API_KEY",
    ]
    for name in variables:
        monkeypatch.delenv(name, raising=False)


def _set_azure_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOAPGIT_LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-05-01")


def _set_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOAPGIT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


def test_llm_doctor_reports_success_for_azure(monkeypatch: pytest.MonkeyPatch) -> None:
    """The doctor command should report all checks as ok in mock mode."""
    _set_azure_env(monkeypatch)
    runner = CliRunner()
    result = runner.invoke(cli_main.app, ["llm", "doctor", "--mock", "--json"])

    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["provider"] == "azure"
    assert payload["mock"] is True
    statuses = {check["status"] for check in payload["checks"]}
    assert statuses == {"ok"}


def test_llm_doctor_reports_success_for_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI provider should also succeed when using the mock client."""
    _set_openai_env(monkeypatch)
    runner = CliRunner()
    result = runner.invoke(
        cli_main.app,
        ["llm", "doctor", "--mock", "--model", "gpt-4o-mini", "--json"],
    )

    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["provider"] == "openai"
    assert payload["mock"] is True
    assert all(check["status"] == "ok" for check in payload["checks"])
