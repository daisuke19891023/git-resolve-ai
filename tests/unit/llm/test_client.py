"""Tests for the LLM client factory."""

from __future__ import annotations

from typing import cast

import pytest
from pydantic import ValidationError

from goapgit.llm.client import LLMSettings, make_client_from_env

openai_module = pytest.importorskip("openai")
AzureOpenAI = openai_module.AzureOpenAI
OpenAI = openai_module.OpenAI


@pytest.fixture(autouse=True)
def clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure LLM-related environment variables do not leak between tests."""
    for variable in [
        "GOAPGIT_LLM_PROVIDER",
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "OPENAI_API_VERSION",
    ]:
        monkeypatch.delenv(variable, raising=False)


def test_make_client_from_env_returns_azure_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Azure settings should create an AzureOpenAI client."""
    monkeypatch.setenv("GOAPGIT_LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-02-01")

    client = make_client_from_env()

    assert isinstance(client, AzureOpenAI)
    azure_client = cast("AzureOpenAI", client)
    assert azure_client.api_key == "azure-key"
    assert str(azure_client.base_url) == "https://example.openai.azure.com/openai/"
    assert azure_client.default_query.get("api-version") == "2024-02-01"


def test_make_client_from_env_returns_openai_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI settings should create an OpenAI client."""
    monkeypatch.setenv("GOAPGIT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    client = make_client_from_env()

    assert isinstance(client, OpenAI)
    openai_client = cast("OpenAI", client)
    assert openai_client.api_key == "openai-key"


def test_make_client_from_env_requires_known_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unsupported providers must raise a validation error."""
    monkeypatch.setenv("GOAPGIT_LLM_PROVIDER", "unsupported")
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")

    with pytest.raises(ValidationError) as exc:
        LLMSettings()

    assert "Unsupported GOAPGIT_LLM_PROVIDER" in str(exc.value)


def test_make_client_from_env_requires_azure_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Azure provider must supply all mandatory Azure variables."""
    monkeypatch.setenv("GOAPGIT_LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")

    with pytest.raises(ValidationError) as exc:
        LLMSettings()

    assert "AZURE_OPENAI_ENDPOINT" in str(exc.value)
    assert "OPENAI_API_VERSION" in str(exc.value)
