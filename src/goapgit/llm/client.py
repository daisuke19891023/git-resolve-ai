"""Client factory for LLM providers."""

from __future__ import annotations

from enum import StrEnum
from typing import cast

from openai import AzureOpenAI, OpenAI
from pydantic import AnyHttpUrl, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(StrEnum):
    """Supported LLM providers."""

    AZURE = "azure"
    OPENAI = "openai"


class LLMSettings(BaseSettings):
    """Resolve LLM configuration from the environment."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", validate_default=True)

    provider: LLMProvider = Field(default=LLMProvider.AZURE, alias="GOAPGIT_LLM_PROVIDER")
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    azure_openai_api_key: SecretStr | None = Field(default=None, alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: AnyHttpUrl | None = Field(default=None, alias="AZURE_OPENAI_ENDPOINT")
    openai_api_version: str | None = Field(default=None, alias="OPENAI_API_VERSION")

    @field_validator("provider", mode="before")
    @classmethod
    def _normalise_provider(cls, value: object) -> object:
        """Normalise provider strings to the canonical enum value."""
        if isinstance(value, str):
            normalised = value.strip().lower()
            if not normalised:
                return None
            try:
                return LLMProvider(normalised)
            except ValueError as exc:  # pragma: no cover - converted to ValidationError
                message = f"Unsupported GOAPGIT_LLM_PROVIDER: {value!r}"
                raise ValueError(message) from exc
        return value

    @model_validator(mode="after")
    def _validate_provider_requirements(self) -> LLMSettings:
        """Ensure provider-specific environment variables are present."""
        if self.provider is LLMProvider.OPENAI:
            if self.openai_api_key is None:
                message = "OPENAI_API_KEY is required when GOAPGIT_LLM_PROVIDER is 'openai'."
                raise ValueError(message)
        else:
            missing: list[str] = []
            if self.azure_openai_api_key is None:
                missing.append("AZURE_OPENAI_API_KEY")
            if self.azure_openai_endpoint is None:
                missing.append("AZURE_OPENAI_ENDPOINT")
            if self.openai_api_version is None:
                missing.append("OPENAI_API_VERSION")
            if missing:
                missing_vars = ", ".join(missing)
                message = (
                    "Missing required Azure OpenAI environment variables when "
                    f"GOAPGIT_LLM_PROVIDER is 'azure': {missing_vars}."
                )
                raise ValueError(message)
        return self


def make_client_from_env(*, settings: LLMSettings | None = None) -> OpenAI | AzureOpenAI:
    """Create an OpenAI client for the configured provider."""
    resolved_settings = settings or LLMSettings()

    if resolved_settings.provider is LLMProvider.OPENAI:
        openai_key = cast("SecretStr", resolved_settings.openai_api_key)
        return OpenAI(api_key=openai_key.get_secret_value())

    azure_key = cast("SecretStr", resolved_settings.azure_openai_api_key)
    azure_endpoint = cast("AnyHttpUrl", resolved_settings.azure_openai_endpoint)
    api_version = cast("str", resolved_settings.openai_api_version)
    return AzureOpenAI(
        api_key=azure_key.get_secret_value(),
        azure_endpoint=str(azure_endpoint),
        api_version=api_version,
    )
