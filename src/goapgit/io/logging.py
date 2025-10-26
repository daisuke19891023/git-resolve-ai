"""Structured logging utilities."""

from __future__ import annotations

import json
from collections.abc import Mapping as MappingABC
from datetime import datetime, UTC
import re
from typing import Any, TextIO, cast
from collections.abc import Mapping

from pydantic import BaseModel, SecretStr, field_validator


class _SanitizedText(BaseModel):
    """Model that normalises sensitive fragments in log text."""

    text: SecretStr

    @field_validator("text", mode="before")
    @classmethod
    def _mask_sensitive_data(cls, value: Any) -> str:
        text = str(value)
        patterns = [
            (r"https://[^:]+:[^@]+@", "https://***:***@"),
            (r"token[=:]\s*\S+", "token=***"),
        ]
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text


def _sanitize_log_output(text: str) -> str:
    """Mask sensitive fragments in the provided text."""
    sanitized = _SanitizedText.model_validate({"text": text})
    return sanitized.text.get_secret_value()


def _sanitize_log_value(value: Any) -> Any:
    """Apply sanitisation recursively to structured log data."""
    if isinstance(value, str):
        return _sanitize_log_output(value)
    if isinstance(value, MappingABC):
        typed_mapping = cast("Mapping[Any, Any]", value)
        sanitised_mapping: dict[Any, Any] = {}
        for key, item in typed_mapping.items():
            sanitised_mapping[key] = _sanitize_log_value(item)
        return sanitised_mapping
    if isinstance(value, tuple):
        typed_tuple = cast("tuple[Any, ...]", value)
        return tuple(_sanitize_log_value(item) for item in typed_tuple)
    if isinstance(value, list):
        typed_list = cast("list[Any]", value)
        return [_sanitize_log_value(item) for item in typed_list]
    if isinstance(value, set):
        typed_set = cast("set[Any]", value)
        return {_sanitize_log_value(item) for item in typed_set}
    return value


class StructuredLogger:
    """Simple structured logger supporting JSON lines and text output."""

    def __init__(self, *, name: str, json_mode: bool = False, stream: TextIO | None = None) -> None:
        """Initialise the structured logger."""
        self._name = name
        self._json_mode = json_mode
        self._stream: TextIO = stream or _default_stream()

    @property
    def name(self) -> str:
        """Return the logger name."""
        return self._name

    @property
    def json_mode(self) -> bool:
        """Return whether JSON mode is enabled."""
        return self._json_mode

    def debug(self, message: str, **fields: Any) -> None:
        """Log a DEBUG-level message."""
        self._emit("DEBUG", message, fields)

    def info(self, message: str, **fields: Any) -> None:
        """Log an INFO-level message."""
        self._emit("INFO", message, fields)

    def warning(self, message: str, **fields: Any) -> None:
        """Log a WARNING-level message."""
        self._emit("WARNING", message, fields)

    def error(self, message: str, **fields: Any) -> None:
        """Log an ERROR-level message."""
        self._emit("ERROR", message, fields)

    def _emit(self, level: str, message: str, fields: dict[str, Any]) -> None:
        timestamp = datetime.now(UTC).isoformat()
        sanitised_message = _sanitize_log_output(message)
        sanitised_fields = {key: _sanitize_log_value(value) for key, value in fields.items()}
        if self._json_mode:
            payload: dict[str, Any] = {
                "timestamp": timestamp,
                "level": level,
                "logger": self._name,
                "message": sanitised_message,
            }
            payload.update(sanitised_fields)
            self._stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        else:
            line = f"[{timestamp}] {level:<7} {self._name}: {sanitised_message}"
            if sanitised_fields:
                extras = " ".join(
                    f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in sanitised_fields.items()
                )
                line = f"{line} | {extras}"
            self._stream.write(line + "\n")
        self._stream.flush()


def _default_stream() -> TextIO:
    import sys

    return sys.stdout


__all__ = ["StructuredLogger"]
