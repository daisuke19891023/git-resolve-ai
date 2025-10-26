"""Configuration loading utilities for goapgit."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, cast
from collections.abc import Mapping

from goapgit.core.models import Config


def load_config(
    *,
    path: Path | str | None = None,
    data: str | bytes | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Config:
    """Load and validate configuration from TOML data.

    Exactly one of ``path`` or ``data`` must be provided. ``overrides`` allows
    callers to patch specific sections before validation, which is useful for
    CLI flags or tests.
    """
    if (path is None and data is None) or (path is not None and data is not None):
        msg = "Provide exactly one of 'path' or 'data' when loading configuration."
        raise ValueError(msg)

    raw_content: dict[str, Any]
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_file():
            msg = f"Configuration path '{path}' is not a file."
            raise ValueError(msg)
        try:
            text = path.read_text()
        except OSError as error:
            reason = getattr(error, "strerror", None) or str(error)
            msg = f"Failed to read configuration file at '{path}': {reason}"
            raise ValueError(msg) from error
        raw_content = tomllib.loads(text)
    else:
        if data is None:
            msg = "Configuration data must be provided when path is omitted."
            raise ValueError(msg)
        text = data if isinstance(data, str) else data.decode()
        raw_content = tomllib.loads(text)

    if overrides is not None:
        typed_overrides: dict[str, Any] = {str(key): value for key, value in overrides.items()}
        raw_content = _merge_dicts(dict(raw_content), typed_overrides)

    normalised = _normalise(raw_content)

    return Config.model_validate(normalised)


def _merge_dicts(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if (
            key in base
            and isinstance(base[key], MappingABC)
            and isinstance(value, MappingABC)
        ):
            nested_base = cast("dict[str, Any]", dict(base[key]))
            nested_updates = cast("Mapping[str, Any]", value)
            base[key] = _merge_dicts(nested_base, nested_updates)
        else:
            base[key] = value
    return base


def _normalise(raw: Mapping[str, Any]) -> dict[str, Any]:
    goal = raw.get("goal", {})
    strategy = raw.get("strategy", {})
    safety = raw.get("safety", {})

    config_dict: dict[str, Any] = {
        "goal": goal,
        "strategy_rules": list(strategy.get("rules", [])),
        "enable_rerere": strategy.get("enable_rerere", raw.get("enable_rerere", True)),
        "conflict_style": strategy.get("conflict_style", raw.get("conflict_style", "zdiff3")),
        "allow_force_push": safety.get("allow_force_push", raw.get("allow_force_push", False)),
        "dry_run": safety.get("dry_run", raw.get("dry_run", True)),
        "max_test_runtime_sec": safety.get(
            "max_test_runtime_sec", raw.get("max_test_runtime_sec", 600),
        ),
    }

    if "strategy_rules" in raw:
        config_dict["strategy_rules"] = raw["strategy_rules"]

    return config_dict


__all__ = ["load_config"]
