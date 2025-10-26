"""Telemetry helpers for Responses API interactions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
import pathlib

from pydantic import BaseModel, ConfigDict, Field


class TelemetryRecord(BaseModel):
    """Structured record describing a single Responses call."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    mode: str
    result: str
    response_id: str | None = None
    previous_response_id: str | None = None
    model: str | None = None
    usage: dict[str, int | float | None] | None = None

    model_config = ConfigDict(extra="forbid")


@dataclass(slots=True)
class TelemetryLogger:
    """Append JSON Lines telemetry records to disk."""

    path: pathlib.Path

    def log(self, record: TelemetryRecord) -> None:
        """Append ``record`` to the telemetry file."""
        payload = record.model_dump(mode="json")
        target = pathlib.Path(self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False)
            stream.write("\n")


__all__ = ["TelemetryLogger", "TelemetryRecord"]
