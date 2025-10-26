"""Tests for telemetry helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from goapgit.llm.safety import UsageMetrics
from goapgit.llm.telemetry import TelemetryLogger, TelemetryRecord

if TYPE_CHECKING:
    from pathlib import Path


def test_telemetry_logger_appends_json_lines(tmp_path: Path) -> None:
    """Telemetry records should be appended as JSON Lines."""
    path = tmp_path / "logs" / "telemetry.jsonl"
    logger = TelemetryLogger(path)

    record = TelemetryRecord(
        mode="resolver",
        result="success",
        response_id="resp_123",
        previous_response_id="resp_122",
        model="gpt-test",
        usage=UsageMetrics(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost=0.02,
        ).model_dump(),
    )

    logger.log(record)

    content = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1

    payload = json.loads(content[0])
    assert payload["mode"] == "resolver"
    assert payload["response_id"] == "resp_123"
    assert payload["usage"]["total_tokens"] == 15
    assert payload["usage"]["cost"] == 0.02
