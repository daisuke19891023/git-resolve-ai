from __future__ import annotations

import io
import json
from typing import Any

from goapgit.io.logging import StructuredLogger


def read_json_lines(buffer: io.StringIO) -> list[dict[str, Any]]:
    """Parse the contents of the buffer into JSON objects."""
    buffer.seek(0)
    return [json.loads(line) for line in buffer.read().splitlines() if line]


def test_structured_logger_outputs_json_lines() -> None:
    """JSON mode should emit one JSON object per line."""
    stream = io.StringIO()
    logger = StructuredLogger(name="goapgit", json_mode=True, stream=stream)

    logger.info("plan generated", action_id="123", details={"steps": 3})
    logger.error("execution failed", error_code="E42")

    records = read_json_lines(stream)
    assert len(records) == 2
    first, second = records
    assert first["level"] == "INFO"
    assert first["action_id"] == "123"
    assert first["message"] == "plan generated"
    assert first["details"] == {"steps": 3}
    assert "timestamp" in first

    assert second["level"] == "ERROR"
    assert second["message"] == "execution failed"
    assert second["error_code"] == "E42"


def test_structured_logger_text_mode() -> None:
    """Text mode should produce a single newline-terminated line."""
    stream = io.StringIO()
    logger = StructuredLogger(name="goapgit", json_mode=False, stream=stream)

    logger.info("dry run complete")

    stream.seek(0)
    output = stream.read()
    assert "INFO" in output
    assert "dry run complete" in output
    assert output.count("\n") == 1


def test_structured_logger_masks_sensitive_data() -> None:
    """Sensitive fragments should be masked before emission."""
    json_stream = io.StringIO()
    text_stream = io.StringIO()
    json_logger = StructuredLogger(name="goapgit", json_mode=True, stream=json_stream)
    text_logger = StructuredLogger(name="goapgit", json_mode=False, stream=text_stream)

    secret_url = "https://" + "alice" + ":" + "supersecret" + "@example.com/repo.git"
    secret_token = "token" + "=" + "abcd1234"
    json_logger.info("Cloning %s", secret_url=secret_url, credentials=secret_token)
    text_logger.error(
        "Failed with token: %s",
        details={"error": secret_token, "url": secret_url},
    )

    records = read_json_lines(json_stream)
    assert records[0]["message"] == "Cloning %s"
    assert records[0]["secret_url"] == "https://" + "***" + ":" + "***" + "@example.com/repo.git"
    assert records[0]["credentials"] == "token" + "=***"

    text_stream.seek(0)
    text_output = text_stream.read()
    assert ("https://" + "***" + ":" + "***" + "@example.com/repo.git") in text_output
    assert ("token" + "=***") in text_output
