from __future__ import annotations

import io
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from goapgit.actions.quality import explain_range_diff
from goapgit.io.logging import StructuredLogger

if TYPE_CHECKING:
    from goapgit.git.facade import GitFacade


class FakeGitFacade:
    """Capture git commands invoked by the quality action."""

    def __init__(self, responses: list[SimpleNamespace] | None = None) -> None:
        """Initialise the fake facade with an optional response sequence."""
        self.commands: list[tuple[list[str], bool]] = []
        self._responses = list(responses) if responses is not None else None

    def run(self, command: list[str], *, check: bool = True, **_: object) -> SimpleNamespace:  # type: ignore[override]
        """Record the command and mimic a git invocation."""
        self.commands.append((list(command), check))
        if self._responses is not None:
            if not self._responses:
                msg = "No more fake responses configured"
                raise AssertionError(msg)
            response = self._responses.pop(0)
        else:
            response = SimpleNamespace(stdout="diff", stderr="", returncode=0)
        response.stdout = getattr(response, "stdout", "")
        response.stderr = getattr(response, "stderr", "")
        response.returncode = getattr(response, "returncode", 0)
        return response


def _make_logger() -> StructuredLogger:
    """Create a structured logger backed by an in-memory buffer."""
    return StructuredLogger(name="test-quality", stream=io.StringIO())


def test_explain_range_diff_uses_separator_and_preserves_ranges() -> None:
    """Ensure the range-diff command keeps the ranges as data."""
    facade = FakeGitFacade()
    logger = _make_logger()
    before = "-c pager.range-diff=echo hacked"
    after = "-c log.showSignature=echo hacked"

    explain_range_diff(cast("GitFacade", facade), logger, before, after)

    assert facade.commands == [
        (["git", "range-diff", "--", before, after], False),
    ]


@pytest.mark.parametrize(
    "value",
    ["range\nwith newline", "range\rwith carriage return"],
)
def test_explain_range_diff_rejects_newlines(value: str) -> None:
    """Reject newline characters that could break command tokenisation."""
    facade = FakeGitFacade()
    logger = _make_logger()

    with pytest.raises(ValueError, match="newline characters"):
        explain_range_diff(cast("GitFacade", facade), logger, value, "main..feature")

    assert facade.commands == []


def test_explain_range_diff_retries_without_separator_when_git_needs_ranges() -> None:
    """Fallback to the legacy invocation when git requires explicit ranges."""
    failure = SimpleNamespace(stdout="", stderr="fatal: need two commit ranges", returncode=129)
    success = SimpleNamespace(stdout="range summary", stderr="", returncode=0)
    facade = FakeGitFacade([failure, success])
    logger = _make_logger()
    before = "base..tip"
    after = "upstream..feature"

    summary = explain_range_diff(cast("GitFacade", facade), logger, before, after)

    assert summary == "range summary"
    assert facade.commands == [
        (["git", "range-diff", "--", before, after], False),
        (["git", "range-diff", before, after], False),
    ]
