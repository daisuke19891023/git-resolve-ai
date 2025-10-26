"""Tests for the patch proposal workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, Any, cast

import pytest

from goapgit.core.models import ConflictType
from goapgit.git.facade import GitFacade
from goapgit.io.logging import StructuredLogger
from goapgit.llm.patch import (
    ConflictExcerpt,
    PatchProposalResult,
    build_initial_prompt,
    build_retry_prompt,
    propose_patch,
)
from goapgit.llm.responses import ResponsesAPI

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


@dataclass
class _StubResponse:
    id: str
    output_text: str


class _StubResponses(ResponsesAPI):
    def __init__(self, *responses: _StubResponse) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **payload: Any) -> _StubResponse:
        self.calls.append(payload)
        if not self._responses:  # pragma: no cover - defensive guard
            raise RuntimeError("No stub responses remaining")
        return self._responses.pop(0)


class _StubClient:
    responses: ResponsesAPI

    def __init__(self, *responses: _StubResponse) -> None:
        self.responses = _StubResponses(*responses)


def _make_payload(patches: Iterable[str]) -> str:
    """Return a JSON string matching the PatchSet schema."""
    payload = {
        "patches": tuple(patches),
        "confidence": "med",
        "rationale": "ok",
    }
    return json.dumps(payload)


def test_build_initial_prompt_includes_conflict_details(tmp_path: Path) -> None:
    """The initial prompt should surface path, header, and snippet details."""
    excerpt = ConflictExcerpt(
        path="docs/readme.md",
        snippet="""<<<<<<< ours\n- old\n=======\n- new\n>>>>>>> theirs""",
        ctype=ConflictType.text,
        hunk_header="@@ -1,3 +1,3 @@",
    )

    prompt = build_initial_prompt(excerpt, repo_path=tmp_path)

    assert str(tmp_path) in prompt
    assert "docs/readme.md" in prompt
    assert "@@ -1,3 +1,3 @@" in prompt
    assert "Minimal conflict hunk" in prompt
    assert "<<<<<<< ours" in prompt


def test_build_retry_prompt_only_mentions_feedback() -> None:
    """Retry prompt should only include the failure feedback context."""
    prompt = build_retry_prompt("patch failed: rejects")

    assert "patch failed" in prompt
    assert "Failure feedback" in prompt
    # Should not mention repository or file context again.
    assert "repository" not in prompt


def test_propose_patch_calls_responses_with_schema_and_rules(tmp_path: Path) -> None:
    """Calling propose_patch should forward schema, instructions, and prompt."""
    excerpt = ConflictExcerpt(
        path="config.json",
        snippet="""<<<<<<< ours\n  \"version\": \"1.0.0\"\n=======\n  \"version\": \"1.1.0\"\n>>>>>>> theirs""",
        ctype=ConflictType.json,
    )
    stub_response = _StubResponse(id="resp_1", output_text=_make_payload(["patch-content"]))
    client = _StubClient(stub_response)

    result = propose_patch(
        client,
        model="responses-test",
        repo_path=tmp_path,
        conflict=excerpt,
    )

    assert isinstance(result, PatchProposalResult)
    assert result.patch_set.patches == ("patch-content",)
    assert result.patch_set.confidence == "med"
    assert result.patch_set.rationale == "ok"
    assert result.response_id == "resp_1"

    stub_responses = cast("_StubResponses", client.responses)
    assert len(stub_responses.calls) == 1
    payload = stub_responses.calls[0]
    assert payload["model"] == "responses-test"
    assert payload["response_format"]["json_schema"]["name"] == "goapgit_patch_set"
    instructions = payload["instructions"]
    assert "git apply --check" in instructions
    assert "Return only unified diff" in instructions
    prompt = payload["input"][0]["content"][0]["text"]
    assert "config.json" in prompt
    assert "JSON" in prompt
    assert "<<<<<<< ours" in prompt


def test_propose_patch_retry_uses_previous_id(tmp_path: Path) -> None:
    """Retries must link to the previous response id and omit redundant context."""
    initial_excerpt = ConflictExcerpt(
        path="README.md",
        snippet="""<<<<<<< ours\n- old\n=======\n- new\n>>>>>>> theirs""",
        ctype=ConflictType.text,
    )
    first_payload = _make_payload(["patch-1"])
    second_payload = _make_payload(["patch-2"])
    client = _StubClient(
        _StubResponse(id="resp_1", output_text=first_payload),
        _StubResponse(id="resp_2", output_text=second_payload),
    )

    first = propose_patch(
        client,
        model="responses-test",
        repo_path=tmp_path,
        conflict=initial_excerpt,
    )

    second = propose_patch(
        client,
        model="responses-test",
        repo_path=tmp_path,
        conflict=initial_excerpt,
        previous_response_id=first.response_id,
        failure_feedback="patch rejected: context missing",
    )

    assert first.patch_set.patches == ("patch-1",)
    assert second.patch_set.patches == ("patch-2",)

    stub_responses = cast("_StubResponses", client.responses)
    assert len(stub_responses.calls) == 2
    retry_payload = stub_responses.calls[1]
    assert retry_payload["previous_response_id"] == "resp_1"
    retry_prompt = retry_payload["input"][0]["content"][0]["text"]
    assert "patch rejected" in retry_prompt
    assert "<<<<<<< ours" not in retry_prompt


def _init_repo(repo_path: Path) -> GitFacade:
    """Initialise an empty git repository at ``repo_path`` and return a facade."""
    facade = _make_git_facade(repo_path)
    facade.run(["git", "init"])
    return facade


def _write_file(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` using UTF-8 encoding."""
    path.write_text(content, encoding="utf-8")


@pytest.mark.parametrize(
    "filename", ["config.json", "README.md", "yarn.lock"],
)
def test_patch_payloads_pass_git_apply_check(tmp_path: Path, filename: str) -> None:
    """Patches from the LLM payload should pass ``git apply --check``."""
    facade = _init_repo(tmp_path)

    conflict_content = {
        "config.json": dedent(
            """\
            {
            <<<<<<< ours
              "version": "1.0.0"
            =======
              "version": "1.1.0"
            >>>>>>> theirs
            }
            """,
        ),
        "README.md": dedent(
            """\
            # Title
            <<<<<<< ours
            - old
            =======
            - new
            >>>>>>> theirs
            """,
        ),
        "yarn.lock": dedent(
            """\
            <<<<<<< ours
            left@1.0.0
            =======
            left@1.1.0
            >>>>>>> theirs
            """,
        ),
    }[filename]

    patch_content = {
        "config.json": (
            "diff --git a/config.json b/config.json\n"
            "--- a/config.json\n"
            "+++ b/config.json\n"
            "@@ -1,7 +1,3 @@\n"
            " {\n"
            "-<<<<<<< ours\n"
            '-  "version": "1.0.0"\n'
            "-=======\n"
            '   "version": "1.1.0"\n'
            "->>>>>>> theirs\n"
            " }\n"
        ),
        "README.md": (
            "diff --git a/README.md b/README.md\n"
            "--- a/README.md\n"
            "+++ b/README.md\n"
            "@@ -1,6 +1,2 @@\n"
            " # Title\n"
            "-<<<<<<< ours\n"
            "-- old\n"
            "-=======\n"
            " - new\n"
            "->>>>>>> theirs\n"
        ),
        "yarn.lock": (
            "diff --git a/yarn.lock b/yarn.lock\n"
            "--- a/yarn.lock\n"
            "+++ b/yarn.lock\n"
            "@@ -1,5 +1 @@\n"
            "-<<<<<<< ours\n"
            "-left@1.0.0\n"
            "-=======\n"
            " left@1.1.0\n"
            "->>>>>>> theirs\n"
        ),
    }[filename]

    target = tmp_path / filename
    _write_file(target, conflict_content)

    payload = json.dumps(
        {
            "patches": (patch_content,),
            "confidence": "med",
            "rationale": "resolved",
        },
    )
    client = _StubClient(_StubResponse(id="resp", output_text=payload))

    conflict = ConflictExcerpt(path=filename, snippet=conflict_content, ctype=_CTYPE_FOR_FILE(filename))

    result = propose_patch(
        client,
        model="responses-test",
        repo_path=tmp_path,
        conflict=conflict,
    )

    for index, patch in enumerate(result.patch_set.patches):
        patch_file = tmp_path / f"patch-{index}.diff"
        patch_file.write_text(patch, encoding="utf-8")
        facade.run(["git", "apply", "--check", str(patch_file)])


def _CTYPE_FOR_FILE(filename: str) -> ConflictType:
    """Infer the conflict type from the filename extension."""
    if filename.endswith(".json"):
        return ConflictType.json
    if filename.endswith(".md"):
        return ConflictType.text
    return ConflictType.lock


def _make_git_facade(repo_path: Path) -> GitFacade:
    """Create a :class:`GitFacade` for ``repo_path`` with a simple logger."""
    logger = StructuredLogger(name="test-llm-patch")
    return GitFacade(repo_path=repo_path, logger=logger)
