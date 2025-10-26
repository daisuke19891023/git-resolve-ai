"""Quality and explainability related git actions."""
from __future__ import annotations

from typing import TYPE_CHECKING

from goapgit.git.facade import GitCommandError

if TYPE_CHECKING:
    import subprocess
    from pathlib import Path
    from goapgit.git.facade import GitFacade
    from goapgit.io.logging import StructuredLogger

RANGE_DIFF_USAGE_EXIT_CODE = 129


def _normalize_range(label: str, value: str) -> str:
    normalized = value.strip()
    if "\n" in normalized or "\r" in normalized:
        msg = f"{label} must not contain newline characters"
        raise ValueError(msg)
    return normalized


def _should_retry_without_separator(
    result: subprocess.CompletedProcess[str],
    before_range: str,
    after_range: str,
) -> bool:
    if result.returncode != RANGE_DIFF_USAGE_EXIT_CODE:
        return False
    stderr = (result.stderr or "").lower()
    if "need two commit ranges" not in stderr:
        return False
    if ".." not in before_range or ".." not in after_range:
        return False
    return not (before_range.startswith("-") or after_range.startswith("-"))


def explain_range_diff(
    facade: GitFacade,
    logger: StructuredLogger,
    before_range: str,
    after_range: str,
    *,
    output_path: Path | None = None,
) -> str:
    """Run ``git range-diff`` and return its textual summary."""
    normalized_before = _normalize_range("before_range", before_range)
    normalized_after = _normalize_range("after_range", after_range)
    command = ["git", "range-diff", "--", normalized_before, normalized_after]
    result = facade.run(command, check=False)
    if _should_retry_without_separator(result, normalized_before, normalized_after):
        command = ["git", "range-diff", normalized_before, normalized_after]
        result = facade.run(command, check=False)
    summary = (result.stdout or "").strip()

    logger.info(
        "computed range diff",
        before=normalized_before,
        after=normalized_after,
        returncode=result.returncode,
        summary=summary,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = result.stdout or ""
        if content and not content.endswith("\n"):
            content = f"{content}\n"
        output_path.write_text(content, encoding="utf-8")

    if result.returncode != 0:
        raise GitCommandError(command, result.returncode, result.stdout or "", result.stderr or "")

    return result.stdout or ""


__all__ = ["explain_range_diff"]
