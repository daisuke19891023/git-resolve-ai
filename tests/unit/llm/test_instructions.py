"""Tests for the instruction composer helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from goapgit.llm.instructions import (
    InstructionRole,
    compose_instructions,
    messenger_instructions,
    planner_instructions,
    resolver_instructions,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize(
    ("factory", "role"),
    [
        (resolver_instructions, InstructionRole.RESOLVER),
        (messenger_instructions, InstructionRole.MESSENGER),
        (planner_instructions, InstructionRole.PLANNER),
    ],
)
def test_shortcuts_delegate_to_compose_instructions(
    factory: Callable[[], str],
    role: InstructionRole,
) -> None:
    """Shortcut helpers must include the shared guard rails and role text."""
    instructions = factory()

    assert f"GOAPGit's {role.value} specialist" in instructions
    assert "Always respond with JSON" in instructions


def test_compose_instructions_supports_extra_rules() -> None:
    """Extra rules should be appended to the core guard rails."""
    instructions = compose_instructions(InstructionRole.RESOLVER, extra_rules=("No HTML",))

    assert "No HTML" in instructions


def test_compose_instructions_rejects_unknown_role() -> None:
    """Using an unsupported role should surface a helpful error."""
    with pytest.raises(ValueError, match="Unsupported instruction role"):
        compose_instructions(cast("InstructionRole", "other"))

