"""Action helpers orchestrating git operations for GOAP planning."""

from .conflict import apply_path_strategy, auto_trivial_resolve
from .quality import explain_range_diff
from .rebase import rebase_continue_or_abort, rebase_onto_upstream
from .safety import create_backup_ref, ensure_clean_or_stash
from .sync import fetch_all, push_with_lease

__all__ = [
    "apply_path_strategy",
    "auto_trivial_resolve",
    "create_backup_ref",
    "ensure_clean_or_stash",
    "explain_range_diff",
    "fetch_all",
    "push_with_lease",
    "rebase_continue_or_abort",
    "rebase_onto_upstream",
]
