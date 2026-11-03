"""The outcome statuses that train nothing."""
from __future__ import annotations


NONTRAINING_OUTCOME_STATUSES = frozenset(
    {
        "critic_rejected",
        "dynamic_primitive_forbidden",
        "primitive_rejected",
        "train_error",
        "unresolved_stop",
    }
)
