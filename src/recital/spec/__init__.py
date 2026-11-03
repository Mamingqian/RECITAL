"""Reward DSL: typed composition over a registry of named primitives."""
from recital.spec.registry import (
    ROLE_CROSS_ORACLE_AGG,
    ROLE_HARD_GATE,
    ROLE_PER_ORACLE_NORM,
    ROLE_PER_SEQ_PENALTY,
    ROLE_PER_SEQ_SHAPING,
    VALID_ROLES,
    get_primitive,
    list_primitives,
    register_primitive,
)
from recital.spec.composer import (
    ComposedRewardFn,
    Hyperparameters,
    RewardSpec,
    SourceRolePolicy,
    validate_source_roles,
    validate_spec,
)

__all__ = [
    "ROLE_PER_ORACLE_NORM",
    "ROLE_CROSS_ORACLE_AGG",
    "ROLE_PER_SEQ_SHAPING",
    "ROLE_PER_SEQ_PENALTY",
    "ROLE_HARD_GATE",
    "VALID_ROLES",
    "register_primitive",
    "get_primitive",
    "list_primitives",
    "ComposedRewardFn",
    "Hyperparameters",
    "RewardSpec",
    "SourceRolePolicy",
    "validate_source_roles",
    "validate_spec",
]
