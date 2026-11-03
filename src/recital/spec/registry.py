"""Primitive registry: reward pipeline roles and terminal library selectors.

Roles:
  - per_oracle_normalization: dict[str, np.ndarray] -> dict[str, np.ndarray]
        Apply a normalization (e.g. EMA z-score) per oracle.
  - per_oracle_utility: raw, target, conditioning -> np.ndarray
        Convert one oracle's raw scores to utility without editing its target.
  - cross_oracle_aggregation: dict[str, np.ndarray] -> np.ndarray
        Combine multiple oracle outputs into one per-sequence scalar.
  - per_sequence_shaping: np.ndarray -> np.ndarray
        Reshape the aggregated scalar into a reward (e.g. target-distance Huber).
  - per_sequence_penalty: list[str] -> np.ndarray
        Subtract a per-sequence penalty (e.g. distance from a reference set).
  - hard_biological_gate: list[str] -> np.ndarray[bool]
        Return True if a sequence violates a biological constraint;
        violation maps to a fixed invalid_penalty in shape config.
        A violating sequence's reward is replaced AFTER the clip, so a violation
        is strictly worse than any merely low-scoring sequence -- which is what
        makes a gate a hard constraint rather than a large penalty.

The framework owns these roles and registers only primitives that mean the same
thing over any alphabet. A primitive whose thresholds are calibrated on one
alphabet registers from its own task package (tasks.dna.primitives,
tasks.protein.primitives) into this same registry.
"""
from __future__ import annotations

from typing import Callable

ROLE_PER_ORACLE_NORM = "per_oracle_normalization"
ROLE_PER_ORACLE_UTILITY = "per_oracle_utility"
ROLE_CROSS_ORACLE_AGG = "cross_oracle_aggregation"
ROLE_PER_SEQ_SHAPING = "per_sequence_shaping"
ROLE_PER_SEQ_PENALTY = "per_sequence_penalty"
ROLE_HARD_GATE = "hard_biological_gate"
ROLE_LIBRARY_SELECTOR = "library_selector"

# Constructor arguments the composer binds from run context. The agent names the
# library; it never supplies its rows, and a specification that supplies one is
# refused. The capability card reads this so a parameter the agent must not write
# is not advertised as writable.
FRAMEWORK_OWNED_PARAMETERS = {
    "seen_penalty": "train_set",
    "reference_distance": "reference_sequences",
}

VALID_ROLES = {
    ROLE_PER_ORACLE_NORM,
    ROLE_PER_ORACLE_UTILITY,
    ROLE_CROSS_ORACLE_AGG,
    ROLE_PER_SEQ_SHAPING,
    ROLE_PER_SEQ_PENALTY,
    ROLE_HARD_GATE,
    ROLE_LIBRARY_SELECTOR,
}

REGISTRY: dict[tuple[str, str], Callable] = {}


def register_primitive(role: str, name: str):
    """Decorator: register a primitive class under (role, name)."""
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid primitive role {role!r}. Valid: {sorted(VALID_ROLES)}")

    def deco(cls):
        key = (role, name)
        if key in REGISTRY:
            raise ValueError(f"Primitive already registered: {key}")
        REGISTRY[key] = cls
        cls._primitive_role = role
        cls._primitive_name = name
        return cls

    return deco


def get_primitive(role: str, name: str):
    key = (role, name)
    if key not in REGISTRY:
        raise KeyError(
            f"Unknown primitive {role!r}/{name!r}. Registered: {sorted(REGISTRY.keys())}"
        )
    return REGISTRY[key]


def list_primitives(role: str | None = None) -> list[tuple[str, str]]:
    if role is None:
        return sorted(REGISTRY.keys())
    return sorted(k for k in REGISTRY if k[0] == role)
