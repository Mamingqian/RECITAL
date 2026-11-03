"""Per-sequence penalty primitives.

Role: per_sequence_penalty.
Inputs are list[str] of generated sequences; outputs are 1-D arrays of nonnegative
penalties to subtract from the shaped reward.

Only alphabet-independent penalties live here. A penalty that reads nucleotide or
residue composition belongs to the task that owns the alphabet and registers
itself from that task's package.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from recital.sequence_distance import METRICS, NORMALIZED_EDIT, bound_shortfall, nearest_distances
from recital.spec.registry import ROLE_PER_SEQ_PENALTY, register_primitive


@register_primitive(ROLE_PER_SEQ_PENALTY, "seen_penalty")
@dataclass
class SeenPenalty:
    """Penalize sequences that exactly match the training set.

    This is ``reference_distance`` at its degenerate threshold: metric
    ``exact_match`` against the ``training_set`` library with ``floor=1.0``
    produces the identical vector. It keeps its own name and class because that
    name is addressed directly elsewhere -- the bounded penalty-correction rule in
    the loop, the DNA contracts and the Strategist's prompt all refer to it by
    name.
    """

    train_set: frozenset[str] = field(default_factory=frozenset)
    weight: float = 0.3

    def __call__(self, seqs: list[str]) -> np.ndarray:
        out = np.zeros(len(seqs), dtype=float)
        for i, s in enumerate(seqs):
            if s in self.train_set:
                out[i] = self.weight
        return out


@register_primitive(ROLE_PER_SEQ_PENALTY, "reference_distance")
@dataclass
class ReferenceDistance:
    """Keep each sequence inside a declared distance band from a reference set.

    ``reference`` names a library the task bound at compose time (``training_set``
    is always available); ``reference_sequences`` is the resolved library and is
    framework-owned, exactly like ``seen_penalty.train_set`` -- letting an LLM
    supply it would make the term arbitrary and serialize a private corpus.

    The penalty is the two-sided shortfall ``weight * (max(0, floor - d) +
    max(0, d - ceiling))``; an absent bound leaves that side unconstrained. CM's
    ``cm_novelty`` is this term at (normalized_edit, 0.2, 0.4) expressed as an
    oracle rather than a penalty, which is why both share one implementation
    of the arithmetic instead of two that could drift apart.
    """

    reference: str = "training_set"
    metric: str = NORMALIZED_EDIT
    floor: float | None = None
    ceiling: float | None = None
    weight: float = 1.0
    reference_sequences: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.metric not in METRICS:
            raise ValueError(f"unknown reference_distance metric {self.metric!r}; expected one of {METRICS}")
        if self.floor is None and self.ceiling is None:
            raise ValueError("reference_distance needs a floor, a ceiling, or both")
        for name in ("floor", "ceiling"):
            bound = getattr(self, name)
            if bound is not None and (isinstance(bound, bool) or not np.isfinite(float(bound))):
                raise ValueError(f"reference_distance {name} must be a finite number or null")
        if self.floor is not None and self.ceiling is not None and self.ceiling < self.floor:
            raise ValueError("reference_distance ceiling must not be below its floor")
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError("reference_distance weight must be finite and nonnegative")

    def __call__(self, seqs: list[str]) -> np.ndarray:
        distances = nearest_distances(list(seqs), list(self.reference_sequences), self.metric)
        return self.weight * bound_shortfall(distances, floor=self.floor, ceiling=self.ceiling)
