"""One definition of "how far is this sequence from a reference set?".

The ``seen_penalty`` reward term (exact match against the training set) and CM's
``cm_novelty`` oracle (normalized whole-sequence edit distance to the natural
family, floored and ceilinged) ask the same question at different thresholds.
They differ only in metric and bounds, and both use the functions here.

The metrics here are alphabet-independent: none of them knows what a base or a
residue is. What differs per task is the reference SET and the bounds, which is
why those are parameters of the ``reference_distance`` primitive rather than of
these functions.
"""
from __future__ import annotations

import os

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

# Metric names a reward specification may declare. Both are
# distances: 0.0 means "identical to something in the reference set".
NORMALIZED_EDIT = "normalized_edit"
EXACT_MATCH = "exact_match"
METRICS = (NORMALIZED_EDIT, EXACT_MATCH)


def nearest_normalized(sequence: str, references: list[str]) -> tuple[float, int]:
    """Closest reference by whole-sequence Levenshtein / max(lengths), and its index.

    Length-tolerant by construction, which is what a variable-length task needs:
    an equal-length-only metric has no comparable neighbour at all for most of a
    library that emits 80-96 residues, and scoring that as maximal drift is a
    silent lie rather than a missing measurement.
    """
    if not references:
        raise ValueError("normalized distance needs at least one reference sequence")
    _, distance, index = process.extractOne(
        sequence, references, scorer=Levenshtein.normalized_distance)
    return float(distance), int(index)


def pairwise_normalized(sequences: list[str]) -> np.ndarray:
    """Whole-sequence Levenshtein / max(lengths) between every pair, as a symmetric matrix with a zero diagonal.

    The same length-tolerant distance as `nearest_normalized`, one worker per core this process may use.
    """
    return process.cdist(sequences, sequences, scorer=Levenshtein.normalized_distance, dtype=np.float64,
                         workers=len(os.sched_getaffinity(0)))


def nearest_distances(
    sequences: list[str], references: list[str], metric: str
) -> np.ndarray:
    """Per-sequence distance to the nearest reference under ``metric``.

    ``exact_match`` is the degenerate metric a reference set induces on its own
    members: 0.0 for a member, 1.0 for everything else. It is what
    ``seen_penalty`` measures, expressed in the same coordinates as the edit
    distance so the two terms are comparable rather than merely adjacent.
    """
    if metric == NORMALIZED_EDIT:
        return np.asarray(
            [nearest_normalized(sequence, references)[0] for sequence in sequences],
            dtype=float,
        )
    if metric == EXACT_MATCH:
        members = frozenset(references)
        return np.asarray(
            [0.0 if sequence in members else 1.0 for sequence in sequences], dtype=float
        )
    raise ValueError(f"unknown sequence distance metric {metric!r}; expected one of {METRICS}")


def bound_shortfall(
    distances: np.ndarray, *, floor: float | None, ceiling: float | None
) -> np.ndarray:
    """Two-sided distance shortfall: how far outside ``[floor, ceiling]`` each is.

    An absent bound leaves that side unconstrained. Zero inside the band, so a
    term built on this is silent until the library actually leaves it.
    """
    distances = np.asarray(distances, dtype=float)
    shortfall = np.zeros_like(distances)
    if floor is not None:
        shortfall = shortfall + np.maximum(0.0, floor - distances)
    if ceiling is not None:
        shortfall = shortfall + np.maximum(0.0, distances - ceiling)
    return shortfall
