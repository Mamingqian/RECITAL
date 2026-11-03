"""Nucleotide-alphabet reward primitives, registered by the DNA task adapters.

Import this module from a DNA task adapter to register its primitives. Every
one of them is calibrated on four-letter statistics, so offering them to a
20-letter task would give an allowlist that cannot bind -- a ``max_run`` of 6
is unreachable over 20 residues and an 8-mer is unique by construction in 96
positions. The framework keeps the roles; the alphabet keeps the thresholds.

Each gate is a hard constraint: it lets the Strategist *forbid* degenerate
sequences (extreme GC, homopolymer runs, low dinucleotide entropy) outright
instead of only penalising them, which is what makes ``add_calibrated_gate`` an
objective edit rather than a hyperparameter tweak. Thresholds are constructor
arguments so a gate can be added loosely and tightened on a later divergence --
the "calibrated" in add_calibrated_gate.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from recital.eval import gc_content
from recital.sequence_complexity import _dinucleotide_entropy, _max_homopolymer
from recital.spec.registry import ROLE_HARD_GATE, ROLE_PER_SEQ_PENALTY, register_primitive


def _longest_run(seq: str, alphabet: str) -> int:
    if not seq:
        return 0
    best = run = 1 if seq[0] in alphabet else 0
    for i in range(1, len(seq)):
        if seq[i] in alphabet and seq[i] == seq[i - 1]:
            run += 1
            best = max(best, run)
        else:
            run = 1 if seq[i] in alphabet else 0
    return best


@register_primitive(ROLE_PER_SEQ_PENALTY, "at_homopoly")
@dataclass
class ATHomopoly:
    """Penalize long AT homopolymer runs in DNA sequences."""

    max_run: int = 4
    weight: float = 0.06

    def __call__(self, seqs: list[str]) -> np.ndarray:
        out = np.zeros(len(seqs), dtype=float)
        for i, s in enumerate(seqs):
            run_a = _longest_run(s, "A")
            run_t = _longest_run(s, "T")
            overshoot = max(0, max(run_a, run_t) - self.max_run)
            out[i] = self.weight * overshoot
        return out


@register_primitive(ROLE_PER_SEQ_PENALTY, "repeated_kmer")
@dataclass
class RepeatedKmer:
    """Subtract excessive tandem/repeated substrings after oracle aggregation.

    The arithmetic is alphabet-independent; the defaults are not. k=8 over 50-200
    nucleotides is a repeat detector, but over 80-96 residues nearly every 8-mer
    is unique and the penalty is identically zero, so the calibration -- and
    therefore the primitive -- belongs to the DNA tasks. Applying a shared
    subtraction to all columns of a contrastive oracle would cancel it; this
    scalar penalty does not.
    """

    k: int = 8
    max_copies: int = 12
    weight: float = 0.5

    def __post_init__(self):
        if type(self.k) is not int or self.k < 1 or type(self.max_copies) is not int or self.max_copies < 1:
            raise ValueError("k and max_copies must be positive integers")
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError("repetition weight must be finite and nonnegative")

    def __call__(self, seqs: list[str]) -> np.ndarray:
        def excess(sequence):
            counts = Counter(sequence[i:i + self.k] for i in range(max(0, len(sequence) - self.k + 1)))
            return max(0, max(counts.values(), default=0) - self.max_copies)
        return np.asarray([self.weight * excess(sequence) for sequence in seqs], dtype=float)


@register_primitive(ROLE_HARD_GATE, "gc_bounds_gate")
@dataclass
class GCBoundsGate:
    """Forbid sequences whose GC fraction leaves a plausible band. A policy that discovers the reward model rewards
    extreme GC can drive every sequence to one composition; a gate makes such a sequence invalid, where a penalty only
    makes it expensive.
    """

    min_gc: float = 0.25
    max_gc: float = 0.75

    def __call__(self, seqs: list[str]) -> np.ndarray:
        gc = gc_content(list(seqs))
        return np.asarray((gc < self.min_gc) | (gc > self.max_gc), dtype=bool)


@register_primitive(ROLE_HARD_GATE, "homopolymer_gate")
@dataclass
class HomopolymerGate:
    """Forbid sequences containing a homopolymer run longer than ``max_run``.

    A hard cap on motif copies. Unlike ``at_homopoly`` (a soft per-base
    penalty on A/T only), this is a hard bound over every base. Run length is
    measured alphabet-agnostically, but 6 is a four-letter threshold: a protein
    task wants ``tasks.protein.primitives.residue_run_gate`` instead.
    """

    max_run: int = 6

    def __call__(self, seqs: list[str]) -> np.ndarray:
        return np.asarray(
            [_max_homopolymer(str(s)) > self.max_run for s in seqs], dtype=bool
        )


@register_primitive(ROLE_HARD_GATE, "dinucleotide_entropy_gate")
@dataclass
class DinucleotideEntropyGate:
    """Forbid low-complexity sequences by dinucleotide entropy. Over a 16-letter alphabet it is bounded above by 4.0;
    a diverse sequence sits near 3.7-3.9 and a collapsed one falls well below.
    """

    min_entropy: float = 3.0

    def __call__(self, seqs: list[str]) -> np.ndarray:
        return np.asarray(
            [_dinucleotide_entropy(str(s)) < self.min_entropy for s in seqs],
            dtype=bool,
        )
