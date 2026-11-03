"""Amino-acid-alphabet reward primitives, registered by the protein adapters.

Import this module from a protein task adapter to register its primitives. The DNA
primitives cannot bind over a protein: ``homopolymer_gate``'s max_run of 6 is
unreachable over 20 letters, ``repeated_kmer``'s 8-mers are unique by construction
in 80-96 positions, and ``seen_penalty`` never fires for a temperature-1 policy that
has not memorised its corpus.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recital.sequence_complexity import _max_homopolymer
from recital.spec.registry import ROLE_HARD_GATE, register_primitive


@register_primitive(ROLE_HARD_GATE, "residue_run_gate")
@dataclass
class ResidueRunGate:
    """Forbid sequences containing a single residue repeated past ``max_run``.

    The protein counterpart of the DNA ``homopolymer_gate``, and the reason the
    two are separate primitives rather than one with a shared default: over 20
    residues a run of 4 is already anomalous (natural AroQ domains do not carry
    one), while the DNA threshold of 6 has probability ~20^-5 per position and so
    can never fire. Long single-residue tracts are also a real liability rather
    than a mere statistical oddity -- poly-Q and poly-N tracts drive aggregation.
    """

    max_run: int = 4

    def __call__(self, seqs: list[str]) -> np.ndarray:
        return np.asarray(
            [_max_homopolymer(str(s)) > self.max_run for s in seqs], dtype=bool
        )
