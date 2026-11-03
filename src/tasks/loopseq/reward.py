"""Loop-seq reward terms using materialized, condition-specific references.

Import this module from the task adapter to register its primitive. The shared
composer does not import task modules or invent biological quality thresholds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import ClassVar

import numpy as np

from recital.sequence_complexity import _dinucleotide_entropy, _kmer_counts
from recital.spec.registry import ROLE_PER_SEQ_PENALTY, register_primitive


@register_primitive(ROLE_PER_SEQ_PENALTY, "loopseq_reference_quality")
@dataclass(frozen=True)
class LoopseqReferenceQualityPenalty:
    """Penalize H2 and unique-kmer deficits against frozen condition cutoffs.

    ``by_condition`` maps exact adapter-supplied condition names to
    ``dinucleotide_entropy_min`` (bits) and ``unique_kmer_min`` (count). For each
    feature x with positive cutoff c, the normalized deficit is max(0, 1-x/c).
    The penalty is weight * (d_H2**2 + d_kmer**2) / 2, a squared hinge that is
    continuous with zero slope at the cutoff in feature space. An explicitly
    declared zero cutoff imposes no deficit for that feature.

    This is a per-sequence training signal. It does not guarantee library
    uniqueness, pairwise diversity, novelty, or any SuccessSpec requirement.
    Missing reference profiles are an input-readiness error, never a zero penalty.
    """

    by_condition: Mapping[str, Mapping[str, float]]
    kmer_size: int = 3
    weight: float = 1.0
    requires_conditioning: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.kmer_size, bool)
            or not isinstance(self.kmer_size, Integral)
            or self.kmer_size < 1
        ):
            raise ValueError("kmer_size must be a positive integer")
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, Real)
            or not np.isfinite(self.weight)
            or self.weight < 0
        ):
            raise ValueError("weight must be finite and nonnegative")
        if not isinstance(self.by_condition, Mapping) or not self.by_condition:
            raise ValueError(
                "quality reference pending: by_condition requires materialized cutoffs"
            )
        required = {"dinucleotide_entropy_min", "unique_kmer_min"}
        frozen_cutoffs = {}
        for condition, cutoffs in self.by_condition.items():
            if not isinstance(condition, str) or not condition:
                raise ValueError("quality reference condition names must be nonempty strings")
            if not isinstance(cutoffs, Mapping) or set(cutoffs) != required:
                raise ValueError(
                    f"quality reference for {condition!r} must declare exactly {sorted(required)}"
                )
            for name, cutoff in cutoffs.items():
                if (
                    isinstance(cutoff, bool)
                    or not isinstance(cutoff, Real)
                    or not np.isfinite(cutoff)
                    or cutoff < 0
                ):
                    raise ValueError(
                        f"quality reference {condition!r}/{name} must be finite and nonnegative"
                    )
            if cutoffs["dinucleotide_entropy_min"] > 4.0:
                raise ValueError("dinucleotide_entropy_min cannot exceed 4 bits for A/C/G/T DNA")
            frozen_cutoffs[condition] = {key: float(cutoffs[key]) for key in required}
        # Snapshot the supplied reference rather than retaining a caller's nested
        # dictionary, which may subsequently be reused to construct another spec.
        object.__setattr__(self, "by_condition", frozen_cutoffs)

    def __call__(
        self,
        sequences: Sequence[str],
        *,
        conditioning: Sequence[str] | None = None,
    ) -> np.ndarray:
        if conditioning is None or len(conditioning) != len(sequences):
            raise ValueError("quality penalty requires one explicit condition per sequence")
        if any(not isinstance(condition, str) for condition in conditioning):
            raise ValueError("quality penalty conditioning must contain exact string keys")
        missing = set(conditioning) - set(self.by_condition)
        if missing:
            raise ValueError(f"quality reference pending for condition(s) {sorted(missing)}")
        out = np.zeros(len(sequences), dtype=float)
        for index, (sequence, condition) in enumerate(zip(sequences, conditioning, strict=True)):
            if not isinstance(sequence, str) or set(sequence) - set("ACGT"):
                raise ValueError(
                    f"quality penalty requires unrepaired A/C/G/T sequence at row {index}"
                )
            if len(sequence) < max(2, self.kmer_size):
                raise ValueError(
                    f"sequence at row {index} is too short for declared quality features"
                )
            cutoffs = self.by_condition[condition]
            entropy_min = cutoffs["dinucleotide_entropy_min"]
            unique_min = cutoffs["unique_kmer_min"]
            entropy = _dinucleotide_entropy(sequence)
            unique = len(_kmer_counts(sequence, self.kmer_size))
            entropy_deficit = max(0.0, 1.0 - entropy / entropy_min) if entropy_min > 0 else 0.0
            unique_deficit = max(0.0, 1.0 - unique / unique_min) if unique_min > 0 else 0.0
            out[index] = self.weight * 0.5 * (entropy_deficit**2 + unique_deficit**2)
        return out
