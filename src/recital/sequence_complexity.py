"""Task-agnostic sequence-complexity measurements for self-validation.

The functions return JSON-serializable per-sequence values plus library-level
summaries. Thresholding is deliberately left to task calibration; this module
only measures the requested failure modes.

The metric NAMES are DNA vocabulary for quantities that are not: "dinucleotide
entropy" is 2-mer entropy and "max homopolymer length" is the longest run of one
symbol, and both are computed for CM's 20-letter libraries (checkpoints.py calls
this module with k=3 for exactly that reason). ``_has_at_run`` is the one
nucleotide-only measurement; it is computed for every task because the trainer
computes this whole block for every task.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import log2
from typing import Any, Iterable

import numpy as np


def _kmer_counts(sequence: str, k: int) -> Counter[str]:
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    if len(sequence) < k:
        return Counter()
    return Counter(sequence[i : i + k] for i in range(len(sequence) - k + 1))


def _dinucleotide_entropy(sequence: str) -> float:
    counts = _kmer_counts(sequence, 2)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return float(-sum((n / total) * log2(n / total) for n in counts.values()))


def _max_homopolymer(sequence: str) -> int:
    if not sequence:
        return 0
    best = run = 1
    for left, right in zip(sequence, sequence[1:], strict=False):
        run = run + 1 if left == right else 1
        best = max(best, run)
    return best


def _has_at_run(sequence: str, minimum: int) -> bool:
    return ("A" * minimum) in sequence or ("T" * minimum) in sequence


def _nearest_hamming(sequence: str, references: list[str]) -> int | None:
    same_length = [ref for ref in references if len(ref) == len(sequence)]
    if not same_length:
        return None
    return min(sum(a != b for a, b in zip(sequence, ref, strict=True)) for ref in same_length)


def _mean_pairwise_hamming(sequences: list[str]) -> float | None:
    comparable = [
        sum(a != b for a, b in zip(left, right, strict=True))
        for left, right in combinations(sequences, 2)
        if len(left) == len(right)
    ]
    return float(np.mean(comparable)) if comparable else None


def _finite_summary(values: Iterable[int | float | None]) -> dict[str, float | None]:
    array = np.asarray([v for v in values if v is not None], dtype=float)
    if array.size == 0:
        return {"mean": None, "median": None, "p10": None, "p90": None}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
    }


def sequence_complexity_metrics(
    sequences: list[str],
    *,
    training_set: Iterable[str] | None = None,
    k: int = 8,
    at_run_min: int = 5,
    max_training_references: int = 4096,
) -> dict[str, Any]:
    """Measure per-sequence and cohort complexity without applying thresholds."""
    seqs = [str(sequence).upper() for sequence in sequences]
    references = list(dict.fromkeys(str(s).upper() for s in (training_set or [])))[
        :max_training_references
    ]

    unique_kmers: list[int] = []
    max_kmer_copies: list[int] = []
    dinucleotide_entropies: list[float] = []
    max_homopolymers: list[int] = []
    has_at_runs: list[bool] = []
    nearest_hamming: list[int | None] = []

    for sequence in seqs:
        counts = _kmer_counts(sequence, k)
        unique_kmers.append(len(counts))
        max_kmer_copies.append(max(counts.values(), default=0))
        dinucleotide_entropies.append(_dinucleotide_entropy(sequence))
        max_homopolymers.append(_max_homopolymer(sequence))
        has_at_runs.append(_has_at_run(sequence, at_run_min))
        nearest_hamming.append(_nearest_hamming(sequence, references))

    per_sequence = {
        "unique_kmer_count": unique_kmers,
        "max_kmer_copy": max_kmer_copies,
        "dinucleotide_entropy": dinucleotide_entropies,
        "max_homopolymer_length": max_homopolymers,
        "has_at_run": has_at_runs,
        "nearest_training_hamming": nearest_hamming,
    }
    summary = {
        "unique_kmer_count": _finite_summary(unique_kmers),
        "max_kmer_copy": _finite_summary(max_kmer_copies),
        "dinucleotide_entropy": _finite_summary(dinucleotide_entropies),
        "max_homopolymer_length": _finite_summary(max_homopolymers),
        "nearest_training_hamming": _finite_summary(nearest_hamming),
        "at_run_fraction": float(np.mean(has_at_runs)) if has_at_runs else None,
        "pairwise_hamming_diversity": _mean_pairwise_hamming(seqs),
    }
    return {
        "k": int(k),
        "at_run_min": int(at_run_min),
        "n_sequences": len(seqs),
        "training_reference_count": len(references),
        "per_sequence": per_sequence,
        "summary": summary,
    }


def stratify_sequence_complexity_metrics(
    metrics: dict[str, Any],
    sequences: list[str],
    conditioning: list[str],
) -> dict[str, dict[str, Any]]:
    """Partition already-computed per-sequence metrics without rescoring.

    In particular, nearest-training Hamming distance can be expensive.  Reusing
    the global per-sequence values makes per-stratum summaries essentially free
    while preserving the exact same measurements.
    """
    if len(sequences) != len(conditioning):
        raise ValueError("sequences and conditioning must have the same length")
    per_sequence = metrics.get("per_sequence") or {}
    out: dict[str, dict[str, Any]] = {}
    labels = [str(value) for value in conditioning]
    for stratum in dict.fromkeys(labels):
        indices = [index for index, label in enumerate(labels) if label == stratum]
        stratum_sequences = [sequences[index] for index in indices]
        subset = {
            name: [values[index] for index in indices]
            for name, values in per_sequence.items()
        }
        out[stratum] = {
            "k": metrics.get("k"),
            "at_run_min": metrics.get("at_run_min"),
            "n_sequences": len(indices),
            "training_reference_count": metrics.get("training_reference_count", 0),
            "per_sequence": subset,
            "summary": {
                "unique_kmer_count": _finite_summary(
                    subset.get("unique_kmer_count", [])
                ),
                "max_kmer_copy": _finite_summary(
                    subset.get("max_kmer_copy", [])
                ),
                "dinucleotide_entropy": _finite_summary(
                    subset.get("dinucleotide_entropy", [])
                ),
                "max_homopolymer_length": _finite_summary(
                    subset.get("max_homopolymer_length", [])
                ),
                "nearest_training_hamming": _finite_summary(
                    subset.get("nearest_training_hamming", [])
                ),
                "at_run_fraction": (
                    float(np.mean(subset.get("has_at_run", [])))
                    if subset.get("has_at_run")
                    else None
                ),
                "pairwise_hamming_diversity": _mean_pairwise_hamming(
                    stratum_sequences
                ),
            },
        }
    return out
