"""Capped per-cell master-TF motif presence, with K562/HepG2 output columns.

Repetition penalties belong to the scalar reward, after contrastive aggregation.
A common subtraction from both columns cancels in target-minus-off-target MinGap.
This scorer performs PWM scans on both strands and returns an (n_sequences, 2)
array. The separate generic repeated_kmer penalty handles excessive repetition.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tasks.promoter import assets

# --- Scorer constants ---------------------------------------------------------
DEFAULT_MEME_PATH = str(
    assets.PROMOTER_ROOT / "dbs" / "JASPAR2024_vertebrates.meme"
)
# JASPAR2024 ids for each cell's master-TF panel. Column order of the output is
# ``CELL_ORDER`` == (K562, HepG2).
PANEL: dict[str, tuple[str, ...]] = {
    "HepG2": (
        "MA1494.2",  # HNF4A
        "MA0114.5",  # HNF4A
        "MA0148.5",  # FOXA1
        "MA0047.4",  # FOXA2
        "MA0102.5",  # CEBPA
        "MA0046.3",  # HNF1A
        "MA1991.2",  # HNF1A
    ),
    "K562": (
        "MA0035.5",  # GATA1
        "MA0091.2",  # TAL1
        "MA0140.3",  # GATA1::TAL1
        "MA0493.3",  # KLF1
        "MA0841.2",  # NFE2
    ),
}
CELL_ORDER: tuple[str, ...] = ("K562", "HepG2")
_BACKGROUND = np.array([0.25, 0.25, 0.25, 0.25])
_TARGET_P = 1e-4
_CAP = 2  # each motif contributes at most this to the presence sum


@dataclass
class GrammarPriorOracle:
    """Per-cell master-TF motif-grammar scorer.

    Mirrors the oracle interface of ``MalinoisOracle`` / ``EnformerMPRAOracle``:
    ``__call__(seqs) -> (n_seqs, 2)`` with columns ``[K562, HepG2]``. Empty input
    returns ``np.empty((0, 2))``. ``device`` is accepted for API symmetry and
    ignored (the scan is pure numpy).
    """

    meme_path: str | Path = DEFAULT_MEME_PATH
    device: str | None = None
    # Width-grouped precomputed panel: {width: {"fwd", "rc", "thr", "cells"}}.
    _by_width: dict[int, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        from tasks.promoter.oracles.motif_lib import (
            parse_meme,
            pwm_to_logodds,
            reverse_complement_logodds,
            score_pvalue_threshold,
        )

        motifs = {m.motif_id: m for m in parse_meme(self.meme_path)}
        # Flatten to (cell, motif_id) pairs so a motif shared by two panels would
        # be scanned (and counted) once per cell. The current panels are disjoint.
        prepared: dict[int, dict[str, list]] = {}
        for cell in CELL_ORDER:
            for motif_id in PANEL[cell]:
                motif = motifs.get(motif_id)
                if motif is None:
                    raise ValueError(f"required grammar motif is absent: {motif_id}")
                logodds = pwm_to_logodds(motif.pwm, _BACKGROUND)
                rc_logodds = reverse_complement_logodds(logodds)
                threshold, _ = score_pvalue_threshold(
                    logodds, _BACKGROUND, _TARGET_P
                )
                bucket = prepared.setdefault(
                    motif.width, {"fwd": [], "rc": [], "thr": [], "cells": []}
                )
                bucket["fwd"].append(logodds)
                bucket["rc"].append(rc_logodds)
                bucket["thr"].append(threshold)
                bucket["cells"].append(cell)

        # Freeze each width bucket into stacked arrays for a single BLAS matmul.
        self._by_width = {
            width: {
                "fwd": np.stack(bucket["fwd"], axis=0),
                "rc": np.stack(bucket["rc"], axis=0),
                "thr": np.asarray(bucket["thr"], dtype=np.float32),
                "cells": tuple(bucket["cells"]),
            }
            for width, bucket in prepared.items()
        }

    def __call__(self, seqs: list[str]) -> np.ndarray:
        """Return per-cell grammar scores with columns ``[K562, HepG2]``."""
        if not seqs:
            return np.empty((0, 2), dtype=float)

        from tasks.promoter.oracles.motif_lib import one_hot_encode, sliding_scores

        sequences = [str(seq) for seq in seqs]
        n = len(sequences)

        # One-hot encode ONCE (n, L, 4); every motif width reuses it.
        one_hot = one_hot_encode(sequences)

        presence: dict[str, np.ndarray] = {cell: np.zeros(n) for cell in CELL_ORDER}
        for group in self._by_width.values():
            # Both strands scored against every same-width motif in one matmul.
            fwd = sliding_scores(one_hot, group["fwd"])  # (n, n_windows, k)
            rev = sliding_scores(one_hot, group["rc"])
            if fwd.shape[1] == 0:  # motif wider than the sequence -> no windows
                continue
            thr = group["thr"][None, None, :]
            hits = (fwd >= thr).sum(axis=1) + (rev >= thr).sum(axis=1)  # (n, k)
            capped = np.minimum(hits, _CAP)
            for column, cell in enumerate(group["cells"]):
                presence[cell] += capped[:, column]

        return np.column_stack([presence[cell] for cell in CELL_ORDER])
