"""Agarwal et al. 2025 lentiMPRA predictor.

The model was trained on a lentiMPRA data source distinct from the Gosai data
used by Malinois.

Architecture: BassetBranched2 CNN, 800bp one-hot input (200bp CRE + 300bp flanks),
trained on K562 and HepG2 log2FC from independent data distribution.

OUTPUT SHAPE: (n_seqs, 2) in order [K562, HepG2].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tasks.promoter import assets


@dataclass
class AgarwalLentiMPRAOracle:
    """Agarwal lentiMPRA scorer wrapper.

    Lazy-loads MalinoisAgarwalScorer from promoter_rl on first call.
    Returns the model's two supported cell types: K562 and HepG2.
    """

    weights_path: str | Path | None = None
    device: str = "cpu"
    batch_size: int = 64
    _scorer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.weights_path is not None:
            self.weights_path = Path(self.weights_path)

    def _load(self) -> None:
        """Lazy-load the MalinoisAgarwalScorer from promoter_rl.

        Mirrors cnn_ensemble.py pattern: import scorer class only on first call,
        instantiate it, set self._scorer, then subsequent __call__s reuse it.
        """
        if self._scorer is not None:
            return

        MalinoisAgarwalScorer = assets.import_scorer(
            "malinois_agarwal"
        ).MalinoisAgarwalScorer

        # Resolve a caller-supplied model directory or weight file.
        weights = Path(self.weights_path or assets.MALINOIS_AGARWAL_WEIGHTS)
        weights_dir = weights if weights.is_dir() else weights.parent

        self._scorer = MalinoisAgarwalScorer(
            model_dir=str(weights_dir),
            device=self.device,
        )

    def __call__(self, seqs: list[str]) -> np.ndarray:
        """Predict MPRA activity for a batch of sequences.

        Args:
            seqs: List of DNA sequences (any length; scorer pads/truncates to 200bp).

        Returns:
            np.ndarray of shape (n_seqs, 2) in order [K562, HepG2].
        """
        if not seqs:
            return np.empty((0, 2), dtype=float)
        self._load()

        # Call the scorer's predict_dict method
        # Returns dict with keys "K562", "HepG2", each of shape (n_seqs,)
        preds_dict = self._scorer.predict_dict(seqs, batch_size=self.batch_size)

        n_seqs = len(seqs)
        supported = {
            cell: np.asarray(preds_dict[cell], dtype=float).reshape(-1)
            for cell in ("K562", "HepG2")
        }
        wrong = {cell: len(values) for cell, values in supported.items() if len(values) != n_seqs}
        if wrong:
            raise ValueError(
                "Agarwal scorer returned a batch length inconsistent with its input: "
                f"expected {n_seqs}, got {wrong}"
            )
        return np.column_stack([supported["K562"], supported["HepG2"]])
