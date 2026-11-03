"""Malinois MPRA oracle (Gosai 2024 CNN).

Predicts per-cell-type log2 fold-change activity for a 200-bp CRE in its 600-bp
assay context. Used as the per-cell-type training oracle for the promoter task.

Wraps promoter_rl.scorers.malinois.MalinoisScorer, lazy-loading on first call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tasks.promoter import assets
from tasks.promoter.assets import MALINOIS_ARTIFACT_DIR
from tasks.promoter.score_contract import MALINOIS_SOURCE_COLUMNS, TARGET_CELLS


@dataclass
class MalinoisOracle:
    """Lazy-loading wrapper around promoter_rl's MalinoisScorer.

    Mirrors tasks.loopseq.oracles.cnn_ensemble.CNNEnsembleOracle: loads the
    underlying scorer on first __call__, then reuses it for all subsequent calls.
    """

    weights_path: str | Path | None = None
    cell_types: tuple[str, ...] = ("K562", "HepG2")
    device: str = "cpu"
    batch_size: int = 256
    _scorer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if (len(set(self.cell_types)) != len(self.cell_types)
                or not set(TARGET_CELLS) <= set(self.cell_types)
                or not set(self.cell_types) <= set(MALINOIS_SOURCE_COLUMNS)):
            raise ValueError("Malinois columns must name supported heads and both design cells")
        if self.weights_path is not None:
            self.weights_path = Path(self.weights_path)
        else:
            self.weights_path = MALINOIS_ARTIFACT_DIR

    def _load(self) -> None:
        """Lazy-load the MalinoisScorer on first call."""
        if self._scorer is not None:
            return

        MalinoisScorer = assets.import_scorer("malinois").MalinoisScorer

        # Respect the requested device.  Passing ``None`` for an explicit CPU
        # request would auto-select CUDA, allocating GPU memory in CPU probes and
        # possibly colliding with the policy process.
        self._scorer = MalinoisScorer.from_artifacts(
            artifact_dir=self.weights_path,
            device=self.device,
        )

    def __call__(self, seqs: list[str]) -> np.ndarray:
        """Return predicted log2FC values in explicitly configured head order.

        Args:
            seqs: List of DNA sequences (200bp or auto-padded/truncated).

        Returns:
            Array shaped (n_seqs, len(cell_types)), default [K562, HepG2].
            A declared SK-N-SH column is retained in its configured position.
        """
        if not seqs:
            return np.empty((0, len(self.cell_types)), dtype=float)
        self._load()
        output = np.asarray(
            self._scorer.predict(seqs, batch_size=self.batch_size, average_rc=True),
            dtype=float,
        )
        # Select explicitly named native heads. Design-condition count must not
        # relabel the native third head or silently remove a declared off-target.
        if output.ndim != 2 or output.shape[1] not in (2, 3):
            raise ValueError(f"Malinois scorer returned unexpected shape {output.shape}")
        columns = [MALINOIS_SOURCE_COLUMNS[cell] for cell in self.cell_types]
        if max(columns) >= output.shape[1]:
            raise ValueError("Malinois scorer lacks a declared output head")
        output = output[:, columns]
        expected = (len(seqs), len(self.cell_types))
        if output.shape != expected:
            raise ValueError(f"Malinois scorer returned shape {output.shape}; expected {expected}")
        if not np.isfinite(output).all():
            raise ValueError("Malinois scorer returned nonfinite active cell scores")
        return output
