"""Internal CNN ensemble cyclizability scorer.

The user-supplied training oracle for Loop-seq. An ensemble of 3 convolutional
regressors trained on the unified dataset; held-out Pearson rho ~ 0.845. Model
code is in tasks.loopseq.oracles.cnn_model; the checkpoints are referenced by path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tasks.loopseq.assets import CNN_SCORER_DIR


@dataclass
class CNNEnsembleOracle:
    checkpoint_dir: str | Path = CNN_SCORER_DIR
    device: str = "cpu"
    batch_size: int = 512
    _ensemble: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.checkpoint_dir = Path(self.checkpoint_dir)

    def _load(self) -> None:
        if self._ensemble is not None:
            return
        from tasks.loopseq.oracles.cnn_model import CyclizabilityEnsemble

        self._ensemble = CyclizabilityEnsemble.from_dir(
            self.checkpoint_dir, device=self.device
        )

    def __call__(self, seqs: list[str]) -> np.ndarray:
        self._load()
        return np.asarray(
            self._ensemble.predict(list(seqs), batch_size=self.batch_size),
            dtype=float,
        )
