"""BendNet bendability oracle (Keras / TensorFlow).

The scorer is built by ``scorer``, a ``package.module:factory`` that wraps the upstream BendNet architecture; the
factory receives ``repo_path``, ``model_prefix`` and ``execution_device``. The fine-tuned weights are referenced by
path in tasks.loopseq.assets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tasks.loopseq.assets import BENDNET_REPO, BENDNET_WEIGHTS
from tasks.loopseq.oracles.external import scorer_factory


@dataclass
class BendNetOracle:
    repo_path: str | Path = BENDNET_REPO
    weights_path: str | Path = BENDNET_WEIGHTS
    batch_size: int = 512
    execution_device: str = "cpu"
    scorer: str = ""
    _scorer: Any = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._scorer is not None:
            return
        self._scorer = scorer_factory(self.scorer, "BendNet")(
            repo_path=self.repo_path,
            model_prefix=self.weights_path,
            execution_device=self.execution_device,
        )

    def __call__(self, seqs: list[str]) -> np.ndarray:
        self._load()
        return np.asarray(
            self._scorer.predict(list(seqs), batch_size=self.batch_size),
            dtype=float,
        )
