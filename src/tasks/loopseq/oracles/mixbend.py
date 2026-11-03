"""MIXBend bendability oracle (DNABERT-6 with bimodal representations).

The scorer is built by ``scorer``, a ``package.module:factory`` that wraps the upstream MIXBend code; the factory
receives ``repo_path``, ``checkpoint_paths``, ``params_path``, ``dnabert_dir`` and ``device``. The local DNABERT-6
pretrained weights and the fine-tuned mixbend_v1 ensemble are referenced by path in tasks.loopseq.assets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tasks.loopseq.assets import (
    MIXBEND_DNABERT_DIR,
    MIXBEND_FINETUNED_PT,
    MIXBEND_PARAMS,
    MIXBEND_REPO,
)
from tasks.loopseq.oracles.external import scorer_factory


@dataclass
class MIXBendOracle:
    repo_path: str | Path = MIXBEND_REPO
    dnabert_dir: str | Path = MIXBEND_DNABERT_DIR
    params_path: str | Path = MIXBEND_PARAMS
    checkpoint_paths: list[str] = field(default_factory=lambda: [str(MIXBEND_FINETUNED_PT)])
    device: str = "cpu"
    batch_size: int = 64
    scorer: str = ""
    _scorer: Any = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._scorer is not None:
            return
        self._scorer = scorer_factory(self.scorer, "MIXBend")(
            repo_path=self.repo_path,
            checkpoint_paths=list(self.checkpoint_paths),
            params_path=self.params_path,
            dnabert_dir=self.dnabert_dir,
            device=self.device,
        )

    def __call__(self, seqs: list[str]) -> np.ndarray:
        self._load()
        return np.asarray(self._scorer.predict(list(seqs), batch_size=self.batch_size), dtype=float)
