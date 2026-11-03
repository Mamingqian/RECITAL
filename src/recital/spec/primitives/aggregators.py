"""Cross-oracle aggregation primitives.

Role: cross_oracle_aggregation.
Inputs are dict[str, np.ndarray] keyed by oracle name; outputs are 1-D arrays of
per-sequence scalars.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from recital.spec.cell_gap import cell_gap
from recital.spec.registry import ROLE_CROSS_ORACLE_AGG, register_primitive
from .ema_zscore import _ema_zscore, _frozen_zscore, _restore_moments


@register_primitive(ROLE_CROSS_ORACLE_AGG, "weighted_sum")
@dataclass
class WeightedSum:
    """Weighted scalar evidence; an omitted enabled-source weight defaults to 1.

    Set a weight explicitly to zero or remove the source from enabled_oracles
    to turn off its contribution. Extra keys never enable an inactive source.
    """

    weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if any(not np.isfinite(value) or value < 0 for value in self.weights.values()):
            raise ValueError("weights must be finite and nonnegative")

    def __call__(self, oracle_outputs: dict[str, np.ndarray]) -> np.ndarray:
        if not oracle_outputs:
            raise ValueError("weighted_sum needs at least one enabled score")
        values = [np.asarray(value, dtype=float) for value in oracle_outputs.values()]
        if any(value.ndim != 1 or value.shape != values[0].shape for value in values):
            raise ValueError("weighted_sum requires equal-length scalar oracle vectors")
        return sum(self.weights.get(name, 1.0) * np.asarray(value, dtype=float)
                   for name, value in oracle_outputs.items())


@register_primitive(ROLE_CROSS_ORACLE_AGG, "trimmed_mean")
@dataclass
class TrimmedMean:
    trim: float = 0.0

    def __call__(self, oracle_outputs: dict[str, np.ndarray]) -> np.ndarray:
        stacked = np.stack([np.asarray(v, dtype=float) for v in oracle_outputs.values()], axis=0)
        if self.trim <= 0.0 or stacked.shape[0] < 3:
            return stacked.mean(axis=0)
        k = int(np.floor(self.trim * stacked.shape[0]))
        sorted_arr = np.sort(stacked, axis=0)
        trimmed = sorted_arr[k : stacked.shape[0] - k] if k > 0 else sorted_arr
        return trimmed.mean(axis=0)


@register_primitive(ROLE_CROSS_ORACLE_AGG, "contrastive_mingap")
@dataclass
class ContrastiveMinGap:
    """For conditional tasks: per-cell-type target minus best off-target activity.

    Expects oracle_outputs values to be 2D arrays shaped (n_sequences, n_cells),
    and the spec to pass `target_indices: list[int]` with one target index per sequence.

    Compute raw gaps before per-oracle EMA normalization and cross-oracle mean.
    Every enabled oracle must measure each target and at least one off-target.
    NaN off-target columns may represent absent heads; missing target evidence
    is an error, never a neutral reward or an implicitly dropped oracle.
    """

    target_indices: list[int] = field(default_factory=list)
    normalize: bool = True
    decay: float = 0.99
    clip: float = 3.0
    eps: float = 1e-8

    def __post_init__(self) -> None:
        # Per-oracle running mean/var of the GAP (not the raw per-cell scores).
        self._mu: dict[str, float] = {}
        self._var: dict[str, float] = {}

    def state_dict(self) -> dict:
        return {"mu": dict(self._mu), "var": dict(self._var)}

    def load_state_dict(self, state: dict) -> None:
        self._mu, self._var = _restore_moments(state)

    def _ema_zscore(self, key: str, gap: np.ndarray) -> np.ndarray:
        return _ema_zscore(gap, key, self._mu, self._var,
                           decay=self.decay, clip=self.clip, eps=self.eps)

    def __call__(self, oracle_outputs: dict[str, np.ndarray]) -> np.ndarray:
        return self._evaluate(oracle_outputs, frozen=False)

    def frozen_inference(self, oracle_outputs: dict[str, np.ndarray]) -> np.ndarray:
        return self._evaluate(oracle_outputs, frozen=True)

    def _evaluate(self, oracle_outputs: dict[str, np.ndarray], *, frozen: bool) -> np.ndarray:
        if not oracle_outputs:
            raise ValueError("contrastive_mingap needs at least one enabled oracle")
        # No declared off-target set: every non-target column is compared, a NaN
        # column is an absent head, and a missing target is an error.
        raw_gaps = {
            name: cell_gap(arr, self.target_indices, missing="raise", label=name)
            for name, arr in oracle_outputs.items()
        }
        # Validate the entire batch before mutating any normalization state.
        return np.stack([
            (_frozen_zscore(gap, name, self._mu, self._var, clip=self.clip, eps=self.eps)
             if frozen else self._ema_zscore(name, gap)) if self.normalize else gap
            for name, gap in raw_gaps.items()
        ]).mean(axis=0)
