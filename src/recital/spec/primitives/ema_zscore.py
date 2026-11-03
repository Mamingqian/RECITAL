"""EMA z-score normalization per oracle.

Role: per_oracle_normalization.
Maintains a running mean and variance per oracle and outputs (x - mu) / sigma,
clipped to [-clip, +clip]. The EMA decay and clip are exposed as parameters so
the Strategist can tune them across iterations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recital.spec.registry import ROLE_PER_ORACLE_NORM, register_primitive


def _restore_moments(state: dict) -> tuple[dict[str, float], dict[str, float]]:
    """Validate the small JSON state shared by the running normalizers."""
    if not isinstance(state, dict) or set(state) != {"mu", "var"}:
        raise ValueError("normalizer state must contain mu and var")
    mu, var = state["mu"], state["var"]
    if not isinstance(mu, dict) or not isinstance(var, dict) or set(mu) != set(var):
        raise ValueError("normalizer means and variances must have the same oracle keys")
    if any(not isinstance(key, str) for key in mu):
        raise ValueError("normalizer oracle keys must be strings")
    mu, var = {key: float(value) for key, value in mu.items()}, {
        key: float(value) for key, value in var.items()
    }
    if any(not np.isfinite(value) for value in (*mu.values(), *var.values())) or any(
        value < 0 for value in var.values()
    ):
        raise ValueError("normalizer state requires finite means and nonnegative finite variances")
    return mu, var


def _frozen_zscore(values: np.ndarray, key: str, mu: dict, var: dict,
                   *, clip: float, eps: float) -> np.ndarray:
    """Use trained moments; an unseen source has no calibrated reward scale."""
    if key not in mu or key not in var:
        raise ValueError(f"frozen reward has no trained normalization statistics for {key!r}")
    return np.clip((np.asarray(values, dtype=float) - mu[key]) /
                   max(float(np.sqrt(var[key])), eps), -clip, clip)


def _ema_zscore(values: np.ndarray, key: str, mu: dict, var: dict,
                *, decay: float, clip: float, eps: float) -> np.ndarray:
    """Z-score ``values`` against running moments kept in ``mu``/``var`` under ``key``.

    The moments are updated in place from the batch's FINITE entries only: one NaN
    or inf would otherwise write NaN into the stored variance permanently, and every
    later batch of that source would score NaN. A batch with no finite entry leaves
    the moments untouched and is returned as it came.
    """
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values
    batch_mu = float(finite.mean())
    batch_var = float(finite.var())
    new_mu = mu.get(key, batch_mu)
    new_var = var.get(key, batch_var)
    new_mu = decay * new_mu + (1.0 - decay) * batch_mu
    new_var = decay * new_var + (1.0 - decay) * batch_var
    mu[key] = new_mu
    var[key] = new_var
    z = (values - new_mu) / max(float(np.sqrt(new_var)), eps)
    return np.clip(z, -clip, clip)


@register_primitive(ROLE_PER_ORACLE_NORM, "ema_zscore")
@dataclass
class EmaZScore:
    decay: float = 0.99
    clip: float = 3.0
    eps: float = 1e-8

    def __post_init__(self) -> None:
        self._mu: dict[str, float] = {}
        self._var: dict[str, float] = {}

    def state_dict(self) -> dict:
        return {"mu": dict(self._mu), "var": dict(self._var)}

    def load_state_dict(self, state: dict) -> None:
        self._mu, self._var = _restore_moments(state)

    def frozen_inference(self, raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {name: _frozen_zscore(values, name, self._mu, self._var,
                                   clip=self.clip, eps=self.eps)
                for name, values in raw.items()}

    def __call__(self, raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {oracle: _ema_zscore(values, oracle, self._mu, self._var,
                                    decay=self.decay, clip=self.clip, eps=self.eps)
                for oracle, values in raw.items()}
