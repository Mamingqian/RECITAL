"""Aggregators of MAC-AMP's reward families, offered to the MAC-AMP port.

MAC-AMP's reward scientist writes `compute_rewards(Sa, Sb, Sc)` and its stage hints name aggregator families: weighted geometric
mean, weighted harmonic mean, and a soft minimum (`rl_refinement_module/prompt_scientist_en.txt:62-68`). Here they are cross-oracle
aggregators over the enabled scores, each in [0, 1] when the scores are. They are registered when this module is imported and
offered only to a run whose contract lists them under `framework.extra_allowed_primitives` (`adapter.AmpAdapter.capabilities`);
no other AMP run sees them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from recital.spec.registry import ROLE_CROSS_ORACLE_AGG, register_primitive

EPS = 1e-6


def _stack(outputs: dict[str, np.ndarray], weights: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    if not outputs:
        raise ValueError("an aggregator needs at least one enabled score")
    values = [np.asarray(value, dtype=float) for value in outputs.values()]
    if any(value.ndim != 1 or value.shape != values[0].shape for value in values):
        raise ValueError("an aggregator requires equal-length scalar score vectors")
    w = np.asarray([weights.get(name, 1.0) for name in outputs], dtype=float)
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be finite, nonnegative and not all zero")
    return np.stack(values, axis=0), w


@register_primitive(ROLE_CROSS_ORACLE_AGG, "weighted_geometric_mean")
@dataclass
class WeightedGeometricMean:
    """exp(sum w log s / sum w); a score below 1e-6 counts as 1e-6, so one failed score pulls the reward down hard."""

    weights: dict[str, float] = field(default_factory=dict)

    def __call__(self, oracle_outputs: dict[str, np.ndarray]) -> np.ndarray:
        stacked, w = _stack(oracle_outputs, self.weights)
        return np.exp((w[:, None] * np.log(np.maximum(stacked, EPS))).sum(axis=0) / w.sum())


@register_primitive(ROLE_CROSS_ORACLE_AGG, "weighted_harmonic_mean")
@dataclass
class WeightedHarmonicMean:
    """sum w / sum (w / s); a score below 1e-6 counts as 1e-6."""

    weights: dict[str, float] = field(default_factory=dict)

    def __call__(self, oracle_outputs: dict[str, np.ndarray]) -> np.ndarray:
        stacked, w = _stack(oracle_outputs, self.weights)
        return w.sum() / (w[:, None] / np.maximum(stacked, EPS)).sum(axis=0)


@register_primitive(ROLE_CROSS_ORACLE_AGG, "soft_minimum")
@dataclass
class SoftMin:
    """-(1/beta) log(sum w exp(-beta s) / sum w): the minimum for a large beta, the weighted mean for beta near 0."""

    weights: dict[str, float] = field(default_factory=dict)
    beta: float = 8.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("beta must be finite and positive")

    def __call__(self, oracle_outputs: dict[str, np.ndarray]) -> np.ndarray:
        stacked, w = _stack(oracle_outputs, self.weights)
        shifted = -self.beta * stacked
        top = shifted.max(axis=0)
        return -(top + np.log((w[:, None] * np.exp(shifted - top)).sum(axis=0) / w.sum())) / self.beta
