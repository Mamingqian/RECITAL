"""Per-sequence shaping primitives.

Role: per_sequence_shaping.
Inputs are 1-D arrays of aggregated scalars; outputs are 1-D arrays of rewards.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from recital.spec.registry import ROLE_PER_SEQ_SHAPING, register_primitive


@register_primitive(ROLE_PER_SEQ_SHAPING, "target_distance")
@dataclass
class TargetDistance:
    """Negative Huber distance to a target. Smooth near target, linear far away."""

    target: float = 0.0
    delta: float = 0.8

    def __call__(self, x: np.ndarray) -> np.ndarray:
        d = np.abs(np.asarray(x, dtype=float) - self.target)
        quad = 0.5 * d * d
        lin = self.delta * (d - 0.5 * self.delta)
        return -np.where(d <= self.delta, quad, lin)


@register_primitive(ROLE_PER_SEQ_SHAPING, "interval_reward")
@dataclass
class IntervalReward:
    """Step function: 1.0 inside [lo, hi], else 0.0. Known to over-optimize."""

    lo: float = -0.5
    hi: float = 0.5
    coordinate: str = "raw_oracle"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.where((x >= self.lo) & (x <= self.hi), 1.0, 0.0)


@register_primitive(ROLE_PER_SEQ_SHAPING, "target_interval")
@dataclass
class TargetInterval:
    """Group-conditional negative-Huber distance to each sequence's TARGET band.

    For a conditional task (e.g. Loop-seq low/mid/high), each sequence has a
    target interval ``[lo, hi]`` chosen by its conditioning group: the reward is 0
    inside the band and a smooth Huber penalty for the distance to the nearest
    bound outside it, so a low-prompted sequence is pushed toward the low band and
    a high-prompted one toward the high band. ``None`` bounds are open (e.g. the
    low group's lower bound is ``None`` = -inf). The Huber ``delta`` defaults to 0.8;
    the plain ``target_distance`` targets a single global scalar and so can only
    ever satisfy the centre group.

    ``intervals`` is the per-group list of ``[lo, hi]`` in conditioning-prompt
    order; ``target_indices`` is the per-sequence group index, bound in per batch
    by the trainer (mirroring ``contrastive_mingap``).
    ``coordinate='aggregate'`` explicitly places those bounds on the composed
    scalar; it does not change any task's raw-coordinate target definition.
    """

    intervals: list = field(default_factory=list)
    target_indices: list = field(default_factory=list)
    delta: float = 0.8
    coordinate: str = "raw_oracle"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        n = len(x)
        if not self.intervals or len(self.target_indices) != n:
            # No per-sequence conditioning bound in (e.g. a run without prompts);
            # contribute no shaping rather than mis-targeting.
            return np.zeros(n, dtype=float)
        out = np.zeros(n, dtype=float)
        for i in range(n):
            lo, hi = self.intervals[int(self.target_indices[i])]
            xi = float(x[i])
            if lo is not None and xi < lo:
                d = lo - xi
            elif hi is not None and xi > hi:
                d = xi - hi
            else:
                d = 0.0
            out[i] = -(0.5 * d * d if d <= self.delta else self.delta * (d - 0.5 * self.delta))
        return out
