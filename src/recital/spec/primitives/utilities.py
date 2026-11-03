"""Editable utility implementations over fixed raw-coordinate target goals.

The utility protocol is ``utility(raw, *, target, conditioning) -> (n,)``.
``target`` is an OracleTargetObjective; its membership, units, and condition
definitions belong to the goal and must not be rewritten by utility primitives.
Implementations may declare their own constructor parameters (scale, curvature,
etc.). The default delegates to the declared arithmetic without recalibration.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from recital.spec.registry import ROLE_PER_ORACLE_UTILITY, register_primitive
from recital.targets import OracleTargetObjective
from .ema_zscore import _ema_zscore, _frozen_zscore, _restore_moments


@register_primitive(ROLE_PER_ORACLE_UTILITY, "target_default")
@dataclass
class TargetDefault:
    """The task's declared per-row utility of the source's value, unchanged (the value itself for a maximize goal)."""

    def __call__(
        self,
        raw: np.ndarray,
        *,
        target: OracleTargetObjective,
        conditioning: Sequence[str] | None = None,
    ) -> np.ndarray:
        return target.evaluate(raw, conditioning=conditioning).utility


@register_primitive(ROLE_PER_ORACLE_UTILITY, "target_margin")
@dataclass
class TargetMargin:
    """Keep a gradient inside an interval target, where ``target_default`` is flat.

    ``target_default`` is the declaration's own arithmetic: the penalty is the Huber
    distance *outside* the interval, clamped to zero inside it, so every in-interval
    sequence scores exactly 0. That flat interior gives the GRPO advantage
    ``(reward - group mean)/group std`` no signal once the group is inside the
    interval, and at selection every eligible row ties at utility 0, so ranking
    degenerates to pool order.

    This primitive leaves the outside penalty identical and replaces the flat
    interior with the depth inside the interval, squashed by ``tanh`` so a one-sided
    bound cannot pay for unbounded drift. The two pieces meet at 0 on the boundary,
    so the utility stays continuous and monotone in "closer to target". Membership is
    unchanged: ``hit`` still comes from the declaration, and so does the depth --
    ``TargetEvaluation.margin`` -- so this primitive and a task's ``margin/<source>``
    selection measurement cannot disagree about where the interior is.

    A directional (maximize/minimize) target already has a gradient everywhere and
    carries no margin, so it is returned exactly as the declaration computes it.
    """

    margin_weight: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.margin_weight, bool) or not isinstance(self.margin_weight, (int, float)):
            raise ValueError("margin_weight must be a number")
        if not np.isfinite(self.margin_weight) or self.margin_weight <= 0:
            raise ValueError("margin_weight must be finite and positive")

    def __call__(
        self,
        raw: np.ndarray,
        *,
        target: OracleTargetObjective,
        conditioning: Sequence[str] | None = None,
    ) -> np.ndarray:
        evaluated = target.evaluate(np.asarray(raw, dtype=float), conditioning=conditioning)
        if evaluated.margin is None:
            return evaluated.utility
        utility = evaluated.utility.copy()
        # Strictly inside: on a bound the declared penalty is already 0 and so is
        # tanh(0), so the two pieces agree there whichever branch owns the row.
        inside = evaluated.margin > 0
        utility[inside] = self.margin_weight * np.tanh(evaluated.margin[inside])
        return utility


@register_primitive(ROLE_PER_ORACLE_UTILITY, "gap_ema_zscore")
@dataclass
class GapEmaZScore:
    """Running z-score of one oracle's target-minus-off-target gap.

    The per-oracle counterpart of ``contrastive_mingap(normalize=True)`` for a
    recipe whose ``reward_input`` has already reduced each oracle to one gap per
    row. The gap is put in the declared target's utility coordinate (the gap itself
    for the framework's reference-0, scale-1 maximize target), standardized by an
    exponential moving mean and variance, and clipped to ``[-clip, clip]``. The
    moments update from finite entries only and are held fixed by
    ``frozen_inference``. The arithmetic is shared with ``contrastive_mingap`` and
    ``ema_zscore``.
    """

    decay: float = 0.99
    clip: float = 3.0
    eps: float = 1e-8

    # One instance serves one oracle; the moments live under this single key.
    _KEY = "gap"

    def __post_init__(self) -> None:
        self._mu: dict[str, float] = {}
        self._var: dict[str, float] = {}

    def state_dict(self) -> dict:
        return {"mu": dict(self._mu), "var": dict(self._var)}

    def load_state_dict(self, state: dict) -> None:
        self._mu, self._var = _restore_moments(state)

    def __call__(
        self,
        raw: np.ndarray,
        *,
        target: OracleTargetObjective,
        conditioning: Sequence[str] | None = None,
    ) -> np.ndarray:
        gap = target.evaluate(raw, conditioning=conditioning).utility
        return _ema_zscore(gap, self._KEY, self._mu, self._var,
                           decay=self.decay, clip=self.clip, eps=self.eps)

    def frozen_inference(
        self,
        raw: np.ndarray,
        *,
        target: OracleTargetObjective,
        conditioning: Sequence[str] | None = None,
    ) -> np.ndarray:
        gap = target.evaluate(raw, conditioning=conditioning).utility
        return _frozen_zscore(gap, self._KEY, self._mu, self._var, clip=self.clip, eps=self.eps)
