"""Scalar targets evaluated in each oracle's declared, fixed raw coordinate.

No task names, token alphabets, or batch-dependent calibration belong here.
Adapters map their conditioning to the exact keys in ``by_condition`` and supply
scalar oracle outputs in the declared units. A fixed positive ``scale`` converts
raw distances to dimensionless utilities; it never changes target membership.
RewardSpec.oracle_utilities selects independently editable training
utilities; it does not redefine the raw goal or its membership predicate.

Intervals are lower-inclusive and upper-exclusive: ``[lower, upper)``. ``None``
denotes an unbounded end. Distance to an excluded upper endpoint is zero, as for
any continuous distance to a half-open set; utility is a training surrogate, not
a substitute for the separately returned ``hit`` predicate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class TargetEvaluation:
    utility: np.ndarray
    hit: np.ndarray | None
    valid: np.ndarray
    # Signed distance from the nearest declared interval bound, inward, in units of
    # ``scale``: positive inside, negative outside, and None for a directional target,
    # which has no bound to be inside of. ``utility`` is clamped to zero inside an
    # interval, so this is the only place the interior's geometry survives. Readers:
    # the ``target_margin`` reward primitive and a task's ``margin/<source>``
    # selection measurement.
    margin: np.ndarray | None = None


@dataclass(frozen=True)
class ScalarTarget:
    """A fixed raw-coordinate target, independent of modality and oracle role.

    ``interval`` utility is negative Huber distance with delta=1, after division
    by ``scale``. ``maximize``/``minimize`` use signed ``(score-reference)/scale``;
    an optional threshold supplies an inclusive qualification predicate. Without
    a threshold their ``hit`` is None: a direction alone does not define success.
    ``reference`` is a fixed origin for directional utility, not an EMA mean.
    """

    kind: Literal["maximize", "minimize", "interval"]
    units: str
    scale: float
    coordinate: Literal["raw"] = "raw"
    lower: float | None = None
    upper: float | None = None
    threshold: float | None = None
    reference: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in {"maximize", "minimize", "interval"}:
            raise ValueError(f"unknown target kind {self.kind!r}")
        if self.coordinate != "raw":
            raise ValueError("target coordinate must be 'raw'; moving coordinates are not targets")
        if not isinstance(self.units, str) or not self.units.strip():
            raise ValueError("target units must be a nonempty declared unit name")
        for name in ("scale", "lower", "upper", "threshold", "reference"):
            value = getattr(self, name)
            if value is None and name in {"lower", "upper", "threshold"}:
                continue
            if isinstance(value, (bool, str)) or not isinstance(value, (int, float)):
                raise ValueError(f"target {name} must be a finite number")
            if not np.isfinite(value):
                raise ValueError(f"target {name} must be finite; use None for unbounded ends")
        if self.scale <= 0:
            raise ValueError("target scale must be positive and frozen before optimization")
        if self.kind == "interval":
            if self.lower is None and self.upper is None:
                raise ValueError("interval target needs at least one finite bound")
            if self.lower is not None and self.upper is not None and self.lower >= self.upper:
                raise ValueError("interval target must satisfy lower < upper")
            if self.threshold is not None or self.reference != 0.0:
                raise ValueError("interval target does not use threshold or reference")
        elif self.lower is not None or self.upper is not None:
            raise ValueError("directional target uses threshold, not lower/upper")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScalarTarget:
        allowed = {
            "kind",
            "units",
            "scale",
            "coordinate",
            "lower",
            "upper",
            "threshold",
            "reference",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown target field(s): {sorted(unknown)}")
        missing = {"kind", "units", "scale", "coordinate"} - set(payload)
        if missing:
            raise ValueError(f"target must explicitly declare {sorted(missing)}")
        return cls(**dict(payload))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.kind == "interval":
            del payload["threshold"]
            del payload["reference"]
        else:
            del payload["lower"]
            del payload["upper"]
        return payload

    def evaluate(self, scores: Sequence[float] | np.ndarray) -> TargetEvaluation:
        values = np.asarray(scores, dtype=float)
        if values.ndim != 1:
            raise ValueError(
                f"scalar target expects a one-dimensional score array, got {values.shape}"
            )
        valid = np.isfinite(values)
        utility = np.full(values.shape, np.nan)
        measured = values[valid]
        margin = None
        if self.kind == "interval":
            distance = np.zeros(measured.shape)
            inward = np.full(measured.shape, np.inf)
            hit = valid.copy()
            if self.lower is not None:
                distance = np.maximum(distance, self.lower - measured)
                inward = np.minimum(inward, measured - self.lower)
                hit &= values >= self.lower
            if self.upper is not None:
                distance = np.maximum(distance, measured - self.upper)
                inward = np.minimum(inward, self.upper - measured)
                hit &= values < self.upper
            margin = np.full(values.shape, np.nan)
            margin[valid] = inward / self.scale
            distance /= self.scale
            huber = distance - 0.5
            near = distance <= 1.0
            huber[near] = 0.5 * distance[near] ** 2
            utility[valid] = -huber
        else:
            direction = 1.0 if self.kind == "maximize" else -1.0
            utility[valid] = direction * (measured - self.reference) / self.scale
            hit = None
            if self.threshold is not None:
                hit = valid & (
                    (values >= self.threshold) if direction > 0 else (values <= self.threshold)
                )
        return TargetEvaluation(utility=utility, hit=hit, valid=valid, margin=margin)


@dataclass(frozen=True)
class OracleTargetObjective:
    """One oracle's unconditional target or complete condition-to-target mapping.

    Serialized format uses common ``kind/coordinate/units/scale`` at the top
    level. ``by_condition`` optionally supplies lower/upper (interval) or
    threshold/reference (directional) per condition. Mixing unconditional bounds
    with conditional bounds is rejected rather than silently overriding them.
    """

    target: ScalarTarget | None = None
    by_condition: Mapping[str, ScalarTarget] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.target is None) == (not self.by_condition):
            raise ValueError("provide exactly one unconditional target or by_condition mapping")
        if self.target is not None and not isinstance(self.target, ScalarTarget):
            raise ValueError("target must be a ScalarTarget")
        for key, target in self.by_condition.items():
            if not isinstance(key, str) or not key or not isinstance(target, ScalarTarget):
                raise ValueError(
                    "by_condition must map nonempty exact condition names to ScalarTarget"
                )
        if self.by_condition:
            declarations = {
                (t.kind, t.coordinate, t.units, t.scale) for t in self.by_condition.values()
            }
            if len(declarations) != 1:
                raise ValueError(
                    "one oracle must use the same kind/coordinate/units/scale for every condition"
                )
            has_hits = [
                t.kind == "interval" or t.threshold is not None for t in self.by_condition.values()
            ]
            if any(has_hits) and not all(has_hits):
                raise ValueError(
                    "conditional target thresholds must be specified for every condition or none"
                )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OracleTargetObjective:
        if not isinstance(payload, Mapping):
            raise ValueError("oracle target objective must be a mapping")
        if "by_condition" not in payload:
            return cls(target=ScalarTarget.from_dict(payload))
        common = {key: value for key, value in payload.items() if key != "by_condition"}
        allowed = {"kind", "coordinate", "units", "scale"}
        if set(common) - allowed:
            raise ValueError(
                "conditional objective allows only kind/coordinate/units/scale beside by_condition"
            )
        per_condition = payload["by_condition"]
        if not isinstance(per_condition, Mapping) or not per_condition:
            raise ValueError("by_condition must be a nonempty mapping")
        targets = {}
        for key, bounds in per_condition.items():
            if not isinstance(bounds, Mapping) or set(bounds) - {
                "lower",
                "upper",
                "threshold",
                "reference",
            }:
                raise ValueError(
                    f"by_condition[{key!r}] must declare only lower/upper/threshold/reference"
                )
            targets[key] = ScalarTarget.from_dict({**common, **bounds})
        return cls(by_condition=targets)

    def to_dict(self) -> dict[str, Any]:
        if self.target is not None:
            return self.target.to_dict()
        first = next(iter(self.by_condition.values())).to_dict()
        common = {key: first[key] for key in ("kind", "coordinate", "units", "scale")}
        return {
            **common,
            "by_condition": {
                key: {name: value for name, value in target.to_dict().items() if name not in common}
                for key, target in self.by_condition.items()
            },
        }

    def validate_conditioning(
        self,
        n_scores: int,
        conditioning: Sequence[str] | None = None,
    ) -> None:
        """Check frozen condition coverage without calculating training utility."""
        if conditioning is not None and len(conditioning) != n_scores:
            raise ValueError("conditioning must have one exact condition key per score")
        if self.target is not None:
            return
        if conditioning is None:
            raise ValueError("conditional target requires explicit conditioning")
        unknown = set(conditioning) - set(self.by_condition)
        if unknown:
            raise ValueError(f"target has no declaration for condition(s) {sorted(unknown)}")

    @property
    def kind(self) -> str:
        """The one declared kind; ``__post_init__`` refuses a mixture across conditions."""
        if self.target is not None:
            return self.target.kind
        return next(iter(self.by_condition.values())).kind

    def evaluate(
        self,
        scores: Sequence[float] | np.ndarray,
        conditioning: Sequence[str] | None = None,
    ) -> TargetEvaluation:
        values = np.asarray(scores, dtype=float)
        if values.ndim != 1:
            raise ValueError("oracle target expects one scalar score per sequence")
        self.validate_conditioning(len(values), conditioning)
        if self.target is not None:
            return self.target.evaluate(values)
        utility = np.full(values.shape, np.nan)
        valid = np.isfinite(values)
        has_hits = [
            t.kind == "interval" or t.threshold is not None for t in self.by_condition.values()
        ]
        hit = np.zeros(values.shape, dtype=bool) if all(has_hits) else None
        kinds = {t.kind for t in self.by_condition.values()}
        margin = np.full(values.shape, np.nan) if kinds == {"interval"} else None
        labels = np.asarray(conditioning)
        for key, target in self.by_condition.items():
            mask = labels == key
            result = target.evaluate(values[mask])
            utility[mask] = result.utility
            if hit is not None:
                hit[mask] = result.hit
            if margin is not None:
                margin[mask] = result.margin
        return TargetEvaluation(utility=utility, hit=hit, valid=valid, margin=margin)
