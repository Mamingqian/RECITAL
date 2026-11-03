"""Explicit pool-to-library selection shared by sequence-design tasks.

Keep pool rows and selected indices so filtered delivery cannot masquerade as raw
generation quality. The primary-score selector remains an explicit strategy;
agent selection plans can use any declared, visible measurement sources.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

import numpy as np


DELIVERY_PROTOCOL = "selected_evaluation_library_v1"


def stable_top_indices(scores: Sequence[float], eligible: Sequence[bool], count: int) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    mask = np.asarray(eligible, dtype=bool)
    if values.ndim != 1 or mask.shape != values.shape:
        raise ValueError("selection scores and eligibility must be aligned vectors")
    if type(count) is not int or count < 1:
        raise ValueError("delivery count must be a positive integer")
    indices = np.flatnonzero(mask & np.isfinite(values))
    # Original pool position resolves ties, independently of every hidden score.
    return indices[np.argsort(-values[indices], kind="stable")[:count]]


@dataclass(frozen=True)
class LibrarySelectionPolicy:
    evaluation_candidates: int
    count: int
    primary_oracle: str
    deduplication: str = "exact"
    exclude_training_copies: bool = True
    protocol: str = DELIVERY_PROTOCOL

    def __post_init__(self):
        if (type(self.evaluation_candidates) is not int or type(self.count) is not int
                or not 1 <= self.count <= self.evaluation_candidates):
            raise ValueError("library selection requires 1 <= count <= evaluation_candidates")
        if not isinstance(self.primary_oracle, str) or not self.primary_oracle:
            raise ValueError("library selection requires an explicit primary oracle")
        if self.deduplication not in {"exact", "reverse_complement"}:
            raise ValueError("unknown sequence deduplication rule")
        if type(self.exclude_training_copies) is not bool or self.protocol != DELIVERY_PROTOCOL:
            raise ValueError("invalid library selection policy")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_contract(cls, contract: Mapping[str, Any]) -> "LibrarySelectionPolicy | None":
        config = contract.get("delivery")
        if config is None:
            return None  # A contract with no delivery block declares no selection policy.
        if config.get("selection_aggregation") != "primary":
            raise ValueError("DNA library selection must use its declared primary score")
        fields = cls.__dataclass_fields__
        return cls(**{key: config[key] for key in fields if key in config})


@dataclass
class PrimaryLibrarySelector:
    policy: LibrarySelectionPolicy
    strata: tuple[str, ...]
    condition_to_stratum: Callable[[str], str | None]
    primary_scores: Callable[[np.ndarray, list[str]], np.ndarray]
    legal_sequence: Callable[[str], bool]
    sequence_identity: Callable[[str], str]
    training_identities: frozenset[str]

    def __call__(self, *, sequences, conditioning, primary_outputs):
        sequences, conditioning = list(sequences), list(conditioning)
        n = len(sequences)
        if len(conditioning) != n:
            raise ValueError("selection requires aligned sequences and conditions")
        groups = [self.condition_to_stratum(str(value)) for value in conditioning]
        if any(group not in self.strata for group in groups):
            raise ValueError("candidate pool contains an unknown condition")
        scores = np.asarray(self.primary_scores(np.asarray(primary_outputs), groups), dtype=float)
        if scores.shape != (n,):
            raise ValueError("primary ranking must produce one finite-or-unavailable score per pool row")
        eligible = np.zeros(n, dtype=bool)
        reasons = []
        seen = {group: set() for group in self.strata}
        for index, (sequence, group) in enumerate(zip(sequences, groups, strict=True)):
            reason = "eligible"
            if not self.legal_sequence(sequence):
                reason = "invalid_sequence"
            else:
                identity = self.sequence_identity(sequence)
                if identity in seen[group]:
                    reason = "duplicate"
                elif self.policy.exclude_training_copies and identity in self.training_identities:
                    reason = "training_copy"
                elif not np.isfinite(scores[index]):
                    reason = "nonfinite_primary"
                seen[group].add(identity)
            reasons.append(reason)
            eligible[index] = reason == "eligible"
        selected = []
        summaries = {}
        for group in self.strata:
            group_mask = np.asarray([value == group for value in groups], dtype=bool)
            if int(group_mask.sum()) != self.policy.evaluation_candidates:
                raise ValueError(f"candidate pool for {group} differs from declared generation count")
            chosen = stable_top_indices(scores, eligible & group_mask, self.policy.count)
            selected.extend(chosen.tolist())
            summaries[group] = {
                "pool_count": int(group_mask.sum()), "eligible_count": int((eligible & group_mask).sum()),
                "selected_count": len(chosen), "required_count": self.policy.count,
                "complete_delivery": len(chosen) == self.policy.count,
                "rejection_counts": {reason: sum(reason == reasons[i] for i in np.flatnonzero(group_mask))
                                     for reason in ("invalid_sequence", "duplicate", "training_copy", "nonfinite_primary")},
            }
        # Keep generation order in the delivered artifact, as CM does after setting
        # its selected-row mask.
        return np.asarray(sorted(selected), dtype=int), {
            "policy": self.policy.to_dict(), "by_stratum": summaries,
            "strata": groups,
            "selected_indices": sorted(selected), "eligibility": eligible.tolist(),
            "rejection_reasons": reasons,
            "selection_scores": [float(value) if np.isfinite(value) else None for value in scores],
            "complete_delivery": all(row["complete_delivery"] for row in summaries.values()),
            "tie_break": "original_pool_position", "selection_scope": "within_each_stratum",
        }


def select_calibration_pool(frame, selector, primary_outputs, *, condition_column="prompt"):
    """Apply the live selector independently to each frozen-SFT sampling seed."""
    frame = frame.copy().reset_index(drop=True)
    values = np.asarray(primary_outputs)
    if len(values) != len(frame) or frame.sampling_seed.isna().any():
        raise ValueError("calibration requires aligned primary outputs and sampling seeds")
    frame["selected"] = False
    summaries = {}
    for seed, group in frame.groupby("sampling_seed", sort=False):
        indices, report = selector(
            sequences=group.sequence.tolist(), conditioning=group[condition_column].tolist(),
            primary_outputs=values[group.index],
        )
        if not report["complete_delivery"]:
            raise ValueError(f"calibration seed {seed} cannot fill the declared selected library")
        frame.loc[group.index[indices], "selected"] = True
        summaries[str(seed)] = report["by_stratum"]
    return frame, summaries


PLAN_PROTOCOL = "agent_library_selection_v1"


def _finite_parameter(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be numeric")
    if not np.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))


@dataclass(frozen=True)
class SelectionComponent:
    """A frozen affine calibration followed by a declared direction.

    Utility is sign * (measurement - offset) / scale. Matrix measurements need
    an explicit column; task-specific utilities can instead be supplied as
    already-computed measurements with their original source dependencies.
    """

    measurement: str
    direction: str = "maximize"
    weight: float = 1.0
    offset: float = 0.0
    scale: float = 1.0
    column: int | None = None
    calibration_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.measurement, str) or not self.measurement:
            raise ValueError("component requires a measurement name")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("component direction must be maximize or minimize")
        _finite_parameter(self.weight, "component weight")
        if self.weight < 0:
            raise ValueError("use direction, not negative component weights")
        _finite_parameter(self.offset, "calibration offset")
        _finite_parameter(self.scale, "calibration scale", positive=True)
        if self.column is not None and (type(self.column) is not int or self.column < 0):
            raise ValueError("measurement column must be a nonnegative integer")
        if self.calibration_id is not None and (not isinstance(self.calibration_id, str) or not self.calibration_id):
            raise ValueError("calibration identity cannot be empty")


@dataclass(frozen=True)
class SelectionFilter:
    """Inclusive bounds on a named measurement in its stored coordinates."""

    measurement: str
    minimum: float | None = None
    maximum: float | None = None
    column: int | None = None
    strata: tuple[str, ...] = ()

    def __post_init__(self):
        SelectionComponent(self.measurement, column=self.column)
        if self.minimum is None and self.maximum is None:
            raise ValueError("a selection filter must specify at least one bound")
        for name in ("minimum", "maximum"):
            if getattr(self, name) is not None:
                _finite_parameter(getattr(self, name), name)
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("selection filter bounds are reversed")
        object.__setattr__(self, "strata", tuple(self.strata))


@dataclass(frozen=True)
class SelectionPlan:
    """Executable library selection; pool size is independent of delivery K.

    Standard strategies rank within each stratum. Pareto uses nondomination
    layers and requires an explicit within-layer rule. Diversity is a declared
    greedy minimum-distance constraint.
    Custom selectors are named, externally resolved sandbox adapters; this module
    does not compile or run arbitrary source code.
    """

    count: int | Mapping[str, int]
    components: tuple[SelectionComponent, ...] = ()
    aggregation: str = "weighted_sum"
    strata: tuple[str, ...] = ()
    filters: tuple[SelectionFilter, ...] = ()
    deduplication: str = "exact"
    deduplication_scope: str = "within_stratum"
    exclude_training_copies: bool = True
    alphabet: str | None = None
    min_length: int = 1
    max_length: int | None = None
    missing: str = "exclude"
    tie_break: str = "original_pool_position"
    pareto_tie_break: str | None = None
    minimum_distance: float = 0.0
    distance_metric: str = "normalized_edit"
    diversity_scope: str = "within_stratum"
    custom_selector: str | None = None
    custom_inputs: tuple[str, ...] = ()
    custom_parameters: Mapping[str, Any] | None = None
    protocol: str = PLAN_PROTOCOL

    def __post_init__(self):
        if isinstance(self.count, Mapping):
            counts = dict(self.count)
            if not counts or any(not isinstance(key, str) or not key for key in counts):
                raise ValueError("count mapping requires named strata")
            values = counts.values()
            object.__setattr__(self, "count", counts)
        else:
            values = [self.count]
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError("delivery counts must be positive integers")
        components = tuple(c if isinstance(c, SelectionComponent) else SelectionComponent(**c)
                           for c in self.components)
        filters = tuple(f if isinstance(f, SelectionFilter) else SelectionFilter(**f) for f in self.filters)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "filters", filters)
        object.__setattr__(self, "strata", tuple(self.strata))
        object.__setattr__(self, "custom_inputs", tuple(self.custom_inputs))
        if len(set(self.strata)) != len(self.strata) or any(not isinstance(s, str) or not s for s in self.strata):
            raise ValueError("strata must be distinct nonempty names")
        if isinstance(self.count, Mapping) and self.strata and set(self.count) != set(self.strata):
            raise ValueError("count mapping must cover exactly the declared strata")
        if self.aggregation not in {"single", "weighted_sum", "min", "lexicographic", "pareto", "custom"}:
            raise ValueError("unknown selection aggregation")
        if self.aggregation != "custom" and not components:
            raise ValueError("standard ranking requires score components")
        if self.aggregation == "single" and len(components) != 1:
            raise ValueError("single ranking requires exactly one component")
        if components and not any(c.weight > 0 for c in components):
            raise ValueError("ranking needs at least one positive component weight")
        if self.tie_break != "original_pool_position":
            raise ValueError("the final deterministic tie break must be original_pool_position")
        choices = {"lexicographic", "weighted_sum", "original_pool_position"}
        if self.aggregation == "pareto" and self.pareto_tie_break not in choices:
            raise ValueError("Pareto ranking requires an explicit within-front tie rule")
        if self.aggregation != "pareto" and self.pareto_tie_break is not None:
            raise ValueError("Pareto tie rule is only meaningful for Pareto ranking")
        if self.aggregation == "custom" and (not isinstance(self.custom_selector, str) or not self.custom_selector):
            raise ValueError("custom ranking requires an externally resolved selector identity")
        if self.aggregation != "custom" and (self.custom_selector or self.custom_inputs or self.custom_parameters):
            raise ValueError("custom inputs only apply to custom ranking")
        if any(not isinstance(name, str) or not name for name in self.custom_inputs):
            raise ValueError("custom measurement names cannot be empty")
        if self.deduplication not in {"none", "exact", "reverse_complement"}:
            raise ValueError("unknown deduplication rule")
        for scope in (self.deduplication_scope, self.diversity_scope):
            if scope not in {"within_stratum", "global"}:
                raise ValueError("selection scope must be within_stratum or global")
        if type(self.exclude_training_copies) is not bool:
            raise ValueError("exclude_training_copies must be boolean")
        if self.alphabet is not None and (not isinstance(self.alphabet, str) or not self.alphabet):
            raise ValueError("alphabet must be a nonempty string")
        if type(self.min_length) is not int or self.min_length < 1:
            raise ValueError("min_length must be a positive integer")
        if self.max_length is not None and (type(self.max_length) is not int or self.max_length < self.min_length):
            raise ValueError("max_length must be at least min_length")
        if self.missing not in {"exclude", "error"}:
            raise ValueError("missing measurements must be excluded or reported as errors")
        _finite_parameter(self.minimum_distance, "minimum distance")
        if not 0 <= self.minimum_distance <= 1:
            raise ValueError("normalized minimum distance must be between zero and one")
        if self.distance_metric not in {"normalized_edit", "normalized_hamming"}:
            raise ValueError("unknown diversity distance")
        if self.protocol != PLAN_PROTOCOL:
            raise ValueError("unknown agent selection protocol")
        # Parameters cross the sandbox adapter boundary as data, never code.
        if self.custom_parameters is not None and not isinstance(self.custom_parameters, Mapping):
            raise ValueError("custom parameters must be an object")
        json.dumps(self.custom_parameters, allow_nan=False)

    def to_dict(self):
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    @property
    def measurement_names(self):
        return tuple(dict.fromkeys([c.measurement for c in self.components]
                                   + [f.measurement for f in self.filters] + list(self.custom_inputs)))


def selection_plan_schema():
    """The public JSON plan surface; constructors enforce cross-field rules."""
    component = {
        "type": "object", "additionalProperties": False, "required": ["measurement"],
        "properties": {
            "measurement": {"type": "string", "minLength": 1},
            "direction": {"enum": ["maximize", "minimize"], "default": "maximize"},
            "weight": {"type": "number", "minimum": 0, "default": 1},
            "offset": {"type": "number", "default": 0},
            "scale": {"type": "number", "exclusiveMinimum": 0, "default": 1},
            "column": {"type": ["integer", "null"], "minimum": 0, "default": None},
            "calibration_id": {"type": ["string", "null"], "default": None},
        },
    }
    filters = {
        "type": "object", "additionalProperties": False, "required": ["measurement"],
        "properties": {
            "measurement": {"type": "string", "minLength": 1},
            "minimum": {"type": ["number", "null"], "default": None},
            "maximum": {"type": ["number", "null"], "default": None},
            "column": {"type": ["integer", "null"], "minimum": 0, "default": None},
            "strata": {"type": "array", "items": {"type": "string"}, "default": []},
        },
    }
    return {
        "type": "object", "additionalProperties": False, "required": ["count"],
        "properties": {
            "count": {"oneOf": [{"type": "integer", "minimum": 1},
                                    {"type": "object", "minProperties": 1,
                                     "additionalProperties": {"type": "integer", "minimum": 1}}]},
            "components": {"type": "array", "items": component, "default": []},
            "aggregation": {"enum": ["single", "weighted_sum", "min", "lexicographic", "pareto", "custom"],
                            "default": "weighted_sum"},
            "strata": {"type": "array", "items": {"type": "string", "minLength": 1},
                       "uniqueItems": True, "default": []},
            "filters": {"type": "array", "items": filters, "default": []},
            "deduplication": {"enum": ["none", "exact", "reverse_complement"], "default": "exact"},
            "deduplication_scope": {"enum": ["within_stratum", "global"], "default": "within_stratum"},
            "exclude_training_copies": {"type": "boolean", "default": True},
            "alphabet": {"type": ["string", "null"], "default": None},
            "min_length": {"type": "integer", "minimum": 1, "default": 1},
            "max_length": {"type": ["integer", "null"], "minimum": 1, "default": None},
            "missing": {"enum": ["exclude", "error"], "default": "exclude"},
            "tie_break": {"const": "original_pool_position", "default": "original_pool_position"},
            "pareto_tie_break": {"enum": [None, "lexicographic", "weighted_sum", "original_pool_position"],
                                 "default": None},
            "minimum_distance": {"type": "number", "minimum": 0, "maximum": 1, "default": 0},
            "distance_metric": {"enum": ["normalized_edit", "normalized_hamming"], "default": "normalized_edit"},
            "diversity_scope": {"enum": ["within_stratum", "global"], "default": "within_stratum"},
            "custom_selector": {"type": ["string", "null"], "default": None},
            "custom_inputs": {"type": "array", "items": {"type": "string", "minLength": 1}, "default": []},
            "custom_parameters": {"type": ["object", "null"], "default": None},
            "protocol": {"const": PLAN_PROTOCOL, "default": PLAN_PROTOCOL},
        },
    }


def selection_capabilities():
    return {
        "protocol": PLAN_PROTOCOL, "plan_schema": selection_plan_schema(),
        "component_utility": "direction_sign * (measurement - offset) / scale * weight",
        "aggregation_semantics": {
            "single": "exactly one component",
            "weighted_sum": "sum calibrated directed component utilities",
            "min": "minimum calibrated directed weighted utility",
            "lexicographic": "components in declared priority order",
            "pareto": "nondomination fronts, then explicit pareto_tie_break",
            "custom": "published isolated library_selector returns unique eligible pool indices",
        },
        "notes": [
            "N is the actual generation pool size; count is K per named stratum, independently configured.",
            "No primary source is mandatory. Only declared visible dependencies can influence selection.",
            "Task-specific derived measurements retain their source dependencies; sequence-only metrics may have none.",
            "Matrix outputs require an explicit column or a separately computed task utility.",
            "Training-copy exclusion requires the full supplied training reference; unknown reference is an error.",
            "Missing or nonfinite required scores are excluded or raised according to missing; never zero-filled.",
            "Strict row-level finiteness applies after sequence legality and training-copy exclusion; missing or failed source columns still raise.",
            "Dedup retains the first eligible pool occurrence; output uses original pool order.",
            "Diversity is greedy rank-order minimum distance; insufficient K is reported, never replenished silently.",
            "Pareto and lexicographic plans do not invent a scalar score for comparing recipes.",
        ],
    }


@dataclass(frozen=True)
class SelectionMeasurement:
    """Row-bound score output with its actual underlying source dependencies.

    ``sources`` must be populated by the trusted scorer/derived-measurement
    resolver, not inferred from an agent-provided display name. Raw arrays in
    the simple API are shorthand for a single source with the mapping's name.
    """

    values: Any
    sources: tuple[str, ...]
    row_ids: tuple[str, ...] | None = None
    status: str = "measured"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self):
        object.__setattr__(self, "sources", tuple(self.sources))
        if any(not isinstance(s, str) or not s for s in self.sources):
            raise ValueError("measurements require explicit source dependencies")
        if self.status not in {"measured", "partial", "missing", "failed", "unavailable"}:
            raise ValueError("unknown measurement status")
        if self.row_ids is not None:
            object.__setattr__(self, "row_ids", tuple(self.row_ids))


def _selection_measurements(measurements, names, n, row_ids, visible_sources, blind_sources, missing):
    if visible_sources is None:
        raise ValueError("selection requires an explicit visible source allowlist")
    allowed, blind = set(visible_sources), set(blind_sources)
    resolved = {}
    for name in names:
        value = measurements.get(name)
        if isinstance(value, SelectionMeasurement):
            measurement = value
        elif isinstance(value, Mapping):
            measurement = SelectionMeasurement(**value)
        else:
            measurement = SelectionMeasurement(values=value, sources=(name,),
                status="missing" if value is None else "measured")
        if name in blind or set(measurement.sources) & blind or not set(measurement.sources) <= allowed:
            raise ValueError(f"selection measurement {name!r} depends on a hidden or blind source")
        if measurement.row_ids is not None and measurement.row_ids != tuple(row_ids):
            raise ValueError(f"measurement {name!r} row identities differ from the candidate pool")
        if measurement.status not in {"measured", "partial"}:
            if missing == "error":
                raise ValueError(f"selection measurement {name!r} is {measurement.status}")
            values = np.full(n, np.nan)
        else:
            values = np.asarray(measurement.values, dtype=float)
            if values.ndim not in {1, 2} or values.shape[0] != n:
                raise ValueError(f"measurement {name!r} must align with all pool rows")
        resolved[name] = (measurement, values)
    return resolved


def _measurement_column(resolved, item):
    measurement, values = resolved[item.measurement]
    if values.ndim == 1:
        if item.column is not None and measurement.status in {"measured", "partial"}:
            raise ValueError(f"measurement {item.measurement!r} is scalar, not a matrix")
        return values
    if item.column is None or item.column >= values.shape[1]:
        raise ValueError(f"measurement {item.measurement!r} needs an in-range matrix column")
    return values[:, item.column]


def _sequence_identity(sequence, rule):
    if rule == "reverse_complement":
        if set(sequence) - set("ACGT"):
            raise ValueError("reverse-complement identity requires uppercase ACGT sequences")
        reverse = sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]
        return min(sequence, reverse)
    return sequence


def _pareto_fronts(utilities):
    # rows dominate columns; equal vectors share a front.
    dominates = (utilities[:, None, :] >= utilities[None, :, :]).all(axis=2)
    dominates &= (utilities[:, None, :] > utilities[None, :, :]).any(axis=2)
    remaining = np.ones(len(utilities), dtype=bool)
    fronts = np.zeros(len(utilities), dtype=int)
    number = 0
    while remaining.any():
        indices = np.flatnonzero(remaining)
        frontier = indices[~dominates[np.ix_(indices, indices)].any(axis=0)]
        fronts[frontier] = number
        remaining[frontier] = False
        number += 1
    return fronts


def _rank_plan_rows(indices, utilities, scalar_scores, plan):
    if plan.aggregation == "lexicographic":
        return sorted(indices, key=lambda i: (*(-utilities[i]).tolist(), int(i)))
    if plan.aggregation == "pareto":
        fronts = _pareto_fronts(utilities[indices])
        def key(position):
            index = indices[position]
            if plan.pareto_tie_break == "lexicographic":
                secondary = tuple(-utilities[index])
            elif plan.pareto_tie_break == "weighted_sum":
                secondary = (-float(utilities[index].sum()),)
            else:
                secondary = ()
            return (int(fronts[position]), *secondary, int(index))
        return [int(indices[position]) for position in sorted(range(len(indices)), key=key)]
    return sorted(indices, key=lambda i: (-float(scalar_scores[i]), int(i)))


def _json_scores(values):
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return [float(v) if np.isfinite(v) else None for v in array]
    return [_json_scores(row) for row in array]


def select_library_plan(sequences, conditions, measurements, plan, *, visible_sources,
                        blind_sources=(), training_sequences=None, row_ids=None,
                        custom_selectors=None):
    """Return generation-order selected indices and a fully replayable report.

    Only plan-referenced measurements are inspected or passed to custom code.
    Caller-supplied source dependencies and custom selector resolvers must come
    from the trusted tool registry. Missing data never become zero scores.
    This operation does not generate, score, or silently replenish sequences.
    """
    plan = plan if isinstance(plan, SelectionPlan) else SelectionPlan.from_dict(plan)
    sequences, groups = list(sequences), [str(c) for c in conditions]
    n = len(sequences)
    if len(groups) != n or any(not isinstance(s, str) for s in sequences):
        raise ValueError("selection requires aligned string sequences and conditions")
    ids = tuple(str(i) for i in range(n)) if row_ids is None else tuple(row_ids)
    if len(ids) != n or len(set(ids)) != n or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("pool row identities must be unique aligned nonempty strings")
    strata = plan.strata or (tuple(plan.count) if isinstance(plan.count, Mapping)
                             else tuple(dict.fromkeys(groups)) or ("all",))
    if any(group not in strata for group in groups):
        raise ValueError("pool contains a condition not covered by the selection plan")
    if any(set(f.strata) - set(strata) for f in plan.filters):
        raise ValueError("filter refers to an unknown stratum")
    counts = dict(plan.count) if isinstance(plan.count, Mapping) else dict.fromkeys(strata, plan.count)
    resolved = _selection_measurements(measurements, plan.measurement_names, n, ids,
                                       visible_sources, blind_sources, plan.missing)
    columns, components = [], {}
    for index, component in enumerate(plan.components):
        raw = _measurement_column(resolved, component)
        with np.errstate(over="ignore", invalid="ignore"):
            value = (raw - component.offset) / component.scale
            value = value * (1 if component.direction == "maximize" else -1) * component.weight
        columns.append(value)
        components[f"{index}:{component.measurement}"] = _json_scores(value)
    utilities = np.column_stack(columns) if columns else np.empty((n, 0))
    with np.errstate(over="ignore", invalid="ignore"):
        scalar_scores = utilities.min(axis=1) if plan.aggregation == "min" else utilities.sum(axis=1)
    # Lexicographic/Pareto plans have no scalar ranking. Do not publish a made-up
    # sum as their scientific objective.
    scalar_ranking = plan.aggregation in {"single", "weighted_sum", "min"}
    reasons = ["eligible"] * n
    rejection_details = [[] for _ in range(n)]
    def reject(index, reason):
        if reasons[index] == "eligible":
            reasons[index] = reason
        rejection_details[index].append(reason)
    alphabet = set(plan.alphabet) if plan.alphabet is not None else None
    if plan.exclude_training_copies and training_sequences is None:
        raise ValueError("training-copy exclusion requires an explicit training reference")
    training_ids = ({_sequence_identity(s, plan.deduplication) for s in training_sequences}
                    if plan.exclude_training_copies else set())
    base_eligible = np.ones(n, dtype=bool)
    for i, sequence in enumerate(sequences):
        legal = (len(sequence) >= plan.min_length
                 and (plan.max_length is None or len(sequence) <= plan.max_length)
                 and (alphabet is None or set(sequence) <= alphabet)
                 and (plan.deduplication != "reverse_complement" or set(sequence) <= set("ACGT")))
        if not legal:
            reject(i, "invalid_sequence")
            base_eligible[i] = False
        elif plan.exclude_training_copies and _sequence_identity(sequence, plan.deduplication) in training_ids:
            reject(i, "training_copy")
            base_eligible[i] = False
        if utilities.shape[1] and not np.isfinite(utilities[i]).all():
            reject(i, "nonfinite_ranking")
        elif scalar_ranking and not np.isfinite(scalar_scores[i]):
            reject(i, "nonfinite_ranking")
    for f in plan.filters:
        values = _measurement_column(resolved, f)
        for i, value in enumerate(values):
            if f.strata and groups[i] not in f.strata:
                continue
            if not np.isfinite(value):
                if plan.missing == "error" and base_eligible[i]:
                    raise ValueError(f"filter measurement {f.measurement!r} contains a nonfinite value")
                reject(i, f"missing_filter:{f.measurement}")
            elif ((f.minimum is not None and value < f.minimum)
                  or (f.maximum is not None and value > f.maximum)):
                reject(i, f"filter:{f.measurement}")
    if plan.aggregation == "custom":
        for name in plan.custom_inputs:
            values = resolved[name][1]
            finite = np.isfinite(values) if values.ndim == 1 else np.isfinite(values).all(axis=1)
            for i in np.flatnonzero(~finite):
                if plan.missing == "error" and base_eligible[i]:
                    raise ValueError(f"custom measurement {name!r} contains a nonfinite value")
                reject(int(i), f"missing_custom:{name}")
    if plan.missing == "error" and any(base_eligible[i] and "nonfinite_ranking" in row
                                       for i, row in enumerate(rejection_details)):
        raise ValueError("derived ranking produced a nonfinite value")
    seen = {}
    if plan.deduplication != "none":
        for i, sequence in enumerate(sequences):
            if reasons[i] != "eligible":
                continue
            key = groups[i] if plan.deduplication_scope == "within_stratum" else "all"
            identity = _sequence_identity(sequence, plan.deduplication)
            if identity in seen.setdefault(key, set()):
                reject(i, "duplicate")
            seen[key].add(identity)
    eligible = np.asarray([reason == "eligible" for reason in reasons], dtype=bool)
    eligible_indices = np.flatnonzero(eligible)
    custom_order = None
    if plan.aggregation == "custom":
        resolver = custom_selectors or {}
        selector = resolver.get(plan.custom_selector)
        if not callable(selector):
            raise ValueError("custom selector is not available through the trusted resolver")
        visible = {name: resolved[name][1].copy() for name in plan.measurement_names}
        for values in visible.values():
            values.setflags(write=False)
        proposed = selector(sequences=tuple(sequences), conditions=tuple(groups), row_ids=ids,
                            measurements=visible, eligible_indices=tuple(eligible_indices.tolist()),
                            measurement_sources={name: resolved[name][0].sources for name in plan.measurement_names},
                            required_counts=dict(counts), parameters=dict(plan.custom_parameters or {}))
        custom_order = list(proposed)
        if (any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in custom_order)
                or len(set(custom_order)) != len(custom_order)
                or any(i < 0 or i >= n or not eligible[i] for i in custom_order)):
            raise ValueError("custom selector returned duplicate, fabricated or ineligible row indices")
        custom_order = [int(i) for i in custom_order]
    if plan.minimum_distance > 0:
        from rapidfuzz.distance import Levenshtein
        def distance(a, b):
            if plan.distance_metric == "normalized_hamming":
                if len(a) != len(b):
                    raise ValueError("normalized Hamming diversity requires equal sequence lengths")
                return sum(x != y for x, y in zip(a, b, strict=True)) / len(a)
            return Levenshtein.normalized_distance(a, b)
    ranked_selected, ranking, summaries, global_chosen = [], {}, {}, []
    skipped_diversity = []
    for group in strata:
        local = np.asarray([i for i in eligible_indices if groups[i] == group], dtype=int)
        order = ([i for i in custom_order if groups[i] == group] if custom_order is not None
                 else _rank_plan_rows(local, utilities, scalar_scores, plan))
        ranking[group] = [int(i) for i in order]
        chosen = []
        for index in order:
            compare = global_chosen if plan.diversity_scope == "global" else chosen
            if plan.minimum_distance > 0 and any(distance(sequences[index], sequences[j]) < plan.minimum_distance
                                                  for j in compare):
                skipped_diversity.append(int(index))
                continue
            chosen.append(int(index))
            global_chosen.append(int(index))
            if len(chosen) == counts[group]:
                break
        ranked_selected.extend(chosen)
        group_indices = [i for i, value in enumerate(groups) if value == group]
        summaries[group] = {
            "pool_count": len(group_indices), "eligible_count": len(local),
            "selected_count": len(chosen), "required_count": counts[group],
            "complete_delivery": len(chosen) == counts[group],
            "rejection_counts": {reason: sum(reasons[i] == reason for i in group_indices)
                                 for reason in sorted(set(reasons) - {"eligible"})},
        }
    selected = sorted(ranked_selected)
    report = {
        "protocol": PLAN_PROTOCOL, "plan": plan.to_dict(), "row_ids": list(ids),
        "pool_binding": hashlib.sha256(json.dumps(
            {"sequences": sequences, "conditions": groups, "row_ids": ids},
            sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode()).hexdigest(),
        "strata": groups, "selected_indices": selected, "selected_row_ids": [ids[i] for i in selected],
        "ranked_selected_indices": ranked_selected, "ranking_indices_by_stratum": ranking,
        "eligibility": eligible.tolist(), "rejection_reasons": reasons,
        "rejection_details": rejection_details, "component_utilities": components,
        "selection_scores": _json_scores(scalar_scores) if scalar_ranking else [None] * n,
        "score_semantics": plan.aggregation if scalar_ranking else "no_scalar_ranking",
        "by_stratum": summaries, "raw_count": n, "selected_count": len(selected),
        "complete_delivery": all(item["complete_delivery"] for item in summaries.values()),
        "missing_measurements": {name: measurement.status for name, (measurement, _) in resolved.items()
                                 if measurement.status != "measured"},
        "measurement_sources": {name: list(measurement.sources) for name, (measurement, _) in resolved.items()},
        "tie_break": plan.tie_break, "duplicate_representative": "first_eligible_pool_position",
        "diversity_skipped_indices": skipped_diversity,
        "diversity_algorithm": "greedy_rank_order_minimum_distance",
        "selection_scope": "within_each_stratum", "output_order": "original_pool_position",
    }
    return np.asarray(selected, dtype=int), report


def isolated_library_selector(constructor):
    """Adapt a catalog's isolated selector constructor to SelectionPlan calls.

    ``constructor`` must be the host-owned ConstructorProxy returned by the
    extension catalog, not an arbitrary generated Python class.
    """
    from recital.synthesis.catalog import ConstructorProxy
    from recital.synthesis.models import ROLE_LIBRARY_SELECTOR
    if not isinstance(constructor, ConstructorProxy) or constructor._primitive_role != ROLE_LIBRARY_SELECTOR:
        raise ValueError("library selector adapter requires an isolated catalog constructor")

    def select(*, sequences, conditions, row_ids, measurements, measurement_sources,
               eligible_indices, required_counts, parameters):
        return constructor(**parameters).select_library(
            sequences, conditioning=conditions, measurements=measurements,
            measurement_sources=measurement_sources, eligible_indices=eligible_indices,
            required_counts=required_counts,
        )
    return select


