"""Compose a typed reward specification into a callable RewardFn.

The spec record is a structured Python dict (or YAML) with eight core fields:

    {
      "enabled_oracles":  [str, ...],            # subset of contract.oracles
      "held_out_oracles": [str, ...],            # scored for validation, never reward
      "transforms":       [primitive_ref, ...],  # per-oracle normalization
      "aggregator":       primitive_ref,         # cross-oracle aggregation
      "shape":            primitive_ref,         # per-sequence shaping
      "penalties":        [primitive_ref, ...],  # per-sequence penalties
      "gates":            [gate_ref, ...],       # hard biological gates
      "hyperparameters":  {"lr": float, "kl": float,
                           "group_size": int, "steps": int},
      "target_objectives": {oracle: {"kind": "interval", "coordinate": "raw",
                           "units": str, "scale": float, "by_condition": {...}}},
      "oracle_utilities": {oracle: {"name": "target_default"}},  # optional overrides
      "reward_input":     {"name": "cell_gap"},          # optional: per-cell output -> one gap
    }

A `primitive_ref` is a dict {"name": str, ...kwargs}. Hard gates are user-supplied
callables looked up by name from a gate registry passed at compose time.
"""
from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Mapping
from copy import copy, deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from recital.targets import OracleTargetObjective
from recital.spec.cell_gap import REWARD_INPUTS
from recital.spec.registry import (
    FRAMEWORK_OWNED_PARAMETERS,
    ROLE_CROSS_ORACLE_AGG,
    ROLE_HARD_GATE,
    ROLE_PER_ORACLE_NORM,
    ROLE_PER_ORACLE_UTILITY,
    ROLE_PER_SEQ_PENALTY,
    ROLE_PER_SEQ_SHAPING,
    get_primitive,
)

# Importing the package registers all built-in primitives.
import recital.spec.primitives  # noqa: F401

LOG = logging.getLogger(__name__)


# Two constants that set the reward's floor and the penalty for a sequence that
# cannot be decoded or fails a gate. They shape the objective as much as any
# declared field, yet they appear in NO recorded spec. One declaration here;
# `effective_reward_constants` in trainer_metrics.json records the values a run
# actually trained against.
DEFAULT_INVALID_PENALTY = -2.0
DEFAULT_REWARD_CLIP_LOW = -1.5


class FrozenRewardUnavailable(ValueError):
    """This recipe/checkpoint cannot supply a calibrated frozen measurement."""


def reward_measurement_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Canonical scoring semantics, excluding training and per-batch bindings."""
    result = RewardSpec.from_dict(spec).to_dict()
    result.pop("hyperparameters", None)
    for field_name in ("transforms", "penalties", "gates"):
        for reference in result[field_name]:
            reference.pop("target_indices", None)
    for field_name in ("aggregator", "shape", "reward_input"):
        if result.get(field_name):
            result[field_name].pop("target_indices", None)
    return result


@dataclass
class Hyperparameters:
    lr: float = 1e-4
    kl: float = 0.02
    group_size: int = 8
    steps: int = 4000


@dataclass(frozen=True)
class SourceRolePolicy:
    """Optional validation context for optimization, held-out, and audit roles.

    ``known_oracles`` is the task's complete declared source pool. Audit sources
    are deliberately absent from ``RewardSpec``: they remain framework-owned and
    cannot be selected by the Strategist. ``require_held_out`` opts a task into
    the protocol, while ``required_held_out_oracles`` and
    ``required_enabled_oracles`` reserve concrete sources that every proposal must
    retain in their declared roles.
    """

    known_oracles: frozenset[str] | None = None
    audit_oracles: frozenset[str] = field(default_factory=frozenset)
    require_held_out: bool = False
    required_held_out_oracles: frozenset[str] = field(default_factory=frozenset)
    required_enabled_oracles: frozenset[str] = field(default_factory=frozenset)
    reward_oracles: frozenset[str] | None = None
    sentinel_oracles: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.known_oracles is not None and not isinstance(self.known_oracles, frozenset):
            object.__setattr__(self, "known_oracles", frozenset(self.known_oracles))
        if not isinstance(self.audit_oracles, frozenset):
            object.__setattr__(self, "audit_oracles", frozenset(self.audit_oracles))
        if not isinstance(self.required_held_out_oracles, frozenset):
            object.__setattr__(
                self,
                "required_held_out_oracles",
                frozenset(self.required_held_out_oracles),
            )
        if not isinstance(self.required_enabled_oracles, frozenset):
            object.__setattr__(
                self,
                "required_enabled_oracles",
                frozenset(self.required_enabled_oracles),
            )
        if self.reward_oracles is not None and not isinstance(self.reward_oracles, frozenset):
            object.__setattr__(self, "reward_oracles", frozenset(self.reward_oracles))
        if self.sentinel_oracles is not None and not isinstance(self.sentinel_oracles, frozenset):
            object.__setattr__(self, "sentinel_oracles", frozenset(self.sentinel_oracles))

        # The audit oracle must not be optimizable. Every consumer that hands the
        # reward pool to something -- the trainer and the proposal screen, which
        # receives the FULL oracle_fns dict -- trusts the caller to keep the pools
        # disjoint, so a context whose pools overlap the audit oracle is refused
        # at construction.
        for role, sources in (
            ("reward_oracles", self.reward_oracles),
            ("sentinel_oracles", self.sentinel_oracles),
        ):
            overlap = sorted((sources or frozenset()) & self.audit_oracles)
            if overlap:
                raise ValueError(
                    f"{role} overlaps audit_oracles on {overlap}: the blind audit "
                    "cannot be optimized against, spent, or consulted by the agent"
                )


def materialize_aggregation_weights(spec: dict[str, Any]) -> dict[str, Any]:
    """Expose weighted_sum defaults before an agent reviews a new proposal.

    Preserve the registered primitive's default of one. Inactive
    sources never enter the reward just because a weight names them.
    """
    copied = deepcopy(spec)
    aggregator = copied.get("aggregator")
    enabled = copied.get("enabled_oracles")
    if not isinstance(aggregator, dict) or aggregator.get("name") != "weighted_sum":
        return copied
    weights = aggregator.get("weights", {})
    if not isinstance(weights, dict) or not isinstance(enabled, list):
        return copied  # The ordinary validator reports malformed fields.
    if not all(isinstance(name, str) for name in enabled):
        return copied
    aggregator["weights"] = {**weights, **{name: weights.get(name, 1.0) for name in enabled}}
    return copied


def materialize_reward_input(spec: dict[str, Any]) -> dict[str, Any]:
    """Expose what a reward input owns before any consumer reads the recipe.

    A recipe that names ``reward_input`` need not write the goal its gaps are
    measured against or a utility for every oracle: an absent ``target_objectives``
    becomes the input's own goal for each enabled oracle, and an enabled oracle
    without an ``oracle_utilities`` entry gets the input's default utility. Written
    values are kept as written, so the ordinary coverage and ownership checks judge
    them. A recipe without a reward input is returned unchanged. Only the fields
    this fills are copied: a recipe in memory can carry runtime objects (the
    trainer's private role map) that cannot be.
    """
    copied = dict(spec)
    reference, enabled = copied.get("reward_input"), copied.get("enabled_oracles")
    if (not isinstance(reference, Mapping) or not isinstance(reference.get("name"), str)
            or reference["name"] not in REWARD_INPUTS or not isinstance(enabled, list)
            or not all(isinstance(name, str) for name in enabled)):
        return copied  # The ordinary validator reports malformed fields.
    owner = REWARD_INPUTS[reference["name"]]
    if not copied.get("target_objectives"):
        copied["target_objectives"] = {name: owner.target().to_dict() for name in enabled}
    utilities = copied.get("oracle_utilities") or {}
    if isinstance(utilities, Mapping):
        copied["oracle_utilities"] = {
            **{name: deepcopy(owner.default_utility) for name in enabled if name not in utilities},
            **deepcopy(dict(utilities)),
        }
    return copied


def derived_default_weight_change(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Only unit defaults for newly enabled sources are part of the role edit.

    Existing coefficients and all other aggregator settings must remain equal.
    This handles authored explicit defaults and runtime materialization equally.
    """
    old = before.get("aggregator")
    new = after.get("aggregator")
    if not isinstance(old, dict) or not isinstance(new, dict):
        return False
    if old.get("name") != "weighted_sum" or new.get("name") != "weighted_sum":
        return False
    added = set(after.get("enabled_oracles") or ()) - set(before.get("enabled_oracles") or ())
    if not added or not isinstance(old.get("weights", {}), dict) or not isinstance(new.get("weights", {}), dict):
        return False
    expected = materialize_aggregation_weights(before)["aggregator"]
    expected["weights"].update({name: expected["weights"].get(name, 1.0) for name in added})
    return materialize_aggregation_weights(after)["aggregator"] == expected


@dataclass
class RewardSpec:
    enabled_oracles: list[str] = field(default_factory=list)
    transforms: list[dict] = field(default_factory=list)
    aggregator: dict | None = None
    shape: dict | None = None
    penalties: list[dict] = field(default_factory=list)
    gates: list[dict] = field(default_factory=list)
    hyperparameters: Hyperparameters = field(default_factory=Hyperparameters)
    invalid_penalty: float = DEFAULT_INVALID_PENALTY
    reward_clip_low: float = DEFAULT_REWARD_CLIP_LOW
    # Prefer ``from_dict`` over positional construction.
    held_out_oracles: list[str] = field(default_factory=list)
    # If present, raw scores are converted to per-oracle target utilities BEFORE
    # aggregation. Transforms/shape must be absent: their coordinates are
    # not implicitly reinterpreted as utility coordinates.
    target_objectives: dict[str, OracleTargetObjective] = field(default_factory=dict)
    # Partial overrides are allowed; omitted enabled oracles use target_default.
    # These references control training utility, never target membership.
    oracle_utilities: dict[str, dict[str, Any]] = field(default_factory=dict)
    # If present, each enabled oracle's (n, n_cells) output is reduced to one value
    # per sequence before any utility or transform runs. None keeps the raw output.
    reward_input: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "RewardSpec":
        d = materialize_reward_input(d)
        hp = d.get("hyperparameters", {}) or {}
        return cls(
            enabled_oracles=list(d.get("enabled_oracles") or []),
            held_out_oracles=list(d.get("held_out_oracles") or []),
            transforms=list(d.get("transforms") or []),
            aggregator=dict(d["aggregator"]) if d.get("aggregator") else None,
            shape=dict(d["shape"]) if d.get("shape") else None,
            penalties=list(d.get("penalties") or []),
            gates=list(d.get("gates") or []),
            hyperparameters=Hyperparameters(
                lr=float(hp.get("lr", 1e-4)),
                kl=float(hp.get("kl", 0.02)),
                group_size=int(hp.get("group_size", 8)),
                steps=int(hp.get("steps", 4000)),
            ),
            invalid_penalty=float(d.get("invalid_penalty", DEFAULT_INVALID_PENALTY)),
            reward_clip_low=float(d.get("reward_clip_low", DEFAULT_REWARD_CLIP_LOW)),
            target_objectives=_parse_target_objectives(d.get("target_objectives")),
            oracle_utilities=_parse_oracle_utilities(d.get("oracle_utilities")),
            reward_input=_parse_reward_input(d.get("reward_input")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the public schema, including executable target declarations."""
        payload = asdict(self)
        payload["target_objectives"] = {
            oracle: target.to_dict()
            for oracle, target in _parse_target_objectives(self.target_objectives).items()
        }
        if payload["reward_input"] is None:
            del payload["reward_input"]  # a recipe without one omits the field
        return payload


def _parse_reward_input(payload: Any) -> dict[str, Any] | None:
    """A validated reward-input reference, or None for the raw-output path.

    The recipe names the input; ``target_indices`` is the per-batch binding the
    trainer adds, so it may appear but is never required.
    """
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("reward_input must be a mapping with a registered name")
    unknown = set(payload) - {"name", "target_indices"}
    if unknown:
        raise ValueError(f"unknown reward_input field(s): {sorted(unknown)}")
    name = payload.get("name")
    if not isinstance(name, str) or name not in REWARD_INPUTS:
        raise ValueError(f"unknown reward_input {name!r}; registered: {sorted(REWARD_INPUTS)}")
    indices = payload.get("target_indices")
    if indices is not None and (
        not isinstance(indices, (list, tuple))
        or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in indices)
    ):
        raise ValueError("reward_input.target_indices must be a list of integers")
    return deepcopy(dict(payload))


def _reward_input_problems(
    reward_input: Any, aggregator: Any, targets: Mapping[str, OracleTargetObjective],
) -> list[str]:
    try:
        reference = _parse_reward_input(reward_input)
    except ValueError as exc:
        return [f"reward_input: {exc}"]
    if reference is None:
        return []
    problems = []
    if isinstance(aggregator, Mapping) and aggregator.get("name") == "contrastive_mingap":
        problems.append(
            "reward_input already reduces each oracle to one gap per sequence, but "
            "contrastive_mingap aggregates per-cell matrices: drop one of them"
        )
    owned = REWARD_INPUTS[reference["name"]].target()
    for oracle, target in targets.items():
        if target != owned:
            problems.append(
                f"target_objectives[{oracle!r}] differs from the goal the reward input "
                f"{reference['name']!r} owns; omit target_objectives to use it"
            )
    return problems


def _parse_oracle_utilities(payload: Any) -> dict[str, dict[str, Any]]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError("oracle_utilities must be a mapping keyed by oracle name")
    result = {}
    for oracle, reference in payload.items():
        if not isinstance(oracle, str) or not oracle:
            raise ValueError("oracle_utilities keys must be nonempty oracle names")
        if (
            not isinstance(reference, Mapping)
            or not isinstance(reference.get("name"), str)
            or not reference["name"]
        ):
            raise ValueError(f"oracle_utilities[{oracle!r}] must be a primitive reference with a name")
        result[oracle] = deepcopy(dict(reference))
    return result


def _utility_scope_problems(
    utilities: Mapping[str, dict[str, Any]],
    enabled_oracles: Iterable[str],
    targets: Mapping[str, OracleTargetObjective],
) -> list[str]:
    problems = []
    if set(utilities) - set(enabled_oracles):
        problems.append("oracle_utilities may reference only enabled_oracles")
    if set(utilities) - set(targets):
        problems.append("oracle_utilities requires a target_objectives declaration for each oracle")
    return problems


def _parse_target_objectives(payload: Any) -> dict[str, OracleTargetObjective]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError("target_objectives must be a mapping keyed by oracle name")
    result = {}
    for oracle, target in payload.items():
        if not isinstance(oracle, str) or not oracle:
            raise ValueError("target_objectives keys must be nonempty oracle names")
        result[oracle] = (
            target if isinstance(target, OracleTargetObjective)
            else OracleTargetObjective.from_dict(target)
        )
    return result


def _target_coordinate_problems(
    *,
    enabled_oracles: list[str],
    target_objectives: dict[str, OracleTargetObjective],
    transforms: list[dict],
    shape: dict | None,
) -> list[str]:
    if target_objectives:
        problems = []
        if set(target_objectives) != set(enabled_oracles):
            problems.append(
                "target_objectives must cover exactly enabled_oracles; "
                "each enabled source needs its own raw-coordinate target"
            )
        if transforms or shape:
            problems.append(
                "target_objectives requires transforms=[] and shape=null: "
                "raw target utility is computed before aggregation; "
                "normalization/shaping cannot silently change its coordinates"
            )
        return problems
    # An interval on transformed or mixed scores needs its own explicit
    # coordinate declaration; raw task boundaries are never reinterpreted.
    if isinstance(shape, Mapping) and shape.get("name") in {"target_interval", "interval_reward"}:
        coordinate = shape.get("coordinate", "raw_oracle")
        if coordinate not in {"raw_oracle", "aggregate"}:
            return ["interval shape coordinate must be raw_oracle or aggregate"]
        if (transforms or len(enabled_oracles) > 1) and coordinate != "aggregate":
            return [
                "interval shape after transforms or multiple raw oracles "
                "has ambiguous target coordinates; declare per-oracle "
                "target_objectives or explicitly declare shape.coordinate='aggregate' "
                "with bounds in that composed coordinate"
            ]
    return []


def _primitive_param_names(cls: Any) -> set[str] | None:
    """The kwargs a primitive class accepts, or None if it cannot be introspected."""
    import inspect

    try:
        parameters = inspect.signature(cls).parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            return None
        return set(parameters)
    except (ValueError, TypeError):
        return None


def _instantiate(
    role: str,
    ref: dict,
    primitive_resolver: Callable[[str, str], Callable] | None = None,
) -> Any:
    if not ref or "name" not in ref:
        raise ValueError(f"primitive reference for role={role} missing 'name': {ref!r}")
    name = ref["name"]
    cls = (primitive_resolver or get_primitive)(role, name)
    kwargs = deepcopy({k: v for k, v in ref.items() if k != "name"})
    # Backstop: drop kwargs the primitive does not accept so a hallucinated param
    # (e.g. {"name": "interval_reward", "delta": 0.8} -- interval_reward takes
    # lo/hi, not delta) cannot crash the whole training run inside cls(**kwargs).
    # validate_spec() catches this earlier and feeds it back to the Strategist;
    # this is the last line of defense if a bad ref reaches instantiation anyway.
    accepted = _primitive_param_names(cls)
    if accepted is not None:
        dropped = [k for k in kwargs if k not in accepted]
        if dropped:
            LOG.warning(
                "primitive %s/%s: ignoring unsupported kwargs %s (accepted: %s)",
                role, name, dropped, sorted(accepted),
            )
            kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(**kwargs)



# Spec field -> primitive role, for static validation.


def validate_source_roles(
    spec_dict: dict,
    *,
    available_oracles: list[str] | tuple[str, ...] | set[str] | frozenset[str] | None = None,
    source_role_policy: SourceRolePolicy | None = None,
) -> list[str]:
    """Validate the three source roles without inspecting reward primitives.

    Enabled sources participate in optimization; held-out sources are visible to
    online diagnostics but must never enter the reward. Audit sources are owned by
    the evaluation harness and must be disjoint from both agent-controlled roles.
    The optional policy turns on task-level pool, audit, and non-empty checks while
    absence of a policy accepts specs that omit held-out sources.
    """

    enabled = [str(o) for o in (spec_dict.get("enabled_oracles") or [])]
    held_out = [str(o) for o in (spec_dict.get("held_out_oracles") or [])]
    enabled_set = set(enabled)
    held_out_set = set(held_out)
    problems: list[str] = []

    if not enabled:
        problems.append("enabled_oracles must be non-empty")
    for role, values in (
        ("enabled_oracles", enabled),
        ("held_out_oracles", held_out),
    ):
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            problems.append(f"{role} contains duplicate source(s) {duplicates}")

    overlap = sorted(enabled_set & held_out_set)
    if overlap:
        problems.append(
            "enabled_oracles and held_out_oracles must be disjoint; "
            f"overlap={overlap}"
        )

    known_sets: list[set[str]] = []
    if available_oracles is not None:
        known_sets.append({str(o) for o in available_oracles})
    if source_role_policy is not None and source_role_policy.known_oracles is not None:
        known_sets.append(set(source_role_policy.known_oracles))
    known = set.intersection(*known_sets) if known_sets else None
    if known is not None:
        for role, values in (
            ("enabled_oracles", enabled),
            ("held_out_oracles", held_out),
        ):
            unknown = sorted(set(values) - known)
            if unknown:
                problems.append(
                    f"{role} contains unknown source(s) {unknown}; "
                    f"known sources: {sorted(known)}"
                )

    if source_role_policy is not None:
        if source_role_policy.require_held_out and not held_out:
            problems.append(
                "held_out_oracles is required by the source-role policy and must be non-empty"
            )
        audit = set(source_role_policy.audit_oracles)
        required = set(source_role_policy.required_held_out_oracles)
        required_enabled = set(source_role_policy.required_enabled_oracles)
        reward_pool = (
            set(source_role_policy.reward_oracles)
            if source_role_policy.reward_oracles is not None
            else None
        )
        sentinel_pool = (
            set(source_role_policy.sentinel_oracles)
            if source_role_policy.sentinel_oracles is not None
            else None
        )
        if reward_pool is not None:
            invalid_enabled = sorted(enabled_set - reward_pool)
            if invalid_enabled:
                problems.append(
                    "enabled_oracles contains source(s) outside the reward pool "
                    f"{invalid_enabled}; reward pool: {sorted(reward_pool)}"
                )
            invalid_required_enabled = sorted(required_enabled - reward_pool)
            if invalid_required_enabled:
                problems.append(
                    "required_enabled_oracles contains source(s) outside the "
                    f"reward pool {invalid_required_enabled}"
                )
        if sentinel_pool is not None:
            invalid_held_out = sorted(held_out_set - sentinel_pool)
            if invalid_held_out:
                problems.append(
                    "held_out_oracles contains source(s) outside the sentinel pool "
                    f"{invalid_held_out}; sentinel pool: {sorted(sentinel_pool)}"
                )
            invalid_required = sorted(required - sentinel_pool)
            if invalid_required:
                problems.append(
                    "required_held_out_oracles contains source(s) outside the "
                    f"sentinel pool {invalid_required}"
                )
        declared_agent_pool = (reward_pool or set()) | (sentinel_pool or set())
        role_audit_overlap = sorted(declared_agent_pool & audit)
        if role_audit_overlap:
            problems.append(
                "reward/sentinel role pools must be disjoint from framework-owned "
                f"audit sources; overlap={role_audit_overlap}"
            )
        if known is not None:
            unknown_required = sorted(required - known)
            if unknown_required:
                problems.append(
                    "required_held_out_oracles contains unknown source(s) "
                    f"{unknown_required}; known sources: {sorted(known)}"
                )
            unknown_required_enabled = sorted(required_enabled - known)
            if unknown_required_enabled:
                problems.append(
                    "required_enabled_oracles contains unknown source(s) "
                    f"{unknown_required_enabled}; known sources: {sorted(known)}"
                )
        required_audit_overlap = sorted(required & audit)
        if required_audit_overlap:
            problems.append(
                "required_held_out_oracles must be disjoint from framework-owned "
                f"audit sources; overlap={required_audit_overlap}"
            )
        required_enabled_overlap = sorted(required & enabled_set)
        if required_enabled_overlap:
            problems.append(
                "required_held_out_oracles must be disjoint from enabled_oracles; "
                f"overlap={required_enabled_overlap}"
            )
        missing_required = sorted(required - held_out_set)
        if missing_required:
            problems.append(
                "held_out_oracles is missing required sentinel source(s) "
                f"{missing_required}"
            )
        missing_required_enabled = sorted(required_enabled - enabled_set)
        if missing_required_enabled:
            problems.append(
                "enabled_oracles is missing required reward source(s) "
                f"{missing_required_enabled}"
            )
        audit_overlap = sorted((enabled_set | held_out_set) & audit)
        if audit_overlap:
            problems.append(
                "enabled_oracles and held_out_oracles must be disjoint from "
                f"framework-owned audit sources; overlap={audit_overlap}"
            )
    return problems


def validate_spec(
    spec_dict: dict,
    pending_primitive: dict | None = None,
    *,
    available_oracles: list[str] | tuple[str, ...] | set[str] | frozenset[str] | None = None,
    source_role_policy: SourceRolePolicy | None = None,
    known_gate_names: set[str] | frozenset[str] | None = None,
    primitive_resolver: Callable[[str, str], Callable] | None = None,
) -> list[str]:
    """Return human-readable problems with a spec dict, without raising.

    Checks source-role invariants and that every registry-backed primitive the
    spec references (oracle_utilities, transforms,
    aggregator, shape, penalties, gates) names a primitive actually registered for
    its role, and that each reference carries a ``name``. An empty list means the
    spec is structurally instantiable from the registry. The Strategist
    occasionally hallucinates a primitive name (e.g. ``at_homopolymer`` for the
    registered ``at_homopoly``); validating here lets the closed loop
    reject-and-re-propose deterministically instead of crashing inside the
    trainer when ``ComposedRewardFn`` calls ``get_primitive``.

    Gates are checked against the ``hard_biological_gate`` registry, plus any
    ``known_gate_names`` a task injects through the composer's ``gate_registry``
    (whose kwargs this function cannot introspect, so only the name is checked).
    A misnamed gate is reported here as re-proposable Critic feedback rather than
    raising a KeyError inside ``ComposedRewardFn.__init__``.

    ``pending_primitive`` is the Strategist's primitive_request for THIS proposal
    (a dict with ``name``/``role``): the Coder synthesizes it only AFTER the spec is
    accepted, so a spec field naming it is legitimate -- it would otherwise be
    rejected as "unknown primitive" and the agent could never introduce a new
    primitive (e.g. a synthesized cross-oracle selectivity aggregator). Such a
    reference is allowed through here (its kwargs are inserted as just ``{"name": X}``
    on admission, so there are no kwargs to validate yet).

    ``primitive_resolver(role, name)`` optionally replaces global lookup, allowing
    run-scoped trusted constructors to validate without global registration. It
    must raise KeyError for unavailable names and delegate built-ins as needed.
    """
    from recital.spec.registry import list_primitives  # avoid import cycle

    spec_dict = materialize_reward_input(spec_dict)
    registered = set(list_primitives())
    resolve = primitive_resolver or get_primitive
    pending_name = (pending_primitive or {}).get("name")
    pending_role = (pending_primitive or {}).get("role")
    problems = validate_source_roles(
        spec_dict,
        available_oracles=available_oracles,
        source_role_policy=source_role_policy,
    )
    try:
        targets = _parse_target_objectives(spec_dict.get("target_objectives"))
        problems.extend(_target_coordinate_problems(
            enabled_oracles=list(spec_dict.get("enabled_oracles") or []),
            target_objectives=targets,
            transforms=list(spec_dict.get("transforms") or []),
            shape=spec_dict.get("shape"),
        ))
    except (TypeError, ValueError) as exc:
        problems.append(f"target_objectives: {exc}")
        targets = {}
    try:
        utilities = _parse_oracle_utilities(spec_dict.get("oracle_utilities"))
        problems.extend(_utility_scope_problems(
            utilities, spec_dict.get("enabled_oracles") or [], targets,
        ))
    except (TypeError, ValueError) as exc:
        problems.append(f"oracle_utilities: {exc}")
        utilities = {}
    problems.extend(_reward_input_problems(spec_dict.get("reward_input"), spec_dict.get("aggregator"), targets))
    try:
        invalid_penalty = float(spec_dict.get("invalid_penalty", DEFAULT_INVALID_PENALTY))
        reward_clip_low = float(spec_dict.get("reward_clip_low", DEFAULT_REWARD_CLIP_LOW))
        if not math.isfinite(invalid_penalty) or not math.isfinite(reward_clip_low):
            problems.append("invalid_penalty and reward_clip_low must be finite")
        elif invalid_penalty > reward_clip_low:
            problems.append(
                "invalid_penalty must be <= reward_clip_low so a gate violation "
                "cannot receive a better reward than a valid clipped sequence"
            )
    except (TypeError, ValueError):
        problems.append("invalid_penalty and reward_clip_low must be numeric")

    def _check(role: str, ref: Any, where: str) -> None:
        if (
            not isinstance(ref, dict)
            or not isinstance(ref.get("name"), str)
            or not ref["name"]
        ):
            problems.append(f"{where}: primitive reference missing 'name': {ref!r}")
            return
        if (
            pending_name is not None and ref["name"] == pending_name
            and (pending_role is None or pending_role == role)
        ):
            return  # about to be synthesized by the Coder this iteration
        try:
            cls = resolve(role, ref["name"])
        except KeyError:
            valid = sorted(n for (r, n) in registered if r == role)
            problems.append(
                f"{where}: unknown {role} primitive {ref['name']!r}; "
                f"valid names for this role: {valid}"
            )
            return
        # Also validate the kwargs: a kwarg the primitive does not accept (e.g.
        # 'delta' on interval_reward, which takes lo/hi) would crash the trainer
        # inside cls(**kwargs). Catch it here so the Strategist re-proposes with a
        # valid param set rather than killing the run. Skip injected runtime keys
        # the trainer patches in per batch (target_indices, intervals).
        accepted = _primitive_param_names(cls)
        if accepted is not None:
            injected = {"target_indices", "intervals"}
            owned = (
                FRAMEWORK_OWNED_PARAMETERS.get(ref["name"])
                if role == ROLE_PER_SEQ_PENALTY else None
            )
            if owned is not None:
                # The reference library is framework-owned run context. Letting
                # an LLM supply it would make the novelty penalty arbitrary and
                # serialize a potentially large/private training sample.
                injected.add(owned)
                if owned in ref:
                    problems.append(
                        f"{where}: {ref['name']}.{owned} is framework-owned and "
                        "must not be supplied by the reward specification"
                    )
            bad = [k for k in ref if k != "name" and k not in accepted and k not in injected]
            if bad:
                problems.append(
                    f"{where}: {role} primitive {ref['name']!r} does not accept "
                    f"param(s) {bad}; accepted params: {sorted(accepted)}"
                )

    for oracle, ref in utilities.items():
        _check(ROLE_PER_ORACLE_UTILITY, ref, f"oracle_utilities[{oracle!r}]")
    for ref in (spec_dict.get("transforms") or []):
        _check(ROLE_PER_ORACLE_NORM, ref, "transforms[]")
    if spec_dict.get("aggregator"):
        _check(ROLE_CROSS_ORACLE_AGG, spec_dict["aggregator"], "aggregator")
    if spec_dict.get("shape"):
        _check(ROLE_PER_SEQ_SHAPING, spec_dict["shape"], "shape")
    for ref in (spec_dict.get("penalties") or []):
        _check(ROLE_PER_SEQ_PENALTY, ref, "penalties[]")
    for ref in (spec_dict.get("gates") or []):
        if isinstance(ref, dict) and ref.get("name") in (known_gate_names or set()):
            continue  # task-injected gate; the composer resolves it, kwargs opaque here
        _check(ROLE_HARD_GATE, ref, "gates[]")
    return problems


class ComposedRewardFn:
    """Concrete reward callable produced from a RewardSpec.

    Usage:
        reward = ComposedRewardFn(spec, gate_registry={"forbidden_motif": _check_motif})
        rewards = reward(sequences=[...], oracle_outputs={"cnn": np.ndarray, ...})

    ``primitive_resolver(role, name)`` returns a constructor and replaces global
    lookup for every role, including implicit target_default utilities. A scoped
    resolver should delegate built-ins to get_primitive itself. Utility instances
    receive ``(raw, target=OracleTargetObjective, conditioning=...)``; aggregators
    receive the resulting enabled-oracle utility map. Constructor kwargs and
    call inputs are copied to protect caller-owned state.

    ``reference_libraries`` maps a binding id to the rows a ``reference_distance``
    penalty measures against. The training set is always bound under the id
    ``training_set``; a task adds its own (CM's natural family) here rather than
    letting a specification carry sequences.

    A spec's ``reward_input`` becomes ``_reward_input``: every enabled oracle's
    ``(n, n_cells)`` output is reduced to one value per sequence before the
    utility, transform and aggregation roles run. Its ``target_indices`` are bound
    for each batch, like those of ``contrastive_mingap``.
    """

    def __init__(
        self,
        spec: RewardSpec,
        gate_registry: dict[str, Callable[[list[str]], np.ndarray]] | None = None,
        training_set: list[str] | tuple[str, ...] | frozenset[str] | None = None,
        *,
        primitive_resolver: Callable[[str, str], Callable] | None = None,
        reference_libraries: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        self.spec = spec = deepcopy(spec)
        self._primitive_resolver = primitive_resolver or get_primitive
        self._gate_registry = gate_registry or {}
        self._training_set = frozenset(
            str(sequence) for sequence in (training_set or ())
        )
        self._reference_libraries = {
            name: tuple(str(sequence) for sequence in rows)
            for name, rows in (reference_libraries or {}).items()
        }
        self._target_objectives = _parse_target_objectives(spec.target_objectives)
        utility_refs = _parse_oracle_utilities(spec.oracle_utilities)
        coordinate_problems = _target_coordinate_problems(
            enabled_oracles=spec.enabled_oracles,
            target_objectives=self._target_objectives,
            transforms=spec.transforms,
            shape=spec.shape,
        )
        coordinate_problems.extend(_utility_scope_problems(
            utility_refs, spec.enabled_oracles, self._target_objectives,
        ))
        coordinate_problems.extend(_reward_input_problems(
            spec.reward_input, spec.aggregator, self._target_objectives))
        if coordinate_problems:
            raise ValueError("; ".join(coordinate_problems))
        reference = _parse_reward_input(spec.reward_input)
        self._reward_input = None if reference is None else REWARD_INPUTS[reference["name"]](
            target_indices=list(reference.get("target_indices") or ()),
        )

        if not math.isfinite(spec.invalid_penalty) or not math.isfinite(
            spec.reward_clip_low
        ):
            raise ValueError("invalid_penalty and reward_clip_low must be finite")
        if spec.invalid_penalty > spec.reward_clip_low:
            raise ValueError(
                "invalid_penalty must be <= reward_clip_low so gate violations "
                "cannot be rewarded"
            )

        def instantiate(role: str, reference: dict) -> Any:
            return _instantiate(role, reference, self._primitive_resolver)

        self._oracle_utilities = {
            oracle: instantiate(
                ROLE_PER_ORACLE_UTILITY,
                utility_refs.get(oracle, {"name": "target_default"}),
            )
            for oracle in self._target_objectives
        }
        self._transforms = [instantiate(ROLE_PER_ORACLE_NORM, t) for t in spec.transforms]
        self._aggregator = (
            instantiate(ROLE_CROSS_ORACLE_AGG, spec.aggregator) if spec.aggregator else None
        )
        self._shape = (
            instantiate(ROLE_PER_SEQ_SHAPING, spec.shape) if spec.shape else None
        )
        self._penalties = []
        for reference in spec.penalties:
            runtime_reference = dict(reference)
            if runtime_reference.get("name") == "seen_penalty":
                # Always override this private constructor argument; it is never
                # agent-controlled. An empty set remains an honest no-op when a
                # task has no frozen reference library.
                runtime_reference["train_set"] = self._training_set
            elif runtime_reference.get("name") == "reference_distance":
                # Unlike seen_penalty, an unbound library here is not a no-op: a
                # distance floor with nothing to measure against would silently
                # score every sequence as maximally novel, so the binding must
                # exist before the recipe can run at all.
                binding = str(runtime_reference.get("reference", "training_set"))
                library = (
                    self._training_set if binding == "training_set"
                    else self._reference_libraries.get(binding)
                )
                if library is None:
                    raise KeyError(
                        f"reference_distance names unbound reference library {binding!r}; "
                        f"bound libraries: {sorted({'training_set', *self._reference_libraries})}"
                    )
                runtime_reference["reference_sequences"] = tuple(sorted(library))
            self._penalties.append(
                instantiate(ROLE_PER_SEQ_PENALTY, runtime_reference)
            )

        self._gates: list[Callable[[list[str]], np.ndarray]] = []
        for g in spec.gates:
            if "name" not in g:
                raise ValueError(f"gate reference missing 'name': {g!r}")
            name = g["name"]
            if name in self._gate_registry:
                self._gates.append(self._gate_registry[name])
                continue
            # Fall back to the primitive registry: a Coder-synthesized
            # hard_biological_gate (e.g. motif_gini_diversity_gate) lives there,
            # not in the user-supplied gate registry. Instantiate it with its spec
            # kwargs; it has the same (seqs) -> bool[] signature as a gate.
            try:
                cls = self._primitive_resolver(ROLE_HARD_GATE, name)
            except KeyError:
                raise KeyError(
                    f"gate {name!r} not in user gate registry nor primitive registry"
                )
            self._gates.append(cls(**deepcopy({k: v for k, v in g.items() if k != "name"})))

    def _components(self) -> dict[str, Any]:
        return {
            **{f"utility/{name}": value for name, value in self._oracle_utilities.items()},
            **{f"transform/{i}": value for i, value in enumerate(self._transforms)},
            "aggregator": self._aggregator,
            "shape": self._shape,
            **{f"penalty/{i}": value for i, value in enumerate(self._penalties)},
            **{f"gate/{i}": value for i, value in enumerate(self._gates)},
        }

    def _checkpoint_components(self) -> dict[str, Any]:
        """Stateful primitives opt into the usual state_dict/load_state_dict pair.

        State must be JSON-serializable; constructor settings come from the
        recipe. Traversing every pipeline role also supports composite
        primitives that implement this pair for their own children.
        """
        stateful = {}
        for name, component in self._components().items():
            save = callable(getattr(component, "state_dict", None))
            restore = callable(getattr(component, "load_state_dict", None))
            if save != restore:
                raise TypeError(f"reward component {name} must implement both state methods")
            if save:
                stateful[name] = component
        return stateful

    def state_dict(self) -> dict[str, Any]:
        """Mutable reward state, separate from the recipe and reference data."""
        return {name: deepcopy(component.state_dict())
                for name, component in self._checkpoint_components().items()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        components = self._checkpoint_components()
        if not isinstance(state, dict) or set(state) != set(components):
            raise ValueError("checkpoint reward state does not match the recipe's stateful components")
        for name, component in components.items():
            component.load_state_dict(deepcopy(state[name]))

    def load_checkpoint_state(self, checkpoint: str | Path) -> dict[str, Any]:
        """Restore scoring state from this exact node, never fresh calibration.

        A checkpoint records the executed reward recipe next to its state, or the
        action manifest supplies the recipe. A stateful checkpoint with neither is
        not sufficient evidence for an exact recipe-reward measurement.
        """
        checkpoint = Path(checkpoint)
        saved = json.loads((checkpoint / "trainer_state.json").read_text())
        record = saved.get("stateful_callbacks", {}).get("RewardStateCallback")
        attributes = record.get("attributes", {}) if isinstance(record, dict) else {}
        saved_spec = attributes.get("reward_spec")
        manifest_path = checkpoint / "training_action_state.json"
        manifest = None
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("hf_step") != saved.get("global_step"):
                raise FrozenRewardUnavailable("checkpoint reward and training-action steps disagree")
        if saved_spec is None and manifest is not None:
            saved_spec = deepcopy(manifest["spec"])
            # The trainer binds these fixed intervals from task context, while
            # row target_indices are rebound for every batch.
            if (saved_spec.get("shape") or {}).get("name") == "target_interval":
                intervals = manifest.get("data_signature", {}).get("group_intervals")
                if intervals is not None:
                    saved_spec["shape"]["intervals"] = intervals
        if saved_spec is not None:
            if reward_measurement_spec(saved_spec) != reward_measurement_spec(self.spec.to_dict()):
                raise FrozenRewardUnavailable("checkpoint recipe does not match this reward measurement")
        elif self._checkpoint_components():
            raise FrozenRewardUnavailable("checkpoint has no recorded recipe for its reward state")
        state = attributes.get("reward_state")
        if state is None:
            if self._checkpoint_components():
                raise FrozenRewardUnavailable("checkpoint has no trained reward state")
            state = {}
        self.load_state_dict(state)
        self._frozen_checkpoint = {"path": str(checkpoint.resolve()), "step": saved.get("global_step")}
        return deepcopy(self._frozen_checkpoint)

    def frozen(self, checkpoint: str | Path | None = None) -> "ComposedRewardFn":
        """Return an independently bound callable with immutable scoring state.

        When checkpoint is omitted, use this object's current trained state.
        Stateless proxy capabilities are shallow-copied, without cloning worker
        processes; stateful components own independent copies of their state.
        Runtime condition indices must still be bound for each scored batch.
        """
        result = copy(self)
        result.spec = deepcopy(self.spec)
        stateful = self._checkpoint_components()

        def clone(name, component):
            return deepcopy(component) if name in stateful else copy(component)

        # Each frozen scorer binds its own batch's target columns.
        result._reward_input = deepcopy(self._reward_input)
        result._oracle_utilities = {name: clone(f"utility/{name}", value)
                                    for name, value in self._oracle_utilities.items()}
        result._transforms = [clone(f"transform/{i}", value) for i, value in enumerate(self._transforms)]
        result._aggregator = clone("aggregator", self._aggregator)
        result._shape = clone("shape", self._shape)
        result._penalties = [clone(f"penalty/{i}", value) for i, value in enumerate(self._penalties)]
        result._gates = [clone(f"gate/{i}", value) for i, value in enumerate(self._gates)]
        result.last_diagnostics = {}
        result._frozen = True
        if checkpoint is not None:
            result.load_checkpoint_state(checkpoint)
        result._validate_frozen_components()
        return result

    def _validate_frozen_components(self) -> None:
        for name, component in self._checkpoint_components().items():
            if not callable(getattr(component, "frozen_inference", None)):
                raise FrozenRewardUnavailable(
                    f"stateful reward component {name} has no frozen_inference implementation")

    def frozen_metadata(self) -> dict[str, Any]:
        """Describe the measurement; its scale is specific to this recipe/node."""
        # Built-ins are pointwise after their learned moments are frozen.
        # Other primitives may rank/normalize their input pool without learned
        # state: score that whole pool once, never silently chunk that operation.
        pointwise = all(component is None or
                        type(component).__module__.startswith("recital.spec.primitives.") or
                        getattr(component, "frozen_batch_independent", False) is True
                        for component in self._components().values())
        return {
            "state": "frozen", "checkpoint": deepcopy(getattr(self, "_frozen_checkpoint", None)),
            "stateful_components": sorted(self._checkpoint_components()),
            "batch_semantics": "pointwise" if pointwise else "whole_input_pool",
            "scale_scope": "recipe_and_checkpoint", "cross_recipe_comparable": False,
        }

    def frozen_inference(
        self, sequences: list[str], oracle_outputs: dict[str, np.ndarray], *,
        conditioning: list[str] | tuple[str, ...] | None = None,
    ) -> np.ndarray:
        """Score against trained statistics without updating them.

        A stateful extension must implement a pure frozen_inference method.
        Pool-dependent stateless extensions retain whole-pool semantics (see
        frozen_metadata); this interface does not invent an out-of-sample scale.
        """
        self._validate_frozen_components()
        before = self.state_dict()
        try:
            result = self._evaluate(sequences, oracle_outputs, conditioning=conditioning, frozen=True)
        except Exception:
            self.load_state_dict(before)
            raise
        if json.dumps(before, sort_keys=True, allow_nan=False) != json.dumps(
                self.state_dict(), sort_keys=True, allow_nan=False):
            self.load_state_dict(before)
            raise FrozenRewardUnavailable("frozen_inference mutated reward component state")
        return result

    def __call__(
        self,
        sequences: list[str],
        oracle_outputs: dict[str, np.ndarray],
        *,
        conditioning: list[str] | tuple[str, ...] | None = None,
    ) -> np.ndarray:
        if getattr(self, "_frozen", False):
            return self.frozen_inference(sequences, oracle_outputs, conditioning=conditioning)
        return self._evaluate(sequences, oracle_outputs, conditioning=conditioning, frozen=False)

    def _evaluate(self, sequences, oracle_outputs, *, conditioning, frozen):
        def invoke(component, *args, context=False, **kwargs):
            if frozen and callable(getattr(component, "state_dict", None)):
                return component.frozen_inference(*args, **kwargs)
            method = component.evaluate_context if context else component
            return method(*args, **kwargs)

        if not self.spec.enabled_oracles:
            raise ValueError("RewardSpec has empty enabled_oracles")
        raw = {o: np.array(oracle_outputs[o], copy=True) for o in self.spec.enabled_oracles}

        # 0. Reward input: one value per sequence from each oracle's per-cell output,
        # for every enabled oracle before any state changes, so everything below
        # sees the (n,) vectors the utility and aggregation roles are defined on.
        if self._reward_input is not None:
            raw = self._reward_input(raw)

        # 1. Raw-coordinate target utility, separately for EVERY enabled oracle when
        # target_objectives is declared; otherwise the configured transforms apply.
        normed: dict[str, np.ndarray] = raw
        # Per row, the share of the enabled oracles whose utility sits on the ceiling its primitive clips to (the +3
        # of gap_ema_zscore). None when no enabled oracle's utility declares a ceiling.
        at_ceiling: list[np.ndarray] = []
        if self._target_objectives:
            normed = {}
            for oracle, objective in self._target_objectives.items():
                values = np.asarray(raw[oracle], dtype=float)
                if values.shape != (len(sequences),):
                    raise ValueError(f"target oracle {oracle!r} must return one scalar per sequence")
                objective.validate_conditioning(len(values), conditioning)
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"target oracle {oracle!r} produced non-finite scores")
                utility = np.asarray(invoke(self._oracle_utilities[oracle],
                    values.copy(),
                    target=deepcopy(objective),
                    conditioning=deepcopy(conditioning),
                ), dtype=float)
                if utility.shape != (len(sequences),):
                    raise ValueError(
                        f"utility for oracle {oracle!r} must produce one scalar per sequence; "
                        f"got {utility.shape}"
                    )
                if not np.all(np.isfinite(utility)):
                    raise ValueError(f"utility for oracle {oracle!r} produced non-finite values")
                normed[oracle] = utility.copy()
                ceiling = getattr(self._oracle_utilities[oracle], "clip", None)
                if ceiling is not None:
                    at_ceiling.append(utility >= ceiling)
        else:
            for tfm in self._transforms:
                normed = invoke(tfm, normed)

        # 2. cross-oracle aggregation
        if self._aggregator is None:
            arrays = np.stack(
                [np.asarray(v, dtype=float).reshape(len(sequences)) for v in normed.values()],
                axis=0,
            )
            aggregated = arrays.mean(axis=0)
        else:
            aggregated = invoke(self._aggregator, normed)

        # 3. per-sequence shaping
        shaped = invoke(self._shape, aggregated) if self._shape is not None else aggregated
        shaped = np.asarray(shaped, dtype=float)
        expected_shape = (len(sequences),)
        if shaped.shape != expected_shape:
            raise ValueError(
                "reward pipeline must produce exactly one scalar per sequence; "
                f"expected shape {expected_shape}, got {shaped.shape}"
            )
        if not np.all(np.isfinite(shaped)):
            raise ValueError("reward pipeline produced non-finite values")

        # 4. subtract per-sequence penalties. Conditional primitives explicitly
        # declare their protocol; the core never infers conditions from a task or
        # primitive name. Unconditional penalties keep their signature.
        penalty_diagnostics = {}
        for index, pen in enumerate(self._penalties):
            if hasattr(pen, "evaluate_context"):
                penalty_values = invoke(pen, list(sequences), context=True, conditioning=deepcopy(conditioning))
            elif getattr(pen, "requires_conditioning", False):
                if conditioning is None or len(conditioning) != len(sequences):
                    raise ValueError(
                        "conditioned penalty requires one explicit condition per sequence"
                    )
                penalty_values = invoke(pen, list(sequences), conditioning=deepcopy(conditioning))
            else:
                penalty_values = invoke(pen, list(sequences))
            penalty = np.asarray(penalty_values, dtype=float)
            if penalty.shape != expected_shape:
                raise ValueError(
                    "penalty must produce exactly one scalar per sequence; "
                    f"expected shape {expected_shape}, got {penalty.shape}"
                )
            if not np.all(np.isfinite(penalty)) or np.any(penalty < 0):
                raise ValueError("penalty values must be finite and nonnegative")
            penalty_diagnostics[str(index)] = {
                "name": self.spec.penalties[index]["name"],
                "mean": float(np.mean(penalty)) if len(penalty) else 0.0,
                "maximum": float(np.max(penalty)) if len(penalty) else 0.0,
                "nonzero_count": int(np.count_nonzero(penalty)),
            }
            shaped = shaped - penalty

        upper_clip = np.mean(at_ceiling, axis=0) if at_ceiling else None
        # Diagnostics describe training utility; they never determine qualification.
        self.last_diagnostics = {
            "count": len(sequences),
            "aggregation": {
                "name": (self.spec.aggregator or {}).get("name", "mean"),
                "mean": float(np.mean(aggregated)) if len(sequences) else 0.0,
                "std": float(np.std(aggregated)) if len(sequences) else 0.0,
            },
            "lower_clip_fraction": float(np.mean(shaped < self.spec.reward_clip_low)) if len(sequences) else 0.0,
            "upper_clip_fraction": (float(np.mean(upper_clip)) if len(sequences) else 0.0)
            if upper_clip is not None else None,
            "pre_clip_reward_std": float(np.std(shaped)) if len(sequences) else 0.0,
            "gate_violation_fraction": 0.0,
            "gates": {},
            "penalties": penalty_diagnostics,
        }
        if (self.spec.aggregator or {}).get("name") == "weighted_sum":
            weights = (self.spec.aggregator or {}).get("weights", {})
            components = {}
            for name, values in normed.items():
                weight = float(weights.get(name, 1.0))
                contribution = weight * np.asarray(values, dtype=float)
                components[name] = {
                    "weight": weight,
                    "mean": float(np.mean(contribution)) if len(sequences) else 0.0,
                    "mean_absolute": float(np.mean(np.abs(contribution))) if len(sequences) else 0.0,
                    "nonzero_count": int(np.count_nonzero(contribution)),
                }
            self.last_diagnostics["aggregation"]["components"] = components
        clip_mask = shaped < self.spec.reward_clip_low
        # 5. lower clip
        shaped = np.maximum(shaped, self.spec.reward_clip_low)

        # 6. apply hard gates LAST (after the clip) so the invalid_penalty is the
        # actual reward for a violating sequence. If gates ran before the clip, an
        # invalid_penalty below reward_clip_low (the default -2.0 < -1.5) would be
        # clipped back up to reward_clip_low, making a gate violation
        # indistinguishable from a merely-bad sequence.
        if self._gates:
            violated = np.zeros(len(sequences), dtype=bool)
            for index, gate in enumerate(self._gates):
                gate_values = np.asarray(
                    invoke(gate, list(sequences), context=True, conditioning=deepcopy(conditioning))
                    if hasattr(gate, "evaluate_context") else invoke(gate, list(sequences))
                )
                if gate_values.shape != expected_shape:
                    raise ValueError(
                        "hard gate must return one violation boolean per sequence; "
                        f"expected shape {expected_shape}, got {gate_values.shape}"
                    )
                if gate_values.dtype.kind != "b":
                    raise ValueError(
                        "hard gate must return boolean violation values"
                    )
                self.last_diagnostics["gates"][str(index)] = {
                    "violation_fraction": float(np.mean(gate_values)) if len(sequences) else 0.0,
                }
                violated |= gate_values
            self.last_diagnostics["gate_violation_fraction"] = float(np.mean(violated)) if len(sequences) else 0.0
            shaped = np.where(violated, self.spec.invalid_penalty, shaped)
        self.last_diagnostics["reward_std"] = float(np.std(shaped)) if len(sequences) else 0.0
        if conditioning is not None:
            labels = np.asarray(conditioning, dtype=str)
            self.last_diagnostics["by_condition"] = {
                label: {"count": int(np.sum(labels == label)),
                        "reward_std": float(np.std(shaped[labels == label])),
                        "lower_clip_fraction": float(np.mean(clip_mask[labels == label])),
                        "upper_clip_fraction": (None if upper_clip is None
                                                else float(np.mean(upper_clip[labels == label])))}
                for label in sorted(set(labels))
            }
        return shaped
