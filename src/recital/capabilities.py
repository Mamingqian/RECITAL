"""Resolved recipe permissions shared by agents and execution validation.

This module describes permitted edits, not task success or source-role policy.
The caller supplies agent-visible frozen requirements; blind measurements stay
in the core success contract. Registry admission and model support remain real
runtime capabilities, not promises made by an agent prompt.
"""
from __future__ import annotations

import copy
import inspect
import math
import re
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from recital.spec.registry import FRAMEWORK_OWNED_PARAMETERS, REGISTRY, VALID_ROLES
from recital.targets import OracleTargetObjective
from recital.search_semantics import LEGACY, VERSIONS


RECIPE_PRIMITIVE_ROLES = {
    "oracle_utilities": "per_oracle_utility",
    "transforms": "per_oracle_normalization",
    "aggregator": "cross_oracle_aggregation",
    "shape": "per_sequence_shaping",
    "penalties": "per_sequence_penalty",
    "gates": "hard_biological_gate",
}


def _canonical_target_map(payload: Any, field_name: str) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must be a mapping keyed by oracle name")
    canonical = {}
    for name, declaration in payload.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{field_name} keys must be nonempty oracle names")
        try:
            canonical[name] = OracleTargetObjective.from_dict(declaration).to_dict()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name}[{name!r}]: {exc}") from exc
    return canonical


@dataclass(frozen=True)
class RecipeCapabilities:
    """One resolved permission declaration; no task-name or DNA defaults.

    ``editable_hyperparameters`` maps names to JSON Schemas. Unlisted fields
    must equal the supplied parent specification. ``runtime_parameters`` maps
    ``role/name`` to adapter-owned parameter bindings/descriptions; agents must
    omit those parameters. It declares ownership, not an injection mechanism.
    ``frozen_requirements`` is read-only prompt context, never a mutable recipe.
    ``frozen_target_objectives`` declares the full immutable target pool. When
    nonempty, each recipe must contain exactly the enabled subset with those
    canonical values. Allow editing ``target_objectives`` to change membership
    alongside an allowed oracle transition; target values remain fixed.
    """

    allowed_reward_fields: tuple[str, ...]
    editable_hyperparameters: dict[str, dict[str, Any]]
    allowed_primitives: tuple[tuple[str, str], ...]
    frozen_requirements: dict[str, Any]
    frozen_hyperparameters: dict[str, Any] = field(default_factory=dict)
    runtime_parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    allow_primitive_requests: bool = False
    frozen_target_objectives: dict[str, dict[str, Any]] = field(default_factory=dict)
    synthesizable_roles: tuple[str, ...] = ()
    # Published extension metadata, never executable classes or agent-authored grants.
    admitted_primitives: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Public immutable grant records; raw reference data never enters the prompt.
    approved_references: dict[str, dict[str, Any]] = field(default_factory=dict)
    search_semantics: str = LEGACY

    def __post_init__(self) -> None:
        if self.search_semantics not in VERSIONS:
            raise ValueError("unknown search_semantics")
        if not isinstance(self.allowed_reward_fields, (list, tuple)) or any(
            not isinstance(name, str) or not name for name in self.allowed_reward_fields
        ):
            raise ValueError("allowed_reward_fields must be a sequence of field names")
        if not isinstance(self.allowed_primitives, (list, tuple)) or any(
            not isinstance(pair, (list, tuple)) or len(pair) != 2
            or not all(isinstance(value, str) and value for value in pair)
            or pair[0] not in VALID_ROLES
            for pair in self.allowed_primitives
        ):
            raise ValueError("allowed_primitives must contain valid (role, name) pairs")
        object.__setattr__(self, "allowed_reward_fields", tuple(self.allowed_reward_fields))
        object.__setattr__(self, "allowed_primitives", tuple(tuple(p) for p in self.allowed_primitives))
        for name in ("editable_hyperparameters", "frozen_requirements",
                     "frozen_hyperparameters", "runtime_parameters"):
            value = getattr(self, name)
            if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
                raise ValueError(f"{name} must be an object with string keys")
            object.__setattr__(self, name, copy.deepcopy(value))
        if not isinstance(self.allow_primitive_requests, bool):
            raise ValueError("allow_primitive_requests must be a boolean")
        if not isinstance(self.synthesizable_roles, (list, tuple)) or any(
            role not in VALID_ROLES for role in self.synthesizable_roles
        ):
            raise ValueError("synthesizable_roles must contain registered role identifiers")
        if len(set(self.synthesizable_roles)) != len(self.synthesizable_roles):
            raise ValueError("synthesizable_roles must be unique")
        object.__setattr__(self, "synthesizable_roles", tuple(self.synthesizable_roles))
        if self.synthesizable_roles and not self.allow_primitive_requests:
            raise ValueError("synthesizable_roles require allow_primitive_requests")
        if not isinstance(self.approved_references, dict):
            raise ValueError("approved_references must be a mapping of binding IDs to grant records")
        if self.approved_references:
            from recital.synthesis.references import ReferenceBinding
            from recital.synthesis.models import SynthesisError

            records = {}
            for binding_id, record in self.approved_references.items():
                if not isinstance(record, dict):
                    raise ValueError("reference grant must be an object")
                try:
                    cleaned = {
                        key: value
                        for key, value in record.items()
                        if key not in {"sha256", "labels_sha256"}
                    }
                    binding = ReferenceBinding(**cleaned)
                except (TypeError, ValueError, SynthesisError) as exc:
                    raise ValueError(f"invalid reference grant: {exc}") from exc
                if binding.binding_id != binding_id:
                    raise ValueError("approved reference key differs from its binding ID")
                records[binding_id] = binding.to_dict()
            object.__setattr__(self, "approved_references", records)
        if not isinstance(self.admitted_primitives, dict):
            raise ValueError("admitted_primitives must be a mapping")
        for key, declaration in self.admitted_primitives.items():
            if not isinstance(key, str) or key.count("/") != 1:
                raise ValueError("admitted primitive keys must be role/name")
            role, name = key.split("/")
            if (role, name) not in self.allowed_primitives:
                raise ValueError("admitted primitives must belong to the effective allowlist")
            if (role, name) in REGISTRY:
                raise ValueError("admitted primitives cannot override built-in registry entries")
            if not isinstance(declaration, dict):
                raise ValueError("admitted primitive declarations must be objects")
            schema = declaration.get("parameters")
            if not isinstance(schema, dict) or schema.get("type") != "object":
                raise ValueError("admitted primitive parameters must be an object JSON Schema")
            if schema.get("additionalProperties") is not False:
                raise ValueError("admitted primitive parameter names must be explicit")
            Draft202012Validator.check_schema(schema)
        object.__setattr__(self, "admitted_primitives", copy.deepcopy(self.admitted_primitives))
        if any(not isinstance(value, dict) for value in self.runtime_parameters.values()):
            raise ValueError("runtime_parameters must map role/name to parameter bindings")
        if "hyperparameters" in self.allowed_reward_fields:
            raise ValueError("declare optimizer permissions in editable_hyperparameters")
        overlap = set(self.editable_hyperparameters) & set(self.frozen_hyperparameters)
        if overlap:
            raise ValueError(f"hyperparameters cannot be both editable and frozen: {sorted(overlap)}")
        for schema in self.editable_hyperparameters.values():
            if not isinstance(schema, dict):
                raise ValueError("editable_hyperparameters must map names to JSON Schema objects")
            try:
                Draft202012Validator.check_schema(schema)
            except SchemaError as exc:
                raise ValueError(f"invalid hyperparameter schema: {exc.message}") from exc
        object.__setattr__(self, "frozen_target_objectives", _canonical_target_map(
            self.frozen_target_objectives, "frozen_target_objectives"
        ))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RecipeCapabilities:
        """Parse YAML/JSON data strictly; constructor validation owns coercion.

        Sequence containers become tuples; field aliases, unknown fields and
        implicit string-to-boolean or numeric conversions are not accepted.
        """
        if not isinstance(payload, dict):
            raise ValueError("recipe capabilities must be an object")
        if any(not isinstance(key, str) for key in payload):
            raise ValueError("recipe capabilities field names must be strings")
        unknown = set(payload) - {entry.name for entry in fields(cls)}
        missing = {"allowed_reward_fields", "editable_hyperparameters",
                   "allowed_primitives", "frozen_requirements"} - set(payload)
        if unknown or missing:
            raise ValueError(
                f"recipe capabilities fields: unknown={sorted(unknown)}, missing={sorted(missing)}"
            )
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def primitive_catalog(capabilities: RecipeCapabilities) -> list[dict[str, Any]]:
    """Describe actual allowed constructors without executing their defaults."""
    catalog = []
    for role, name in sorted(capabilities.allowed_primitives):
        admitted = capabilities.admitted_primitives.get(f"{role}/{name}")
        if admitted is not None:
            catalog.append({"role": role, "name": name, **copy.deepcopy(admitted),
                            "origin": "admitted_run_extension"})
            continue
        cls = REGISTRY.get((role, name))
        if cls is None:
            raise ValueError(f"declared primitive is not registered: {role}/{name}")
        runtime = capabilities.runtime_parameters.get(f"{role}/{name}", {})
        # The composer binds some constructor arguments from run context and
        # refuses a specification that supplies them, whether or not the contract
        # declares them. Advertising those as writable costs the agent a turn.
        framework_owned = FRAMEWORK_OWNED_PARAMETERS.get(name)
        parameters = {}
        for key, parameter in inspect.signature(cls).parameters.items():
            info: dict[str, Any] = {
                "required": parameter.default is inspect.Parameter.empty,
                "runtime_owned": key in runtime or key == framework_owned,
            }
            if parameter.annotation is not inspect.Parameter.empty:
                info["annotation"] = str(parameter.annotation)
            if parameter.default is not inspect.Parameter.empty:
                default = parameter.default
                if default is None or isinstance(default, (str, int, float, bool, list, dict, tuple)):
                    info["default"] = default
                else:
                    info["default_factory_or_object"] = True
            parameters[key] = info
        description = " ".join(inspect.cleandoc(cls.__doc__ or "").split("\n\n")[0].split())
        catalog.append({
            "role": role, "name": name,
            "description": description,
            "parameters": parameters,
        })
    return catalog


def primitive_request_problems(
    request: dict[str, Any], capabilities: RecipeCapabilities, spec: dict[str, Any],
) -> list[str]:
    """Reject request-owner mistakes before spending a code-generation attempt."""
    if not isinstance(request, dict):
        return ["primitive request must be an object"]
    problems = []
    role = request.get("role")
    if not capabilities.allow_primitive_requests or role not in capabilities.synthesizable_roles:
        problems.append("requested primitive role is not authorized for synthesis")
    field = next((key for key, value in RECIPE_PRIMITIVE_ROLES.items() if value == role), None)
    if field not in capabilities.allowed_reward_fields:
        problems.append("synthesis cannot edit a frozen recipe field")
    name = request.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,95}", name):
        problems.append("extension name must be a simple identifier")
    elif isinstance(role, str):
        if (role, name) in REGISTRY or f"{role}/{name}" in capabilities.admitted_primitives:
            problems.append("primitive already exists; reuse it in the recipe instead of requesting synthesis")
        refs = spec.get(field)
        if isinstance(refs, list) and sum(
            isinstance(ref, dict) and ref.get("name") == name for ref in refs
        ) > 1:
            problems.append("synthesized primitive has duplicate recipe references")
    oracle = request.get("oracle")
    if role == "per_oracle_utility":
        if oracle not in (spec.get("enabled_oracles") or []):
            problems.append("utility request must name an enabled optimizer oracle")
    elif oracle is not None:
        problems.append("only a utility request may bind an oracle; omit oracle or use null for other roles")
    references = request.get("reference_bindings", [])
    if not isinstance(references, list) or any(not isinstance(value, str) for value in references):
        problems.append("reference_bindings must be an array of approved IDs")
    elif len(references) != len(set(references)) or any(
        value not in capabilities.approved_references for value in references
    ):
        problems.append("reference_bindings must contain unique approved reference IDs")
    return problems


def validate_recipe_capabilities(
    spec: dict[str, Any],
    capabilities: RecipeCapabilities,
    *,
    current_spec: dict[str, Any] | None = None,
    pending_primitive: dict[str, Any] | None = None,
) -> list[str]:
    """Check permissions and constructor arguments; never repair a proposal.

    A parent is required to establish the immutable-field reference, including
    for the initial proposal (pass the resolved initial spec). This is not a
    replacement for source-role, target-coordinate, compiler or success checks.
    """
    problems = []
    if current_spec is None:
        return ["capability validation requires current_spec as immutable-field reference"]
    if pending_primitive is not None and capabilities.synthesizable_roles:
        problems.extend(primitive_request_problems(pending_primitive, capabilities, spec))
    allowed_fields = set(capabilities.allowed_reward_fields) | {"hyperparameters"}
    for key in sorted(set(spec) | set(current_spec)):
        if key not in allowed_fields and (
            (key in spec) != (key in current_spec) or spec.get(key) != current_spec.get(key)
        ):
            from recital.spec.composer import derived_default_weight_change
            if (key == "aggregator" and "enabled_oracles" in allowed_fields
                    and derived_default_weight_change(current_spec, spec)):
                continue
            problems.append(f"recipe field {key!r} is frozen")
    if capabilities.frozen_target_objectives:
        enabled = spec.get("enabled_oracles")
        if not isinstance(enabled, list) or any(not isinstance(name, str) for name in enabled):
            problems.append("enabled_oracles must be a list of oracle names")
        else:
            try:
                targets = _canonical_target_map(spec.get("target_objectives", {}), "target_objectives")
            except ValueError as exc:
                problems.append(str(exc))
            else:
                if set(targets) != set(enabled):
                    problems.append("target_objectives must cover exactly enabled_oracles")
                frozen = capabilities.frozen_target_objectives
                for name in sorted(set(enabled) | set(targets)):
                    if name not in frozen:
                        problems.append(f"target_objectives[{name!r}] has no frozen declaration")
                    elif name in targets and targets[name] != frozen[name]:
                        problems.append(f"target_objectives[{name!r}] differs from its frozen declaration")
    hp = spec.get("hyperparameters")
    if not isinstance(hp, dict):
        return [*problems, "hyperparameters must be an object"]
    parent_hp = current_spec.get("hyperparameters") or {}
    frozen_hp = {**parent_hp, **capabilities.frozen_hyperparameters}
    editable = capabilities.editable_hyperparameters
    for key in sorted(set(hp) | set(frozen_hp)):
        if key in frozen_hp and key not in hp:
            problems.append(f"hyperparameters.{key} must be retained from the parent")
        elif key not in editable and (
            (key in hp) != (key in frozen_hp) or hp.get(key) != frozen_hp.get(key)
        ):
            problems.append(f"hyperparameters.{key} is frozen")
        elif key in hp and key in editable:
            if isinstance(hp[key], float) and not math.isfinite(hp[key]):
                problems.append(f"hyperparameters.{key} must be finite")
            for error in Draft202012Validator(editable[key]).iter_errors(hp[key]):
                problems.append(f"hyperparameters.{key}: {error.message}")

    allowed_primitives = set(capabilities.allowed_primitives)
    for field_name, role in RECIPE_PRIMITIVE_ROLES.items():
        value = spec.get(field_name)
        if field_name == "oracle_utilities":
            if value is None:
                refs = []
            elif not isinstance(value, dict):
                problems.append("oracle_utilities must be a mapping")
                continue
            else:
                if set(value) - set(spec.get("enabled_oracles") or []):
                    problems.append("oracle_utilities may reference only enabled oracles")
                refs = list(value.values())
        elif field_name in {"aggregator", "shape"}:
            refs = [] if value is None else [value]
        else:
            if value is None:
                refs = []
            elif not isinstance(value, list):
                problems.append(f"{field_name} must be a list")
                continue
            else:
                refs = value
        for ref_index, ref in enumerate(refs):
            if not isinstance(ref, dict) or not isinstance(ref.get("name"), str):
                problems.append(f"{field_name}: primitive reference needs a string name")
                continue
            name = ref["name"]
            if (pending_primitive is not None
                    and (role, name) == (pending_primitive.get("role"), pending_primitive.get("name"))
                    and capabilities.allow_primitive_requests
                    and role in capabilities.synthesizable_roles
                    and field_name in capabilities.allowed_reward_fields
                    and (role, name) not in allowed_primitives):
                # Placeholder only: no constructor or execution capability is granted.
                # Final post-admission validation is called without this exception.
                if role == "per_oracle_utility" and (
                    pending_primitive.get("oracle") not in (spec.get("enabled_oracles") or [])
                    or list(value)[ref_index] != pending_primitive.get("oracle")
                ):
                    problems.append("pending utility must bind its declared enabled oracle")
                continue
            if (role, name) not in allowed_primitives:
                offered = sorted(n for r, n in allowed_primitives if r == role)
                problems.append(f"{field_name}: primitive {role}/{name} is not allowed; "
                                f"the contract allows for this role: {offered or 'none'}")
                continue
            admitted = capabilities.admitted_primitives.get(f"{role}/{name}")
            if admitted is not None:
                kwargs = {key: value for key, value in ref.items() if key != "name"}
                for error in Draft202012Validator(admitted["parameters"]).iter_errors(kwargs):
                    problems.append(f"{field_name}: {error.message}")
                runtime = capabilities.runtime_parameters.get(f"{role}/{name}", {})
                for key in sorted(set(kwargs) & set(runtime)):
                    problems.append(f"{field_name}.{key} is runtime-owned and must be omitted")
                continue
            cls = REGISTRY.get((role, name))
            if cls is None:
                problems.append(f"{field_name}: primitive {role}/{name} is not registered")
                continue
            runtime = capabilities.runtime_parameters.get(f"{role}/{name}", {})
            authored = set(ref) - {"name"}
            for key in sorted(authored & runtime.keys()):
                problems.append(f"{field_name}.{key} is runtime-owned and must be omitted")
            signature = inspect.signature(cls)
            kwargs = {key: value for key, value in ref.items() if key != "name"}
            kwargs.update(runtime)
            try:
                signature.bind(**kwargs)
            except TypeError as exc:
                problems.append(f"{field_name}: {exc}")
    return problems


def with_admitted_primitives(
    capabilities: RecipeCapabilities,
    declarations: dict[str, dict[str, Any]],
    *,
    published_baseline: bool = False,
) -> RecipeCapabilities:
    """Derive effective permissions from verified catalog metadata.

    Callers obtain declarations from the runtime catalog after admission; this
    function does not admit code. Published primitives may form the common frozen
    baseline, including when synthesis of new primitives is not authorized.
    """
    admitted = copy.deepcopy(capabilities.admitted_primitives)
    pairs = list(capabilities.allowed_primitives)
    for key, declaration in declarations.items():
        if not isinstance(key, str) or key.count("/") != 1:
            raise ValueError("extension keys must be role/name")
        role, name = key.split("/")
        if not name or (not published_baseline and role not in capabilities.synthesizable_roles):
            raise ValueError(f"primitive synthesis not authorized for {role!r}")
        if key in admitted and admitted[key] != declaration:
            raise ValueError("an admitted primitive identity cannot be replaced")
        admitted[key] = copy.deepcopy(declaration)
        if (role, name) not in pairs:
            pairs.append((role, name))
    return replace(capabilities, allowed_primitives=tuple(pairs), admitted_primitives=admitted)
