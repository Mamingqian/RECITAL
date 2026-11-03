"""Strict-compatible JSON schemas for the Strategist, Critic and Coder turns, and their retrieval tools.

Every schema is built from run state and uses only: type (a name, or a name plus "null"), enum,
properties, required, additionalProperties false, items, description and maxItems. Which fields a
response must fill for its `kind` is a cross-field rule the deterministic validators check; the
schema only guarantees shape, so it can be sent as an OpenAI strict json_schema.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .transport import ToolSpec

START_MODES = ("from_initial", "from_parent", "continue_parent")
STOP_OUTCOMES = ("deliver", "no_qualified_candidate")
ALLOWED_KEYS = frozenset({"type", "enum", "properties", "required", "additionalProperties",
                          "items", "description", "maxItems"})


def _object(properties: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": dict(properties)}


def _nullable(schema: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(schema)
    result["type"] = [schema["type"], "null"]
    return result


def _name(names: Sequence[str] | None, description: str) -> dict[str, Any]:
    if names:
        return {"type": "string", "enum": list(names), "description": description}
    return {"type": "string", "description": description}


def _call(names: Sequence[str] | None, what: str, params: str | None = None) -> dict[str, Any]:
    return _object({
        "name": _name(names, f"{what} name"),
        "params_json": {"type": "string",
                        "description": params or f"JSON object text of the {what}'s parameters; '{{}}' for defaults"},
    })


def _aggregator_params(form: Mapping[str, Any] | None) -> str | None:
    """What an aggregator's `params_json` holds, shown in the form of the initial recipe's own aggregator.

    `recipes._params` parses the text with `json.loads`, which refuses any character after the object's closing
    brace as extra data.
    """
    if form is None:
        return None
    parameters = {key: value for key, value in form.items() if key != "name"}
    return ("a single JSON object as text with nothing after its closing brace, holding the aggregator's "
            f"parameters; the initial recipe's {form['name']} reads {json.dumps(parameters)}; '{{}}' for defaults")


def round_response_schema(
    *,
    conditions: Sequence[str],
    reward_oracles: Sequence[str],
    aggregators: Sequence[str] = (),
    utilities: Sequence[str] = (),
    penalties: Sequence[str] = (),
    gates: Sequence[str] = (),
    max_candidates: int = 3,
    stop_only: bool = False,
    synthesis_roles: Sequence[str] | None = None,
    approved_references: Sequence[str] = (),
    aggregator_form: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The Strategist's per-round response.

    `synthesis_roles=None` removes `develop_component` and the `component` field entirely.
    `stop_only` leaves `stop` as the only kind and no room for candidates.
    `aggregator_form` is the initial recipe's aggregator, when it has one; the aggregator's `params_json`
    description then shows its parameters.
    """
    kinds = ["stop"] if stop_only else ["propose", "stop"]
    if synthesis_roles is not None and not stop_only:
        kinds.append("develop_component")
    candidate = _object({
        "parent": {"type": "string", "description": "'initial' or a completed candidate id"},
        "start": {"type": "string", "enum": list(START_MODES),
                  "description": "from_initial restarts from the SFT policy; from_parent starts a new stage "
                                 "from the parent's final weights with a fresh optimizer and learning-rate "
                                 "schedule; continue_parent resumes the parent's optimizer and schedule, so it "
                                 "keeps the parent's lr and may add only the updates left in that schedule "
                                 "(otherwise it is refused: use from_parent or from_initial)"},
        "hypothesis": {"type": "string", "description": "one or two sentences naming what this candidate tests"},
        "steps": _nullable({"type": "integer",
                            "description": "optimizer updates; null uses the declared default; with continue_parent "
                                           "at most the updates left in the parent's learning-rate schedule"}),
        "lr": _nullable({"type": "number",
                         "description": "learning rate; null inherits; with continue_parent it must equal the "
                                        "parent's, so a changed lr needs start from_parent or from_initial"}),
        "kl": _nullable({"type": "number", "description": "KL coefficient; null inherits"}),
        "group_size": _nullable({"type": "integer", "description": "rollouts per prompt; null inherits"}),
        "enabled_oracles": _nullable({"type": "array", "items": _name(reward_oracles, "reward oracle"),
                                      "description": "oracles that make up the reward; null inherits"}),
        "aggregator": _nullable(_call(aggregators, "aggregator", _aggregator_params(aggregator_form))),
        "oracle_utilities": _nullable({"type": "array", "items": _object({
            "oracle": _name(reward_oracles, "oracle the utility applies to"),
            "name": _name(utilities, "utility name"),
            "params_json": {"type": "string", "description": "JSON object text of the utility's parameters"},
        })}),
        "penalties": _nullable({"type": "array", "items": _call(penalties, "penalty"),
                                "description": "empty array removes all penalties; null inherits"}),
        "gates": _nullable({"type": "array", "items": _call(gates, "gate"),
                            "description": "empty array removes all gates; null inherits"}),
    })
    properties: dict[str, Any] = {
        "kind": {"type": "string", "enum": kinds},
        "rationale": {"type": "string", "description": "at most 120 words"},
        "candidates": {"type": "array", "maxItems": 0 if stop_only else max_candidates, "items": candidate,
                       "description": f"exactly {max_candidates} candidates when kind is propose; otherwise empty"},
        "stop": _nullable(_object({
            "outcome": {"type": "string", "enum": list(STOP_OUTCOMES)},
            "assignment": {"type": "array", "items": _object({
                "condition": {"type": "string", "enum": list(conditions)},
                "candidate": {"type": "string", "description": "completed candidate id delivering this condition"},
            }), "description": "one entry per condition when outcome is deliver; empty otherwise"},
            "summary": {"type": "string"},
        })),
    }
    if synthesis_roles is not None:
        properties["component"] = _nullable(_object({
            "role": {"type": "string", "enum": list(synthesis_roles)},
            "name": {"type": "string", "description": "new primitive name; must not exist yet"},
            "description": {"type": "string"},
            "biological_motivation": {"type": "string"},
            "oracle": _nullable({"type": "string", "description": "oracle a per_oracle_utility applies to"}),
            "reference_candidate": {"type": "string", "description": "candidate whose recipe the component is tested in"},
            "reference_bindings": {"type": "array", "items": _name(approved_references, "approved reference"),
                                   "description": "approved reference datasets the code may use"},
        }))
    return _object(properties)


def critic_schema() -> dict[str, Any]:
    return _object({
        "concerns": {"type": "array", "maxItems": 8, "items": {"type": "string"}},
        "requested_evidence": {"type": "array", "maxItems": 4, "items": {"type": "string"}},
        "rationale": {"type": "string", "description": "at most 80 words"},
    })


def coder_output_schema(roles: Sequence[str]) -> dict[str, Any]:
    return _object({
        "primitive_name": {"type": "string"},
        "role": {"type": "string", "enum": list(roles)},
        "code": {"type": "string"},
        "unit_test": {"type": "string"},
        "parameters_json": {"type": "string", "description": "JSON object text of constructor defaults; '{}' for none"},
        "parameter_schema_json": {"type": "string",
                                  "description": "JSON Schema object text for the parameters; '{}' for none"},
    })


def retrieval_tools(conditions: Sequence[str], sort_fields: Sequence[str]) -> tuple[ToolSpec, ToolSpec]:
    candidate = ToolSpec(
        "get_candidate",
        "Effective recipe, start and training-curve tail of one completed candidate; its ledger row is already "
        "in the ledger.",
        _object({"id": {"type": "string"}}))
    rows = ToolSpec(
        "get_rows",
        "Sequences and measurements of up to 50 rows of one candidate's pool for one condition.",
        _object({
            "candidate": {"type": "string"},
            "condition": {"type": "string", "enum": list(conditions)},
            "n": {"type": "integer", "description": "1 to 50"},
            "sort": {"type": "string", "enum": [*sort_fields, "random"]},
        }))
    return candidate, rows


def assert_strict_compatible(schema: Mapping[str, Any], *, path: str = "$") -> None:
    """Raise ValueError unless `schema` stays inside the strict structured-output subset."""
    unknown = set(schema) - ALLOWED_KEYS
    if unknown:
        raise ValueError(f"{path}: keys outside the strict subset: {sorted(unknown)}")
    kind = schema.get("type")
    if isinstance(kind, list):
        if len(kind) != 2 or "null" not in kind or kind.count("null") != 1:
            raise ValueError(f"{path}: a type list must be one type plus null, got {kind}")
        kind = [name for name in kind if name != "null"][0]
    if kind == "object":
        properties = schema.get("properties")
        if schema.get("additionalProperties") is not False or not isinstance(properties, Mapping):
            raise ValueError(f"{path}: an object needs additionalProperties false and explicit properties")
        if set(schema.get("required", ())) != set(properties):
            raise ValueError(f"{path}: every property must be required")
        for name, child in properties.items():
            assert_strict_compatible(child, path=f"{path}.{name}")
    elif kind == "array":
        if "items" not in schema:
            raise ValueError(f"{path}: an array needs items")
        assert_strict_compatible(schema["items"], path=f"{path}[]")
    elif kind not in {"string", "integer", "number", "boolean"}:
        raise ValueError(f"{path}: unsupported type {kind!r}")
