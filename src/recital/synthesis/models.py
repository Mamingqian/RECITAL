"""Trusted JSON records and admission policy for isolated primitives."""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

ROLE_PER_ORACLE_UTILITY = "per_oracle_utility"
ROLE_CROSS_ORACLE_AGG = "cross_oracle_aggregation"
ROLE_PER_SEQ_PENALTY = "per_sequence_penalty"
ROLE_HARD_GATE = "hard_biological_gate"
ROLE_LIBRARY_SELECTOR = "library_selector"
ROLES = frozenset({ROLE_PER_ORACLE_UTILITY, ROLE_CROSS_ORACLE_AGG,
                   ROLE_PER_SEQ_PENALTY, ROLE_HARD_GATE, ROLE_LIBRARY_SELECTOR})
MAX_MESSAGE_BYTES = 8 * 1024 * 1024


class SynthesisError(ValueError):
    """An extension failed closed at a trusted runtime boundary."""


class IsolationError(SynthesisError):
    """The required operating-system isolation could not be established."""


def canonical_json(value: Any, *, max_bytes: int | None = None) -> bytes:
    """Serialize plain JSON only; never invoke user serialization hooks."""
    def check(item: Any, depth: int = 0) -> None:
        if depth > 64:
            raise SynthesisError("JSON nesting exceeds 64")
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) in (list, tuple):
            for child in item:
                check(child, depth + 1)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                check(child, depth + 1)
            return
        raise SynthesisError("only finite, plain JSON values are allowed")

    check(value)
    result = json.dumps(value, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=True, allow_nan=False).encode("utf-8")
    if max_bytes is not None and len(result) > max_bytes:
        raise SynthesisError("JSON message exceeds byte limit")
    return result


def probe_representation(representation: Mapping[str, Any]) -> dict[str, Any]:
    """The declared sequence space the isolated admission probe is drawn from.

    The admission probe has to be a sequence the task's own primitives can
    actually accept: a protein penalty that assumes 80-96 residues is not
    exercised by a 4-mer, it is rejected by one.  Only plain JSON scalars are
    accepted because this crosses the sandbox boundary inside the worker
    configuration.
    """
    if type(representation) is not dict or set(representation) != {
        "alphabet", "min_length", "max_length"
    }:
        raise SynthesisError("probe representation requires exactly alphabet/min_length/max_length")
    alphabet = representation["alphabet"]
    minimum, maximum = representation["min_length"], representation["max_length"]
    if type(alphabet) is not str or not alphabet or len(set(alphabet)) != len(alphabet):
        raise SynthesisError("probe alphabet must be a nonempty string of distinct characters")
    if type(minimum) is not int or type(maximum) is not int or not 1 <= minimum <= maximum:
        raise SynthesisError("probe lengths must be integers with 1 <= min_length <= max_length")
    return {"alphabet": alphabet, "max_length": maximum, "min_length": minimum}


@dataclass(frozen=True)
class RuntimeLimits:
    memory_bytes: int = 1024 ** 3
    cpu_seconds: int = 2
    wall_seconds: float = 5.0
    admission_seconds: float = 10.0
    max_message_bytes: int = MAX_MESSAGE_BYTES

    def __post_init__(self) -> None:
        for key in ("memory_bytes", "cpu_seconds", "wall_seconds",
                    "admission_seconds", "max_message_bytes"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise SynthesisError(f"invalid runtime limit {key}")
        for key in ("memory_bytes", "cpu_seconds", "max_message_bytes"):
            if type(getattr(self, key)) is not int:
                raise SynthesisError(f"{key} must be an integer")


@dataclass(frozen=True)
class PrimitiveRequest:
    role: str
    name: str
    code: str
    unit_test: str
    parameters: dict[str, Any] = field(default_factory=dict)
    reference_bindings: tuple[str, ...] = ()
    parameter_schema: dict[str, Any] | None = None
    oracle: str | None = None
    selector_input_schema: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise SynthesisError(f"unsupported generated role: {self.role}")
        if self.oracle is not None and (
            self.role != ROLE_PER_ORACLE_UTILITY or not isinstance(self.oracle, str)
            or not self.oracle
        ):
            raise SynthesisError("oracle is only valid as a nonempty per-oracle utility binding")
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,95}", self.name):
            raise SynthesisError("extension name must be a simple identifier")
        for key in ("code", "unit_test"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value.strip() or len(value.encode()) > 256 * 1024:
                raise SynthesisError(f"{key} must be nonempty and at most 256 KiB")
        if type(self.parameters) is not dict or any(
            not isinstance(key, str) or not key.isidentifier()
            or key.startswith("_") or key in {"oracle_name", "name"}
            for key in self.parameters
        ):
            raise SynthesisError("parameters must be plain constructor kwargs")
        object.__setattr__(self, "parameters", json.loads(canonical_json(self.parameters)))
        if self.parameter_schema is not None:
            object.__setattr__(self, "parameter_schema", json.loads(canonical_json(self.parameter_schema)))
        if not isinstance(self.reference_bindings, (list, tuple)) or any(
            not isinstance(key, str) or not key for key in self.reference_bindings
        ) or len(set(self.reference_bindings)) != len(self.reference_bindings):
            raise SynthesisError("reference_bindings must be unique approved IDs")
        object.__setattr__(self, "reference_bindings", tuple(self.reference_bindings))
        if self.selector_input_schema is not None:
            schema = json.loads(canonical_json(self.selector_input_schema))
            if self.role != ROLE_LIBRARY_SELECTOR or type(schema) is not dict or set(schema) != {"conditions", "measurements"}:
                raise SynthesisError("selector input schema requires conditions/measurements and a selector role")
            conditions, measurements = schema["conditions"], schema["measurements"]
            if (type(conditions) is not list or not conditions or
                    any(type(value) is not str or not value for value in conditions) or
                    len(conditions) != len(set(conditions)) or type(measurements) is not dict):
                raise SynthesisError("selector input schema requires unique condition labels and a measurement mapping")
            for name, declaration in measurements.items():
                if not name or type(declaration) is not dict or set(declaration) != {"shape", "sources"}:
                    raise SynthesisError("selector measurement schema requires shape/sources")
                shape, sources = declaration["shape"], declaration["sources"]
                if (type(shape) is not list or len(shape) > 1 or
                        any(type(size) is not int or size < 1 for size in shape)):
                    raise SynthesisError("selector measurement shape must be [] or [column_count]")
                if (type(sources) is not list or any(type(source) is not str or not source for source in sources)
                        or len(sources) != len(set(sources))):
                    raise SynthesisError("selector measurement sources must be unique visible names")
            object.__setattr__(self, "selector_input_schema", schema)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PrimitiveRequest":
        try:
            return cls(**dict(payload))
        except TypeError as exc:
            raise SynthesisError(f"invalid primitive request: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
