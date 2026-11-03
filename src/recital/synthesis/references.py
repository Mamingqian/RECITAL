"""Adapter-owned reference grants; generated requests can select IDs only."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from .models import SynthesisError, canonical_json


@dataclass(frozen=True)
class ReferenceBinding:
    binding_id: str
    source: str
    condition: str | None
    schema: dict[str, Any]
    count: int
    condition_coverage: tuple[str, ...] = ()
    condition_counts: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if not self.binding_id or not self.source:
            raise SynthesisError("reference ID and source must be nonempty")
        if type(self.count) is not int or self.count <= 0:
            raise SynthesisError("reference count must be positive")
        if self.condition is not None and not isinstance(self.condition, str):
            raise SynthesisError("reference condition must be a string or null")
        coverage = self.condition_coverage
        if (not isinstance(coverage, (list, tuple))
                or any(type(label) is not str or not label for label in coverage)
                or len(set(coverage)) != len(coverage)):
            raise SynthesisError("reference condition coverage must contain unique nonempty labels")
        coverage = tuple(sorted(coverage))
        if coverage and self.condition is not None and coverage != (self.condition,):
            raise SynthesisError("reference condition coverage differs from its declared condition")
        object.__setattr__(self, "condition_coverage", coverage)
        if self.condition_counts is not None:
            counts = self.condition_counts
            if (type(counts) is not dict or set(counts) != set(coverage)
                    or any(type(n) is not int or n <= 0 for n in counts.values())
                    or sum(counts.values()) != self.count):
                raise SynthesisError("reference condition counts must exactly cover all granted rows")
            object.__setattr__(self, "condition_counts", dict(sorted(counts.items())))
        schema = json.loads(canonical_json(self.schema))
        # No external schema resolution (including file:// or HTTP requests).
        def check(value: Any) -> None:
            if isinstance(value, dict):
                if "$ref" in value or "$dynamicRef" in value:
                    raise SynthesisError("reference schemas must be self-contained")
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)
        check(schema)
        Draft202012Validator.check_schema(schema)
        object.__setattr__(self, "schema", schema)

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        # Unset condition_coverage and condition_counts are omitted, so a reference without condition labels
        # keeps a byte-for-byte reproducible record.
        if self.condition_coverage:
            record["condition_coverage"] = list(self.condition_coverage)
        else:
            record.pop("condition_coverage")
        if record["condition_counts"] is None:
            record.pop("condition_counts")
        return record


class ReferenceBinder:
    """Bind complete datasets against immutable records approved by an adapter.

    ``grants`` and ``allowed_sources`` are framework inputs, never agent inputs.
    Data is JSON rows or aligned JSON columns. No API accepts filesystem paths.
    """

    def __init__(self, grants: Mapping[str, ReferenceBinding] | None = None,
                 *, allowed_sources: set[str] | frozenset[str] = frozenset()):
        self._grants: dict[str, bytes] = {}
        self._bound: dict[str, bytes] = {}
        self._records: dict[str, bytes] = {}
        allowed_sources = frozenset(allowed_sources)
        for key, grant in (grants or {}).items():
            if key != grant.binding_id or grant.source not in allowed_sources:
                raise SynthesisError("reference grant ID/source is not approved")
            self._grants[key] = canonical_json(grant.to_dict())

    def approve(self, binding_id: str, data: Any, *,
                labels: list[str] | tuple[str, ...] | None = None) -> None:
        """Bind complete data and optional adapter-supplied row-aligned labels.

        Labels are checked against the actual row count and any declared coverage
        or counts. They are not exposed publicly or injected into generated
        reference values;
        adapters needing labels in generated code must include them in the
        schema-validated data itself (or provide separate per-condition grants).
        """
        if binding_id not in self._grants:
            raise SynthesisError("unknown reference grant")
        record = json.loads(self._grants[binding_id])
        encoded = canonical_json(data)
        clean = json.loads(encoded)
        if isinstance(clean, list):
            count = len(clean)
        elif isinstance(clean, dict) and clean and all(isinstance(v, list) for v in clean.values()):
            sizes = {len(v) for v in clean.values()}
            count = sizes.pop() if len(sizes) == 1 else -1
        else:
            raise SynthesisError("reference must contain nonempty rows or aligned columns")
        if count <= 0 or count != record["count"]:
            raise SynthesisError("reference count is incomplete or empty")
        errors = list(Draft202012Validator(record["schema"]).iter_errors(clean))
        if errors:
            raise SynthesisError(f"reference schema mismatch: {errors[0].message}")
        if labels is None:
            if any(record.get(key) for key in ("condition_coverage", "condition_counts")):
                raise SynthesisError("reference condition coverage requires aligned labels")
        else:
            if (not isinstance(labels, (list, tuple)) or len(labels) != count
                    or any(type(label) is not str or not label for label in labels)):
                raise SynthesisError("reference labels must contain one nonempty string per row")
            clean_labels = json.loads(canonical_json(labels))
            counts = dict(sorted(Counter(clean_labels).items()))
            if record["condition"] is not None and set(counts) != {record["condition"]}:
                raise SynthesisError("reference labels differ from the granted condition")
            verified = {"condition_coverage": sorted(counts), "condition_counts": counts}
            for key, value in verified.items():
                if key in record and record[key] != value:
                    raise SynthesisError(f"reference {key} differs from aligned labels")
            record.update(verified)
        metadata = canonical_json(record)
        if binding_id in self._bound and self._bound[binding_id] != encoded:
            raise SynthesisError("reference binding is immutable")
        if binding_id in self._records and self._records[binding_id] != metadata:
            raise SynthesisError("reference label binding is immutable")
        self._bound[binding_id] = encoded
        self._records[binding_id] = metadata

    def declarations(self) -> dict[str, dict[str, Any]]:
        """Detached public grant metadata for completely approved bindings only.

        Never returns reference values, aligned labels, or filesystem locations.
        Unbound and failed grants confer no discoverable runtime capability.
        """
        return {key: json.loads(self._records[key]) for key in sorted(self._bound)}

    def resolve(self, binding_ids: tuple[str, ...] | list[str]) -> dict[str, Any]:
        result = {}
        for key in binding_ids:
            if key not in self._bound:
                raise SynthesisError(f"reference {key!r} has no complete approved binding")
            result[key] = {"record": json.loads(self._records[key]),
                           "data": json.loads(self._bound[key])}
        return result
