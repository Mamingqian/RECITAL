"""Run-local staged admission and trusted constructor proxies."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping

from jsonschema import Draft202012Validator
import numpy as np

from .bundle import RuntimeBundle
from .models import (PrimitiveRequest, ROLE_CROSS_ORACLE_AGG, ROLE_HARD_GATE,
                     ROLE_PER_ORACLE_UTILITY, ROLE_PER_SEQ_PENALTY, ROLE_LIBRARY_SELECTOR, RuntimeLimits,
                     SynthesisError, canonical_json, probe_representation)
from .references import ReferenceBinder
from .runtime import IsolatedWorker


_FRAMEWORK_PARAMETER_KEYS = frozenset({"name"})


def _schema_instance(schema: dict[str, Any], parameters: dict[str, Any], name: str) -> dict[str, Any]:
    """Build the schema instance without passing framework metadata to code.

    Coder responses may declare ``name`` as a required schema field so the
    framework can bind the primitive identity.  It is metadata owned by the
    catalog, never a generated constructor argument.  Keeping this conversion
    in one helper makes stage admission and later constructor calls agree.
    """
    if ("name" in schema.get("properties", {})
            or "name" in schema.get("required", [])):
        instance = dict(parameters)
        instance["name"] = name
        return instance
    return dict(parameters)


def parameter_schema(request: PrimitiveRequest) -> dict[str, Any]:
    schema = request.parameter_schema
    if schema is None:
        schema = {"type": "object", "properties": {
            key: {"const": value, "default": value}
            for key, value in request.parameters.items()}, "additionalProperties": False}
    schema = json.loads(canonical_json(schema))
    if (schema.get("type") != "object" or schema.get("additionalProperties") is not False
            or type(schema.get("properties")) is not dict or "patternProperties" in schema):
        raise SynthesisError("parameter schema must be a closed object with properties")
    if any(not key.isidentifier() or key.startswith("_") or key == "oracle_name"
           for key in schema["properties"]):
        raise SynthesisError("invalid or reserved constructor parameter")
    def no_external(value):
        if isinstance(value, dict):
            if any(key in value for key in ("$ref", "$dynamicRef")):
                raise SynthesisError("parameter schema must be self-contained")
            for child in value.values():
                no_external(child)
        elif isinstance(value, list):
            for child in value:
                no_external(child)
    no_external(schema)
    Draft202012Validator.check_schema(schema)
    try:
        Draft202012Validator(schema).validate(
            _schema_instance(schema, request.parameters, request.name)
        )
    except Exception as exc:
        raise SynthesisError(f"admission parameters violate schema: {exc}") from exc
    return schema


def _numeric_vector(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
        raise SynthesisError(f"{label} must be a finite numeric vector")
    return array


def _validated_output(role: str, result: Any, count: int) -> np.ndarray:
    if role == ROLE_LIBRARY_SELECTOR:
        if type(result) is list and not result:
            return np.empty(0, dtype=int)
        array = np.asarray(result)
        if (array.ndim != 1 or array.dtype.kind not in "iu" or len(array) > count
                or len(np.unique(array)) != len(array) or np.any(array < 0) or np.any(array >= count)):
            raise SynthesisError("selector output must be unique in-range integer pool indices")
        return array.astype(int)
    if role == ROLE_HARD_GATE and count == 0 and type(result) is list and not result:
        # JSON has no dtype for an empty list; the child validates its actual dtype.
        return np.empty(0, dtype=bool)
    array = np.asarray(result)
    if array.shape != (count,):
        raise SynthesisError(f"role output must have shape ({count},)")
    if role == ROLE_HARD_GATE:
        if array.dtype.kind != "b":
            raise SynthesisError("gate output must be boolean (True means violation)")
    else:
        array = _numeric_vector(result, "output")
        if role == ROLE_PER_SEQ_PENALTY and np.any(array < 0):
            raise SynthesisError("penalty output must be nonnegative")
    return array


class PrimitiveProxy:
    """Host-owned callable; no generated Python objects cross this boundary."""

    def __init__(self, stage: "StagedExtension", parameters: dict[str, Any], oracle_name: str | None):
        self._stage = stage
        self._parameters = json.loads(canonical_json(parameters))
        self._oracle_name = oracle_name

    def evaluate_context(
        self, sequences: list[str] | tuple[str, ...], *,
        conditioning: list[str] | tuple[str, ...] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate a penalty/gate with explicit, aligned batch context."""
        stage = self._stage
        if stage.closed or stage.catalog.closed:
            raise SynthesisError("extension is closed")
        if stage.request.role not in (ROLE_PER_SEQ_PENALTY, ROLE_HARD_GATE):
            raise SynthesisError("evaluate_context is only valid for penalties and gates")
        if not isinstance(sequences, (list, tuple)) or any(
            type(sequence) is not str for sequence in sequences
        ):
            raise SynthesisError("penalty/gate input must be a sequence list")
        if isinstance(conditioning, np.ndarray):
            if conditioning.ndim != 1:
                raise SynthesisError("conditioning must be an aligned one-dimensional list")
            conditioning = conditioning.tolist()
        if conditioning is not None:
            if (not isinstance(conditioning, (list, tuple))
                    or len(conditioning) != len(sequences)
                    or any(type(label) is not str for label in conditioning)):
                raise SynthesisError("conditioning must contain one string per sequence")
            conditioning = list(conditioning)
        payload = {"sequences": list(sequences), "conditioning": conditioning}
        canonical_json(payload)
        result = stage.worker.request("call", parameters=self._parameters, payload=payload)
        return _validated_output(stage.request.role, result.get("output"), len(sequences))

    def __call__(self, *args, **kwargs) -> np.ndarray:
        stage = self._stage
        if stage.closed or stage.catalog.closed:
            raise SynthesisError("extension is closed")
        role = stage.request.role
        if role == ROLE_LIBRARY_SELECTOR:
            return self.select_library(*args, **kwargs)
        if role == ROLE_PER_ORACLE_UTILITY:
            signature = inspect.Signature([
                inspect.Parameter("raw", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter("target", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter("conditioning", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None),
            ])
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            raw = _numeric_vector(bound.arguments["raw"], "raw optimizer scores")
            target = bound.arguments["target"]
            if hasattr(target, "to_dict"):
                target = target.to_dict()
            if not isinstance(target, Mapping):
                raise SynthesisError("target must be a JSON target declaration")
            conditioning = bound.arguments["conditioning"]
            if isinstance(conditioning, np.ndarray):
                conditioning = conditioning.tolist()
            count = len(raw)
            payload = {"raw": raw.tolist(), "target": dict(target), "conditioning": conditioning}
        else:
            if len(args) != 1 or kwargs:
                raise SynthesisError("aggregation/penalty/gate expects one positional input")
            value = args[0]
            if role == ROLE_CROSS_ORACLE_AGG:
                if not isinstance(value, Mapping) or not value:
                    raise SynthesisError("aggregation requires nonempty optimizer utility mapping")
                if any(name not in stage.catalog.allowed_oracles for name in value):
                    raise SynthesisError("aggregation contains a non-optimizer oracle")
                arrays = {name: _numeric_vector(values, "utility") for name, values in value.items()}
                counts = {len(array) for array in arrays.values()}
                if len(counts) != 1:
                    raise SynthesisError("utility vectors must have equal lengths")
                count = counts.pop()
                payload = {"utilities": {name: array.tolist() for name, array in arrays.items()}}
            else:
                return self.evaluate_context(value)
        # Validate every input before serialization, and output again in the host.
        canonical_json(payload)
        result = stage.worker.request("call", parameters=self._parameters, payload=payload)
        return _validated_output(role, result.get("output"), count)

    def select_library(self, sequences, *, conditioning, measurements, eligible_indices,
                       required_counts, measurement_sources=None):
        """Execute the selector in isolation and revalidate its row decisions."""
        stage = self._stage
        if stage.closed or stage.catalog.closed or stage.request.role != ROLE_LIBRARY_SELECTOR:
            raise SynthesisError("library selection requires an open selector extension")
        if (not isinstance(sequences, (list, tuple)) or any(type(s) is not str for s in sequences)
                or not isinstance(conditioning, (list, tuple)) or len(conditioning) != len(sequences)
                or any(type(c) is not str for c in conditioning)):
            raise SynthesisError("selector sequences and conditioning must be aligned string lists")
        count = len(sequences)
        if (not isinstance(required_counts, Mapping) or not required_counts
                or any(type(k) is not str or not k or type(v) is not int or v < 1
                       for k, v in required_counts.items())
                or any(group not in required_counts for group in conditioning)):
            raise SynthesisError("selector required counts must cover every condition")
        indices = list(eligible_indices)
        if (any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in indices)
                or len(set(indices)) != len(indices) or any(i < 0 or i >= count for i in indices)):
            raise SynthesisError("eligible indices must be unique in-range integers")
        if not isinstance(measurements, Mapping):
            raise SynthesisError("selector measurements must be a visible score mapping")
        arrays = {}
        for name, values in measurements.items():
            sources = (measurement_sources or {}).get(name, (name,))
            if (not isinstance(name, str) or not isinstance(sources, (list, tuple))
                    or not set(sources) <= stage.catalog.allowed_oracles):
                raise SynthesisError("selector measurement depends on a hidden source")
            array = np.asarray(values)
            # Eligibility belongs to the plan, which only requires a measurement where it
            # uses it: a stratum-scoped filter skips other strata, and ranking reads one
            # column. Demanding every measurement be finite on every eligible row would refuse
            # plans the plan itself accepts. What a selector returns is still checked below.
            if array.ndim not in {1, 2} or len(array) != count or array.dtype.kind not in "fiu":
                raise SynthesisError("selector measurements must align with rows")
            # JSON null represents unavailable scores only on ineligible rows;
            # the worker converts it back to NaN, never a favorable zero.
            arrays[name] = np.where(np.isfinite(array), array, None).tolist()
        payload = {"sequences": list(sequences), "conditioning": list(conditioning),
                   "measurements": arrays, "eligible_indices": [int(i) for i in indices],
                   "required_counts": dict(required_counts)}
        canonical_json(payload)
        result = stage.worker.request("call", parameters=self._parameters, payload=payload)
        selected = _validated_output(ROLE_LIBRARY_SELECTOR, result.get("output"), count)
        if not set(selected) <= set(indices):
            raise SynthesisError("selector returned an ineligible row")
        for group, maximum in required_counts.items():
            if sum(conditioning[i] == group for i in selected) > maximum:
                raise SynthesisError("selector exceeded a group's delivery count")
        return selected


class ConstructorProxy:
    """Inspectable constructor with a closed, declarative JSON kwargs schema."""

    def __init__(self, stage: "StagedExtension"):
        self._stage = stage
        self.__name__ = stage.request.name
        self.__doc__ = "Isolated generated primitive constructor (run scoped)."
        self.parameter_schema = json.loads(canonical_json(stage.schema))
        parameters = []
        for key, schema in sorted(stage.schema["properties"].items()):
            # ``name`` is a framework-owned identity field.  It may be
            # required by the declared schema, but is never forwarded to the
            # generated constructor and is intentionally absent from this
            # public signature.
            if key in _FRAMEWORK_PARAMETER_KEYS:
                continue
            default = stage.request.parameters.get(key, schema.get("default", inspect.Parameter.empty))
            parameters.append(inspect.Parameter(key, inspect.Parameter.KEYWORD_ONLY, default=default))
        if stage.request.role == ROLE_PER_ORACLE_UTILITY:
            default = stage.request.oracle or inspect.Parameter.empty
            parameters.append(inspect.Parameter("oracle_name", inspect.Parameter.KEYWORD_ONLY,
                                                default=default))
        self.__signature__ = inspect.Signature(parameters)
        self._primitive_role = stage.request.role
        self._primitive_name = stage.request.name

    def __call__(self, **kwargs: Any) -> PrimitiveProxy:
        stage = self._stage
        if stage.closed or stage.catalog.closed:
            raise SynthesisError("extension is closed")
        has_oracle_name = "oracle_name" in kwargs
        oracle_name = kwargs.pop("oracle_name", stage.request.oracle)
        if stage.request.role == ROLE_PER_ORACLE_UTILITY:
            if oracle_name not in stage.catalog.allowed_oracles:
                raise SynthesisError("per-oracle utility requires an approved optimizer oracle_name")
            if stage.request.oracle is not None and oracle_name != stage.request.oracle:
                raise SynthesisError("cannot replace the admitted optimizer oracle binding")
        elif has_oracle_name:
            raise SynthesisError("oracle_name is only valid for per-oracle utility")
        parameters = {**stage.request.parameters, **kwargs}
        if any(key in _FRAMEWORK_PARAMETER_KEYS for key in parameters):
            raise SynthesisError("framework metadata cannot be supplied as constructor parameters")
        canonical_json(parameters)
        errors = list(Draft202012Validator(stage.schema).iter_errors(
            _schema_instance(stage.schema, parameters, stage.request.name)
        ))
        if errors:
            raise SynthesisError(f"constructor parameters violate schema: {errors[0].message}")
        return PrimitiveProxy(stage, parameters, oracle_name)


class StagedExtension:
    """Screenable resolver whose candidate becomes visible only after publish."""

    def __init__(self, catalog: "ExtensionCatalog", request: PrimitiveRequest,
                 schema: dict, manifest: dict, worker: IsolatedWorker,
                 admission_sample: np.ndarray, artifact_id: str | None = None):
        self.catalog = catalog
        self.request = request
        self.schema = schema
        self.manifest = manifest
        # The primitive's own output on the declared-space admission probe.
        self.admission_sample = admission_sample
        self.artifact_id = artifact_id or f"{request.role}__{request.name}"
        self.worker = worker
        self.closed = False
        self.published = False
        self.constructor = ConstructorProxy(self)

    @property
    def key(self) -> tuple[str, str]:
        return self.request.role, self.request.name

    @property
    def artifact_path(self) -> Path:
        return self.catalog.artifact_dir / self.artifact_id

    def resolve(self, role: str, name: str) -> Callable:
        if self.closed:
            raise SynthesisError("staging handle is closed")
        return self.constructor if (role, name) == self.key else self.catalog.resolve(role, name)

    resolver = resolve

    def capability_declarations(self) -> dict[str, Any]:
        declarations = self.catalog.capability_declarations()
        if self.closed:
            raise SynthesisError("staging handle is closed")
        declarations[f"{self.request.role}/{self.request.name}"] = self.declaration()
        return declarations

    def declaration(self) -> dict[str, Any]:
        result = {"parameters": json.loads(canonical_json(self.schema)),
                  "description": "Run-scoped isolated generated Python primitive",
                  "artifact_id": self.artifact_id}
        if self.request.selector_input_schema is not None:
            result["selector_input_schema"] = json.loads(canonical_json(self.request.selector_input_schema))
        return result

    def publish(self) -> "StagedExtension":
        if self.closed or self.catalog.closed:
            raise SynthesisError("cannot publish a closed stage")
        existing = self.catalog._entries.get(self.key)
        if existing is not None and existing is not self:
            raise SynthesisError("role/name already published in this run")
        marker = self.artifact_path / "PUBLISHED"
        if marker.exists():
            if marker.read_text() != self.artifact_id + "\n":
                raise SynthesisError("published artifact marker is corrupt")
        else:
            with marker.open("x") as stream:
                stream.write(self.artifact_id + "\n")
            marker.chmod(0o444)
        self.catalog._entries[self.key] = self
        self.published = True
        return self

    def abort(self) -> None:
        if self.published:
            raise SynthesisError("cannot abort a published extension; close the catalog")
        self.close()

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.worker.close()

    def __enter__(self) -> "StagedExtension":
        return self

    def __exit__(self, *exc: Any) -> None:
        if not self.published:
            self.abort()


class ExtensionCatalog:
    """A run-local extension catalog. Generated classes never enter REGISTRY."""

    def __init__(self, artifact_dir: str | Path, *, allowed_oracles: set[str] | frozenset[str],
                 representation: Mapping[str, Any],
                 reference_binder: ReferenceBinder | None = None,
                 limits: RuntimeLimits | None = None, runtime_bundle: RuntimeBundle | None = None):
        if not allowed_oracles or any(not isinstance(name, str) or not name for name in allowed_oracles):
            raise SynthesisError("catalog requires the trusted nonempty optimizer oracle allowlist")
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.allowed_oracles = frozenset(allowed_oracles)
        # The task's declared sequence space; penalties and gates are probed
        # against it at admission, so it is part of the admitted configuration.
        self.representation = probe_representation(representation)
        self.reference_binder = reference_binder or ReferenceBinder()
        self.limits = limits or RuntimeLimits()
        self._bundle = runtime_bundle
        self._owns_bundle = runtime_bundle is None
        self._entries: dict[tuple[str, str], StagedExtension] = {}
        self._stages: list[StagedExtension] = []
        self.closed = False

    @property
    def runtime_bundle(self) -> RuntimeBundle:
        if self.closed:
            raise SynthesisError("catalog is closed")
        if self._bundle is None:
            self._bundle = RuntimeBundle()
        return self._bundle

    def stage(self, request: PrimitiveRequest | Mapping[str, Any], *,
              artifact_id: str | None = None, allow_identical_replay: bool = False) -> StagedExtension:
        if self.closed:
            raise SynthesisError("catalog is closed")
        # Snapshot mutable request dictionaries before computing provenance.
        request = PrimitiveRequest.from_dict(request.to_dict() if isinstance(request, PrimitiveRequest) else request)
        from recital.spec.registry import REGISTRY
        if (request.role, request.name) in REGISTRY:
            raise SynthesisError("generated extensions cannot override built-in primitives")
        if (request.role, request.name) in self._entries:
            existing = self._entries[(request.role, request.name)]
            if allow_identical_replay:
                prior = existing.request
                fields = ("code", "unit_test", "parameters", "oracle", "reference_bindings", "selector_input_schema")
                identical = all(getattr(request, field) == getattr(prior, field) for field in fields)
                if identical and parameter_schema(request) == existing.schema:
                    if artifact_id is None or artifact_id == existing.artifact_id:
                        return existing
            raise SynthesisError("role/name already published in this run")
        if request.oracle is not None and request.oracle not in self.allowed_oracles:
            raise SynthesisError("utility oracle is not an approved optimizer source")
        if request.selector_input_schema is not None:
            for declaration in request.selector_input_schema["measurements"].values():
                if not set(declaration["sources"]) <= self.allowed_oracles:
                    raise SynthesisError("selector admission measurement depends on a hidden source")
        schema = parameter_schema(request)
        references = self.reference_binder.resolve(request.reference_bindings)
        configuration = {"role": request.role, "name": request.name,
                         "parameters": request.parameters, "parameter_schema": schema,
                         "oracle": request.oracle,
                         "allowed_oracles": sorted(self.allowed_oracles),
                         "reference_bindings": list(request.reference_bindings),
                         "representation": self.representation,
                         "limits": asdict(self.limits)}
        if request.selector_input_schema is not None:
            configuration["selector_input_schema"] = request.selector_input_schema
        bundle = self.runtime_bundle
        manifest = {"format_version": 1}
        worker = IsolatedWorker(bundle, {**configuration, "code": request.code,
                                        "unit_test": request.unit_test,
                                        "references": references}, self.limits)
        try:
            admission = worker.request("admit")
            count = 3 * len(request.selector_input_schema["conditions"]) if request.selector_input_schema else 3
            sample = _validated_output(request.role, admission.get("sample"), count)
            if type(admission.get("test_count")) is not int or admission["test_count"] <= 0:
                raise SynthesisError("generated admission did not execute tests")
            files = {"code.py": request.code.encode(), "unit_test.py": request.unit_test.encode(),
                     "config.json": canonical_json(configuration),
                     "references.json": canonical_json(references),
                     "manifest.json": canonical_json(manifest)}
            # A corrected implementation under the same name is a new version, so the
            # artifact is addressed by its contents: the repair keeps its own directory
            # and the failed attempt stays readable instead of blocking it.
            digest = hashlib.sha256()
            for name in sorted(files):
                digest.update(name.encode())
                digest.update(files[name])
            stage = StagedExtension(
                self, request, schema, manifest, worker, sample,
                artifact_id=artifact_id or f"{request.role}__{request.name}__{digest.hexdigest()[:12]}",
            )
            self._persist(stage.artifact_path, files)
            self._stages.append(stage)
            return stage
        except BaseException:
            worker.close()
            raise

    def _persist(self, destination: Path, files: dict[str, bytes]) -> None:
        if destination.exists():
            self._verify_files(destination, files)
            return
        temporary = Path(tempfile.mkdtemp(prefix=".stage-", dir=self.artifact_dir))
        try:
            for name, data in files.items():
                path = temporary / name
                with path.open("xb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                path.chmod(0o444)
            try:
                temporary.rename(destination)
            except FileExistsError:
                self._verify_files(destination, files)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    @staticmethod
    def _verify_files(directory: Path, files: dict[str, bytes]) -> None:
        if directory.is_symlink():
            raise SynthesisError("artifact directories cannot be symlinks")
        for name, data in files.items():
            path = directory / name
            if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
                raise SynthesisError(f"immutable extension artifact mismatch: {name}")

    def load(self, artifact_id: str) -> StagedExtension:
        """Replay a published artifact under the current grants and same runtime."""
        if not isinstance(artifact_id, str) or not artifact_id or "/" in artifact_id or ".." in artifact_id:
            raise SynthesisError("load requires a catalog artifact id, never an agent path")
        directory = self.artifact_dir / artifact_id
        if directory.is_symlink():
            raise SynthesisError("artifact directories cannot be symlinks")
        files = {}
        for name in ("code.py", "unit_test.py", "config.json", "references.json", "manifest.json", "PUBLISHED"):
            path = directory / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size > self.limits.max_message_bytes:
                raise SynthesisError("missing or invalid immutable extension artifact")
            files[name] = path.read_bytes()
        manifest = json.loads(files["manifest.json"])
        config = json.loads(files["config.json"])
        references = json.loads(files["references.json"])
        if (manifest.get("format_version") != 1
                or files["PUBLISHED"] != (artifact_id + "\n").encode()):
            raise SynthesisError("extension artifact identity mismatch")
        if (config["allowed_oracles"] != sorted(self.allowed_oracles)
                or config["limits"] != asdict(self.limits)
                or config["representation"] != self.representation
                or self.reference_binder.resolve(config["reference_bindings"]) != references):
            raise SynthesisError("replay grants or runtime configuration changed")
        request = PrimitiveRequest(role=config["role"], name=config["name"],
                                   code=files["code.py"].decode(), unit_test=files["unit_test.py"].decode(),
                                   parameters=config["parameters"], parameter_schema=config["parameter_schema"],
                                   oracle=config["oracle"],
                                   selector_input_schema=config.get("selector_input_schema"),
                                   reference_bindings=tuple(config["reference_bindings"]))
        return self.stage(request, artifact_id=artifact_id)

    def resolve(self, role: str, name: str) -> Callable:
        if self.closed:
            raise SynthesisError("catalog is closed")
        if (role, name) in self._entries:
            return self._entries[(role, name)].constructor
        from recital.spec.registry import get_primitive
        return get_primitive(role, name)

    resolver = resolve

    def capability_declarations(self) -> dict[str, Any]:
        if self.closed:
            raise SynthesisError("catalog is closed")
        return {f"{role}/{name}": stage.declaration() for (role, name), stage in sorted(self._entries.items())}

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            for stage in self._stages:
                stage.close()
            if self._bundle is not None and self._owns_bundle:
                self._bundle.close()

    def __enter__(self) -> "ExtensionCatalog":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
