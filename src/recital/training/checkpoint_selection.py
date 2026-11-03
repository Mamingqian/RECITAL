"""Select scheduled checkpoints across an uninterrupted or resumed training run.

The callback never trains, reloads weights, or constructs an optimizer,
scheduler, reference model or RNG stream. It samples scheduled policy states
through injected task adapters and uses the shared SuccessSpec selector.

Adapter protocol::

    evaluator(model, step) -> {
        "measurements": {metric: {stratum: Measurement | dict}},
        "library": path_reference(library_path),
        "library_count": number_of_raw_candidates,
        "raw_artifacts": {name: path_reference(path)},  # optional
    }
    checkpoint_writer(model, step, *, evaluation) -> {
        "step": step,
        "checkpoint": path_reference(checkpoint_file_or_directory),
    }

The writer saves the supplied model without mutating it and uses a separate
path per step. Records contain paths, steps and measurements. HF checkpoints
also persist the schedule cursor and existing observations; normal Trainer
resume restores the optimizer, scheduler and RNG independently.
"""
from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from transformers import TrainerCallback, TrainerControl, TrainerState

from recital.success import Measurement, SuccessSpec, assess_candidate, select_candidate
from recital.randomness import evaluation_mode, isolated_rng
from recital.terminal_eligibility import EvidenceEligibility, assess_evidence
from recital.training.callback_state import CheckpointStateMixin


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def path_reference(path: str | Path) -> dict[str, Any]:
    """Resolve an existing file/nonempty directory without reading model bytes."""
    path = Path(path).resolve(strict=True)
    if path.is_file():
        kind = "file"
    elif path.is_dir():
        files = sorted(entry.relative_to(path).as_posix() for entry in path.rglob("*") if entry.is_file())
        if not files:
            raise ValueError("artifact directory is empty")
        return {"path": str(path), "kind": "directory", "files": files}
    else:
        raise ValueError("artifact must be a file or nonempty directory")
    return {"path": str(path), "kind": kind}


def _verify_reference(raw: Any) -> dict[str, Any]:
    if isinstance(raw, (str, Path)):
        return path_reference(raw)
    if not isinstance(raw, Mapping) or "path" not in raw:
        raise ValueError("artifact reference requires a path")
    observed = path_reference(raw["path"])
    if raw.get("kind", observed["kind"]) != observed["kind"]:
        raise ValueError(f"artifact kind changed: {raw['path']}")
    if any(not (Path(observed["path"]) / filename).is_file() for filename in raw.get("files", [])):
        raise ValueError(f"artifact directory is missing a saved file: {raw['path']}")
    return observed


def _measurements(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("measurements must be a metric mapping")
    result = {}
    for metric, groups in raw.items():
        if not isinstance(metric, str) or not isinstance(groups, Mapping):
            raise ValueError("measurements require metric names and stratum mappings")
        result[metric] = {}
        for group, value in groups.items():
            if not isinstance(group, str):
                raise ValueError("measurement stratum names must be strings")
            try:
                entry = Measurement.model_validate(value)
                if entry.value is not None and not math.isfinite(entry.value):
                    raise ValueError("nonfinite measurement")
                if entry.status == "measured" and (entry.value is None or entry.count == 0):
                    raise ValueError("empty measurement")
            except ValueError as exc:
                entry = Measurement(status="error", reason=str(exc))
            result[metric][group] = entry.model_dump()
    return result


class CheckpointSelectionCallback(CheckpointStateMixin, TrainerCallback):
    """Evaluate exact scheduled steps, saving each library/checkpoint together.

    ``unknown_policy`` is mandatory and applies to missing/invalid required or
    ranking evidence, undersampling, adapter errors, artifact drift and missed
    scheduled steps. Unknowns never qualify. Biological failures alone do not
    stop training. Stop-on-qualification is a separately declared run policy.

    ``online_sources`` must come from the caller's protocol role boundary; the
    callback cannot sandbox arbitrary Python closures. Only one process is
    supported: distributed state collection/writing requires a separate adapter.
    """

    def __init__(
        self, *, evaluator: Callable[..., Mapping[str, Any]],
        checkpoint_writer: Callable[..., Mapping[str, Any]],
        success_spec: SuccessSpec | Mapping[str, Any], scheduled_steps: Sequence[int],
        stop_when_qualified: bool, unknown_policy: Literal["stop", "continue"],
        online_sources: Sequence[str], forbidden_sources: Sequence[str] = (),
        evidence_policy: EvidenceEligibility | None = None, require_heldout: bool = False,
        required_held_out_oracles: Sequence[str] = (),
    ) -> None:
        steps = tuple(scheduled_steps)
        if not steps or any(type(step) is not int or step < 1 for step in steps) or any(
            right <= left for left, right in zip(steps, steps[1:])
        ):
            raise ValueError("scheduled_steps must be positive, strictly increasing integers")
        if type(stop_when_qualified) is not bool or unknown_policy not in {"stop", "continue"}:
            raise ValueError("explicit stop_when_qualified and unknown_policy are required")
        if not callable(evaluator) or not callable(checkpoint_writer):
            raise ValueError("evaluator and checkpoint_writer must be callable adapters")
        raw = success_spec.model_dump() if isinstance(success_spec, SuccessSpec) else success_spec
        self._spec = SuccessSpec.model_validate(copy.deepcopy(raw))
        self._spec.validate_information_access(
            known_sources=set(online_sources), audit_sources=set(forbidden_sources),
            private_sources=set(),
        )
        self._evidence_policy = evidence_policy or EvidenceEligibility()
        self._require_heldout = require_heldout
        self._required_held_out_oracles = tuple(required_held_out_oracles)
        self._steps = steps
        self._stop_when_qualified = stop_when_qualified
        self._unknown_policy = unknown_policy
        self._evaluator = evaluator
        self._writer = checkpoint_writer
        self._records: list[dict[str, Any]] = []
        self._invalidated: dict[int, str] = {}
        self._stop_evidence: dict[str, Any] | None = None
        self._last_actual_step = 0
        self._next = 0

    def _evidence_contract(self) -> dict[str, Any]:
        return {"policy": self._evidence_policy.to_dict(), "require_heldout": self._require_heldout,
                "required_held_out_oracles": list(self._required_held_out_oracles)}

    def state_dict(self) -> dict[str, Any]:
        return {"records": self._records, "invalidated": self._invalidated,
                "evidence_contract": _json_copy(self._evidence_contract()),
                "stop_evidence": self._stop_evidence,
                "last_actual_step": self._last_actual_step}

    def load_state_dict(self, state: dict | None, *, checkpoint_step: int) -> None:
        if state is not None:
            if state.get("evidence_contract") is not None and state["evidence_contract"] != _json_copy(self._evidence_contract()):
                raise ValueError("checkpoint evidence contract changed on recovery")
            self._records = _json_copy(state["records"])
            self._invalidated = {int(step): reason for step, reason in state["invalidated"].items()}
            self._stop_evidence = _json_copy(state["stop_evidence"])
            if state["last_actual_step"] != checkpoint_step or any(
                row["actual_step"] > checkpoint_step for row in self._records
            ):
                raise ValueError("selection observations exceed or differ from the saved optimizer step")
        self._last_actual_step = checkpoint_step
        recorded = {row["scheduled_step"] for row in self._records}
        # Missing past observations cannot be recovered by evaluating the
        # current model and labelling it with a prior step.
        missing = [step for step in self._steps if step <= checkpoint_step and step not in recorded]
        self._records.extend(self._missed(step, checkpoint_step, "checkpoint lacks saved selection evidence")
                             for step in missing)
        self._records.sort(key=lambda row: row["scheduled_step"])
        self._next = sum(step <= checkpoint_step for step in self._steps)
        if missing and self._unknown_policy == "stop":
            self._stop_evidence = {"step": checkpoint_step, "reason": "operational_unknown",
                                   "unknown_steps": missing}

    def _missed(self, scheduled_step: int, actual_step: int, reason: str) -> dict:
        return {"scheduled_step": scheduled_step, "actual_step": actual_step,
                "status": "unknown", "qualified": False, "measurements": {},
                "library_count": 0, "artifacts": {}, "errors": [reason],
                "assessment": assess_candidate(self._spec, {}).to_dict()}

    def _inspect_measurements(self, measurements: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        assessment = assess_candidate(self._spec, measurements).to_dict()
        unknowns = [f"{item['metric']}/{item['stratum']}: {item['reason']}"
                    for item in assessment["evidence"] if item["status"] == "unknown"]
        for metric, groups in measurements.items():
            unknowns.extend(f"{metric}/{group}: {entry['reason'] or entry['status']}"
                            for group, entry in groups.items() if entry["status"] != "measured")
        for criterion in self._spec.ranking:
            for group in criterion.strata:
                value = measurements.get(criterion.metric, {}).get(group)
                if value is None or value["status"] != "measured" or value["count"] < 1:
                    unknowns.append(f"ranking {criterion.metric}/{group} is unavailable")
        return assessment, list(dict.fromkeys(unknowns))

    def _evaluate(self, model: Any, step: int) -> dict[str, Any]:
        record = {
            "scheduled_step": step, "actual_step": step, "status": "unknown",
            "measurements": {}, "library_count": 0, "errors": [], "artifacts": {},
            "assessment": assess_candidate(self._spec, {}).to_dict(), "qualified": False,
        }
        try:
            if model is None:
                raise ValueError("model is unavailable at scheduled evaluation")
            with isolated_rng(), evaluation_mode(model):
                raw = self._evaluator(model, step)
                if not isinstance(raw, Mapping):
                    raise ValueError("evaluator must return a mapping")
                record["measurements"] = _measurements(raw.get("measurements"))
                evidence = raw.get("statistics") or {}
                flags = raw.get("flags") or []
                if not isinstance(evidence, Mapping) or not isinstance(flags, list) or any(not isinstance(v, str) for v in flags):
                    raise ValueError("checkpoint evidence requires a statistics mapping and flag names")
                record["statistics"] = _json_copy(evidence)
                record["flags"] = list(flags)
                assessment, errors = self._inspect_measurements(record["measurements"])
                record.update(assessment=assessment, errors=errors)
                count = raw.get("library_count")
                if type(count) is not int or count < 0:
                    raise ValueError("library_count must be an explicit nonnegative integer")
                record["library_count"] = count
                if count == 0:
                    record["errors"].append("evaluated library is empty")
                library = _verify_reference(raw.get("library"))
                raw_refs = {name: _verify_reference(ref)
                            for name, ref in dict(raw.get("raw_artifacts") or {}).items()}
                record["artifacts"] = {"library": library, **{
                    f"raw:{name}": ref for name, ref in raw_refs.items()
                }}
                evaluation = {
                    "step": step,
                    "measurements": record["measurements"], "library": library,
                    "library_count": count, "raw_artifacts": raw_refs,
                    "statistics": record["statistics"], "flags": record["flags"],
                }
                record["evaluation_completed"] = True
                written = self._writer(
                    model, step, evaluation=_json_copy(evaluation)
                )
                if not isinstance(written, Mapping) or type(written.get("step")) is not int or (
                    written["step"] != step
                ):
                    raise ValueError("writer returned a different checkpoint step")
                checkpoint = _verify_reference(written.get("checkpoint"))
                primary_paths = {library["path"], checkpoint["path"]}
                prior_paths = {ref["path"] for prior in self._records
                               for name, ref in prior["artifacts"].items()
                               if name in {"library", "checkpoint"}}
                if len(primary_paths) != 2 or primary_paths & prior_paths:
                    for previous in self._records:
                        if any(ref["path"] in primary_paths for ref in previous["artifacts"].values()):
                            self._invalidated[previous["scheduled_step"]] = "artifact path reused by a later step"
                    raise ValueError("each library and checkpoint needs a separate path per step")
                refs = {"library": library, "checkpoint": checkpoint, **{
                    f"raw:{name}": ref for name, ref in raw_refs.items()
                }}
                for ref in refs.values():
                    _verify_reference(ref)
                binding = {
                    "evaluation": evaluation, "checkpoint": checkpoint,
                }
                record.update(
                    artifacts=refs, binding=binding,
                    qualified=assessment["status"] == "pass" and not record["errors"],
                    status="unknown" if record["errors"] else assessment["status"],
                )
        except Exception as exc:  # noqa: BLE001 - operational failures remain auditable unknowns
            record["errors"].append(f"{type(exc).__name__}: {exc}")
            record["qualified"] = False
            record["status"] = "unknown"
        return record

    def _selection(self) -> dict[str, Any]:
        views = copy.deepcopy(self._records)
        for record in views:
            step = record["scheduled_step"]
            assessment, errors = self._inspect_measurements(record["measurements"])
            record["assessment"] = assessment
            record["status"] = "unknown" if errors or record["errors"] else assessment["status"]
            evidence = assess_evidence(SimpleNamespace(statistics=record.get("statistics") or {}, flags=record.get("flags") or []),
                self._evidence_policy, require_heldout=self._require_heldout,
                required_held_out_oracles=self._required_held_out_oracles,
                required_strata=tuple(dict.fromkeys(group for criterion in self._spec.ranking for group in criterion.strata)))
            record["evidence_assessment"] = evidence
            record["qualified"] = bool(record.get("binding") and assessment["status"] == "pass"
                and not errors and not record["errors"] and evidence["eligible"])
            if not evidence["eligible"]:
                record["status"] = "unknown" if evidence["state"] == "unavailable" else "fail"
            if record.get("binding") and step not in self._invalidated:
                try:
                    for ref in record["artifacts"].values():
                        _verify_reference(ref)
                except Exception as exc:  # noqa: BLE001 - missing files are unusable
                    self._invalidated[step] = f"{type(exc).__name__}: {exc}"
            if step in self._invalidated:
                record["status"], record["qualified"] = "unknown", False
                record["errors"].append(self._invalidated[step])
                record["artifact_status"] = "missing"
            else:
                record["artifact_status"] = "available" if record.get("binding") else "unbound"
        qualified = [record for record in views if record["qualified"]]
        decision = select_candidate([
            SimpleNamespace(iter=record["actual_step"], candidate_status="scored",
                            statistics={"candidate_measurements": record["measurements"]})
            for record in qualified
        ], self._spec)
        selected = next((record for record in qualified
                         if record["actual_step"] == decision.selected_iteration), None)
        return {
            "records": views, "selected": selected,
            "selected_step": selected["actual_step"] if selected else None,
            "selected_checkpoint": selected["artifacts"]["checkpoint"] if selected else None,
            "status": "qualified" if selected else "no_qualified_checkpoint",
        }

    def summary(self) -> dict[str, Any]:
        """Return a detached, JSON-safe evidence manifest, rechecking artifacts."""
        # Read first qualification off the SAME rechecked views the selection
        # uses. self._records holds the pre-invalidation state -- _selection only
        # mutates a deepcopy -- so reading it could report a
        # first_qualification_step whose own records[i]["qualified"] is False,
        # and attach first_qualification_cost to that dead step.
        selection = self._selection()
        views = selection["records"]
        first_index = next((index for index, record in enumerate(views)
                            if record["qualified"]), None)
        first = views[first_index] if first_index is not None else None
        prefix = views[:first_index + 1] if first_index is not None else []
        return _json_copy({
            "schema_version": 2, "scope": "one_candidate_training_run",
            "success_spec": self._spec.model_dump(mode="json"),
            "resume_status": getattr(self, "restore_status", "not_resumed"),
            "scheduled_steps": self._steps, "actual_steps_completed": self._last_actual_step,
            "pending_steps": self._steps[self._next:], "stop_when_qualified": self._stop_when_qualified,
            "unknown_policy": self._unknown_policy, "stop_requested": self._stop_evidence is not None,
            # The first qualification recorded over the run is distinct from the
            # currently selected, available checkpoint after further training.
            "first_qualification_step": first["actual_step"] if first else None,
            "first_qualification_checkpoint": first["artifacts"].get("checkpoint") if first else None,
            "first_qualification_cost": {
                "optimizer_global_step": first["actual_step"],
                "completed_library_evaluations": sum(item.get("evaluation_completed", False) for item in prefix),
                "raw_candidates_evaluated": sum(item["library_count"] for item in prefix),
                "scope": "historical first qualification; optimizer step and callback evaluation counts only",
            } if first else None,
            "stop_evidence": self._stop_evidence, **selection,
        })

    def on_step_end(
        self, args: Any, state: TrainerState, control: TrainerControl, **kwargs: Any,
    ) -> TrainerControl:
        step = state.global_step
        if type(step) is not int or step < self._last_actual_step:
            raise ValueError("Trainer global_step must be a nondecreasing integer")
        self._last_actual_step = step
        if self._stop_evidence is not None:
            control.should_training_stop = True
            return control
        if self._next >= len(self._steps) or step < self._steps[self._next]:
            return control
        if getattr(args, "world_size", 1) != 1:
            raise ValueError("checkpoint selection requires a single-process evaluator/writer")
        current = []
        while self._next < len(self._steps) and self._steps[self._next] < step:
            missed = self._steps[self._next]
            current.append(self._missed(missed, step, "scheduled evaluation was missed; no checkpoint was evaluated"))
            self._next += 1
        if self._next < len(self._steps) and self._steps[self._next] == step:
            current.append(self._evaluate(kwargs.get("model"), step))
            self._next += 1
        self._records.extend(current)
        selection = self._selection()
        unknown_steps = [record["scheduled_step"] for record in selection["records"]
                         if record["scheduled_step"] in {entry["scheduled_step"] for entry in current}
                         and record["status"] == "unknown"]
        if self._invalidated:
            unknown_steps.extend(self._invalidated)
        if self._unknown_policy == "stop" and unknown_steps:
            self._stop_evidence = {"step": step, "reason": "operational_unknown", "unknown_steps": sorted(set(unknown_steps))}
        elif self._stop_when_qualified and selection["selected"] is not None:
            self._stop_evidence = {
                "step": step, "reason": "qualified_checkpoint",
                "selected_step": selection["selected_step"],
                "selected_checkpoint": selection["selected_checkpoint"],
            }
        if self._stop_evidence is not None:
            control.should_training_stop = True
        return control
