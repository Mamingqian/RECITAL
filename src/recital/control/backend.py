"""Model execution for one job: train a candidate, evaluate its pool, build a delivery.

The task supplies loaders, score interpretation and qualification. Checkpoints,
generated pools and per-source scores are persisted before dependent work starts.
A candidate's directory holds `pool.json`, `scores/`, `training_receipt.json`, `evaluation.json` and
`delivery_artifact.json`.

A job owns one output directory. Everything shared between jobs of a run
(`initial_policy_sample.json`, `composition_reference/`) is written by the baseline
job before any candidate starts and only read afterwards.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from recital.actions import ExperimentContract
from recital.control.errors import ControlError, ScoringFailed
from recital.control.health import UNIQUE_KMER_MEASUREMENT, HealthRule, assess, summary_entries, with_requirements
from recital.control.jobs import (
    BASELINE_ACTION_ID, INITIAL_CHECKPOINT, JobSpec, file_sha256, write_json,
)
from recital.dna_session import json_safe
from recital.library_selection import SelectionMeasurement, SelectionPlan, select_library_plan
from recital.randomness import isolated_rng
from recital.training.policy_precision import resolve_bf16


_write = write_json


def _read(path):
    return json.loads(Path(path).read_text())


def prediction_pool_binding(pool):
    """Bind predictions to the exact ordered sequences and conditions scored."""
    identity = {key: pool[key] for key in ("row_ids", "sequences", "conditions")}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class EvaluationActionError(ControlError):
    """A measurement or selection step failed, with its partial evidence on disk.

    It is not a ValueError: nothing here converts it into a revision the agent could
    make. The worker reports it as the candidate's failure.
    """


class DeliveryError(ControlError):
    """An assignment or a cached artifact cannot be assembled into a delivery."""


class SpecMismatch(ControlError):
    """A job spec disagrees with the run's own config or contract."""


class LazyOracle:
    def __init__(self, factory):
        self.factory, self.instance = factory, None

    def __call__(self, sequences):
        if self.instance is None:
            self.instance = self.factory()
        return self.instance(sequences)

    def release(self):
        self.instance = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _measure_record(value, *, source, row_ids):
    if isinstance(value, SelectionMeasurement):
        return {
            "values": json_safe(value.values), "sources": list(value.sources),
            "row_ids": list(value.row_ids) if value.row_ids is not None else list(row_ids),
            "status": value.status, "metadata": json_safe(value.metadata),
        }
    if isinstance(value, dict) and "values" in value:
        result = deepcopy(value)
        result.setdefault("sources", [source] if source else [])
        result.setdefault("row_ids", list(row_ids))
        result.setdefault("status", "measured")
        metadata = dict(result.get("metadata") or {})
        for key in list(result):
            if key not in {"values", "sources", "row_ids", "status", "metadata"}:
                metadata[key] = result.pop(key)
        result["metadata"] = metadata
        return json_safe(result)
    return {"values": json_safe(np.asarray(value)), "sources": [source] if source else [],
            "row_ids": list(row_ids), "status": "measured", "metadata": {}}


def summarize_measurements(measurements, conditions, condition_names):
    result = {}
    for name, record in measurements.items():
        values = np.asarray(record["values"], dtype=float)
        groups = {}
        for condition in condition_names:
            rows = np.flatnonzero(np.asarray(conditions) == condition)
            subset = values[rows]
            finite = np.isfinite(subset)
            if subset.ndim == 1:
                observed = subset[finite]
                summary = {"mean": float(observed.mean()) if len(observed) else None,
                           "median": float(np.median(observed)) if len(observed) else None,
                           "q05": float(np.quantile(observed, 0.05)) if len(observed) else None,
                           "q95": float(np.quantile(observed, 0.95)) if len(observed) else None,
                           "standard_error": float(observed.std(ddof=1) / np.sqrt(len(observed)))
                           if len(observed) > 1 else None,
                           "measured_count": int(finite.sum())}
            else:
                summary = {"columns": [{
                    "mean": float(column[np.isfinite(column)].mean()) if np.isfinite(column).any() else None,
                    "median": float(np.median(column[np.isfinite(column)])) if np.isfinite(column).any() else None,
                    "measured_count": int(np.isfinite(column).sum()),
                } for column in subset.T]}
            groups[condition] = {"count": len(rows), **summary}
        result[name] = {"sources": record.get("sources", []), "status": record.get("status"),
                        "metadata": record.get("metadata", {}), "by_condition": groups}
    return result


def _summarized_execution_cost(cost, path):
    """Roll the per-call oracle log up before it reaches an agent's context.

    `oracle_events` holds one record per oracle call, so it scales with training volume.

    The roll-up keeps what a decision needs -- how many calls each oracle took, how long
    they took, how many sequences they scored, and whether any raised -- and names the
    file that still holds every event. A failed call carries its own `outcome`, so it
    lands in its own row; `attempt` is not summarized because `costs.py` sets it to the
    running event index, not to a retry count.
    """
    events = cost.get("oracle_events")
    if not isinstance(events, list):
        return cost
    rolled = {}
    for event in events:
        key = (event.get("oracle"), event.get("phase"), event.get("outcome"))
        row = rolled.setdefault(key, {"oracle": key[0], "phase": key[1], "outcome": key[2],
                                      "calls": 0, "sequences_requested": 0,
                                      "elapsed_seconds": 0.0, "exceptions": 0})
        row["calls"] += 1
        row["sequences_requested"] += int(event.get("sequences_requested") or 0)
        row["elapsed_seconds"] += float(event.get("elapsed_seconds") or 0.0)
        row["exceptions"] += int(event.get("exception_type") is not None)
    summarized = dict(cost)
    summarized["oracle_events"] = {
        "summarized": True, "events": len(events),
        "by_oracle_phase_outcome": sorted(rolled.values(),
                                          key=lambda row: (str(row["oracle"]), str(row["phase"]),
                                                           str(row["outcome"]))),
        "every_event_is_recorded_in": str(path),
    }
    return summarized


def candidate_checkpoint_records(spec: JobSpec, receipt: dict) -> tuple[list[dict], str]:
    """The checkpoints a finished training receipt publishes, as the controller registers them.

    Ids are `<candidate id>:step<N>`; `lineage_updates` adds the step's distance from the
    action start to the start checkpoint's. Returns the records and the end checkpoint's id.
    """
    train = spec.train
    parent = spec.parent_checkpoints[train.start_checkpoint]
    completed = int(receipt["additional_updates_completed"])
    if completed != train.additional_updates:
        raise ValueError("completed training receipt does not match the declared action")
    published = receipt.get("checkpoints", [])
    if isinstance(published, dict):
        published = [{"path": path, "step": int(step)} for step, path in published.items()]
    end_path = str(receipt["end_checkpoint"])
    end_step = int(receipt["action_end_step"])
    if not any(str(record.get("path")) == end_path for record in published):
        published = [*published, {"path": end_path, "step": end_step}]
    records, end_id = [], None
    for record in published:
        step = int(record["step"])
        local_step = step - int(receipt["action_start_step"])
        identifier = f"{spec.candidate.id}:step{step}"
        is_end = str(record["path"]) == end_path
        records.append({
            "id": identifier, "path": str(record["path"]), "parent": train.start_checkpoint,
            "action_id": spec.candidate.action_id, "step": step,
            "lineage_updates": parent.lineage_updates + local_step,
            "full_state_available": bool(record.get("full_state_available", True)),
            "weights_available": True, "recipe": deepcopy(train.recipe),
            "schedule": deepcopy(record.get("schedule", receipt.get("schedule"))),
            "kl_reference": train.kl_reference,
            "training_summary": deepcopy(record.get("training_summary", receipt.get("training_summary", {})
                                                    if is_end else {})),
            "measurement_refs": [],
        })
        if is_end:
            end_id = identifier
    if end_id is None:
        raise ValueError("training did not publish an end checkpoint")
    return records, end_id


def qualification_by_condition(artifact: dict, conditions, required_counts: dict) -> dict:
    """Split a library's qualification into one verdict per condition.

    A requirement counts against a condition when its stratum is that condition or
    `all`. A qualification payload with no requirement list (a task that reports one
    verdict) gives every condition that verdict.
    """
    qualification = artifact["qualification"]
    strata = artifact["selection_report"]["strata"]
    selected = Counter(strata[i] for i in artifact["selected_indices"])
    requirements = qualification.get("requirements")
    result = {}
    for condition in conditions:
        count_met = condition not in required_counts or selected[condition] == required_counts[condition]
        if requirements is None:
            failing, qualified = [], bool(qualification["qualified"]) and count_met
        else:
            failing = [item["requirement_id"] for item in requirements
                       if item["stratum"] in (condition, "all") and item["status"] != "pass"]
            qualified = count_met and not failing
        result[condition] = {"qualified": qualified, "count_met": count_met,
                             "selected": selected[condition], "failing_requirements": failing}
    return result


class SequenceExperimentBackend:
    def __init__(self, runtime: dict[str, Any], *, output: Path, device: str = "cuda", synthesis=None):
        self.runtime = runtime
        self.contract: ExperimentContract = runtime["contract"]
        self.output, self.device = Path(output).resolve(), device
        self.load_policy = runtime["load_policy"]
        self.prompts = runtime["prompts"]
        self.training_sequences = runtime["training_sequences"]
        self.representation = runtime["representation"]
        self.context_extras = runtime.get("context_extras", {})
        self.oracles = {name: LazyOracle(factory) for name, factory in runtime["oracle_factories"].items()}
        if set(self.oracles) != {tool.name for tool in self.contract.tools}:
            raise ValueError("oracle runtime must implement the complete declared non-blind inventory")
        self.synthesis = synthesis

    def actual_training_updates(self, directory):
        receipt = Path(directory) / "training_receipt.json"
        if receipt.exists():
            saved = _read(receipt)
            return int(saved.get("total_actual_updates_spent", saved["additional_updates_completed"]))
        ledger = Path(directory) / "training_calls.jsonl"
        calls = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
        measured = sum(int(call.get("measurements", {}).get("optimizer_steps", {}).get("value") or 0) for call in calls)
        return max([measured, *[int(call.get("cumulative_action_updates", 0)) for call in calls]])

    def training_cost_status(self, directory):
        directory = Path(directory)
        receipt = directory / "training_receipt.json"
        if receipt.exists():
            return _read(receipt).get("cost_status", "measured")
        if (directory / "interrupted_training_cost.json").exists():
            return "lower_bound"
        ledger = directory / "training_calls.jsonl"
        calls = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
        return "lower_bound" if any(call.get("cumulative_status") == "lower_bound" for call in calls) else "measured"

    def training_evidence(self, directory):
        path = Path(directory) / "training_statistics.json"
        if not path.exists():
            return {"status": "unavailable", "reason": "training receipt was committed before summary construction"}
        value = _read(path)
        keys = ("training_sequence_validity", "training_log", "reward_execution", "training_progress",
                "base_policy_kl", "base_policy_kl_max", "reference_policy_kl", "reference_policy_kl_max",
                "base_policy_kl_unavailable_reason", "frac_reward_zero_std", "reward_std", "group_size",
                "condition_repeats", "seed", "train_metrics", "optimizer_steps_completed",
                "optimizer_steps_configured", "training_start", "reward_state_resume", "execution_cost")
        keys = (*keys, "base_policy_sampling", "effective_reward_constants")
        result = {"status": "measured", **{key: deepcopy(value[key]) for key in keys if key in value}}
        if isinstance(result.get("execution_cost"), dict):
            result["execution_cost"] = _summarized_execution_cost(result["execution_cost"], path)
        baseline_path = self.output / "initial_policy_sample.json"
        quality = self.runtime.get("quality_measure")
        if baseline_path.exists() and quality is not None:
            baseline = _read(baseline_path)
            sequences = baseline["sequences"]
            records = {name: _measure_record(value, source=None,
                       row_ids=[f"initial_quality:{i}" for i in range(len(sequences))])
                       for name, value in quality(sequences).items()}
            result["initial_policy_quality"] = {
                "checkpoint": "initial", "population": "initial SFT sampled raw library",
                "scope": "pooled across conditions; original row conditions were not retained",
                "sampling": baseline.get("sampling"), "count": len(sequences),
                "measurements": summarize_measurements(records, ["all"] * len(sequences), ("all",)),
            }
        return result

    def _extras(self, model, tokenizer):
        result = deepcopy(self.context_extras)
        factory = self.runtime.get("context_extras_factory")
        if factory is not None:
            result.update(factory(model, tokenizer))
        return result

    def _release(self, model):
        if hasattr(model, "to"):
            model.to("cpu")
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _seed(self, action_id, phase):
        ordinal = int(action_id.rsplit("_", 1)[-1])
        return int((self.contract.seed + ordinal * 1009 + phase * 1000003) % (2**32))

    @contextmanager
    def _evaluation_policy(self, checkpoint):
        with isolated_rng():
            model, tokenizer = self.load_policy(checkpoint)
            try:
                yield model, tokenizer
            finally:
                self._release(model)

    def _check_spec(self, spec: JobSpec, kind: str) -> dict[str, dict]:
        """Refuse a spec that disagrees with this run; return its checkpoints as plain dicts."""
        if spec.kind != kind:
            raise SpecMismatch(f"a {spec.kind} spec was given to a {kind} entry point")
        if Path(spec.run_dir).resolve() != self.output:
            raise SpecMismatch("the spec's run_dir is not this backend's output")
        if spec.candidate.seed != self.contract.seed:
            raise SpecMismatch(f"the spec's seed {spec.candidate.seed} is not the contract's {self.contract.seed}")
        checkpoints = {name: record.model_dump(mode="json") for name, record in spec.parent_checkpoints.items()}
        if checkpoints[INITIAL_CHECKPOINT]["path"] != str(self.contract.initial_checkpoint):
            raise SpecMismatch("the spec's initial checkpoint is not the contract's")
        if set(spec.evaluation.generation_plan["condition_counts"]) != set(self.contract.conditions):
            raise SpecMismatch("the spec's generation plan does not cover the contract's conditions")
        self.contract.validate_sources(spec.evaluation.scorers)
        if spec.train is not None:
            recipe = spec.train.recipe
            self.contract.validate_sources(recipe["enabled_oracles"], "reward_active")
            self.contract.validate_sources(recipe.get("held_out_oracles", []))
        if spec.evaluation.selection_plan:
            plan = SelectionPlan.from_dict(spec.evaluation.selection_plan)
            self.contract.validate_sources(set(plan.measurement_names) & set(self.oracles), "selection_used")
        return checkpoints

    @staticmethod
    def _directory(spec: JobSpec) -> Path:
        directory = Path(spec.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def train_candidate(self, spec: JobSpec) -> dict:
        """Train one candidate to its end checkpoint and return the training receipt.

        A finished receipt is returned as it is. Otherwise training resumes from the
        newest complete checkpoint of this action if there is one, and the work done
        before an interruption is carried into `total_actual_updates_spent`.
        """
        from recital.spec.composer import ComposedRewardFn, RewardSpec
        from recital.trainer import TrainContext, TRLDAPOTrainer
        from recital.training.action_state import latest_action_checkpoint

        checkpoints = self._check_spec(spec, "train_candidate")
        directory, action_id = self._directory(spec), spec.candidate.action_id
        parameters = spec.train.model_dump(mode="json")
        receipt_path = directory / "training_receipt.json"
        if receipt_path.exists():
            return _read(receipt_path)
        baseline_path = self.output / "initial_policy_sample.json"
        if not baseline_path.exists():
            raise ControlError("initial_policy_sample.json is absent; the baseline job writes it before any candidate")
        source = checkpoints[parameters["start_checkpoint"]]
        seed = self._seed(action_id, 1)
        model, tokenizer = self.load_policy(source["path"])
        extras = self._extras(model, tokenizer)
        baseline = _read(baseline_path)
        if baseline["checkpoint"] != checkpoints["initial"]["path"]:
            raise ValueError("fixed initial-policy reference changed")
        extras["base_policy_sequences"] = baseline["sequences"]
        resolver = self.synthesis.catalog.resolve if self.synthesis is not None else None
        recipe = deepcopy(parameters["recipe"])
        training_action = {key: deepcopy(parameters[key]) for key in (
            "start_mode", "additional_updates", "optimizer_policy", "schedule_policy",
            "reward_state_policy", "randomness_policy", "checkpoint_steps",
        )}
        training_action.update(
            action_id=action_id, start_checkpoint=(None if parameters["start_mode"] == "initial_restart" else source["path"]),
            kl_reference=checkpoints[parameters["kl_reference"]]["path"],
            initial_reference=checkpoints["initial"]["path"],
        )
        recovery = latest_action_checkpoint(directory / "training" / "actions" / action_id)
        if recovery is not None:
            training_action.update(start_mode="crash_recovery", start_checkpoint=recovery)
        recovery_step = int(_read(Path(recovery) / "trainer_state.json")["global_step"]) if recovery else (
            int(source["step"]) if parameters["start_mode"] == "full_state_continuation" else 0)
        active_call = directory / "training_call_active.json"
        if active_call.exists():
            interrupted = _read(active_call)
            spent_lower_bound = max(0, recovery_step - interrupted["start_step"])
            _write(directory / "interrupted_training_cost.json", {
                "total_lower_bound": interrupted["cost_before"] + spent_lower_bound,
                "status": "lower_bound", "reason": "process terminated before the invocation cost receipt; unsaved updates may be unmeasured",
            })
        cost_before = self.actual_training_updates(directory)
        interrupted_cost_path = directory / "interrupted_training_cost.json"
        if interrupted_cost_path.exists():
            cost_before = max(cost_before, _read(interrupted_cost_path)["total_lower_bound"])

        def saved(receipt):
            value = deepcopy(receipt)
            value["total_actual_updates_spent"] = cost_before + int(value["actual_updates_spent"])
            value["cost_status"] = "lower_bound" if interrupted_cost_path.exists() else "measured"
            for checkpoint in value.get("checkpoints", []):
                step_state = Path(checkpoint["path"]) / "trainer_state.json"
                if step_state.exists():
                    step_value = _read(step_state)
                    history = [row for row in step_value.get("log_history", [])
                               if row.get("step", 0) > value["action_start_step"]]
                    checkpoint["training_summary"] = {"log_history": history[-24:],
                                                       "action_start_step": value["action_start_step"],
                                                       "global_step": step_value["global_step"]}
            state_path = Path(value["end_checkpoint"]) / "trainer_state.json"
            if state_path.exists():
                trainer_state = _read(state_path)
                history = [row for row in trainer_state.get("log_history", [])
                           if row.get("step", 0) > value["action_start_step"]]
                value["training_summary"] = {"log_history": history[-24:],
                                             "action_start_step": value["action_start_step"],
                                             "global_step": trainer_state["global_step"]}
            _write(receipt_path, value)

        extras.update(
            tokenizer=tokenizer, seed=seed, evaluation_seed=self._seed(action_id, 2),
            training_action=training_action, training_completed_callback=saved,
            training_set=self.training_sequences, primitive_resolver=resolver,
            condition_names=list(self.contract.conditions),
            model_architecture=type(model), reference_model_architecture=type(model),
            model_loader=self.runtime.get("load_training_model", lambda checkpoint: self.load_policy(checkpoint)[0]),
            audit_oracles=list(self.contract.terminal_blind_sources),
            protocol_condition=self.contract.protocol_condition,
        )
        # The action trainer restores the declared weights and complete state.
        # Keep the configured template on CPU so its generation settings survive.
        self._release(model)
        context = TrainContext(
            policy=model, train_data=None,
            conditioning_prompts=[self.prompts[name] for name in self.contract.conditions],
            token_alphabet=self.representation["alphabet"],
            oracle_fns={name: oracle for name, oracle in self.oracles.items() if name in self.contract.visible_sources},
            extras=extras,
        )
        reward = ComposedRewardFn(
            RewardSpec.from_dict(recipe), training_set=self.training_sequences,
            primitive_resolver=resolver, reference_libraries=self.runtime.get("reference_libraries"),
        )
        trainer = TRLDAPOTrainer(
            output_dir=directory / "training", use_cpu=self.device == "cpu",
            bf16=resolve_bf16(self.runtime, device=self.device),
            max_completion_length=self.representation["max_completion_length"],
        )
        # The marker covers the window in which optimizer updates can happen unseen, so
        # it is written once everything before the trainer has succeeded. A failure
        # earlier than this leaves no unmeasured work and must not downgrade the run's
        # cost provenance to a lower bound for the rest of the experiment.
        _write(active_call, {"start_step": recovery_step, "cost_before": cost_before})
        try:
            trajectory = trainer.run(spec=recipe, reward_fn=reward, context=context,
                                     perform_final_evaluation=False)
            if not receipt_path.exists():
                saved(trajectory["training_action_receipt"])
            _write(directory / "training_statistics.json", {
                key: value for key, value in trajectory.items()
                if key not in {"policy", "oracle_outputs", "sequences", "conditioning", "training_set", "base_policy_sequences"}
            })
            return _read(receipt_path)
        finally:
            if trainer.last_execution_cost is not None:
                path = directory / "training_calls.jsonl"
                with path.open("a") as handle:
                    cost = json_safe(trainer.last_execution_cost)
                    cost["cumulative_action_updates"] = cost_before + int(cost.get("measurements", {}).get("optimizer_steps", {}).get("value") or 0)
                    cost["cumulative_status"] = "lower_bound" if interrupted_cost_path.exists() else "measured"
                    handle.write(json.dumps(cost, allow_nan=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                if active_call.exists():
                    active_call.unlink()
            for oracle in self.oracles.values():
                oracle.release()
            self._release(model)

    def _generate(self, params, *, action_id, directory, checkpoints):
        from recital.trainer import TRLDAPOTrainer

        path = directory / "pool.json"
        if path.exists():
            return _read(path)
        assignment, generation = params["checkpoint_assignment"], params["generation_plan"]
        sequences, conditions, sources = [], [], []
        completed_parts = directory / "generated_parts"
        completed_parts.mkdir(parents=True, exist_ok=True)
        for index, condition in enumerate(self.contract.conditions):
            count = generation["condition_counts"][condition]
            part_path = completed_parts / f"condition_{index}.json"
            if part_path.exists():
                part = _read(part_path)
            else:
                identifier = assignment[condition]
                with self._evaluation_policy(checkpoints[identifier]["path"]) as (model, tokenizer):
                    model.to(self.device)
                    trainer = TRLDAPOTrainer(max_completion_length=self.representation["max_completion_length"],
                                             use_cpu=self.device == "cpu")
                    extras = self._extras(model, tokenizer)
                    metadata = {}
                    if count:
                        with isolated_rng():
                            from recital.runutil import seed_everything
                            seed_everything(self._seed(action_id, 2) + index)
                            seqs, _ = trainer._sample_final_batch(
                                trainer=SimpleNamespace(model=model), tokenizer=tokenizer,
                                conditioning_prompts=[self.prompts[condition]], samples_per_prompt=count,
                                alphabet=self.representation["alphabet"],
                                min_new_tokens=extras.get("min_new_tokens", self.representation["min_length"]),
                                sampling_temperature=generation["temperature"], sampling_top_p=generation["top_p"],
                                sampling_batch_size=generation["batch_size"],
                                codec=extras.get("sequence_codec"),
                                sampling_add_special_tokens=extras.get("sampling_add_special_tokens"),
                                sampling_use_cache=extras.get("sampling_use_cache"),
                                sampling_metadata=metadata,
                            )
                    else:
                        seqs = []
                    part = {"sequences": seqs, "condition": condition, "checkpoint": identifier,
                            "sampling": metadata}
                    _write(part_path, part)
            sequences.extend(part["sequences"])
            conditions.extend([condition] * len(part["sequences"]))
            sources.extend([part["checkpoint"]] * len(part["sequences"]))
        pool = {"id": f"{action_id}:pool", "sequences": sequences, "conditions": conditions,
                "checkpoint_assignment": deepcopy(assignment), "source_checkpoints": sources,
                "row_ids": [f"{action_id}:row{i}" for i in range(len(sequences))],
                "generation_plan": deepcopy(generation)}
        _write(path, pool)
        return pool

    def _valid_rows(self, pool):
        rep = self.representation
        return np.asarray([isinstance(sequence, str) and
                           rep["min_length"] <= len(sequence) <= rep["max_length"] and
                           set(sequence) <= set(rep["alphabet"]) for sequence in pool["sequences"]])

    def _score_record(self, name, full, pool, valid):
        record = _measure_record(full, source=name, row_ids=pool["row_ids"])
        record["status"] = "measured" if np.isfinite(full).all() else "partial"
        record["metadata"] = {"source_version": self.runtime.get("oracle_versions", {}).get(name, name),
                              "pool_binding": prediction_pool_binding(pool),
                              "invalid_sequence_count": int((~valid).sum())}
        return record

    def _score(self, name, pool, directory):
        path = directory / "scores" / f"{name}.json"
        binding = prediction_pool_binding(pool)
        version = self.runtime.get("oracle_versions", {}).get(name, name)
        cached = None
        if path.exists():
            record = _read(path)
            if record["row_ids"] != pool["row_ids"] or record.get("metadata", {}).get("pool_binding") != binding:
                raise ValueError("cached scores refer to a different generated pool")
            if record.get("metadata", {}).get("source_version") != version:
                raise ValueError("cached prediction source changed; use its original version or a new named source")
            cached = np.asarray(record["values"], dtype=float)
            if cached.ndim not in (1, 2) or len(cached) != len(pool["sequences"]):
                raise ValueError("cached scorer output shape is invalid")
        valid = self._valid_rows(pool)
        missing = valid.copy()
        if cached is not None:
            finite = np.isfinite(cached)
            missing &= ~finite if cached.ndim == 1 else ~finite.all(axis=1)
        indices = np.flatnonzero(missing)
        full = cached.copy() if cached is not None else None
        if len(indices):
            try:
                values = np.asarray(self.oracles[name]([pool["sequences"][i] for i in indices]), dtype=float)
            finally:
                self.oracles[name].release()
            if (values.ndim not in (1, 2) or len(values) != len(indices)
                    or (cached is not None and values.shape[1:] != cached.shape[1:])):
                raise ValueError("scorer output shape does not align with its actual inputs")
            if full is None:
                full = np.full((len(valid), *values.shape[1:]), np.nan)
            full[indices] = np.where(np.isfinite(full[indices]), full[indices], values)
        if full is None:
            full = np.full(len(valid), np.nan)
        record = self._score_record(name, full, pool, valid)
        _write(path, record)
        return record

    def _identity_applied(self, selection_plan):
        """Apply the task's declared sequence identity to a selection plan.

        The task defines what counts as one delivered sequence, so a plan inherits
        that definition when it says nothing and is refused when it asks for less.
        Everything else about the plan -- sources, ranking, filters, diversity --
        stays the plan's own.
        """
        strictness = {"none": 0, "exact": 1, "reverse_complement": 2}
        declared = self.contract.delivery_requirements.get("identity") or {}
        plan = dict(selection_plan)
        for field, default in (("deduplication", "exact"), ("exclude_training_copies", True)):
            plan.setdefault(field, declared.get(field, default))
        if strictness[plan["deduplication"]] < strictness.get(declared.get("deduplication"), 0):
            raise ValueError(f"the task declares {declared['deduplication']} sequence identity; "
                             f"a delivery cannot be selected under {plan['deduplication']}")
        if declared.get("exclude_training_copies") and not plan["exclude_training_copies"]:
            raise ValueError("the task declares that a delivery excludes training-set copies")
        return plan

    def _score_sources(self, pool, scorers, scoring_directory):
        """Score the pool with every scorer the job names; a scorer that fails fails the job.

        The training that produced the pool is already on disk, so a retry only scores again. Carrying on without
        a source would leave measurements the controller counts on missing, with nobody to repair them.
        """
        measurements = {}
        for name in scorers:
            try:
                measurements[name] = self._score(name, pool, scoring_directory)
            except Exception as exc:
                raise ScoringFailed(f"scoring {name} failed: {type(exc).__name__}: {exc}") from exc
        return measurements, {}

    def evaluate_pool(self, spec: JobSpec) -> dict:
        """Generate the candidate's pool from its end checkpoint, score it, select, qualify.

        Reads the training receipt in the job's directory. A pool, a score or a
        measurement already on disk is reused, so a repeat call completes an
        interrupted one.
        """
        checkpoints = self._check_spec(spec, "train_candidate")
        directory, action_id = self._directory(spec), spec.candidate.action_id
        receipt = _read(directory / "training_receipt.json")
        records, end_id = candidate_checkpoint_records(spec, receipt)
        checkpoints.update({record["id"]: record for record in records})
        evaluation = spec.evaluation
        params = {"checkpoint_assignment": {condition: end_id for condition in self.contract.conditions},
                  "generation_plan": deepcopy(evaluation.generation_plan)}
        return self._evaluate_pool(spec, params, action_id=action_id, directory=directory, checkpoints=checkpoints)

    def _evaluate_pool(self, spec, params, *, action_id, directory, checkpoints):
        evaluation = spec.evaluation
        pool = self._generate(params, action_id=action_id, directory=directory, checkpoints=checkpoints)
        measurements, errors = self._score_sources(pool, evaluation.scorers, directory)
        return self._measure_and_select(
            pool, measurements=measurements, errors=errors, directory=directory, scoring_directory=directory,
            pool_path=directory / "pool.json", action_id=action_id,
            selection_plan=deepcopy(evaluation.selection_plan), health=evaluation.health)

    def _measure_and_select(self, pool, *, measurements, errors, directory, scoring_directory, pool_path,
                            action_id, selection_plan, health: HealthRule | None):
        raw = {name: np.asarray(record["values"], dtype=float) for name, record in measurements.items()}
        summary = {
            "id": pool["id"], "path": str(pool_path),
            "checkpoint_assignment": pool["checkpoint_assignment"],
            "generation_plan": pool["generation_plan"], "raw_count": len(pool["sequences"]),
            "raw": summarize_measurements(measurements, pool["conditions"], self.contract.conditions),
            "missing_scores": dict(errors),
        }
        result = {"pool": summary, "observed_sources": list(raw), "selection_sources": []}
        try:
            derived = self.runtime["measure"](pool["sequences"], pool["conditions"], raw)
            for name, value in derived.items():
                record = _measure_record(value, source=None, row_ids=pool["row_ids"])
                self.contract.validate_sources(record["sources"])
                if name in measurements:
                    prior = measurements[name]
                    if record["sources"] != prior["sources"] or not np.array_equal(
                        np.asarray(record["values"], dtype=float), np.asarray(prior["values"], dtype=float), equal_nan=True
                    ):
                        raise ValueError("derived measurement cannot change a raw model output")
                    record["metadata"] = {**(prior.get("metadata") or {}), **(record.get("metadata") or {})}
                measurements[name] = record
            summary["raw"] = summarize_measurements(measurements, pool["conditions"], self.contract.conditions)
            snapshot = scoring_directory / "public_measurements.json"
            # The per-row reader serves this file, so a later narrower evaluation of the
            # same pool must not drop the rows an earlier one measured.
            kept = _read(snapshot)["measurements"] if snapshot.exists() else {}
            _write(snapshot, {
                "pool_binding": prediction_pool_binding(pool), "action_id": action_id,
                "measurements": {**kept, **measurements},
            })
            if selection_plan is not None:
                if health is None:
                    raise ValueError("a library is selected under a library-health rule; none was given")
                plan = SelectionPlan.from_dict(self._identity_applied(selection_plan))
                indices, report = select_library_plan(
                    pool["sequences"], pool["conditions"], measurements, plan,
                    visible_sources=self.contract.visible_sources,
                    blind_sources=self.contract.terminal_blind_sources,
                    training_sequences=self.training_sequences, row_ids=pool["row_ids"],
                )
                self.contract.validate_sources(
                    {source for sources in report["measurement_sources"].values() for source in sources},
                    "selection_used",
                )
                # Counts in a raw record are over the pool, so they are relabelled rather
                # than carried into the library, where they would read as its own.
                pool_scope = ("invalid_sequence_count", "unscored_rows", "unscored_value")
                selected_measurements = {name: {
                    **record, "values": np.asarray(record["values"])[indices].tolist(),
                    "row_ids": [pool["row_ids"][i] for i in indices],
                    "metadata": {
                        **{key: value for key, value in (record.get("metadata") or {}).items()
                           if key not in pool_scope},
                        **({"source_pool_counts": {key: value for key, value in record["metadata"].items()
                                                   if key in pool_scope}}
                           if any(key in (record.get("metadata") or {}) for key in pool_scope) else {}),
                    },
                } for name, record in measurements.items()}
                for record in selected_measurements.values():
                    if record["status"] in {"measured", "partial"}:
                        record["status"] = ("measured" if np.isfinite(np.asarray(record["values"], dtype=float)).all()
                                            else "partial")
                selected_sequences = [pool["sequences"][i] for i in indices]
                selected_conditions = [pool["conditions"][i] for i in indices]
                selected_raw = {name: values[indices] for name, values in raw.items()}
                qualification = self.runtime["qualify"](selected_sequences, selected_conditions, selected_raw, report)
                required_counts = self.contract.delivery_requirements.get("required_counts", self.contract.delivery_requirements.get("counts"))
                count_ok = all(selected_conditions.count(condition) == count for condition, count in required_counts.items())
                assessment = self._assess_health(health, plan, pool, measurements, indices, report, required_counts)
                qualification = with_requirements(qualification, assessment)
                qualified = bool(qualification["qualified"] and count_ok)
                selected_summary = summarize_measurements(selected_measurements, selected_conditions, self.contract.conditions)
                selected_summary.update(summary_entries(assessment, "library"))
                summary["raw"].update(summary_entries(assessment, "pool"))
                artifact = {
                    "id": f"{action_id}:delivery", "path": str(directory / "delivery_artifact.json"),
                    "pool_id": pool["id"], "pool_path": str(pool_path),
                    "checkpoint_assignment": pool["checkpoint_assignment"],
                    "selection_plan": plan.to_dict(), "selection_report": report, "health": assessment,
                    "selected_indices": indices.tolist(), "selected_row_ids": [pool["row_ids"][i] for i in indices],
                    "selected": selected_summary, "qualified": qualified,
                    "qualification": {**qualification, "count_requirements_met": count_ok},
                }
                _write(directory / "selection_measurements.json", measurements)
                _write(directory / "delivery_artifact.json", artifact)
                result["delivery"] = artifact
                summary["selection"] = {"artifact_id": artifact["id"], "qualified": qualified,
                                        "selected_count": len(indices), "selected": selected_summary}
                result["selection_sources"] = sorted({source for sources in report["measurement_sources"].values()
                                                       for source in sources})
        except (ValueError, KeyError, TypeError) as exc:
            result["evaluation_error"] = {"type": type(exc).__name__, "message": str(exc)}
            _write(directory / "evaluation.json", result)
            raise EvaluationActionError(str(exc)) from exc
        _write(directory / "evaluation.json", result)
        return result

    @staticmethod
    def _assess_health(rule: HealthRule, plan: SelectionPlan, pool, measurements, indices, report, required_counts):
        """Library health of each condition's legal pool rows and of its selected rows, and the requirement it sets.

        Legality is the task's own per-row `legal` measurement, the one the baseline reference was taken over. A
        task with no `unique_kmer_count` measurement (UNIQUE_KMER_MEASUREMENT) reports that ratio as unavailable.
        """
        if "legal" not in measurements:
            raise ValueError("library health needs the task's per-row `legal` measurement, which it did not emit")
        legal = np.asarray(measurements["legal"]["values"], dtype=float) == 1.0
        kmers = measurements.get(UNIQUE_KMER_MEASUREMENT)
        counts = None if kmers is None else np.asarray(kmers["values"], dtype=float)
        chosen = set(indices.tolist())

        def rows(keep):
            result = {}
            for condition in rule.reference:
                picked = [i for i, value in enumerate(pool["conditions"]) if value == condition and keep(i)]
                result[condition] = ([pool["sequences"][i] for i in picked],
                                     None if counts is None else [float(counts[i]) for i in picked])
            return result

        skipped = Counter(report["strata"][i] for i in report["diversity_skipped_indices"])
        return assess(rule, pool=rows(lambda i: bool(legal[i])), library=rows(lambda i: i in chosen),
                      required=required_counts, skipped=skipped, minimum_distance=plan.minimum_distance)

    def _composition_reference(self):
        """Training rows spread across the conditions, and where their scores are kept.

        This reference tests executable composition, not candidate performance.
        """
        rows = np.linspace(0, len(self.training_sequences) - 1,
                           min(24, len(self.training_sequences)), dtype=int)
        sequences = [self.training_sequences[i] for condition in self.contract.conditions for i in rows]
        conditions = [condition for condition in self.contract.conditions for i in rows]
        pool = {"sequences": sequences, "conditions": conditions,
                "row_ids": [f"reference:{i}" for i in range(len(sequences))]}
        return pool, self.output / "composition_reference"

    def _proposal_screen(self, spec):
        from recital.proposal_screen import ProposalScreen

        pool, directory = self._composition_reference()
        sequences, conditions = pool["sequences"], pool["conditions"]
        scores = {name: np.asarray(self._score(name, pool, directory)["values"], dtype=float)
                  for name in spec["enabled_oracles"]}
        original_columns = self.context_extras.get("oracle_target_indices")
        columns = ({condition: original_columns[self.prompts[condition]] for condition in self.contract.conditions}
                   if original_columns is not None else None)
        return ProposalScreen(
            sequences=sequences, oracle_outputs=scores, conditioning=conditions,
            conditioning_prompts=list(self.contract.conditions), training_set=self.training_sequences,
            reference_libraries=self.runtime.get("reference_libraries", {}), oracle_target_indices=columns,
            require_training_reference=True, reject_reference_constants=False,
        )

    def _required_counts(self):
        requirements = self.contract.delivery_requirements
        return requirements.get("required_counts", requirements.get("counts"))

    def by_condition(self, artifact):
        """`qualification_by_condition` under this run's conditions and required counts."""
        return qualification_by_condition(artifact, list(self.contract.conditions), self._required_counts())

    def reward_eligible_sources(self):
        """Visible sources a recipe may train on."""
        return sorted(tool.name for tool in self.contract.tools
                      if tool.name in self.contract.visible_sources and tool.available
                      and "reward_active" in tool.uses)

    def _initial_policy_sample(self, initial):
        """The initial policy's sampled library, written once per run.

        Drawn by the trainer's own base-policy sampler, on the model and device its
        first training call would use, with the options and seed that call would use:
        `baseline_seed` (default 0), `base_library_samples_per_prompt` (default 128)
        and the runtime's sampling extras. A candidate that finds this file supplies
        it to the trainer instead of sampling.
        """
        from recital.trainer import TrainContext, TRLDAPOTrainer

        path = self.output / "initial_policy_sample.json"
        if path.exists():
            baseline = _read(path)
            if baseline["checkpoint"] != initial["path"]:
                raise ValueError("fixed initial-policy reference changed")
            return baseline
        model, tokenizer = self.load_policy(initial["path"])
        extras = self._extras(model, tokenizer)
        extras.update(tokenizer=tokenizer)
        self._release(model)
        context = TrainContext(
            policy=model, train_data=None,
            conditioning_prompts=[self.prompts[name] for name in self.contract.conditions],
            token_alphabet=self.representation["alphabet"], oracle_fns={}, extras=extras,
        )
        trainer = TRLDAPOTrainer(use_cpu=self.device == "cpu",
                                 max_completion_length=self.representation["max_completion_length"])
        loader = self.runtime.get("load_training_model", lambda checkpoint: self.load_policy(checkpoint)[0])
        sampler = loader(str(initial["path"]))
        import torch
        if self.device == "cpu":
            sampler = sampler.to("cpu")
        elif torch.cuda.is_available():
            sampler = sampler.to("cuda")
        metadata = {}
        try:
            with isolated_rng():
                sequences = trainer._base_policy_library(
                    model=sampler, tokenizer=tokenizer, context=context,
                    samples_per_prompt=int(extras.get("base_library_samples_per_prompt") or 128),
                    seed=int(extras.get("baseline_seed", 0)), reference_identity=None,
                    sampling_metadata=metadata, execution_cost=None,
                )
        finally:
            self._release(sampler)
        baseline = {"checkpoint": initial["path"], "sequences": sequences, "sampling": metadata}
        _write(path, baseline)
        return baseline

    def baseline_pool(self, spec: JobSpec) -> dict:
        """The initial policy's sample, its scored pool, and the composition reference.

        Runs before any candidate. Returns the pool's evaluation, like `evaluate_pool`.
        """
        checkpoints = self._check_spec(spec, "baseline_pool")
        directory = self._directory(spec)
        self._initial_policy_sample(checkpoints[INITIAL_CHECKPOINT])
        params = {"checkpoint_assignment": {condition: INITIAL_CHECKPOINT for condition in self.contract.conditions},
                  "generation_plan": deepcopy(spec.evaluation.generation_plan)}
        result = self._evaluate_pool(spec, params, action_id=BASELINE_ACTION_ID, directory=directory,
                                     checkpoints=checkpoints)
        reference, directory = self._composition_reference()
        for name in self.reward_eligible_sources():
            self._score(name, reference, directory)
        return result

    def _verified_scores(self, directory, pool, name):
        """A candidate's cached scores for `name`, checked against its own pool."""
        path = Path(directory) / "scores" / f"{name}.json"
        if not path.exists():
            raise DeliveryError(f"{path} is absent")
        record = _read(path)
        metadata = record.get("metadata") or {}
        if record["row_ids"] != pool["row_ids"] or metadata.get("pool_binding") != prediction_pool_binding(pool):
            raise DeliveryError(f"cached scores {path} refer to a different generated pool")
        if metadata.get("source_version") != self.runtime.get("oracle_versions", {}).get(name, name):
            raise DeliveryError(f"cached scores {path} come from a different prediction source")
        values = np.asarray(record["values"], dtype=float)
        if values.ndim not in (1, 2) or len(values) != len(pool["sequences"]):
            raise DeliveryError(f"cached scores {path} have an invalid shape")
        return values

    def assemble_delivery(self, assignment: dict[str, str], candidate_dirs: dict[str, Path],
                          selection_plan: dict, output: Path, *, health: HealthRule) -> dict:
        """Deliver each condition from the candidate assigned to it.

        `assignment` maps every condition to a candidate id and `candidate_dirs` maps
        candidate ids to their job directories. Each condition's rows come from that
        candidate's pool, row ids and source checkpoints unchanged; scores come from the
        candidates' cached files after their pool binding and source version are
        checked, so no model runs. The derived measurements are recomputed on the
        concatenated pool, the selection plan is applied to it, and a library that
        qualifies under the task's requirements and `health` is frozen into `output` as
        `final_sequences.parquet` and `delivery.json`.

        Returns `{"artifact", "frozen", "by_condition", "assignment"}`; an unqualified
        library is written under `output/delivery` and not frozen.
        """
        conditions = list(self.contract.conditions)
        if set(assignment) != set(conditions):
            raise DeliveryError("the assignment must name each condition exactly once")
        missing = sorted(set(assignment.values()) - set(candidate_dirs))
        if missing:
            raise DeliveryError(f"no directory for candidate(s) {missing}")
        output = Path(output).resolve()
        directory = output / "delivery"
        directory.mkdir(parents=True, exist_ok=True)
        candidates = sorted(set(assignment.values()))
        pools, end_ids, observed = {}, {}, []
        for candidate in candidates:
            source = Path(candidate_dirs[candidate])
            pool = _read(source / "pool.json")
            owners = set(pool["checkpoint_assignment"].values())
            if len(owners) != 1:
                raise DeliveryError(f"candidate {candidate} did not generate from a single end checkpoint")
            pools[candidate], end_ids[candidate] = pool, owners.pop()
            observed.append(set(_read(source / "evaluation.json")["observed_sources"]))
        sources = sorted(set.intersection(*observed) & set(self.contract.visible_sources))
        plans = {json.dumps(pool["generation_plan"], sort_keys=True) for pool in pools.values()}
        if len(plans) != 1:
            raise DeliveryError("the assigned candidates were generated under different plans")
        rows = {key: [] for key in ("sequences", "conditions", "row_ids", "source_checkpoints")}
        picks = {}
        for condition in conditions:
            pool = pools[assignment[condition]]
            picks[condition] = [i for i, value in enumerate(pool["conditions"]) if value == condition]
            for key in rows:
                rows[key].extend(pool[key][i] for i in picks[condition])
        delivery_pool = {"id": "delivery:pool", **rows,
                         "checkpoint_assignment": {condition: end_ids[assignment[condition]] for condition in conditions},
                         "generation_plan": deepcopy(pools[candidates[0]]["generation_plan"])}
        if len(set(delivery_pool["row_ids"])) != len(delivery_pool["row_ids"]):
            raise DeliveryError("the assigned pools share row identities")

        def gather(name):
            parts = [self._verified_scores(candidate_dirs[assignment[condition]], pools[assignment[condition]],
                                           name)[picks[condition]] for condition in conditions]
            if len({part.shape[1:] for part in parts}) != 1:
                raise DeliveryError(f"candidates scored {name} with different output shapes")
            return np.concatenate(parts)

        valid = self._valid_rows(delivery_pool)
        measurements = {name: self._score_record(name, gather(name), delivery_pool, valid) for name in sources}
        _write(directory / "pool.json", delivery_pool)
        for name, record in measurements.items():
            _write(directory / "scores" / f"{name}.json", record)
        _write(directory / "assignment.json", {
            "assignment": dict(assignment), "checkpoint_assignment": delivery_pool["checkpoint_assignment"],
            "candidates": {candidate: {"directory": str(candidate_dirs[candidate]),
                                       "end_checkpoint_id": end_ids[candidate], "pool_id": pools[candidate]["id"],
                                       "pool_sha256": file_sha256(Path(candidate_dirs[candidate]) / "pool.json")}
                           for candidate in candidates},
        })

        result = self._measure_and_select(
            delivery_pool, measurements=measurements, errors={}, directory=directory, scoring_directory=directory,
            pool_path=directory / "pool.json", action_id="delivery",
            selection_plan=deepcopy(selection_plan), health=health)
        artifact = result["delivery"]
        frozen = bool(artifact["qualified"])
        if frozen:
            self._freeze_library(artifact, output=output)
        return {"artifact": artifact, "frozen": frozen, "assignment": dict(assignment),
                "by_condition": self.by_condition(artifact)}

    def _freeze_library(self, artifact, *, output):
        import pandas as pd

        saved = _read(artifact["path"])
        if saved != json_safe(artifact) or not saved["qualified"]:
            raise ValueError("delivery artifact changed or was not qualified")
        pool = _read(saved["pool_path"])
        indices = saved["selected_indices"]
        if [pool["row_ids"][i] for i in indices] != saved["selected_row_ids"]:
            raise ValueError("delivery rows no longer match the generated pool")
        measurements = _read(Path(artifact["path"]).parent / "selection_measurements.json")
        checked_indices, checked_report = select_library_plan(
            pool["sequences"], pool["conditions"], measurements, saved["selection_plan"],
            visible_sources=self.contract.visible_sources,
            blind_sources=self.contract.terminal_blind_sources,
            training_sequences=self.training_sequences, row_ids=pool["row_ids"],
        )
        if checked_indices.tolist() != indices or json_safe(checked_report) != saved["selection_report"]:
            raise ValueError("delivery pool or its measurements changed after evaluation")
        frame = pd.DataFrame({"row_id": saved["selected_row_ids"],
                              "sequence": [pool["sequences"][i] for i in indices],
                              "condition": [pool["conditions"][i] for i in indices],
                              "checkpoint": [pool["source_checkpoints"][i] for i in indices]})
        path = output / "final_sequences.parquet"
        if path.exists():
            if not pd.read_parquet(path).equals(frame):
                raise ValueError("frozen sequences differ from the approved artifact")
        else:
            temporary = path.with_suffix(".parquet.tmp")
            frame.to_parquet(temporary, index=False)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            temporary.replace(path)
        _write(output / "delivery.json", {**saved, "outcome": "qualified_delivery", "frozen": True,
                                           "final_sequences": str(path), "blind_audit": {"status": "not_run"}})
