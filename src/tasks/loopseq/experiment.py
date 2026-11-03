"""Loop-seq runtime: assets, immutable targets, measurements and quality checks."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from recital.actions import ToolCapability
from recital.experiment_tasks import (
    asset_manifest, counts_from_config, delivery_success, dna_quality_definitions,
    dna_quality_measurements, make_contract, measurement,
    oracle_factory, path_from_root, qualification, declared_audit_statistic, declared_source_role, summarize_audit,
)
from .adapter import LoopseqAdapter
from .evaluation import LoopseqSuccessEvaluator, canonical_dna, interval_objective
from .sft import _resolve_asset_path, resolve_asset_root


def build_runtime(config: dict, repo_root: Path, output: Path, device: str) -> dict:
    repo_root = Path(repo_root).resolve()
    adapter = LoopseqAdapter(repo_root=repo_root, contract_path=Path(config["contract_path"]),
                             initial_spec_path=Path(config.get("initial_spec_path", "configs/loopseq_initial_spec.json")))
    declared = adapter.contract()
    backend = resolve_asset_root(repo_root)
    training_record = declared["data"]["success_measurement"]["training_reference"]
    training_record["path"] = str(_resolve_asset_path(training_record["path"], backend=backend, repo_root=repo_root))
    evaluator = LoopseqSuccessEvaluator.from_contract(declared, repo_root=repo_root)
    if evaluator is None or evaluator.quality_thresholds is None or evaluator.training_identities is None:
        raise ValueError("Loop experiment requires its complete frozen quality/training references")
    raw_initial = (declared["data"].get("initial_reward_spec") if "initial_spec_path" not in config else None)
    if raw_initial is None:
        raw_initial = json.loads(path_from_root(adapter.initial_spec_path, repo_root).read_text())
    initial = adapter.compile_initial_recipe(raw_initial, contract=declared, evaluator=evaluator)
    conditions = tuple(evaluator.groups)
    prompts = {group: prompt for prompt, group in evaluator.conditioning_groups.items()}
    entries = {entry["id"]: entry for entry in declared["oracles"]["candidate_pool"]}
    # Loop-seq states the per-candidate role in the candidate pool and names the
    # sentinels it reserves in `self_validation`; both are the contract's word.
    fold = (declared.get("self_validation") or {}).get("primary_fold") or {}
    required = next((fold[key] for key in ("required_sentinels", "required_held_out_oracles")
                     if fold.get(key) is not None), ())
    blind = tuple(name for name, item in entries.items() if item.get("use_for") in {"audit", "optional_audit"})
    # `feedback` is the candidate-pool entries declared use_for == "sentinel"; they are
    # hidden from the agent unless protocol_condition is self_validating.
    feedback = tuple(name for name, item in entries.items()
                     if item.get("use_for") == "sentinel")
    visible = set(entries) - set(blind)
    if config.get("protocol_condition", "self_validating") != "self_validating":
        visible -= set(feedback)
    public_evaluator = replace(evaluator,
        oracle_boundaries={k: v for k, v in evaluator.oracle_boundaries.items() if k in visible},
        oracle_objectives={k: v for k, v in evaluator.oracle_objectives.items() if k in visible})
    success = delivery_success(declared["success"], counts_from_config(config, conditions))
    task = {
        "name": "loopseq", "description": "Design 50 bp DNA for low, mid and high cyclizability.",
        "target_objective": "Improve agreement with the requested raw-coordinate target bands across available predictors while retaining qualified sequence quality. Compare model support and choose the final method using evidence.",
        "biological_constraints": ["Exactly 50 uppercase A/C/G/T bases", "Declared conditional complexity, diversity and training-overlap requirements apply to the delivered library"],
        "reward_target_objectives": public_evaluator.reward_targets(sorted(visible)),
        "measurement_semantics": {"utility/<source>": "Fixed negative Huber distance to the requested raw-coordinate interval, clamped to zero inside it, so every in-target row scores 0 and ranking on it decides nothing", "margin/<source>": "Signed distance from the nearest requested interval bound, inward, in units of the declared target scale: positive inside, negative outside, higher is better. Ranking on this separates rows that utility/<source> ties", "hit/<source>": "Binary membership in the requested half-open interval", "oracle_margin/<source>": "The delivered library's per-condition median of margin/<source>. It is the declared ranking's second criterion, applied after oracle_hit/<source>. The mid band is two-sided, so its margin is bounded above by half the band width; the low and high bands are half-open, so their margin has no upper bound", "raw_scores": "Separate predictor coordinates; do not average without explicit scale treatment"},
    }
    # The contract's own note for each source. Only entries that carry a note are kept,
    # so the fallback below applies to an entry with `notes: ""` (the schema default);
    # an empty string is not a valid `ToolCapability` description.
    descriptions = {name: " ".join(entry["notes"].split())
                    for name, entry in entries.items() if entry.get("notes")}
    tools = [ToolCapability(name, descriptions.get(name, "Declared cyclizability predictor"),
                            declared_role=declared_source_role(
                                name, required=required,
                                use_for=(entries.get(name) or {}).get("use_for")),
                            units=entries[name].get("score_units", "predictor raw coordinates"),
                            direction="condition-dependent interval") for name in entries if name not in blind]
    sft = _resolve_asset_path(config.get("sft_checkpoint", adapter.sft_recipe().checkpoint), backend=backend, repo_root=repo_root)
    training = declared["data"]["success_measurement"]["training_reference"]
    data = _resolve_asset_path(training["path"], backend=backend, repo_root=repo_root)
    frame = pd.read_parquet(data, columns=[training["sequence_column"], training["split_column"]])
    sequences = frame.loc[frame[training["split_column"]].eq(training["split_value"]), training["sequence_column"]].tolist()
    bounds = deepcopy(declared["recipe_capabilities"]["editable_hyperparameters"])
    # A proposal carries no steps and no condition_repeats (control/recipes.py EDITED_HYPERPARAMETERS), so the
    # agent is shown no bound for either; a candidate keeps its parent's condition_repeats.
    for name in ("steps", "condition_repeats"):
        bounds.pop(name, None)
    contract = make_contract(config, task=task, conditions=conditions, initial_checkpoint=sft,
        initial_recipe=initial, tools=tools, feedback_sources=feedback, blind_sources=blind,
        success=success, bounds=bounds,
        identity={key: declared["delivery"][key] for key in ("deduplication", "exclude_training_copies")
                  if key in declared["delivery"]})
    oracle_options = {
        "cnn": {"device": device, "checkpoint_dir": backend / "outputs/scorer"},
        "mixbend": {"device": device, "repo_path": backend / "MIXBend",
                    "dnabert_dir": backend / "MIXBend/pretrain_model/dnabert",
                    "params_path": backend / "MIXBend/params/MIXBend.json",
                    "checkpoint_paths": [str(backend / "outputs/external_models/mixbend_v1/best.pt")]},
        "bendnet": {"repo_path": backend, "weights_path": backend / "outputs/external_models/bendnet_v1/best.weights.h5"},
        "dnacycp2": {"repo_path": backend / "external/DNAcycP2-Python",
                     "python_bin": backend / "external/DNAcycP2-Python/.venv311/bin/python"},
    }
    factories = {name: oracle_factory(entry["class"], **oracle_options[name]) for name, entry in entries.items()}
    margin_units = "declared target scales from the nearest bound"
    definitions = {name: {"sources": [], "shape": [], "units": units, "meaning": meaning} for name, units, meaning in (
        ("legal", "boolean", "Exactly 50 uppercase DNA bases"),
        ("training_copy", "boolean", "Exact or reverse-complement identity in the complete TRAIN reference"))}
    definitions.update(dna_quality_definitions(evaluator.kmer_size))
    for name in sorted(visible):
        definitions[name] = {"sources": [name], "shape": [], "units": entries[name]["score_units"], "meaning": "Raw cyclizability prediction"}
        definitions[f"utility/{name}"] = {"sources": [name], "shape": [], "units": "fixed dimensionless interval utility", "meaning": task["measurement_semantics"]["utility/<source>"]}
        definitions[f"hit/{name}"] = {"sources": [name], "shape": [], "units": "target membership", "meaning": task["measurement_semantics"]["hit/<source>"]}
        # Declared only where the declaration is an interval: a directional target has no
        # bound to be inside of and carries no margin.
        if public_evaluator.oracle_objectives[name].kind == "interval":
            definitions[f"margin/{name}"] = {"sources": [name], "shape": [], "units": margin_units, "meaning": task["measurement_semantics"]["margin/<source>"]}
    assets = asset_manifest({
        "initial_policy": [sft], "training_reference": [data],
        "quality_reference": [path_from_root(declared["data"]["success_measurement"]["quality_reference"]["path"], repo_root)],
        "cnn": [backend / f"outputs/scorer/member_{i}.pt" for i in range(3)],
        "mixbend": [oracle_options["mixbend"]["dnabert_dir"], oracle_options["mixbend"]["params_path"], *oracle_options["mixbend"]["checkpoint_paths"]],
        "bendnet": [oracle_options["bendnet"]["weights_path"]],
        "dnacycp2": [oracle_options["dnacycp2"]["repo_path"] / name for name in ("irlstm", "irlstm_smooth")]
                      + [oracle_options["dnacycp2"]["python_bin"]],
    })

    def load_policy(checkpoint, device_override=None):
        from recital.policy_loading import load_remote_code_policy
        model, tokenizer = load_remote_code_policy(str(checkpoint))
        return model.to(device if device_override is None else device_override), tokenizer

    def quality_measure(sequences):
        return dna_quality_measurements(sequences, k=evaluator.kmer_size)

    def measure(sequences, conditions, oracle_outputs):
        result = quality_measure(sequences)
        for name, raw in oracle_outputs.items():
            if name not in visible:
                continue
            values = np.asarray(raw, dtype=float)
            if values.shape != (len(sequences),):
                raise ValueError("Loop raw scores must align one-to-one with sequence rows")
            mapped = public_evaluator.oracle_objectives[name].evaluate(values, conditions)
            result[name] = measurement(values, (name,), units=entries[name]["score_units"])
            result[f"utility/{name}"] = measurement(mapped.utility, (name,), units="fixed dimensionless interval utility", direction="maximize")
            result[f"hit/{name}"] = measurement(mapped.hit.astype(float), (name,), units="target membership", direction="maximize")
            if mapped.margin is not None:
                result[f"margin/{name}"] = measurement(mapped.margin, (name,), units=margin_units, direction="maximize")
        legal = [len(s) == 50 and set(s) <= set("ACGT") for s in sequences]
        result["legal"] = measurement(legal)
        result["training_copy"] = measurement([canonical_dna(s) in evaluator.training_identities if ok else False for s, ok in zip(sequences, legal)])
        return result

    def qualify(sequences, conditions, oracle_outputs, selection_report):
        measured = public_evaluator.evaluate({"sequences": sequences, "conditioning": conditions, "oracle_outputs": oracle_outputs})
        return qualification(success, measured)

    audit_objectives = {name: interval_objective(entries[name]) for name in blind}

    def audit_measure(sequences, conditions, oracle_outputs):
        """Per blind source, the fraction of rows inside the condition's band (primary) and the median signed
        distance inside it (secondary), each by condition and as the minimum over conditions.

        A row with no finite score or an unknown condition carries neither value, so the condition is reported
        incomplete against its original denominator.
        """
        report = config["blind_report"]
        primary_statistic, secondary_statistic = declared_audit_statistic(report), declared_audit_statistic(report["secondary"])
        hits, margins = {}, {}
        for name in blind:
            raw = np.asarray(oracle_outputs.get(name, [np.nan] * len(sequences)), dtype=float)
            if raw.shape != (len(sequences),):
                raise ValueError("Loop terminal scores must align with frozen sequences")
            usable = np.array([bool(np.isfinite(value)) and group in contract.conditions
                               for value, group in zip(raw, conditions)], dtype=bool)
            mapped = audit_objectives[name].evaluate(raw[usable], [group for group, ok in zip(conditions, usable) if ok])
            hit, margin = np.full(len(raw), np.nan), np.full(len(raw), np.nan)
            hit[usable], margin[usable] = mapped.hit, mapped.margin
            hits[name] = summarize_audit(hit, conditions, contract.conditions, statistic=primary_statistic)
            margins[name] = summarize_audit(margin, conditions, contract.conditions, statistic=secondary_statistic)
        primary = report["primary_source"]
        return {"sources": hits, "primary": hits[primary], "definition": dict(report),
                "secondary": {"sources": margins, "primary": margins[primary], "definition": dict(report["secondary"])}}

    return {"adapter": adapter, "contract": contract, "load_policy": load_policy,
            "load_training_model": lambda checkpoint: load_policy(checkpoint, device_override="cpu")[0],
            "oracle_factories": {name: fn for name, fn in factories.items() if name not in blind},
            "audit_oracle_factories": {name: factories[name] for name in blind},
            "audit_output_shapes": dict.fromkeys(blind, ()), "measurement_definitions": definitions,
            "oracle_versions": {name: {"class": entries[name]["class"], "assets": assets[name]} for name in entries},
            "asset_manifest": assets,
            "prompts": prompts, "training_sequences": sequences,
            "representation": {"alphabet": "ACGT", "min_length": 50, "max_length": 50, "max_completion_length": 50},
            "context_extras": {"sampling_add_special_tokens": False, "sampling_use_cache": False},
            "measure": measure, "quality_measure": quality_measure,
            "qualify": qualify, "audit_measure": audit_measure}
