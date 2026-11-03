"""CRE task runtime: explicit cell coordinates, measurements, qualification and terminal blind evaluation."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import os

import numpy as np
import pandas as pd

from recital.actions import ToolCapability
from recital.experiment_tasks import (
    asset_manifest, counts_from_config, delivery_success, dna_quality_definitions,
    dna_quality_measurements, make_contract, measurement, oracle_factory,
    median_measurement, qualification, scalar_measurement, declared_audit_statistic, declared_source_role, summarize_audit,
)
from .adapter import PromoterAdapter
from .assets import resolve_asset_path
from .score_contract import panel_gap


def build_runtime(config: dict, repo_root: Path, output: Path, device: str) -> dict:
    repo_root = Path(repo_root).resolve()
    adapter = PromoterAdapter(repo_root=repo_root, contract_path=Path(config["contract_path"]),
                              initial_spec_path=Path(config["initial_spec_path"]))
    declared, initial = adapter.contract(), adapter.initial_reward_spec()
    conditions = ("K562", "HepG2")
    prompts = {cell: f"<CELL_{cell}>" for cell in conditions}
    entries = {entry["id"]: entry for entry in declared["oracles"]["candidate_pool"]}
    roles = declared["oracles"]["source_roles"]["primary_fold"]
    blind = tuple(roles["audit_oracles"])
    # `feedback` is the declared sentinels that are not in the reward pool; they are
    # hidden from the agent unless protocol_condition is self_validating.
    feedback = tuple(name for name in roles["sentinel_pool"] if name not in roles["reward_pool"])
    visible = set(entries) - set(blind)
    if config.get("protocol_condition", "self_validating") != "self_validating":
        visible -= set(feedback)
    declared_panels = declared["task"]["score_semantics"]["oracle_panels"]
    panels = {name: deepcopy(value) for name, value in declared_panels.items() if name in visible}
    success = delivery_success(declared["task"]["success"], counts_from_config(config, conditions))
    task = {
        "name": "cre", "description": "Design 200 bp cis-regulatory DNA for K562 and HepG2.",
        "target_objective": "Improve target-cell activity and specificity using available computational evidence. Compare each target cell from actual measurements. The system selects each delivered library from a candidate's pool by the stated rule; predictor values are not measured enhancer activity.",
        "biological_constraints": ["Exactly 200 uppercase A/C/G/T bases", "Complete finite declared qualification measurements and required counts for both cells", "No target-activity threshold has been established"],
        "score_semantics": {"target_cells": list(conditions), "oracle_panels": panels,
                            "gap_definition": "raw target head minus maximum raw declared off-target head",
                            "target_activity_floor_by_cell": dict.fromkeys(conditions)},
        "measurement_semantics": {"mingap/<source>": "Per-row raw target-minus-maximum-offtarget gap; higher is better", "activity/<source>": "Raw target-cell head", "offtarget/<source>": "Maximum raw declared off-target head", "raw_scores": "Matrices retain declared cell-column order; rows retain their generated target condition"},
    }
    # The contract's own note for each source, not a second description beside it.
    # Only entries that carry a note are kept, so the fallback below applies to an
    # entry with `notes: ""` (the schema default); an empty string is not a valid
    # `ToolCapability` description.
    descriptions = {name: " ".join(entry["notes"].split())
                    for name, entry in entries.items() if entry.get("notes")}
    tools = [ToolCapability(name, descriptions.get(name, "Declared CRE predictor"),
                           units="model-specific raw head coordinates", direction="target-dependent raw MinGap",
                           declared_role=declared_source_role(
                               name, reward_pool=roles.get("reward_pool", ()),
                               sentinel_pool=roles.get("sentinel_pool", ()),
                               required=roles.get("required_sentinels", ()),
                               use_for=(entries.get(name) or {}).get("use_for")))
             for name in entries if name not in blind]
    sft = resolve_asset_path(config.get("sft_checkpoint", adapter.sft_recipe().checkpoint), repo_root=repo_root)
    data = resolve_asset_path(config.get("training_data", declared["data"]["unified_parquet"]), repo_root=repo_root)
    sequence_column = declared["data"].get("sequence_column", "sequence")
    frame = pd.read_parquet(data, columns=[sequence_column, "split"])
    training_sequences = frame.loc[frame["split"].eq("train"), sequence_column].tolist()
    if not training_sequences:
        raise ValueError("CRE training reference has no rows in its declared train split")
    bounds = deepcopy(declared["recipe_capabilities"]["editable_hyperparameters"])
    bounds.pop("steps", None)
    contract = make_contract(config, task=task, conditions=conditions, initial_checkpoint=sft,
        initial_recipe=initial, tools=tools, feedback_sources=feedback, blind_sources=blind,
        success=success, bounds=bounds,
        identity={key: declared["delivery"][key] for key in ("deduplication", "exclude_training_copies")
                  if key in declared["delivery"]})
    backend = resolve_asset_path("backends/promoter_rl", repo_root=repo_root)
    oracle_options = {
        "malinois": {"weights_path": backend / "tools/malinois_weights/artifacts"},
        "agarwal": {"weights_path": backend / "tools/malinois_agarwal_weights/best_model.pt"},
        "enformer_mpra": {"checkpoint_path": Path(os.environ.get("ENFORMER_MPRA_CHECKPOINT", str(repo_root / "backends/enformer_mpra/mpra_model_best_epoch_07_val_MalinoisMPRA_mean_SpearmanR_0_73185.ckpt")))},
        "grammar_prior": {"meme_path": backend / "dbs/JASPAR2024_vertebrates.meme"},
        "chrombpnet": {"weights_dir": backend / "tools/chrombpnet_models"},
    }
    factories = {name: oracle_factory(entry["class"], device=device, **oracle_options[name]) for name, entry in entries.items()}
    definitions = {"legal": {"sources": [], "shape": [], "units": "boolean", "meaning": "Exactly 200 uppercase DNA bases"}}
    definitions.update(dna_quality_definitions(8))
    for name, panel in panels.items():
        columns = panel["oracle_columns"]
        definitions[name] = {"sources": [name], "shape": [len(columns)], "column_names": list(columns),
                             "units": "raw model head coordinates", "meaning": task["measurement_semantics"]["raw_scores"]}
        for prefix in ("activity", "offtarget", "mingap"):
            definitions[f"{prefix}/{name}"] = {"sources": [name], "shape": [], "units": "raw model head coordinates",
                                               "meaning": task["measurement_semantics"][f"{prefix}/<source>"]}
    assets = asset_manifest({
        "initial_policy": [sft], "training_reference": [data],
        "malinois": [oracle_options["malinois"]["weights_path"] / "torch_checkpoint.pt"],
        "agarwal": [oracle_options["agarwal"]["weights_path"]],
        "enformer_mpra": [oracle_options["enformer_mpra"]["checkpoint_path"]],
        "grammar_prior": [oracle_options["grammar_prior"]["meme_path"]],
        "chrombpnet": [oracle_options["chrombpnet"]["weights_dir"] / cell / "models/chrombpnet_nobias.h5" for cell in ("K562", "HepG2", "SKNSH")],
    })

    def load_policy(checkpoint, device_override=None):
        from recital.policy_loading import load_remote_code_policy
        model, tokenizer = load_remote_code_policy(str(checkpoint))
        return model.to(device if device_override is None else device_override), tokenizer

    def quality_measure(sequences):
        return dna_quality_measurements(sequences, k=8)

    def measure(sequences, conditions, oracle_outputs):
        result = quality_measure(sequences)
        result["legal"] = measurement([len(s) == 200 and set(s) <= set("ACGT") for s in sequences])
        for name, raw in oracle_outputs.items():
            if name not in visible:
                continue
            panel = panels[name]
            columns = panel["oracle_columns"]
            values = np.asarray(raw, dtype=float)
            if values.shape != (len(sequences), len(columns)):
                raise ValueError(f"{name} output does not match its declared cell-column panel")
            # A row without its declared heads has no gap; it stays NaN and is reported as missing.
            target, off_target, gap = panel_gap(
                values, conditions, oracle_columns=columns,
                off_target_cells_by_target=panel["off_target_cells_by_target"], missing="nan",
            )
            result[name] = measurement(values, (name,), columns=list(columns), units="raw model head coordinates")
            result[f"activity/{name}"] = measurement(target, (name,), units="raw target activity", direction="maximize")
            result[f"offtarget/{name}"] = measurement(off_target, (name,), units="raw maximum off-target activity", direction="minimize")
            result[f"mingap/{name}"] = measurement(gap, (name,), units="raw target-minus-offtarget gap", direction="maximize", panel=panel)
        return result

    def qualify(sequences, conditions, oracle_outputs, selection_report):
        derived = measure(sequences, conditions, oracle_outputs)
        gaps = np.asarray(derived["mingap/malinois"].values, dtype=float) if "mingap/malinois" in derived else np.full(len(sequences), np.nan)
        legal = np.asarray(derived["legal"].values, dtype=bool)
        measured = {name: {} for name in ("legal_ratio", "generated_count", "malinois_finite_ratio", "malinois_median_mingap_raw")}
        for cell in contract.conditions:
            rows = np.flatnonzero(np.asarray(conditions) == cell)
            n = len(rows)
            finite = gaps[rows][np.isfinite(gaps[rows])]
            measured["generated_count"][cell] = scalar_measurement(n, 1)
            measured["legal_ratio"][cell] = scalar_measurement(float(legal[rows].mean()) if n else None, n)
            measured["malinois_finite_ratio"][cell] = scalar_measurement(len(finite) / n if n else None, n)
            measured["malinois_median_mingap_raw"][cell] = median_measurement(finite)
        measured["condition_valid_ratio"] = {"all": scalar_measurement(
            sum(cell in contract.conditions for cell in conditions) / len(conditions) if conditions else None, len(conditions))}
        return qualification(success, measured)

    def audit_measure(sequences, conditions, oracle_outputs):
        summaries = {}
        audit_panel = config["blind_report"]["oracle_columns"]
        off_targets = config["blind_report"]["off_target_cells_by_target"]
        for name in blind:
            values = np.asarray(oracle_outputs.get(name, np.full((len(sequences), len(audit_panel)), np.nan)), dtype=float)
            if values.shape != (len(sequences), len(audit_panel)):
                raise ValueError("CRE terminal output columns differ from the declared blind panel")
            # The declared off-target list excludes SK-N-SH. A missing blind output is NaN
            # and summarize_audit reports it as incomplete.
            gap = panel_gap(values, conditions, oracle_columns=audit_panel,
                            off_target_cells_by_target=off_targets, missing="nan").gap
            summaries[name] = summarize_audit(gap, conditions, contract.conditions,
                                              statistic=declared_audit_statistic(config["blind_report"]))
        primary = config["blind_report"]["primary_source"]
        return {"sources": summaries, "primary": summaries[primary], "definition": dict(config["blind_report"])}

    return {"adapter": adapter, "contract": contract, "load_policy": load_policy,
            "load_training_model": lambda checkpoint: load_policy(checkpoint, device_override="cpu")[0],
            "oracle_factories": {name: fn for name, fn in factories.items() if name not in blind},
            "audit_oracle_factories": {name: factories[name] for name in blind},
            "audit_output_shapes": {name: (len(config["blind_report"]["oracle_columns"]),) for name in blind},
            "measurement_definitions": definitions,
            "prompts": prompts, "training_sequences": training_sequences,
            "representation": {"alphabet": "ACGT", "min_length": 200, "max_length": 200, "max_completion_length": 200},
            "context_extras": {**adapter.sampling_options(), "oracle_target_indices": dict(adapter.oracle_target_indices())},
            "oracle_versions": {name: {"class": entry["class"], "assets": assets[name]} for name, entry in entries.items()},
            "asset_manifest": assets,
            "measure": measure, "quality_measure": quality_measure,
            "qualify": qualify, "audit_measure": audit_measure}
