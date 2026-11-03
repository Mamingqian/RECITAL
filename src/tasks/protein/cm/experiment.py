"""CM task runtime: assets, measurements, qualification and terminal blind evaluation."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from recital.actions import ToolCapability
from recital.experiment_tasks import (
    asset_manifest, counts_from_config, delivery_success, make_contract, measurement, path_from_root,
    qualification, scalar_measurement, declared_audit_statistic, declared_source_role, summarize_audit,
)
from .adapter import CMAdapter
from .data import ALPHABET, read_fasta


def build_runtime(config: dict, repo_root: Path, output: Path, device: str) -> dict:
    repo_root = Path(repo_root).resolve()
    root = path_from_root(config["asset_root"], repo_root)
    adapter = CMAdapter(repo_root=repo_root, config_path=Path(config["contract_path"]))
    declared, initial = adapter.contract(), adapter.initial_reward_spec()
    if "novelty_parameters" in config:
        declared["oracles"]["novelty"] = dict(config["novelty_parameters"])
    conditions = ("CM",)
    roles = declared["oracles"]["source_roles"]["primary"]
    blind = tuple(roles["audit_oracles"])
    names = tuple(dict.fromkeys([*roles["reward_pool"], *roles["sentinel_pool"]]))
    # `feedback` is the declared sentinels that are not in the reward pool; they are
    # hidden from the agent unless protocol_condition is self_validating.
    feedback = tuple(name for name in roles["sentinel_pool"] if name not in roles["reward_pool"])
    visible = set(names)
    if config.get("protocol_condition", "self_validating") != "self_validating":
        visible -= set(feedback)
    counts = counts_from_config(config, conditions)
    # Read the contract, as loopseq and promoter do. The per-row ratios are the
    # conjunction `cm_eligible` states; the condition check is declared beside it.
    success = delivery_success(declared["success"], counts)
    novelty_floor = float(declared["delivery"]["min_natural_distance"])
    task = {
        "name": "cm", "description": "Design complete 80–96 amino-acid AroQ chorismate-mutase domains.",
        "target_objective": "Improve computational support for complementation function while preserving qualified novelty and sequence quality; use available functional and family evidence to choose the final method.",
        "biological_constraints": ["Generate the complete domain from <bos>1 without a wild-type prefix", "Standard amino-acid alphabet and length 80–96", "Unique sequences excluding exact SFT and functional-predictor training copies", f"Normalized full-sequence edit distance at least {novelty_floor:g} from the natural family"],
        "novelty": {"delivery_minimum": novelty_floor, "reward_shortfall": declared["oracles"].get("novelty", {}), "delivery_maximum": declared["delivery"].get("max_natural_distance")},
        "measurement_semantics": {"calibrated/<source>": "Same fitted predictor with its own development calibration; not a probability or measured activity", "natural_distance": "Full-sequence Levenshtein divided by maximum sequence length against the nearest natural family member", "training_copy": "Exact match in the complete exclusion reference"},
    }
    # The contract's own note for each source. The cm_novelty note states that its
    # upper bound is a reward term, not a gate: delivery keeps sequences within the
    # contract's delivery.max_natural_distance, and the reward penalizes distance
    # past the novelty ceiling.
    entries = {entry["id"]: entry for entry in declared["oracles"]["candidate_pool"]}
    # Only entries that carry a note are kept, so the fallback below applies to an
    # entry with `notes: ""` (the schema default); an empty string is not a valid
    # `ToolCapability` description.
    descriptions = {name: " ".join(entry["notes"].split())
                    for name, entry in entries.items() if entry.get("notes")}
    tools = [ToolCapability(name, descriptions.get(name, "Declared CM family predictor"),
                            declared_role=declared_source_role(
                                name, reward_pool=roles.get("reward_pool", ()),
                                sentinel_pool=roles.get("sentinel_pool", ()),
                                required=roles.get("required_sentinels", ())),
                            units="model-specific prediction", direction="maximize",
                            dependencies=("cm_bm",) if name == "cm_max" else ()) for name in names]
    sft = path_from_root(config.get("sft_checkpoint", root / "sft/model"), repo_root)
    table = pd.read_csv(root / "data/sft.csv")
    sft_training = table.loc[table["split"].eq("train"), "sequence"].tolist()
    full_exclusion = [sequence for _, sequence in read_fasta((root / "data/training_sequences.fasta").read_text())]
    natural_sequences = [sequence.replace("-", "") for _, sequence in read_fasta((root / "data/natural_aligned.fasta").read_text())]
    training_sequences = list(dict.fromkeys([*sft_training, *full_exclusion]))
    if not training_sequences:
        raise ValueError("CM complete training exclusion reference is empty")
    training_set = frozenset(training_sequences)
    calibration = {}
    versions = {}
    for name in [*names, *blind]:
        head = root / "oracles" / f"{name}.joblib"
        versions[name] = {"asset_root": str(root), "head": str(head) if head.is_file() else None}
        if name in visible and name.startswith("function_"):
            calibration[name] = dict(joblib.load(head)["calibration"])
            versions[name]["calibration"] = calibration[name]
    assets = asset_manifest({
        "initial_policy": [sft], "training_reference": [root / "data/sft.csv", root / "data/training_sequences.fasta"],
        "natural_reference": [root / "data/natural_aligned.fasta"],
        "cm_novelty": [root / "data/natural_aligned.fasta"],
        **{name: [root / "oracles" / f"{name}.joblib", root / "assets" / ("prot-t5" if name.endswith("t5") else "esm2-650m")]
           for name in [*names, *blind] if name.startswith(("function_", "audit_"))},
        "cm_bm": [root / "assets/bmDCA.npz", root / "development/prior_scale.json"],
        "cm_max": [root / "assets/bmDCA.npz", root / "assets/meDCA.npz", root / "development/prior_scale.json"],
        "cm_poet": [declared["oracles"]["poet"][key] for key in ("python", "source", "checkpoint")]
                   + [root / "data/qualification.csv"],
    })
    for name in versions:
        versions[name]["assets"] = assets.get(name, [])
        if name.startswith("cm_"):
            versions[name]["natural_reference"] = assets["natural_reference"]
        if name == "cm_novelty":
            versions[name]["settings"] = declared["oracles"].get("novelty", {})
        if name == "cm_poet":
            versions[name]["settings"] = declared["oracles"]["poet"]
    bounds = {"lr": {"type": "number", "minimum": declared["framework"]["learning_rate_bounds"][0], "maximum": declared["framework"]["learning_rate_bounds"][1]},
              "kl": {"type": "number", "minimum": declared["framework"]["kl_bounds"][0], "maximum": declared["framework"]["kl_bounds"][1]},
              "group_size": {"type": "integer", "minimum": 2, "maximum": 64}}
    contract = make_contract(config, task=task, conditions=conditions, initial_checkpoint=sft,
        initial_recipe=initial, tools=tools, feedback_sources=feedback, blind_sources=blind,
        success=success, bounds=bounds)
    definitions = {name: {"sources": [], "shape": [], "units": units, "meaning": meaning} for name, units, meaning in (
        ("legal", "boolean", "Standard amino acids with complete-domain length 80–96"),
        ("natural_distance", "normalized edit distance", task["measurement_semantics"]["natural_distance"]),
        ("training_copy", "boolean", task["measurement_semantics"]["training_copy"]))}
    for name in sorted(visible):
        definitions[name] = {"sources": [name], "shape": [], "units": "raw predictor coordinates", "meaning": descriptions[name]}
        if name in calibration:
            definitions[f"calibrated/{name}"] = {"sources": [name], "shape": [], "units": "calibrated complementation prediction",
                                                  "meaning": task["measurement_semantics"]["calibrated/<source>"], "calibration": calibration[name]}

    @lru_cache(maxsize=1)
    def family():
        from .family import FamilyReference
        return FamilyReference(root)

    def factory(name):
        def create():
            from .oracles import load_oracle
            return load_oracle(root, name, device=device,
                family=family() if name.startswith("cm_") else None,
                novelty_config=declared["oracles"].get("novelty"), poet_config=declared["oracles"].get("poet"))
        return create

    def load_policy(checkpoint, device_override=None):
        from tasks.protein.progen import ProGen3CausalLM
        from .runner import load_tokenizer
        tokenizer = load_tokenizer(Path(checkpoint))
        model = ProGen3CausalLM.from_pretrained(str(checkpoint), local_files_only=True)
        model.generation_config.forced_eos_token_id = tokenizer.eos_token_id
        model.generation_config.do_sample = True
        model.generation_config.top_k = 0
        return model.to(device if device_override is None else device_override), tokenizer

    def context_extras_factory(model, tokenizer):
        from tasks.protein.progen import ProGenSequenceCodec
        return {"sequence_codec": ProGenSequenceCodec(tokenizer),
                "generation_kwargs": {"forced_eos_token_id": tokenizer.eos_token_id, "top_k": 0}}

    def measure(sequences, conditions, oracle_outputs):
        legal = np.asarray([80 <= len(s) <= 96 and set(s) <= set(ALPHABET) for s in sequences])
        distances = np.full(len(sequences), np.nan)
        indices = np.flatnonzero(legal)
        if len(indices):
            distances[indices] = family().distances([sequences[i] for i in indices])
        result = {"legal": measurement(legal), "natural_distance": measurement(distances),
                  "training_copy": measurement([s in training_set for s in sequences])}
        for name, raw in oracle_outputs.items():
            if name not in visible:
                continue
            values = np.asarray(raw, dtype=float)
            if values.shape != (len(sequences),):
                raise ValueError(f"CM {name} scores must align with sequence rows")
            result[name] = measurement(values, (name,), units="raw predictor coordinates")
            if name in calibration:
                calibrated = values * calibration[name]["slope"] + calibration[name]["intercept"]
                result[f"calibrated/{name}"] = measurement(calibrated, (name,), units="calibrated complementation prediction", calibration=calibration[name], model_asset=versions[name]["head"])
        return result

    def qualify(sequences, conditions, oracle_outputs, selection_report):
        # Imported here because `checkpoints` pulls in torch and the shared checkpoint
        # selector, which this module otherwise never needs and which are slow to load.
        from .checkpoints import eligible_pool
        derived = measure(sequences, conditions, oracle_outputs)
        n = len(sequences)
        function = np.asarray(oracle_outputs.get("function_2026_esm", [np.nan] * n), dtype=float)
        # The same conjunction `eligible_pool` applies to a candidate pool, applied here to
        # the delivered library, rather than a second statement of it.
        table = pd.DataFrame({
            "valid": np.asarray(derived["legal"].values, dtype=bool),
            "duplicate": pd.Series(sequences, dtype=object).duplicated().to_numpy(),
            "training_copy": np.asarray(derived["training_copy"].values, dtype=bool),
            "natural_distance": np.asarray(derived["natural_distance"].values, dtype=float),
            "function_2026_esm": function})
        pool, _ = eligible_pool(table, min_distance=novelty_floor)
        # The contract's declared per-sequence score under `selection_aggregation: primary`
        # is the validation-calibrated function_2026_esm, and its candidate score is that
        # column's median over the delivered library -- the `selection_score_median`
        # that `select_library` reports.
        calibrated = np.asarray(derived["calibrated/function_2026_esm"].values, dtype=float) \
            if "calibrated/function_2026_esm" in derived else np.full(n, np.nan)
        scored = calibrated[np.isfinite(calibrated)]
        values = {"cm_delivered": (n, 1),
                  "cm_eligible": (float(len(pool)), n),
                  "cm_selection_score": (float(np.median(scored)) if len(scored) else None, len(scored)),
                  "condition_valid_ratio": (sum(c == "CM" for c in conditions) / n if n else None, n)}
        # The contract decides qualification through cm_eligible, which is the conjunction
        # of the five below. They stay published because a single count cannot say WHICH
        # conjunct failed, and the agent has to be able to tell a novelty shortfall from a
        # training copy. Measured, not required: assess_candidate reads only what the
        # contract declares.
        values.update({
            "legal_ratio": (float(table.valid.mean()) if n else None, n),
            "unique_ratio": (float((~table.duplicate).mean()) if n else None, n),
            "training_copy_ratio": (float(table.training_copy.mean()) if n else None, n),
            "natural_distance_pass_ratio": (
                float(table.natural_distance.ge(novelty_floor).mean()) if n else None, n),
            "function_finite_ratio": (float(np.isfinite(function).mean()) if n else None, n)})
        measured = {key: {"CM": scalar_measurement(value, count)} for key, (value, count) in values.items()}
        return qualification(success, measured)

    def audit_measure(sequences, conditions, oracle_outputs):
        statistic = declared_audit_statistic(config["blind_report"])
        summaries = {name: summarize_audit(oracle_outputs.get(name, [np.nan] * len(sequences)),
                                          conditions, contract.conditions, statistic=statistic)
                     for name in blind}
        primary = config["blind_report"]["primary_source"]
        return {"sources": summaries, "primary": summaries[primary], "definition": dict(config["blind_report"])}

    return {"adapter": adapter, "contract": contract, "load_policy": load_policy,
            "load_training_model": lambda checkpoint: load_policy(checkpoint, device_override="cpu")[0],
            "oracle_factories": {name: factory(name) for name in names},
            "audit_oracle_factories": {name: factory(name) for name in blind},
            "audit_output_shapes": dict.fromkeys(blind, ()), "measurement_definitions": definitions,
            "oracle_versions": versions, "prompts": {"CM": "<bos>1"}, "training_sequences": training_sequences,
            "asset_manifest": assets, "reference_libraries": {"cm_natural_family": natural_sequences},
            "representation": {"alphabet": ALPHABET, "min_length": 80, "max_length": 96, "max_completion_length": 97},
            "context_extras": {"min_new_tokens": 80, "min_sequence_length": 80, "max_sequence_length": 96,
                               "sampling_add_special_tokens": False, "sampling_use_cache": True},
            "context_extras_factory": context_extras_factory, "bf16": False,
            "measure": measure, "qualify": qualify,
            "audit_measure": audit_measure}
