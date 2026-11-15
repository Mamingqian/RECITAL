"""The amp runtime for the controller and its workers: AMP-Prompt, five visible predictors, APEX at the terminal audit."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from recital.actions import ToolCapability
from recital.experiment_tasks import (
    asset_manifest, counts_from_config, declared_audit_statistic, declared_source_role, delivery_success, make_contract,
    measurement, path_from_root, qualification, scalar_measurement, summarize_audit,
)
from recital.sequence_complexity import _max_homopolymer

from . import policy
from .adapter import CONDITION, MIN3, MIN3_MEASUREMENT, MIN3_METRIC, MIN3_SOURCES, PATHOGEN_NAMES, AmpAdapter
from .oracles import load_oracle

MIN_NONTOXIC_MEAN, MIN_AMP_LIKELIHOOD_MEAN = 0.62, 0.5


def lowest_activity(oracle_outputs: dict) -> np.ndarray | None:
    """Per peptide, the lowest of the three activity scores; a row any head leaves unscored is unscored. `None` when a head is absent."""
    heads = [oracle_outputs.get(name) for name in MIN3_SOURCES]
    if any(head is None for head in heads):
        return None
    return np.minimum.reduce([np.asarray(head, dtype=float) for head in heads])


def known_peptides(table: Path) -> list[str]:
    """The distinct standard-residue sequences of the public MIC table the activity predictors were fitted on."""
    sequences = pd.read_csv(table, usecols=["sequence"])["sequence"].astype(str).str.upper()
    return sorted(set(sequences[sequences.str.fullmatch(f"[{policy.RESIDUES}]+")]))


def build_runtime(config: dict, repo_root: Path, output: Path, device: str) -> dict:
    repo_root = Path(repo_root).resolve()
    pathogen = config.get("pathogen", "ecoli")
    zenodo = path_from_root(config["asset_root"], repo_root)
    # An interpreter path is kept without following symlinks (resolving would follow a virtual environment's `python`
    # symlink to the base interpreter); relative paths are anchored at `repo_root` without `resolve()`.
    workers = {key: (Path(value) if Path(value).is_absolute() else repo_root / value) if key.endswith("_python")
               else path_from_root(value, repo_root)
               for key, value in config["workers"].items()}
    adapter = AmpAdapter(repo_root=repo_root, config_path=Path(config["contract_path"]), pathogen=pathogen)
    declared, initial = adapter.contract(), adapter.initial_reward_spec()
    conditions = (CONDITION,)
    roles = declared["oracles"]["source_roles"]["primary"]
    blind = tuple(roles["audit_oracles"])
    names = tuple(dict.fromkeys([*roles["reward_pool"], *roles["sentinel_pool"]]))
    feedback = tuple(name for name in roles["sentinel_pool"] if name not in roles["reward_pool"])
    visible = set(names) - (set(feedback) if config.get("protocol_condition", "self_validating") != "self_validating" else set())
    counts = counts_from_config(config, conditions)
    success = delivery_success(declared["success"], counts)
    esmfold = path_from_root(config["esmfold_dir"], repo_root) if config.get("esmfold_dir") else None
    task = {
        "name": "amp", "description": declared["task"]["description"],
        "target_objective": declared["task"]["target_objective"],
        "biological_constraints": declared["task"]["biological_constraints"],
        "measurement_semantics": {
            "activity_<pathogen>": "1 / (1 + predicted MIC in micromolar); a predictor output, not measured activity",
            "amp_likelihood": "AMP-classifier probability of the Macrel package",
            "nonhemolytic": "one minus the hemolysis probability of the Macrel package",
            "nontoxic": "one minus the ToxinPred3 Hybrid Score",
            "training_copy": "exact match to a peptide of the public MIC table",
            **({"plddt_esmfold": "mean pLDDT of ESMFold v1 over the atoms it places, in [0, 1]"} if "plddt_esmfold" in names else {}),
            **({MIN3_MEASUREMENT: "the lowest of a peptide's E. coli, S. aureus and P. aeruginosa activity scores; the score the delivered "
                                  "library is ranked by"} if pathogen == MIN3 else {}),
        },
    }
    entries = {entry["id"]: entry for entry in declared["oracles"]["candidate_pool"]}
    descriptions = {name: " ".join(entry["notes"].split()) for name, entry in entries.items() if entry.get("notes")}
    tools = [ToolCapability(name, descriptions.get(name, "Declared peptide predictor"),
                            declared_role=declared_source_role(name, reward_pool=roles.get("reward_pool", ()),
                                                               sentinel_pool=roles.get("sentinel_pool", ()),
                                                               required=roles.get("required_sentinels", ())),
                            units="predictor output in [0, 1]", direction="maximize") for name in names]
    sft = path_from_root(config.get("sft_checkpoint", declared["sft_checkpoint"]), repo_root)
    known = known_peptides(path_from_root(config["known_peptides"], repo_root))
    if not known:
        raise ValueError("the table of known peptides is empty")
    training_set = frozenset(known)
    # The reference sequences the engine scores (the Coder's composition screen takes rows spread across them) are those the policy can
    # generate: a known peptide of 1 to 4 or of 33 or more residues gets no score from any scorer, and the screen fails on a missing score.
    # The copy flag above still knows every table peptide.
    training_sequences = [s for s in known if policy.MIN_LENGTH <= len(s) <= policy.MAX_LENGTH]
    regressor = {name: [zenodo / f"{label}_regress.pth", zenodo / "config.json"]
                 for name, label in (("activity_ecoli", "Ecoli"), ("activity_saureus", "Saureus"),
                                     ("activity_paeruginosa", "Paeruginosa"))}
    assets = asset_manifest({
        "initial_policy": [sft], "training_reference": [path_from_root(config["known_peptides"], repo_root)], **regressor,
        "amp_likelihood": [workers["macrel_python"]], "nonhemolytic": [workers["macrel_python"]],
        "nontoxic": [workers["toxinpred3_python"]],
        **({"plddt_esmfold": [esmfold / "pytorch_model.bin"]} if esmfold is not None else {}),
        **{name: [workers["apex_dir"] / "trained_models"] for name in blind},
    })
    versions = {name: {"assets": assets.get(name, [])} for name in [*names, *blind]}
    versions.update({
        "amp_likelihood": {**versions["amp_likelihood"], "package": "macrel 1.6.1 (ONNX AMP model)"},
        "nonhemolytic": {**versions["nonhemolytic"], "package": "macrel 1.6.1 (ONNX hemolysis model)"},
        "nontoxic": {**versions["nontoxic"], "package": "toxinpred3 1.4, -m 2 -t 0.38, Hybrid Score"},
        **{name: {**versions[name], "package": "APEX 1.1 ensemble, 8 configurations x 5 members, MIC in micromolar"}
           for name in blind},
        **{name: {**versions[name], "layout": f"every row padded to {34} tokens"} for name in regressor},
    })
    bounds = {"lr": {"type": "number", "minimum": declared["framework"]["learning_rate_bounds"][0],
                     "maximum": declared["framework"]["learning_rate_bounds"][1]},
              "kl": {"type": "number", "minimum": declared["framework"]["kl_bounds"][0],
                     "maximum": declared["framework"]["kl_bounds"][1]},
              "group_size": {"type": "integer", "minimum": 2, "maximum": 64}}
    contract = make_contract(config, task=task, conditions=conditions, initial_checkpoint=sft, initial_recipe=initial,
                             tools=tools, feedback_sources=feedback, blind_sources=blind, success=success, bounds=bounds)
    definitions = {name: {"sources": [], "shape": [], "units": units, "meaning": meaning} for name, units, meaning in (
        ("legal", "boolean", "Standard amino acids, 5 to 32 residues"),
        ("length", "residues", "Number of residues"),
        ("max_run", "residues", "Longest run of one residue"),
        ("training_copy", "boolean", task["measurement_semantics"]["training_copy"]))}
    for name in sorted(visible):
        definitions[name] = {"sources": [name], "shape": [], "units": "predictor output in [0, 1]", "meaning": descriptions[name]}
    if pathogen == MIN3:
        definitions[MIN3_MEASUREMENT] = {"sources": list(MIN3_SOURCES), "shape": [], "units": "predictor output in [0, 1]",
                                         "meaning": task["measurement_semantics"][MIN3_MEASUREMENT]}

    def factory(name):
        def create():
            return load_oracle(name, zenodo=zenodo, device=device, python={
                "macrel": workers["macrel_python"], "toxinpred3": workers["toxinpred3_python"], "apex": workers["apex_python"]},
                apex_dir=workers["apex_dir"], scratch=Path(output) / "scorer_scratch", esmfold=esmfold)
        return create

    def load_policy(checkpoint, device_override=None):
        return policy.load_policy(checkpoint, device if device_override is None else device_override)

    def context_extras_factory(model, tokenizer):
        return {"generation_kwargs": {"top_k": 0}}

    def measure(sequences, conditions, oracle_outputs):
        legal = np.asarray([policy.MIN_LENGTH <= len(s) <= policy.MAX_LENGTH and set(s) <= set(policy.RESIDUES)
                            for s in sequences])
        result = {"legal": measurement(legal), "length": measurement([len(s) for s in sequences]),
                  "max_run": measurement([_max_homopolymer(s) for s in sequences]),
                  "training_copy": measurement([s in training_set for s in sequences])}
        for name, raw in oracle_outputs.items():
            if name not in visible:
                continue
            values = np.asarray(raw, dtype=float)
            if values.shape != (len(sequences),):
                raise ValueError(f"amp {name} scores must align with sequence rows")
            result[name] = measurement(values, (name,), units="predictor output in [0, 1]")
        lowest = lowest_activity(oracle_outputs) if pathogen == MIN3 else None
        if lowest is not None:
            if lowest.shape != (len(sequences),):
                raise ValueError(f"amp {MIN3_MEASUREMENT} scores must align with sequence rows")
            result[MIN3_MEASUREMENT] = measurement(lowest, MIN3_SOURCES, units="predictor output in [0, 1]")
        return result

    def library_mean(oracle_outputs, name):
        values = np.asarray(oracle_outputs.get(name, []), dtype=float)
        values = values[np.isfinite(values)]
        return (float(values.mean()) if len(values) else None), len(values)

    def qualify(sequences, conditions, oracle_outputs, selection_report):
        n = len(sequences)
        derived = measure(sequences, conditions, oracle_outputs)
        legal = np.asarray(derived["legal"].values, dtype=bool)
        copies = np.asarray(derived["training_copy"].values, dtype=bool)
        if pathogen == MIN3:
            lowest = lowest_activity(oracle_outputs)
            selection = {MIN3_METRIC: library_mean({MIN3_MEASUREMENT: [] if lowest is None else lowest}, MIN3_MEASUREMENT)}
        else:
            selection = {"amp_selection_score": library_mean(oracle_outputs, f"activity_{pathogen}")}
        values = {"generated_count": (float(n), n), **selection,
                  "nontoxic_mean": library_mean(oracle_outputs, "nontoxic"),
                  "amp_likelihood_mean": library_mean(oracle_outputs, "amp_likelihood"),
                  "condition_valid_ratio": (sum(c == CONDITION for c in conditions) / n if n else None, n)}
        # Measured, not required: which conjunct of a short or failing delivery gave way.
        values.update({
            "legal_ratio": (float(legal.mean()) if n else None, n),
            "unique_ratio": (float((~pd.Series(sequences, dtype=object).duplicated()).mean()) if n else None, n),
            "training_copy_ratio": (float(copies.mean()) if n else None, n),
            "nonhemolytic_mean": library_mean(oracle_outputs, "nonhemolytic")})
        # The pLDDT library mean is measured for a contract that lists it among its requirements, so a contract without that floor
        # qualifies, and records its qualification, as it did.
        if any(requirement.metric == "plddt_mean" for requirement in success.requirements):
            values["plddt_mean"] = library_mean(oracle_outputs, "plddt_esmfold")
        measured = {key: {CONDITION: scalar_measurement(value, count)} for key, (value, count) in values.items()}
        return qualification(success, measured)

    def audit_measure(sequences, conditions, oracle_outputs):
        statistic = declared_audit_statistic(config["blind_report"])
        summaries = {name: summarize_audit(oracle_outputs.get(name, [np.nan] * len(sequences)),
                                          conditions, contract.conditions, statistic=statistic) for name in blind}
        primary = config["blind_report"]["primary_source"]
        return {"sources": summaries, "primary": summaries[primary], "definition": dict(config["blind_report"])}

    return {"adapter": adapter, "contract": contract, "load_policy": load_policy,
            "load_training_model": lambda checkpoint: load_policy(checkpoint, "cpu")[0],
            "oracle_factories": {name: factory(name) for name in names},
            "audit_oracle_factories": {name: factory(name) for name in blind},
            "audit_output_shapes": dict.fromkeys(blind, ()), "measurement_definitions": definitions,
            "oracle_versions": versions, "prompts": {CONDITION: policy.PROMPT_TEXT},
            "training_sequences": training_sequences, "asset_manifest": assets,
            "reference_libraries": {"public_mic_table_peptides": training_sequences},
            "representation": {"alphabet": policy.RESIDUES, "min_length": policy.MIN_LENGTH, "max_length": policy.MAX_LENGTH,
                               "max_completion_length": policy.MAX_COMPLETION_TOKENS},
            "context_extras": {"min_new_tokens": policy.MIN_LENGTH, "min_sequence_length": policy.MIN_LENGTH,
                               "max_sequence_length": policy.MAX_LENGTH, "sampling_add_special_tokens": False,
                               "sampling_use_cache": True},
            "context_extras_factory": context_extras_factory, "bf16": False,
            "measure": measure, "qualify": qualify, "audit_measure": audit_measure}
