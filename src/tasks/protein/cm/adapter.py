"""Chorismate-mutase (CM) task semantics for the shared causal-LM execution engine."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import yaml

from recital.capabilities import RecipeCapabilities
from recital.search_budget import SearchBudgetConfig, validate_final_candidate_budget
from recital.sft import SFTRecipe
from recital.task_adapter import source_role_policy_from_contract, task_spec_from_contract
from recital.task_provider import ObjectiveDescriptor
from recital.token_sequence import RepresentationSpec

# capabilities() offers ("hard_biological_gate", "residue_run_gate"), so the
# residue-calibrated primitives must be registered before anyone reads that
# allowlist. It registers HERE rather than in tasks/protein/__init__.py because
# that package is a parent of tasks.protein.cm.poet2, which has to import inside
# the PoET-2 interpreter where recital's dependencies do not exist.
from tasks.protein import primitives as _protein_primitives  # noqa: F401

CM_OBJECTIVE = ObjectiveDescriptor(
    name="cm_function", stat_key="cm_function_score", worst_key="library_median",
    per_group_key="per_condition_median", objective_kind="maximize",
)

# The protocol condition each CM arm runs under. `no_synthesis` and `no_spend` close an action channel, not the
# sentinel channel, so they self-validate exactly like `full`; `no_feedback` neither shows nor measures it.
ARM_PROTOCOL_CONDITIONS = {
    "full": "self_validating",
    "no_synthesis": "self_validating",
    "no_spend": "self_validating",
    "no_feedback": "no_self_validation",
}


def declared_search_budget(contract: dict) -> SearchBudgetConfig | None:
    framework = contract.get("framework") or {}
    declared = framework.get("search_budget")
    if declared is None:
        return None  # No search_budget declared: candidate-only protocol.
    config = SearchBudgetConfig(**declared)
    initial = contract["initial_reward_spec"]["hyperparameters"]["steps"]
    if config.evaluation_mode != "final" or framework.get("checkpoint_evaluation"):
        raise ValueError("CM final-library update budgets cannot enable checkpoint selection")
    if config.first_candidate_updates != initial:
        raise ValueError("CM budget initial depth differs from the initial recipe")
    if config.max_trained_candidates != framework["iteration_budget"]:
        raise ValueError("CM budget candidate limit differs from iteration_budget")
    validate_final_candidate_budget(config, max_steps=framework["max_steps_per_iter"],
                                    proposals=framework["iteration_budget"] * 4)
    minimum = framework.get("min_steps_per_iter", 1)
    if min((initial, *config.allowed_candidate_updates)) < minimum:
        raise ValueError("CM budget duration is below the declared minimum")
    if framework.get("fixed_steps") and any(n != initial for n in config.allowed_candidate_updates):
        raise ValueError("a fixed CM recipe cannot declare variable budget durations")
    return config


def protocol_condition(contract: dict) -> str:
    """The arm's protocol condition, from its declaration rather than a literal."""
    framework = contract.get("framework") or {}
    declared = framework.get("protocol_condition")
    arm = (framework.get("ablation_matrix") or {}).get("arm")
    if declared is not None:
        if declared not in set(ARM_PROTOCOL_CONDITIONS.values()):
            raise ValueError(f"CM contract declares unknown protocol_condition {declared!r}")
        # An override is for a contract with no arm. With an arm present the two must
        # agree: a contract declaring no_feedback + self_validating would run a mixed arm
        # whose sentinel and operational channels match neither declaration.
        if arm is not None and ARM_PROTOCOL_CONDITIONS.get(arm) != declared:
            raise ValueError(
                f"CM contract declares protocol_condition {declared!r} but ablation arm "
                f"{arm!r} runs as {ARM_PROTOCOL_CONDITIONS.get(arm)!r}; an arm and an "
                "override that disagree produce a mixed arm, not a configured one")
        return str(declared)
    if arm is None:
        # A contract with no ablation matrix declares no arm, so the condition is
        # self_validating.
        return "self_validating"
    if arm not in ARM_PROTOCOL_CONDITIONS:
        raise ValueError(f"CM ablation arm {arm!r} has no declared protocol condition")
    return ARM_PROTOCOL_CONDITIONS[arm]


def novelty_bounds(contract: dict) -> tuple[float, float | None]:
    """Declared cm_novelty distance floor/ceiling; ceiling None means one-sided."""
    from .oracles import NOVELTY_FLOOR

    settings = (contract.get("oracles") or {}).get("novelty") or {}
    ceiling = settings.get("ceiling")
    return float(settings.get("floor", NOVELTY_FLOOR)), None if ceiling is None else float(ceiling)


DELIVERY_PROTOCOL = "selected_evaluation_library_v1"


def selection_semantics(contract: dict) -> dict:
    """Name the score and sources consumed by the declared library selection."""
    aggregation = contract["delivery"]["selection_aggregation"]
    if aggregation not in {"primary", "min", "mean"}:
        raise ValueError(f"Unknown CM selection aggregation: {aggregation}")
    sources = ["function_2026_esm"] if aggregation == "primary" else ["function_2026_esm", "function_2026_t5"]
    function = "validation-calibrated 2026 ESM" if aggregation == "primary" else f"{aggregation} of validation-calibrated 2026 ESM/T5"
    return {"metric": "cm_selection_score", "aggregation": aggregation,
            "oracle_sources": sources, "per_sequence_score": function,
            "candidate_score": "median over the selected eligible library",
            "library_count": contract["delivery"]["count"],
            "delivery_protocol": DELIVERY_PROTOCOL, "additional_delivery_draws": 0,
            "direction": "maximize", "audit_used_for_selection": False}


def resolve_contract(contract: dict) -> dict:
    """Freeze effective objective text and score provenance from one declaration."""
    contract = copy.deepcopy(contract)
    contract["delivery"]["protocol"] = DELIVERY_PROTOCOL
    contract["delivery"].pop("candidate_count", None)
    semantics = selection_semantics(contract)
    contract["delivery"]["ranking"] = semantics["per_sequence_score"]
    contract["task"]["target_objective"] = (
        "Maximize delivered-library median of " + semantics["per_sequence_score"])
    contract["task"]["selection_semantics"] = semantics
    return _resolve_success(contract)


def _resolve_success(contract: dict) -> dict:
    # As in the DNA adapters, the task supplies qualification and ranking to
    # the shared engine. Callers may explicitly override this declaration.
    # The delivery metrics consume exactly the oracles the selection RULE reads; the
    # blinded condition must reject a selection metric drawn from a source its agent
    # cannot see.
    selection_sources = selection_semantics(contract)["oracle_sources"]
    contract.setdefault("success", {
        "requirements": [{"id": "complete_delivery", "metric": "cm_delivered",
                          "strata": ["CM"], "minimum": contract["delivery"]["count"]}],
        "ranking": [{"metric": "cm_selection_score", "strata": ["CM"], "direction": "maximize"}],
        "stop_when_qualified": False,
        "metric_sources": {"cm_delivered": list(selection_sources),
                           "cm_selection_score": list(selection_sources)},
    })
    for requirement in contract["success"]["requirements"]:
        if requirement["id"] in {"complete_delivery", "novel_unique_pool"}:
            requirement["minimum"] = contract["delivery"]["count"]
    contract["success"]["metric_sources"].setdefault("cm_primary", ["function_2026_esm"])
    if contract.get("framework", {}).get("checkpoint_evaluation"):
        roles = contract["oracles"]["source_roles"]["primary"]
        for name in dict.fromkeys(roles["reward_pool"] + roles["sentinel_pool"]):
            contract["success"]["metric_sources"][name] = [name]
        contract["success"]["metric_sources"]["natural_distance"] = []
    if contract["delivery"].get("family_requirement"):
        source = contract["delivery"]["family_requirement"]["oracle"]
        for metric in ("cm_delivered", "cm_selection_score"):
            contract["success"]["metric_sources"][metric] = list(dict.fromkeys(
                contract["success"]["metric_sources"][metric] + [source]))
    for metric in ("cm_delivered", "cm_selection_score"):
        declared = contract["success"]["metric_sources"].get(metric)
        expected = set(selection_sources)
        if contract["delivery"].get("family_requirement"):
            expected.add(contract["delivery"]["family_requirement"]["oracle"])
        if declared is not None and set(declared) != expected:
            raise ValueError(f"CM {metric} sources disagree with the effective selection rule")
    if contract.get("framework", {}).get("ablation_matrix"):
        # cm_eligible never reads a cm_novelty SCORE. It counts unique, non-training
        # candidates at or past the declared distance FLOOR that ESM could score -- the
        # ceiling is a reward term and eligible_pool refuses it outright -- so its
        # only oracle input is function_2026_esm; the distance itself is sequence-only
        # (compare natural_distance, declared with no sources at all above).
        contract["success"]["metric_sources"]["cm_eligible"] = ["function_2026_esm"]
        if not any(r["metric"] == "cm_eligible" for r in contract["success"]["requirements"]):
            contract["success"]["requirements"].append({"id": "novel_unique_pool", "metric": "cm_eligible",
                "strata": ["CM"], "minimum": contract["delivery"]["count"]})
    return contract


@dataclass
class CMAdapter:
    repo_root: Path = Path(".")
    config_path: Path | None = None

    def source_paths(self):
        path = self.config_path or Path("configs/cm_contract.yaml")
        path = path if path.is_absolute() else self.repo_root / path
        return path, path

    def contract(self):
        return resolve_contract(yaml.safe_load(self.source_paths()[0].read_text()))

    def initial_reward_spec(self):
        return copy.deepcopy(self.contract()["initial_reward_spec"])

    def sft_recipe(self):
        contract = self.contract()
        checkpoint = contract.get("sft_checkpoint")
        return SFTRecipe(name="cm_sft", status="reuse_existing" if checkpoint else "train",
                         checkpoint=checkpoint, data_source=contract["data"]["source"],
                         conditioning_prompts=("<bos>1",), alphabet=self.contract()["representation"]["alphabet"],
                         sequence_length=96, epochs=3, seed=1101)

    def capabilities(self, steps: int | None = None):
        # Imported here, not at module scope, so describing the recipe surface
        # never drags in the alignment/Potts stack family.py carries.
        from .family import NATURAL_FAMILY_REFERENCE as natural_family_reference

        contract = self.contract()
        framework = contract.get("framework", {})
        initial_hp = contract["initial_reward_spec"]["hyperparameters"]
        steps = steps or initial_hp["steps"]
        lr_low, lr_high = framework.get("learning_rate_bounds", (1e-6, 5e-5))
        kl_low, kl_high = framework.get("kl_bounds", (.005, .2))
        editable = {"lr": {"type": "number", "minimum": lr_low, "maximum": lr_high,
                           "description": "Initial optimizer learning rate; lower means smaller updates."},
                    "kl": {"type": "number", "minimum": kl_low, "maximum": kl_high,
                           "description": "KL penalty coefficient against CM-SFT. Higher anchors the policy more strongly; lowering it is not stronger regularization."}}
        frozen = {"group_size": initial_hp["group_size"]}
        if framework.get("fixed_steps"):
            frozen["steps"] = steps
        else:
            # Optimization DEPTH is the lever the reward-over-optimization turning
            # point sits on. The bounds are declared beside max_steps_per_iter; the
            # fallbacks are the minimum steps and twice the initial steps.
            editable["steps"] = {"type": "integer",
                                 "minimum": framework.get("min_steps_per_iter", steps),
                                 "maximum": framework.get("max_steps_per_iter", steps * 2),
                                 "description": "Optimizer updates for this candidate. Deeper training buys more reward optimization and more divergence from CM-SFT; the checkpoint schedule and the update budget follow this number."}
        if framework.get("search_budget"):
            budget = declared_search_budget(contract)
            if "steps" in editable:
                editable["steps"]["enum"] = sorted(set((budget.first_candidate_updates, *budget.allowed_candidate_updates)))
        return RecipeCapabilities(
            allowed_reward_fields=("enabled_oracles", "held_out_oracles", "transforms", "aggregator",
                                   "shape", "penalties", "gates", "reward_clip_low", "invalid_penalty"),
            editable_hyperparameters=editable,
            frozen_hyperparameters=frozen,
            # Every primitive offered here must be able to BIND on 80-96 residues
            # over 20 letters. The DNA set cannot: homopolymer_gate's max_run of 6
            # is unreachable, repeated_kmer's k=8 is unique by construction,
            # and seen_penalty cannot fire for a temperature-1 policy that has
            # not memorised its corpus.
            # residue_run_gate is the protein-calibrated run bound, and
            # reference_distance is the family-distance term; seen_penalty stays
            # because exact training copies are a declared delivery exclusion.
            allowed_primitives=(("cross_oracle_aggregation", "weighted_sum"),
                                ("cross_oracle_aggregation", "trimmed_mean"),
                                ("per_sequence_penalty", "reference_distance"),
                                ("per_sequence_penalty", "seen_penalty"),
                                ("hard_biological_gate", "residue_run_gate")),
            runtime_parameters={"per_sequence_penalty/reference_distance": {
                "reference_sequences": "Adapter-owned reference rows for the named library; never author this parameter."}},
            frozen_requirements={"target": "maximize CM normalized complementation function",
                                 "alphabet": self.contract()["representation"]["alphabet"],
                                 "length": [80, 96],
                                 # Both distance bounds are shown to the agent.
                                 "min_natural_edit_distance": novelty_bounds(contract)[0],
                                 "max_natural_edit_distance": novelty_bounds(contract)[1],
                                 "natural_edit_distance": (
                                     "cm_novelty is a two-sided distance shortfall when a ceiling is declared: it is "
                                     "negative below the floor AND above the ceiling, zero in between. Published "
                                     "measured functional rates fall about tenfold across the 0.4 boundary (Russ 2020 "
                                     "35.2% -> 3.8%; Netti 2026 28.4% -> 7.2%). That ceiling is a REWARD term: drift "
                                     "past it is penalized, not forbidden. Delivery separately keeps only sequences "
                                     "within [floor, max_natural_distance] of the natural family "
                                     f"({contract['delivery'].get('max_natural_distance')}), so a pool with fewer than "
                                     "128 such rows is unqualified. "
                                     "Requesting a distance-ceiling primitive duplicates a term that already exists."),
                                 "delivery": f"{contract['delivery']['count']} unique novel sequences; all residue positions open",
                                 "cm_novelty": "computed negative shortfall, not a learned biological predictor",
                                 "reference_libraries": (
                                     f"reference_distance measures each sequence against a NAMED library and "
                                     f"penalizes the two-sided shortfall outside [floor, ceiling]. Bound ids: "
                                     f"{natural_family_reference!r} (the natural AroQ family, the same reference "
                                     f"cm_novelty scores against) and 'training_set' (the CM-SFT corpus). Metrics: "
                                     f"'normalized_edit' (whole-sequence Levenshtein / max length) and 'exact_match'. "
                                     f"Naming the natural family with cm_novelty already enabled duplicates a term "
                                     f"the reward has; the training_set library does not."),
                                 "hard_biological_gate": "A hard gate marks a sequence ineligible outright, rather than scaling its reward: it returns one boolean per sequence and a True forbids that sequence. Use it for a constraint that admits no trade-off against predicted function, such as a residue-composition liability or a bound on distance from the natural family. It is evaluated on every training generation, so its per-call cost matters; a gate that must compare each sequence against thousands of references is unlikely to fit the isolated execution budget.",
                                 "family_priors": "family statistics; higher is more natural-like, not an activity label",
                                 "selection": contract["delivery"]["ranking"],
                                 "comparison": "Candidates restart from CM-SFT with different training seeds. Sequence bootstraps do not measure between-training-seed variability or establish causal recipe effects.",
                                 "feedback_timing": ("Policy checkpoints are measured during training at the declared steps. All updates are consumed; agents receive the source-filtered checkpoint time series after the candidate, and common final selection may deliver an earlier checkpoint."
                                     if framework.get("checkpoint_evaluation") else
                                     "T5 and PoET are evaluated after each completed candidate; saved intermediate checkpoints are not scored online.")},
            allow_primitive_requests=framework.get("ablation_matrix", {}).get("arm") != "no_synthesis",
            synthesizable_roles=() if framework.get("ablation_matrix", {}).get("arm") == "no_synthesis" else
                ("cross_oracle_aggregation", "per_sequence_penalty", "hard_biological_gate"),
        )

    def task_spec(self):
        contract = self.contract()
        representation = RepresentationSpec(modality="protein", **contract["representation"])
        return task_spec_from_contract(
            contract, objective=CM_OBJECTIVE, sequence_length=96,
            initial_reward_spec=self.initial_reward_spec(),
            source_role_policy=source_role_policy_from_contract(contract),
            representation=representation, sft_recipe=self.sft_recipe(),
            recipe_capabilities=self.capabilities(),
            primitive_allowlist=None,
        )
