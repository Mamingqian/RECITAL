"""The amp task on the shared causal-LM engine: AMP-Prompt designing 5 to 32 residue antimicrobial peptides."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import yaml

from recital.capabilities import RecipeCapabilities
from recital.sft import SFTRecipe
from recital.task_adapter import source_role_policy_from_contract, task_spec_from_contract
from recital.task_provider import ObjectiveDescriptor
from recital.token_sequence import RepresentationSpec

# capabilities() offers ("hard_biological_gate", "residue_run_gate"), so the residue-calibrated primitives are registered
# here, by the adapter that offers them, and not in tasks/protein/__init__.py (see the note there).
from tasks.protein import primitives as _protein_primitives  # noqa: F401

from . import families as _families  # noqa: F401  (registers the MAC-AMP aggregator families; offered only through `framework.extra_allowed_primitives`)
from . import policy

AMP_OBJECTIVE = ObjectiveDescriptor(
    name="amp_activity", stat_key="amp_selection_score", worst_key="library_mean",
    per_group_key="per_condition_mean", objective_kind="maximize",
)
PATHOGEN_NAMES = {"ecoli": "E. coli", "saureus": "S. aureus", "paeruginosa": "P. aeruginosa"}
# The `pathogen` setting `min3` ranks and delivers by the weakest of the three activity scores of a peptide.
MIN3 = "min3"
MIN3_SOURCES = tuple(f"activity_{pathogen}" for pathogen in PATHOGEN_NAMES)
MIN3_MEASUREMENT = "activity_min3"
MIN3_METRIC = "amp_selection_score_min3"
MIN3_SCORE = ("per-peptide minimum of the E. coli, S. aureus and P. aeruginosa activity scores, each 1 / (1 + predicted MIC in "
              "micromolar)")
AMP_OBJECTIVE_MIN3 = ObjectiveDescriptor(
    name="amp_activity_min3", stat_key=MIN3_METRIC, worst_key="library_mean",
    per_group_key="per_condition_mean", objective_kind="maximize",
)
DELIVERY_PROTOCOL = "selected_evaluation_library_v1"
# How each library-mean requirement of the contract reads in the qualification sentence of the capabilities.
FLOOR_NAMES = {"nontoxic_mean": "nontoxic score", "amp_likelihood_mean": "amp_likelihood", "nonhemolytic_mean": "nonhemolytic score",
               "plddt_mean": "plddt_esmfold score"}


def qualification_text(contract: dict) -> str:
    """The qualification rule as the contract's library-mean requirements state it, in the order the contract lists them."""
    floors = [f"its mean {FLOOR_NAMES.get(item['metric'], item['metric'])} is at least {item['minimum']:g}"
              for item in contract["success"]["requirements"] if item["metric"].endswith("_mean")]
    return ("a delivered library qualifies when it has the declared number of peptides, "
            + (floors[0] if len(floors) == 1 else ", ".join(floors[:-1]) + " and " + floors[-1]))
CONDITION = "AMP"


def _swap_pathogen(value, pathogen: str):
    """The contract is written for E. coli; the same text for another pathogen names that pathogen's activity source."""
    if isinstance(value, str):
        return value.replace("activity_ecoli", f"activity_{pathogen}").replace("E. coli", PATHOGEN_NAMES[pathogen])
    if isinstance(value, list):
        return [_swap_pathogen(item, pathogen) for item in value]
    if isinstance(value, dict):
        return {_swap_pathogen(key, pathogen): _swap_pathogen(item, pathogen) for key, item in value.items()}
    return value


def selection_semantics(contract: dict, pathogen: str) -> dict:
    """Name the score and sources consumed by the declared library selection."""
    if pathogen == MIN3:
        metric, sources, score = MIN3_METRIC, list(MIN3_SOURCES), MIN3_SCORE
    else:
        metric, sources = "amp_selection_score", [f"activity_{pathogen}"]
        score = f"{PATHOGEN_NAMES[pathogen]} activity, 1 / (1 + predicted MIC in micromolar)"
    return {"metric": metric, "aggregation": "primary", "oracle_sources": sources, "per_sequence_score": score,
            "candidate_score": "mean over the selected library", "library_count": contract["delivery"]["count"],
            "delivery_protocol": DELIVERY_PROTOCOL, "additional_delivery_draws": 0, "direction": "maximize",
            "audit_used_for_selection": False}


def resolve_contract(contract: dict, pathogen: str = "ecoli") -> dict:
    """Freeze the effective objective text and score provenance from one declaration."""
    if pathogen not in (*PATHOGEN_NAMES, MIN3):
        raise ValueError(f"pathogen must be one of {sorted((*PATHOGEN_NAMES, MIN3))}")
    contract = copy.deepcopy(contract)
    if pathogen not in ("ecoli", MIN3):
        contract = _swap_pathogen(contract, pathogen)
    contract["delivery"]["protocol"] = DELIVERY_PROTOCOL
    semantics = selection_semantics(contract, pathogen)
    contract["delivery"]["ranking"] = semantics["per_sequence_score"]
    contract["task"]["target_objective"] = "Maximize the delivered library's mean " + semantics["per_sequence_score"]
    contract["task"]["selection_semantics"] = semantics
    for requirement in contract["success"]["requirements"]:
        if requirement["id"] == "complete_delivery":
            requirement["minimum"] = contract["delivery"]["count"]
    return contract


@dataclass
class AmpAdapter:
    repo_root: Path = Path(".")
    config_path: Path | None = None
    pathogen: str = "ecoli"

    def source_paths(self):
        path = self.config_path or Path("configs/amp_contract.yaml")
        path = path if path.is_absolute() else self.repo_root / path
        return path, path

    def contract(self):
        return resolve_contract(yaml.safe_load(self.source_paths()[0].read_text()), self.pathogen)

    def initial_reward_spec(self):
        return copy.deepcopy(self.contract()["initial_reward_spec"])

    def sft_recipe(self):
        contract = self.contract()
        return SFTRecipe(name="amp_prompt", status="reuse_existing", checkpoint=contract["sft_checkpoint"],
                         data_source=contract["data"]["source"], conditioning_prompts=(policy.PROMPT_TEXT,),
                         alphabet=contract["representation"]["alphabet"], sequence_length=policy.MAX_LENGTH, epochs=1, seed=0)

    def capabilities(self, steps: int | None = None):
        contract = self.contract()
        framework = contract["framework"]
        initial_hp = contract["initial_reward_spec"]["hyperparameters"]
        steps = steps or initial_hp["steps"]
        lr_low, lr_high = framework["learning_rate_bounds"]
        kl_low, kl_high = framework["kl_bounds"]
        editable = {
            "lr": {"type": "number", "minimum": lr_low, "maximum": lr_high,
                   "description": "Initial optimizer learning rate; lower means smaller updates."},
            "kl": {"type": "number", "minimum": kl_low, "maximum": kl_high,
                   "description": "KL penalty coefficient against the starting policy. Higher anchors the policy more strongly; lowering it is not stronger regularization."},
            "steps": {"type": "integer", "minimum": framework["min_steps_per_iter"], "maximum": framework["max_steps_per_iter"],
                      "description": "Optimizer updates for this candidate. Deeper training buys more reward optimization and more divergence from the starting policy."}}
        frozen = {"group_size": initial_hp["group_size"]}
        if self.pathogen == MIN3:
            target = ("maximize the predicted activity of the delivered peptides against the weakest of E. coli, S. aureus and "
                      "P. aeruginosa while keeping them AMP-like and non-toxic")
            ranked_by = "the minimum of their predicted E. coli, S. aureus and P. aeruginosa activity"
        else:
            pathogen = PATHOGEN_NAMES[self.pathogen]
            target = f"maximize the predicted {pathogen} activity of the delivered peptides while keeping them AMP-like and non-toxic"
            ranked_by = f"predicted {pathogen} activity"
        return RecipeCapabilities(
            allowed_reward_fields=("enabled_oracles", "held_out_oracles", "transforms", "aggregator",
                                   "shape", "penalties", "gates", "reward_clip_low", "invalid_penalty"),
            editable_hyperparameters=editable,
            frozen_hyperparameters=frozen,
            # Primitives that can bind on peptides of 5 to 32 residues over 20 letters: the DNA gates and k-mer terms cannot
            # (a run of 6 is unreachable in a 20-letter alphabet, 8-mers are unique by construction at this length).
            # residue_run_gate is the protein-calibrated run bound; seen_penalty stays because exact copies of the public
            # MIC table's peptides are excluded from delivery.
            allowed_primitives=(("cross_oracle_aggregation", "weighted_sum"),
                                ("cross_oracle_aggregation", "trimmed_mean"),
                                ("per_sequence_penalty", "seen_penalty"),
                                ("hard_biological_gate", "residue_run_gate"),
                                *(tuple(pair) for pair in framework.get("extra_allowed_primitives", ()))),
            frozen_requirements={
                "target": target,
                "alphabet": contract["representation"]["alphabet"],
                "length": [policy.MIN_LENGTH, policy.MAX_LENGTH],
                "delivery": (f"{contract['delivery']['count']} distinct peptides chosen from the candidate's pool of "
                             f"{contract['delivery']['evaluation_candidates']} by {ranked_by}; peptides that "
                             "occur in the public MIC table are not delivered"),
                "qualification": qualification_text(contract),
                "activity": ("activity_<pathogen> is 1 / (1 + predicted MIC) with the MIC in micromolar: a score in (0, 1) that "
                             "saturates, so a gain near 1 needs a large drop in predicted MIC; predicted, not measured activity"
                             + (f"; {MIN3_MEASUREMENT} is the lowest of a peptide's three activity scores, the score the delivered "
                                "library is ranked by" if self.pathogen == MIN3 else "")),
                "scores": "every source is a predictor output in [0, 1] oriented so that higher is better",
                "hard_biological_gate": "A hard gate marks a sequence ineligible outright, rather than scaling its reward: it returns one boolean per sequence and a True forbids that sequence. It is evaluated on every training generation, so its per-call cost matters.",
                "comparison": "Candidates restart from the starting policy with different training seeds. Sequence bootstraps do not measure between-training-seed variability or establish causal recipe effects.",
                "feedback_timing": "Every source is evaluated after each completed candidate on its pool; saved intermediate checkpoints are not scored online.",
            },
            allow_primitive_requests=True,
            synthesizable_roles=("cross_oracle_aggregation", "per_sequence_penalty", "hard_biological_gate"),
        )

    def task_spec(self):
        contract = self.contract()
        representation = RepresentationSpec(modality="protein", **contract["representation"])
        return task_spec_from_contract(
            contract, objective=AMP_OBJECTIVE_MIN3 if self.pathogen == MIN3 else AMP_OBJECTIVE, sequence_length=policy.MAX_LENGTH,
            initial_reward_spec=self.initial_reward_spec(),
            source_role_policy=source_role_policy_from_contract(contract),
            representation=representation, sft_recipe=self.sft_recipe(),
            recipe_capabilities=self.capabilities(),
            primitive_allowlist=None,
        )
