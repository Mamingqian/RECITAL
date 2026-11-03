"""CRE implementation of the shared causal-LM task adapter."""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from recital.failure_modes import FailureModeSpec, FailureProfile, SHARED_FAILURE_MODES
from recital.capabilities import RecipeCapabilities
from recital.library_selection import LibrarySelectionPolicy
from recital.sft import SFTRecipe
from recital.task_adapter import (
    CausalLMTaskSpec,
    source_role_policy_from_contract,
    task_spec_from_contract,
)
from recital.task_provider import CRE_MINGAP_OBJECTIVE
from tasks.promoter.score_contract import PROMPT_TO_COLUMN


# All active CRE oracle wrappers retain this output-column order, including
# singleton task instances. Scheduling order must never redefine these columns.
_ORACLE_TARGET_INDICES = MappingProxyType(PROMPT_TO_COLUMN)
_SAMPLING_OPTIONS = MappingProxyType({
    "sampling_add_special_tokens": False,
    "sampling_batch_size": 64,
    "sampling_use_cache": False,
    "sampling_repetition_penalty": 1.0,
})


PROMOTER_FAILURE_PROFILE = FailureProfile(
    shared=SHARED_FAILURE_MODES,
    task_specific=(
        FailureModeSpec(
            id="cell_specificity_gap",
            evidence_keys=("mean_mingap_per_cell", "target_cell"),
            candidate_actions=("change_reward_structure", "change_condition_repeats"),
            description="The target-cell objective is met unevenly across requested cells.",
            task_specific=True,
        ),
        FailureModeSpec(
            id="contrastive_reward_floor",
            evidence_keys=("reward_std", "frac_reward_zero_std", "group_size"),
            candidate_actions=("increase_group_size", "change_learning_rate"),
            description="The contrastive target-minus-off-target reward has insufficient spread.",
            task_specific=True,
        ),
        FailureModeSpec(
            id="accessibility_mismatch",
            evidence_keys=("reward_progress", "accessibility_audit_progress"),
            candidate_actions=("add_reward_oracle", "add_gate", "add_penalty"),
            description="Expression-predictor progress is not supported by accessibility evidence.",
            task_specific=True,
        ),
    ),
)


@dataclass(frozen=True)
class PromoterAdapter:
    """CRE task adapter: contract, initial recipe, recipe capabilities and SFT recipe."""

    repo_root: Path = Path(".")
    contract_path: Path = Path("src/tasks/promoter/contract.yaml")
    initial_spec_path: Path = Path("configs/cre_initial_spec.json")
    # Empty means the canonical joint K562/HepG2 task; a singleton creates an
    # independent per-cell task instance without changing the shared engine.
    target_cells: tuple[str, ...] = ()

    def _root(self) -> Path:
        return self.repo_root.resolve()

    def source_paths(self) -> tuple[Path, Path]:
        root = self._root()
        return root / self.contract_path, root / self.initial_spec_path

    def contract(self) -> dict[str, Any]:
        payload = yaml.safe_load(self.source_paths()[0].read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("CRE contract must be a mapping")
        return payload

    def initial_reward_spec(self) -> dict[str, Any]:
        payload = json.loads(self.source_paths()[1].read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("CRE initial reward spec must be a mapping")
        return payload

    def oracle_target_indices(self) -> Mapping[str, int]:
        """Immutable complete oracle-column map, even for a singleton task.

        These indices address scorer outputs, not condition scheduling or scalar
        target bands. In particular, HepG2 stays column 1 when it is the only
        requested condition. No mapping is taken from an agent reward proposal.
        """
        prompts = self.task_spec().policy.conditioning_prompts
        unknown = sorted(set(prompts) - set(_ORACLE_TARGET_INDICES))
        if unknown:
            raise ValueError(f"CRE oracle columns are undefined for prompts: {unknown}")
        return _ORACLE_TARGET_INDICES

    def sampling_options(self) -> Mapping[str, Any]:
        """Declared atomic-prompt generation settings, shared by every sampler."""
        return _SAMPLING_OPTIONS

    def resolved_recipe_capabilities(
        self, *, contract: dict[str, Any] | None = None,
        max_steps_per_iter: int | None = None,
    ) -> RecipeCapabilities | None:
        """Resolve the contract-owned Coder action space for CRE."""
        contract = self.contract() if contract is None else contract
        raw = contract.get("recipe_capabilities")
        if raw is None:
            return None
        payload = copy.deepcopy(raw)
        payload["frozen_requirements"] = {
            "score_semantics": copy.deepcopy(contract["task"]["score_semantics"]),
            "success": copy.deepcopy(contract["task"]["success"]),
        }
        if contract.get("delivery"):
            payload["frozen_requirements"]["delivery"] = copy.deepcopy(contract["delivery"])
        if max_steps_per_iter is not None:
            schema = payload.get("editable_hyperparameters", {}).get("steps")
            if schema is not None:
                schema["maximum"] = min(int(schema["maximum"]), int(max_steps_per_iter))
        paired = (contract.get("framework") or {}).get("paired_synthesis")
        if paired:
            budget = paired["search_budget"]
            durations = sorted(set([budget["first_candidate_updates"], *budget["allowed_candidate_updates"]]))
            schema = payload.get("editable_hyperparameters", {}).get("steps")
            if schema is not None:
                if any(n < schema.get("minimum", 1) or n > schema.get("maximum", n) for n in durations):
                    raise ValueError("declared candidate durations exceed effective step capabilities")
                schema["enum"] = durations
        return RecipeCapabilities.from_dict(payload)

    def sft_recipe(self) -> SFTRecipe:
        """Describe the mounted cell-conditioned CRE SFT policy.

        The recipe is a serializable description of the policy stage.  It does
        not train or load a model.
        """
        contract = self.contract()
        data = contract.get("data") or {}
        recipe = SFTRecipe.from_contract(
            contract,
            name="promoter_celltype_completion_only",
            sequence_length=200,
            sequence_column=str(data.get("sequence_column") or "sequence"),
            condition_column=str(data.get("condition_column") or "cell_type"),
            loader="recital.policy_loading.load_remote_code_policy",
            epochs=20,
            metadata={
                # Records that the checkpoint loads; qualification for the task
                # is a separate step.
                "checkpoint_status": "loadable_frozen_rl_start",
                # The backend SFT trainer consumes all three cell tokens.
                # ``active_cells`` is the downstream RL/evaluation scope, which
                # may be a per-cell subset.
                "training_cells": ["K562", "HepG2", "SKNSH"],
                "training_conditioning_prompts": [
                    "<CELL_K562>",
                    "<CELL_HepG2>",
                    "<CELL_SKNSH>",
                ],
                "active_cells": list(self.target_cells)
                if self.target_cells
                else ["K562", "HepG2"],
            },
        )
        # Keep the contract's checkout-relative paths by default, but make an
        # explicit backend override visible to callers that serialize or load
        # this recipe.  This also lets a provider use a mounted promoter_rl
        # tree without accidentally joining ``backends/promoter_rl`` twice.
        if os.environ.get("PROMOTER_ASSET_ROOT"):
            from tasks.promoter.assets import resolve_asset_path

            recipe = replace(
                recipe,
                checkpoint=(
                    str(resolve_asset_path(recipe.checkpoint, repo_root=self._root()))
                    if recipe.checkpoint
                    else None
                ),
                data_source=(
                    str(resolve_asset_path(recipe.data_source, repo_root=self._root()))
                    if recipe.data_source
                    else None
                ),
            )
        return recipe

    def task_spec(self) -> CausalLMTaskSpec:
        contract = self.contract()
        # CRE keeps the biological success block under ``task`` so the task
        # contract owns objective semantics and qualification together.  The
        # shared adapter schema exposes the same block at its top level.
        contract_for_framework = copy.deepcopy(contract)
        contract_for_framework["success"] = copy.deepcopy(
            contract["task"].get("success")
        )
        initial_spec = self.initial_reward_spec()
        sft_recipe = self.sft_recipe()
        data = contract.get("data") or {}
        spec = task_spec_from_contract(
            contract_for_framework,
            objective=CRE_MINGAP_OBJECTIVE,
            sequence_length=200,
            sequence_column=str(data.get("sequence_column") or "sequence"),
            condition_column=str(data.get("condition_column") or "cell_type"),
            initial_reward_spec=initial_spec,
            source_role_policy=source_role_policy_from_contract(contract),
            failure_profile=PROMOTER_FAILURE_PROFILE,
            sft_recipe=sft_recipe,
        )
        if not self.target_cells:
            return spec
        canonical = tuple(str(cell) for cell in self.target_cells)
        unknown = sorted(set(canonical) - {"K562", "HepG2"})
        if unknown:
            raise ValueError(f"unknown CRE target cell(s): {unknown}")
        prompt_by_cell = {
            "K562": "<CELL_K562>",
            "HepG2": "<CELL_HepG2>",
        }
        prompts = tuple(prompt_by_cell[cell] for cell in canonical)
        if len(set(prompts)) != len(prompts):
            raise ValueError("CRE target_cells must be unique")
        scoped_name = f"{spec.name}_{'_'.join(cell.lower() for cell in canonical)}"
        scoped_program = replace(
            spec.program,
            name=scoped_name,
            description=(
                f"{spec.program.description} Active target cell(s): "
                f"{', '.join(canonical)}."
            ),
        )
        scoped_policy = replace(spec.policy, conditioning_prompts=prompts)
        scoped_success = spec.success_spec
        if scoped_success is not None:
            # A one-cell exploratory task cannot satisfy joint K562/HepG2
            # strata. Keep the same structural rules while restricting cell
            # strata and the all-cell count to the active scope.
            selection_policy = LibrarySelectionPolicy.from_contract(contract)
            eval_per_prompt = selection_policy.count if selection_policy else int(
                (contract.get("framework") or {}).get(
                    "eval_samples_per_prompt", 128
                )
            )
            requirements = []
            for requirement in scoped_success.requirements:
                strata = (
                    requirement.strata
                    if requirement.strata == ("all",)
                    else tuple(
                        stratum for stratum in requirement.strata
                        if stratum in canonical
                    )
                )
                if not strata:
                    continue
                min_count = (
                    eval_per_prompt * len(canonical)
                    if requirement.strata == ("all",)
                    else requirement.min_count
                )
                requirements.append(
                    requirement.model_copy(
                        update={"strata": strata, "min_count": min_count}
                    )
                )
            ranking = []
            for criterion in scoped_success.ranking:
                strata = tuple(
                    stratum for stratum in criterion.strata
                    if stratum in canonical
                )
                if strata:
                    ranking.append(criterion.model_copy(update={"strata": strata}))
            scoped_success = scoped_success.model_copy(
                update={
                    "requirements": tuple(requirements),
                    "ranking": tuple(ranking),
                }
            )
        scoped_recipe = replace(
            spec.sft_recipe,
            name=f"{spec.sft_recipe.name}_{'_'.join(cell.lower() for cell in canonical)}"
            if spec.sft_recipe is not None
            else "promoter_celltype_sft",
            metadata={
                **(dict(spec.sft_recipe.metadata) if spec.sft_recipe is not None else {}),
                "active_cells": list(canonical),
            },
        ) if spec.sft_recipe is not None else None
        return replace(
            spec,
            name=scoped_name,
            program=scoped_program,
            policy=scoped_policy,
            success_spec=scoped_success,
            sft_recipe=scoped_recipe,
        )
