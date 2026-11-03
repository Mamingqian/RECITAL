"""Loop-seq implementation of the shared causal-LM task adapter."""

from __future__ import annotations

import json
import copy
from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Any

import yaml

from recital.failure_modes import FailureModeSpec, FailureProfile, SHARED_FAILURE_MODES
from recital.sft import SFTRecipe
from recital.task_adapter import (
    CausalLMTaskSpec,
    source_role_policy_from_contract,
    task_spec_from_contract,
)
from recital.task_provider import HIT_RATE_OBJECTIVE
from recital.capabilities import RecipeCapabilities
from recital.success import SuccessSpec

from .sft import _resolve_asset_path, resolve_asset_root
from . import reward as _reward_primitives  # noqa: F401 - register task-owned primitives


LOOPSEQ_FAILURE_PROFILE = FailureProfile(
    shared=SHARED_FAILURE_MODES,
    task_specific=(
        FailureModeSpec(
            id="low_sequence_complexity",
            evidence_keys=("dinucleotide_entropy", "unique_kmer_count", "GC"),
            candidate_actions=("add_gate", "add_penalty"),
            description="The generated DNA library collapses into a low-complexity mode.",
            task_specific=True,
        ),
        FailureModeSpec(
            id="periodicity_hack",
            evidence_keys=("high_low_periodicity_power",),
            candidate_actions=("add_penalty", "add_gate"),
            description="A high score is not accompanied by the expected bendability periodicity.",
            task_specific=True,
        ),
        FailureModeSpec(
            id="group_target_miss",
            evidence_keys=("per_group_hit_rate",),
            candidate_actions=("change_condition_repeats", "increase_group_size"),
            description="One requested cyclizability group is underrepresented in the generated library.",
            task_specific=True,
        ),
    ),
)


@dataclass(frozen=True)
class LoopseqAdapter:
    """Loop-seq task adapter: contract, initial recipe, recipe capabilities and SFT recipe."""

    repo_root: Path = Path(".")
    contract_path: Path = Path("src/tasks/loopseq/contract.yaml")
    initial_spec_path: Path = Path("configs/loopseq_initial_spec.json")

    def _root(self) -> Path:
        return self.repo_root.resolve()

    def source_paths(self) -> tuple[Path, Path]:
        root = self._root()
        return root / self.contract_path, root / self.initial_spec_path

    def contract(self) -> dict[str, Any]:
        payload = yaml.safe_load(self.source_paths()[0].read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Loop-seq contract must be a mapping")
        return payload

    def initial_reward_spec(self, *, allow_pending_references: bool = False) -> dict[str, Any]:
        contract = self.contract()
        embedded = (contract.get("data") or {}).get("initial_reward_spec")
        if embedded is not None and self.initial_spec_path == Path(
            "configs/loopseq_initial_spec.json"
        ):
            if not isinstance(embedded, dict):
                raise ValueError("data.initial_reward_spec must be a mapping")
            payload = embedded
        else:
            payload = json.loads(self.source_paths()[1].read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Loop-seq initial reward spec must be a mapping")
        return self.compile_initial_recipe(
            payload, contract=contract, allow_pending_references=allow_pending_references,
        )

    def compile_initial_recipe(
        self, payload: dict[str, Any], *, contract=None, evaluator=None,
        allow_pending_references: bool = False,
    ) -> dict[str, Any]:
        """Bind the initial recipe to input targets and conditional quality cuts.

        An explicit initial JSON may choose optimizer settings but cannot omit
        input quality constraints or redefine their initial thresholds. Existing
        conflicting definitions are rejected instead of silently overwritten.
        Later recipe edits are governed by the declared capabilities.
        """
        from .evaluation import LoopseqSuccessEvaluator
        from recital.targets import OracleTargetObjective

        contract = self.contract() if contract is None else contract
        if evaluator is None:
            evaluator = LoopseqSuccessEvaluator.from_contract(contract, repo_root=self._root())
        result = copy.deepcopy(payload)
        if evaluator is None:
            return result
        if result.get("transforms") or result.get("shape") is not None:
            raise ValueError("success initial recipe requires raw target utilities")
        targets = evaluator.reward_targets(result["enabled_oracles"])
        if "target_objectives" in result:
            supplied = {
                key: OracleTargetObjective.from_dict(value).to_dict()
                for key, value in result["target_objectives"].items()
            }
            if supplied != targets:
                raise ValueError("initial target_objectives conflict with frozen contract targets")
        result["target_objectives"] = targets
        if "initial_quality_penalty_weight" not in contract["data"]:
            return result
        quality = evaluator.quality_thresholds
        if quality is None:
            if not allow_pending_references:
                raise ValueError(
                    "initial quality reward requires a frozen conditional quality reference"
                )
            return result
        reference_penalty = {
            "name": "loopseq_reference_quality",
            "by_condition": {
                group: {
                    key: quality[group][key]
                    for key in ("dinucleotide_entropy_min", "unique_kmer_min")
                }
                for group in evaluator.groups
            },
            "kmer_size": evaluator.kmer_size,
            "weight": contract["data"]["initial_quality_penalty_weight"],
        }
        penalties = result.setdefault("penalties", [])
        existing = [p for p in penalties if p.get("name") == "loopseq_reference_quality"]
        if existing and existing != [reference_penalty]:
            raise ValueError("initial quality penalty conflicts with frozen input reference")
        if not existing:
            penalties.append(reference_penalty)
        return result

    def resolved_recipe_capabilities(
        self,
        *,
        contract: dict[str, Any] | None = None,
        evaluator=None,
        max_steps_per_iter: int | None = None,
    ) -> RecipeCapabilities | None:
        """Resolve permissions once; derive scientific constants from task input.

        The YAML declares the action space. Success and target definitions are
        compiled from their authoritative sections, never copied into a second
        independently editable scientific specification. A contract that declares
        no ``success`` section has no frozen requirements or frozen raw targets to
        publish, so those two fields must be empty and the YAML grant is the whole
        declaration; it may not assert requirements the contract cannot support.
        """
        contract = self.contract() if contract is None else contract
        raw = contract.get("recipe_capabilities")
        if raw is None:
            return None
        if evaluator is None and contract.get("success") is not None:
            from .evaluation import LoopseqSuccessEvaluator

            evaluator = LoopseqSuccessEvaluator.from_contract(
                contract,
                repo_root=self._root(),
            )
        if evaluator is None:
            frozen: dict[str, Any] = {}
            targets: dict[str, Any] = {}
            for name in ("frozen_requirements", "frozen_target_objectives"):
                if raw.get(name):
                    raise ValueError(
                        f"recipe_capabilities.{name} requires an explicit success contract"
                    )
        else:
            policy = source_role_policy_from_contract(contract)
            targets = evaluator.reward_targets(sorted(policy.reward_oracles or ()))
            frozen = {
                "success": SuccessSpec.model_validate(contract["success"]).model_dump(mode="json"),
                "success_measurement": copy.deepcopy(contract["data"]["success_measurement"]),
            }
            if contract.get("delivery"):
                frozen["delivery"] = copy.deepcopy(contract["delivery"])
            for name, expected in (
                ("frozen_requirements", frozen),
                ("frozen_target_objectives", targets),
            ):
                if raw.get(name) and raw[name] != expected:
                    raise ValueError(f"recipe_capabilities.{name} conflicts with task input")
        payload = copy.deepcopy(raw)
        payload["allowed_reward_fields"] = tuple(payload["allowed_reward_fields"])
        payload["allowed_primitives"] = tuple(tuple(item) for item in payload["allowed_primitives"])
        payload["frozen_requirements"] = frozen
        payload["frozen_target_objectives"] = targets
        if max_steps_per_iter is not None:
            schema = payload["editable_hyperparameters"].get("steps")
            if schema is not None:
                schema["maximum"] = min(schema["maximum"], max_steps_per_iter)
                if schema["maximum"] < schema.get("minimum", 1):
                    raise ValueError("CLI step budget is below the contract's allowed minimum")
        paired = (contract.get("framework") or {}).get("paired_synthesis")
        if paired:
            budget = paired["search_budget"]
            durations = sorted(set([budget["first_candidate_updates"], *budget["allowed_candidate_updates"]]))
            schema = payload["editable_hyperparameters"].get("steps")
            if schema is not None:
                if any(n < schema.get("minimum", 1) or n > schema.get("maximum", n) for n in durations):
                    raise ValueError("declared candidate durations exceed effective step capabilities")
                schema["enum"] = durations
        return RecipeCapabilities.from_dict(payload)

    def sft_recipe(self) -> SFTRecipe:
        """Describe the mounted group-conditioned SFT starting policy.

        Metadata only: calling it never loads weights or starts a training run.
        """
        recipe = SFTRecipe.from_contract(
            self.contract(),
            name="loopseq_group3_completion_only",
            sequence_length=50,
            sequence_column="seq50",
            # The trainer derives this column from z_cyc before group3 SFT;
            # it is not a stored ``stratum`` column in the unified parquet.
            condition_column="target_group",
            loader="recital.policy_loading.load_remote_code_policy",
            epochs=3,
            metadata={
                "conditioning_mode": "group3",
                "n_target_bins": 3,
                "condition_source": "z_cyc_to_target_group",
                # Records that the checkpoint loads; qualification for the task
                # is a separate step.
                "checkpoint_status": "loadable_frozen_rl_start",
            },
        )
        if os.environ.get("LOOPSEQ_ASSET_ROOT"):
            backend = resolve_asset_root(self._root())
            recipe = replace(
                recipe,
                checkpoint=(
                    str(
                        _resolve_asset_path(
                            recipe.checkpoint,
                            backend=backend,
                            repo_root=self._root(),
                        )
                    )
                    if recipe.checkpoint
                    else None
                ),
                data_source=(
                    str(
                        _resolve_asset_path(
                            recipe.data_source,
                            backend=backend,
                            repo_root=self._root(),
                        )
                    )
                    if recipe.data_source
                    else None
                ),
            )
        return recipe

    def task_spec(self) -> CausalLMTaskSpec:
        contract = self.contract()
        # Describing a task is allowed before reference materialization. The
        # runner refuses training until the required references are available.
        initial_spec = self.initial_reward_spec(allow_pending_references=True)
        sft_recipe = self.sft_recipe()
        capabilities = self.resolved_recipe_capabilities(contract=contract)
        task = task_spec_from_contract(
            contract,
            objective=HIT_RATE_OBJECTIVE,
            sequence_length=50,
            sequence_column="seq50",
            condition_column="stratum",
            initial_reward_spec=initial_spec,
            source_role_policy=source_role_policy_from_contract(contract),
            failure_profile=LOOPSEQ_FAILURE_PROFILE,
            sft_recipe=sft_recipe,
            recipe_capabilities=capabilities,
        )
        return task
