"""Execute a proposed reward on a cached reference before training.

The screen checks composition, candidate count and finite values. Task adapters
supply the same semantic conditions used during training. Missing reference
scores or conditions make the screen abstain, not reject a valid proposal.

``reject_reference_constants`` rejects a spec whose reward is constant on the
reference library. Constancy on a finite reference library does not establish
constancy under the policy. Contract-mode Loop-seq disables that check; target
utility is often zero on reference sequences that already meet their target.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from recital.synthesis.models import IsolationError


def _needs_conditioning(spec: dict[str, Any], primitive_resolver=None) -> bool:
    """Does this spec carry a primitive the trainer binds per batch?

    Asked of the registry rather than a hardcoded name list, so a new conditional
    primitive is covered the day it is registered instead of silently becoming a
    hole in the screen.
    """
    from dataclasses import fields as dataclass_fields, is_dataclass

    from recital.spec.registry import (
        ROLE_CROSS_ORACLE_AGG,
        ROLE_PER_SEQ_PENALTY,
        ROLE_PER_SEQ_SHAPING,
        get_primitive,
    )

    if spec.get("reward_input"):
        return True
    if any(target.get("by_condition") for target in (spec.get("target_objectives") or {}).values()):
        return True

    for role, block in (
        (ROLE_CROSS_ORACLE_AGG, spec.get("aggregator")),
        (ROLE_PER_SEQ_SHAPING, spec.get("shape")),
        *((ROLE_PER_SEQ_PENALTY, item) for item in spec.get("penalties", [])),
    ):
        name = (block or {}).get("name") if isinstance(block, dict) else None
        if not name:
            continue
        try:
            cls = (primitive_resolver or get_primitive)(role, str(name))
        except Exception:  # noqa: BLE001 - an unknown primitive fails later, loudly
            continue
        if getattr(cls, "requires_conditioning", False) or any(
            f.name == "target_indices" for f in (dataclass_fields(cls) if is_dataclass(cls) else ())
        ):
            return True
    return False


@dataclass
class ProposalScreen:
    """Callable returning the list of reasons a spec cannot train (empty = fine).

    `oracle_outputs` are the reward-pool oracles' scores on `sequences`, computed
    once and reused: the screen costs no oracle calls and no gradient steps.
    """

    sequences: list[str]
    oracle_outputs: dict[str, np.ndarray]
    conditioning: list[str] = field(default_factory=list)
    conditioning_prompts: list[str] = field(default_factory=list)
    group_intervals: Any = None
    training_set: list[str] = field(default_factory=list)
    # Named reference libraries a reference_distance penalty may measure against,
    # bound exactly as training binds them: screening with a different library
    # would score a spec the trainer will never run.
    reference_libraries: Mapping[str, list[str]] = field(default_factory=dict)
    min_reward_std: float = 1e-9
    # Constancy on a finite reference is not proof of zero reward variance under
    # the current generation policy.
    reject_reference_constants: bool = True
    # Oracle output columns stay fixed when active prompts are subset/reordered;
    # scalar target shapes continue to use their local condition indices.
    oracle_target_indices: Mapping[str, int] | None = None
    primitive_resolver: Callable[[str, str], Callable] | None = None
    # Strict execution paths use the same nonempty reference as training;
    # otherwise the screen does not depend on a reference.
    require_training_reference: bool = False
    last_diagnostics: dict[str, Any] = field(default_factory=dict, init=False)

    def __call__(self, spec: dict[str, Any]) -> list[str]:
        from recital.spec.composer import ComposedRewardFn, RewardSpec
        from recital.trainer import _bind_prompt_conditioning

        self.last_diagnostics = {"status": "unavailable"}
        if self.require_training_reference and not self.training_set:
            return ["required training reference is missing or empty"]
        if not self.sequences or not self.oracle_outputs:
            return (["required execution screen has no bound reference scores/sequences"]
                    if self.require_training_reference else [])

        enabled = list(spec.get("enabled_oracles") or [])
        missing = [name for name in enabled if name not in self.oracle_outputs]
        if missing:
            # Not the proposal's fault -- the screen has no scores for this oracle,
            # so it must abstain rather than reject.
            return ([f"required execution screen lacks scores for {missing}"]
                    if self.require_training_reference else [])
        if _needs_conditioning(spec, self.primitive_resolver) and not (
            self.conditioning and self.conditioning_prompts
        ):
            # Same rule. An unbound conditional primitive collapses to a constant
            # reward, which is precisely the signature this screen rejects on -- so
            # without the binding it would reject every conditional spec. When the
            # screen cannot see straight it says nothing.
            return (["required execution screen lacks declared conditioning"]
                    if self.require_training_reference else [])

        try:
            # The same binding the trainer performs per batch: a contrastive_mingap
            # aggregator needs one target index per sequence, and a target_interval
            # shape needs the per-group intervals. Screening an unbound spec would
            # reject every conditional task for the wrong reason.
            bound = (
                _bind_prompt_conditioning(
                    spec,
                    self.conditioning,
                    self.conditioning_prompts,
                    group_intervals=self.group_intervals,
                    oracle_target_indices=self.oracle_target_indices,
                    primitive_resolver=self.primitive_resolver,
                )
                if self.conditioning and self.conditioning_prompts
                else spec
            )
            # Strict screens bind the actual reference and allow constant reward
            # there; otherwise the reference is empty.
            reward_kwargs = {"training_set": self.training_set if self.require_training_reference else [],
                             "reference_libraries": dict(self.reference_libraries)}
            if self.primitive_resolver is not None:
                reward_kwargs["primitive_resolver"] = self.primitive_resolver
            reward_fn = ComposedRewardFn(RewardSpec.from_dict(bound), **reward_kwargs)
            rewards = np.asarray(
                reward_fn(
                    sequences=list(self.sequences),
                    oracle_outputs={k: self.oracle_outputs[k] for k in enabled},
                    conditioning=list(self.conditioning) if self.conditioning else None,
                ),
                dtype=float,
            )
        except IsolationError:
            raise
        except Exception as exc:  # noqa: BLE001 - the reward DSL is the boundary
            return [
                f"this specification fails to execute on a reference library of "
                f"{len(self.sequences)} sequences: {type(exc).__name__}: {exc}"
            ]

        if rewards.shape != (len(self.sequences),):
            return [
                "this specification does not produce exactly one reward per "
                f"sequence (got shape {rewards.shape})"
            ]
        if not np.all(np.isfinite(rewards)):
            n_bad = int(np.sum(~np.isfinite(rewards)))
            return [
                f"this specification produces non-finite rewards on {n_bad} of "
                f"{len(self.sequences)} reference sequences"
            ]
        training_reference = set(self.training_set) if self.require_training_reference else set()
        self.last_diagnostics = {
            **getattr(reward_fn, "last_diagnostics", {}),
            "population": "cached_reference_sequences",
            "interpretation": "Component activation here describes this reference library; use reward_execution for generated training candidates.",
            "status": "passed", "count": len(rewards),
            "reward_std": float(np.std(rewards)), "reward_min": float(np.min(rewards)),
            "reward_max": float(np.max(rewards)),
            "reference_count": len(self.training_set) if self.require_training_reference else 0,
            "training_copy_count": sum(s in training_reference for s in self.sequences)
            if self.require_training_reference else None,
            "condition_reward_std": {
                condition: float(np.std(rewards[np.array(self.conditioning) == condition]))
                for condition in sorted(set(self.conditioning))
            } if len(self.conditioning) == len(rewards) else {},
        }
        if self.reject_reference_constants and float(np.std(rewards)) <= self.min_reward_std:
            return [
                "this specification gives every sequence in the reference library "
                f"the identical reward ({float(rewards[0]):.4g}). GRPO standardizes "
                "advantages within a group. This reference-library "
                "heuristic rejects the proposal, but does not establish zero "
                "advantage on newly generated policy samples."
            ]
        return []


