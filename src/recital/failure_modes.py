"""Shared failure-mode vocabulary for sequence-design tasks.

The vocabulary names mechanisms that can recur across tasks.  A task adapter
supplies the concrete evidence and calibration; the shared loop never assumes
that one task's detector is valid for another.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FailureModeSpec:
    """One named failure family and the evidence/actions exposed to an agent."""

    id: str
    evidence_keys: tuple[str, ...] = ()
    candidate_actions: tuple[str, ...] = ()
    description: str = ""
    task_specific: bool = False


@dataclass(frozen=True)
class FailureProfile:
    """Shared failure families plus task-specific observational extensions."""

    shared: tuple[FailureModeSpec, ...] = field(default_factory=tuple)
    task_specific: tuple[FailureModeSpec, ...] = field(default_factory=tuple)

    def all_modes(self) -> tuple[FailureModeSpec, ...]:
        return (*self.shared, *self.task_specific)

    def by_id(self) -> dict[str, FailureModeSpec]:
        return {mode.id: mode for mode in self.all_modes()}

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "shared": [_mode_dict(mode) for mode in self.shared],
            "task_specific": [_mode_dict(mode) for mode in self.task_specific],
        }


def _mode_dict(mode: FailureModeSpec) -> dict[str, Any]:
    return {
        "id": mode.id,
        "evidence_keys": list(mode.evidence_keys),
        "candidate_actions": list(mode.candidate_actions),
        "description": mode.description,
        "task_specific": mode.task_specific,
    }


SHARED_FAILURE_MODES: tuple[FailureModeSpec, ...] = (
    FailureModeSpec(
        id="objective_disagreement",
        evidence_keys=("reward_progress", "heldout_progress", "audit_progress"),
        candidate_actions=("add_reward_oracle", "add_penalty", "add_gate"),
        description="The optimized objective and an unoptimized measurement disagree.",
    ),
    FailureModeSpec(
        id="objective_saturation",
        evidence_keys=("objective_value", "objective_variance", "reward_std"),
        candidate_actions=("change_reward_structure", "change_learning_rate"),
        description="The objective no longer provides useful ranking information.",
    ),
    FailureModeSpec(
        id="advantage_starvation",
        evidence_keys=("frac_reward_zero_std", "reward_std", "group_size"),
        candidate_actions=("increase_group_size", "change_learning_rate"),
        description="Within-group reward variance is too small to provide an advantage.",
    ),
    FailureModeSpec(
        id="policy_drift",
        evidence_keys=("base_policy_kl", "base_policy_kl_max"),
        candidate_actions=("increase_kl", "decrease_learning_rate"),
        description="The policy moved too far from its causal-LM starting point.",
    ),
    FailureModeSpec(
        id="distribution_shift",
        evidence_keys=("reference_distribution", "generated_distribution"),
        candidate_actions=("add_penalty", "add_gate", "increase_kl"),
        description="Generated sequences leave the declared reference distribution.",
    ),
    FailureModeSpec(
        id="constraint_violation",
        evidence_keys=("invalid_fraction", "constraint_statistics"),
        candidate_actions=("add_gate", "change_generation_constraints"),
        description="Generated completions violate a declared task constraint.",
    ),
    FailureModeSpec(
        id="oracle_unavailable",
        evidence_keys=("oracle_status", "finite_fraction"),
        candidate_actions=("stop", "change_evaluator"),
        description="A required measurement cannot score the current candidates.",
    ),
    FailureModeSpec(
        id="oracle_scale_mismatch",
        evidence_keys=("oracle_units", "calibration_status"),
        candidate_actions=("calibrate_oracle", "change_aggregator"),
        description="Oracle values cannot be combined without task-declared calibration.",
    ),
)


def shared_failure_profile() -> FailureProfile:
    """Return a fresh immutable profile containing the shared failure families."""
    return FailureProfile(shared=SHARED_FAILURE_MODES)
