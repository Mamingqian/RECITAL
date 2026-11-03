"""The declared update budget of a final-library search: how deep a candidate may train and how many may.

CM declares it under `framework.search_budget`; the adapter reads it to bound the step choices it leaves the
agent and to check that the declared depths fit its `max_steps_per_iter`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class SearchBudgetConfig:
    max_total_updates: int
    first_candidate_updates: int
    allowed_candidate_updates: tuple[int, ...]
    eval_interval_updates: int
    max_eval_slots: int
    max_trained_candidates: int
    max_proposals: int
    max_synthesis_attempts: int
    evaluation_mode: str = "checkpoint"

    def __post_init__(self) -> None:
        for name in (
            "max_total_updates", "first_candidate_updates", "eval_interval_updates",
            "max_eval_slots", "max_trained_candidates", "max_proposals",
            "max_synthesis_attempts",
        ):
            _integer(getattr(self, name), name, 1)
        if self.evaluation_mode not in {"checkpoint", "final"}:
            raise ValueError("evaluation_mode must be checkpoint or final")
        choices = tuple(self.allowed_candidate_updates)
        if not choices or len(set(choices)) != len(choices):
            raise ValueError("allowed_candidate_updates must be nonempty and unique")
        for value in choices:
            _integer(value, "allowed_candidate_updates", 1)
        if self.first_candidate_updates > self.max_total_updates:
            raise ValueError("first candidate exceeds the total update budget")
        object.__setattr__(self, "allowed_candidate_updates", choices)


def validate_final_candidate_budget(config: SearchBudgetConfig, *, max_steps: int, proposals: int) -> None:
    """Check a final-only search without equating initial depth with the depth cap."""
    if proposals != config.max_proposals + 1:
        raise ValueError("outer iteration budget must equal max_proposals + 1")
    if max((config.first_candidate_updates, *config.allowed_candidate_updates)) > max_steps:
        raise ValueError("CLI step ceiling is below a declared candidate duration")
    if config.evaluation_mode != "final" and (
        config.allowed_candidate_updates != (config.first_candidate_updates,)
        or config.eval_interval_updates != config.first_candidate_updates
    ):
        raise ValueError("variable-depth final-only training requires evaluation_mode=final")
