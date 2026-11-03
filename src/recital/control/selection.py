"""Per-condition qualification, the declared ranking, and validation of delivery assignments.

A candidate's measured library is qualified or not per condition; qualified candidates are ranked per
condition by the success spec's declared criteria (tolerance chain, earlier candidate on a final tie); a tolerance
of "auto" ties candidates whose gap to the best is inside the standard error of that gap. The agent assigns one
completed candidate to each condition; this module says whether that assignment is legal and what the ranking would
pick, without ever consulting a source the arm cannot see.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from recital.success import SuccessSpec, best_value_error, ranking_error, ranking_value, tied_with_best

from .events import Problem

ALL_STRATUM = "all"


@dataclass(frozen=True)
class Evidence:
    """What selection needs from one finished candidate."""

    id: str
    ordinal: int
    ok: bool
    qualification: Mapping[str, Any] | None  # the `qualification` block of its delivery artifact


@dataclass(frozen=True)
class ConditionStatus:
    qualified: bool
    failing: tuple[str, ...]
    unknown: tuple[str, ...]


@dataclass(frozen=True)
class Ranked:
    candidate: str
    rank: int
    values: tuple[float | None, ...]
    tied_with: int  # candidates inside the final tolerance pool when this one was chosen, itself included


def _problem(code: str, path: str = "", detail: str = "") -> Problem:
    return Problem(code=code, path=path, detail=detail[:300])


def condition_status(qualification: Mapping[str, Any] | None, condition: str) -> ConditionStatus:
    """Requirement evidence for this condition's stratum and for the pooled `all` stratum."""
    if not qualification:
        return ConditionStatus(False, (), ("no_measurement",))
    rows = [row for row in qualification.get("requirements", ()) if row["stratum"] in (condition, ALL_STRATUM)]
    failing = tuple(sorted({row["requirement_id"] for row in rows if row["status"] == "fail"}))
    unknown = tuple(sorted({row["requirement_id"] for row in rows if row["status"] == "unknown"}))
    return ConditionStatus(bool(rows) and not failing and not unknown, failing, unknown)


def condition_values(success: SuccessSpec, qualification: Mapping[str, Any] | None,
                     condition: str) -> tuple[float | None, ...]:
    """One value per declared ranking criterion, restricted to this condition's stratum."""
    measured = (qualification or {}).get("measurements") or {}
    return tuple(ranking_value(criterion.model_copy(update={"strata": (condition,)}), measured)
                 for criterion in success.ranking)


def condition_errors(success: SuccessSpec, qualification: Mapping[str, Any] | None,
                     condition: str) -> tuple[float | None, ...]:
    """The standard error of each `condition_values` entry, None where the measurement carries none."""
    measured = (qualification or {}).get("measurements") or {}
    return tuple(ranking_error(criterion.model_copy(update={"strata": (condition,)}), measured)
                 for criterion in success.ranking)


def _winner(success: SuccessSpec,
            pool: Sequence[tuple[Evidence, tuple[float | None, ...], tuple[float | None, ...]]]):
    contenders = list(pool)
    for index, criterion in enumerate(success.ranking):
        best = (max if criterion.direction == "maximize" else min)(values[index] for _, values, _ in contenders)
        best_error = best_value_error(best, [(values[index], errors[index]) for _, values, errors in contenders])
        contenders = [(item, values, errors) for item, values, errors in contenders
                      if tied_with_best(criterion, values[index], errors[index], best, best_error)]
    return min(contenders, key=lambda entry: entry[0].ordinal), len(contenders)


def rank_condition(success: SuccessSpec, candidates: Sequence[Evidence], condition: str) -> list[Ranked]:
    """Qualified candidates with every declared ranking value, best first.

    A candidate missing a declared ranking value is unknown, not worst: it is left out rather than
    allowed to win on the remaining criteria.
    """
    remaining = []
    for item in candidates:
        if not item.ok or not condition_status(item.qualification, condition).qualified:
            continue
        values = condition_values(success, item.qualification, condition)
        if all(value is not None for value in values):
            remaining.append((item, values, condition_errors(success, item.qualification, condition)))
    ranked: list[Ranked] = []
    while remaining:
        (item, values, _), tied = _winner(success, remaining)
        ranked.append(Ranked(item.id, len(ranked) + 1, values, tied))
        remaining = [entry for entry in remaining if entry[0].id != item.id]
    return ranked


def leaderboard(success: SuccessSpec, candidates: Sequence[Evidence], conditions: Sequence[str],
                *, top: int = 5) -> dict[str, list[dict[str, Any]]]:
    metrics = [criterion.metric for criterion in success.ranking]
    return {condition: [{"candidate": row.candidate, "rank": row.rank,
                         "values": dict(zip(metrics, row.values)), "tied_with": row.tied_with}
                        for row in rank_condition(success, candidates, condition)[:top]]
            for condition in conditions}


def best_assignment(success: SuccessSpec, candidates: Sequence[Evidence],
                    conditions: Sequence[str]) -> dict[str, str] | None:
    """The declared ranking's pick per condition; None when any condition has no qualified candidate."""
    picks = {}
    for condition in conditions:
        ranked = rank_condition(success, candidates, condition)
        if not ranked:
            return None
        picks[condition] = ranked[0].candidate
    return picks


def validate_assignment(assignment: Mapping[str, str], candidates: Sequence[Evidence],
                        conditions: Sequence[str], *, require_qualified: bool) -> list[Problem]:
    problems: list[Problem] = []
    if set(assignment) != set(conditions):
        problems.append(_problem("assignment_conditions", "assignment",
                                 f"assign exactly the conditions {sorted(conditions)} once each; got {sorted(assignment)}"))
        return problems
    by_id = {item.id: item for item in candidates}
    for condition in conditions:
        name = assignment[condition]
        item = by_id.get(name)
        if item is None or not item.ok:
            problems.append(_problem("assignment_candidate", f"assignment.{condition}",
                                     f"{name!r} is not a completed candidate; completed: {sorted(i for i, e in by_id.items() if e.ok)}"))
        elif require_qualified:
            status = condition_status(item.qualification, condition)
            if not status.qualified:
                reasons = [*status.failing, *(f"unknown:{requirement}" for requirement in status.unknown)]
                problems.append(_problem("assignment_unqualified", f"assignment.{condition}",
                                         f"{name} does not qualify for {condition}: {', '.join(reasons) or 'no measurement'}"))
    return problems


def validate_no_qualified(candidates: Sequence[Evidence], conditions: Sequence[str]) -> list[Problem]:
    """`no_qualified_candidate` is legal only while some condition has no qualified candidate."""
    covered = [condition for condition in conditions
               if any(item.ok and condition_status(item.qualification, condition).qualified for item in candidates)]
    if len(covered) == len(conditions):
        return [_problem("qualified_exists", "stop.outcome",
                         "every condition has a qualified candidate; deliver an assignment instead")]
    return []
