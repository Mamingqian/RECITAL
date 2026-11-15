"""M3's shadow training: how the board's veto is measured, and what a bad outcome is.

A veto keeps a candidate from training, so nothing says whether it was right. The design that does:

  1. When the arbiter vetoes a candidate, a seeded draw decides whether a *shadow* of it is trained anyway. The draw is
     a function of the run seed, the decision and the candidate's position (`draw`), so the allocation is the same on
     every replay of the log and does not depend on the model. `shadow_fraction` of the vetoes are drawn (default 0.5).
  2. Shadows train under their own budget of extra updates, `shadow_budget_fraction` of the run's total updates
     (default 0.15); a veto whose steps do not fit in what is left is not shadowed (`allocate`). Shadow updates are
     not committed updates: `RunState.committed_updates`, the remaining budget, the rounds and the candidate budget
     never see them.
  3. A shadow is not a candidate. It is never in `RunState.candidates`, so `projection.evidence_of`, the ledger, the
     leaderboard, selection and delivery cannot contain it, and no prompt names it. Its job runs under
     `<run>/shadow/<id>`; the module's own events record its allocation, submission and result.
  4. When a shadow finishes, `realized_outcome` labels it with the same rule that labels every accepted candidate:
     the candidate ends unqualified in some condition, or its declared ranking is below its reference's. The share of
     shadows with a bad outcome is the veto's precision; the share of vetoed candidates that would have beaten their
     reference is the false-veto cost; the risk bin of every accepted candidate against its own outcome is the
     reviewers' calibration curve.

This file holds the pure parts: the outcome rule, the draw and the allocation. The module (`board.py`) writes the
events and runs the jobs.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from recital.success import SuccessSpec, ranking_error, ranking_value, tied_with_best

from ..selection import condition_status

SHADOW_FRACTION = 0.5
SHADOW_BUDGET_FRACTION = 0.15


# ----- the outcome rule -----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Outcome:
    """What a trained candidate came to, against the row it was built from.

    `bad` is the label every board measurement uses: the job did not finish, a condition ended unqualified, or the
    declared ranking is below the reference's. `comparison` is -1, 0 or 1 for below, tied with or above the reference
    (None when there is no reference to compare with).
    """

    bad: bool
    why: str                 # ok, failed, unqualified or below_reference
    comparison: int | None


def compare_rankings(success: SuccessSpec, qualification: Mapping[str, Any], reference: Mapping[str, Any]) -> int | None:
    """-1, 0 or 1: the declared ranking chain of `qualification` against `reference` (the tolerances decide ties).

    The library's value of a criterion is `ranking_value` over the criterion's own strata, which applies the declared
    reduction. None when the reference carries no value to compare with; a candidate that lacks a value the reference
    has is below it.
    """
    measured, other = qualification.get("measurements") or {}, reference.get("measurements") or {}
    for criterion in success.ranking:
        mine, theirs = ranking_value(criterion, measured), ranking_value(criterion, other)
        if theirs is None:
            return None
        if mine is None:
            return -1
        try:
            tied = tied_with_best(criterion, mine, ranking_error(criterion, measured), theirs,
                                  ranking_error(criterion, other))
        except ValueError:      # an 'auto' tolerance with no standard error on one side: compare exactly
            tied = mine == theirs
        if not tied:
            return 1 if (mine > theirs) == (criterion.direction == "maximize") else -1
    return 0


def realized_outcome(success: SuccessSpec, conditions: Sequence[str], row: Mapping[str, Any],
                     reference: Mapping[str, Any] | None) -> Outcome:
    """The outcome of the candidate whose frozen ledger row is `row`, against the row `reference` (or None)."""
    if row["status"] != "ok":
        return Outcome(True, "failed", None)
    qualification = row["qualification"]
    comparison = None
    if reference is not None and reference.get("status") == "ok":
        comparison = compare_rankings(success, qualification, reference["qualification"])
    if not all(condition_status(qualification, condition).qualified for condition in conditions):
        return Outcome(True, "unqualified", comparison)
    if comparison is not None and comparison < 0:
        return Outcome(True, "below_reference", comparison)
    return Outcome(False, "ok", comparison)


# ----- the allocation ------------------------------------------------------------------------------------------------

def draw(seed: int, decision_id: str, index: int) -> float:
    """The uniform draw of a vetoed candidate: a function of the run seed, the decision and the candidate's position."""
    return random.Random(f"shadow:{seed}:{decision_id}:{index}").random()


@dataclass(frozen=True)
class Allocation:
    shadow: bool
    why: str                 # drawn, not_drawn, over_budget or off


def allocate(*, seed: int, decision_id: str, index: int, steps: int, fraction: float, spent: int,
             budget: int) -> Allocation:
    """Whether the vetoed candidate at `index` of `decision_id` is trained as a shadow.

    `spent` is the steps of the shadows already allocated and `budget` the run's shadow budget in updates.
    """
    if fraction <= 0:
        return Allocation(False, "off")
    if draw(seed, decision_id, index) >= fraction:
        return Allocation(False, "not_drawn")
    if spent + steps > budget:
        return Allocation(False, "over_budget")
    return Allocation(True, "drawn")


def shadow_budget(total_updates: int, fraction: float) -> int:
    return int(total_updates * fraction)


def shadow_id(ordinal: int) -> str:
    """`shadow_<ordinal>`: a training action identifier (`jobs.ACTION_ID`) whose integer is the ordinal the candidate
    would have had in its slot, so its training and evaluation streams are the ones a candidate there would draw."""
    return f"shadow_{ordinal:06d}"


# ----- what the vetoes taught ----------------------------------------------------------------------------------------

def veto_report(shadows: Sequence[Mapping[str, Any]], accepted: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The measured veto from finished shadows and the outcomes of the candidates that were accepted.

    `shadows` and `accepted` are mappings with `bad` (bool) and `risk` (the board's mean risk); `shadows` also carry
    `comparison`. Precision is the share of finished shadows with a bad outcome; `base_rate` is the share of accepted
    candidates with one; `false_vetoes` are shadows that did not end bad.
    """
    n = len(shadows)
    bad = sum(1 for item in shadows if item["bad"])
    rate = (sum(1 for item in accepted if item["bad"]) / len(accepted)) if accepted else None
    return {"shadows": n, "bad": bad, "precision": bad / n if n else None, "base_rate": rate,
            "false_vetoes": n - bad,
            "better_than_reference": sum(1 for item in shadows if (item.get("comparison") or 0) > 0)}
