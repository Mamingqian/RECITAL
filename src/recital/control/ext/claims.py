"""Module M1, claims: what the Strategist says a candidate will measure, scored against the frozen ledger row.

A claim is one typed prediction about one measurement of one condition of one candidate's pool, against the
candidate's parent or the initial policy. When the candidate's row freezes, `resolve` turns the claim into hit, miss
or unresolvable from the row alone: a contrast of two pool medians is divided by its standard error and cut at 2, 4
and 10 standard errors, so a bin says how far the pool moved in units of what the pool itself resolves. Nothing here
calls a model and nothing reads a measurement a row does not carry.

Everything the agent is later shown about its claims (a verdict beside a row, a calibration table, a lessons table,
a dead-end list) is computed here from rows. A lesson, whoever wrote it, is kept only when `validate_lesson`
reproduces it from the candidates it cites.

The functions are pure. A row is the full ledger row (`ledger.build_row`), with the pool standard errors under
`row["ext"]["claims"]["pool_se"]`; the initial policy is a row-shaped mapping with the same two parts.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Collection, Iterable, Literal, Mapping, Sequence

import numpy as np

from recital.experiment_tasks import median_measurement
from recital.sequence_distance import pairwise_normalized

from .. import events as ev
from ..errors import BlindingViolation, ControlError, LedgerCorruption, PromptBudgetExceeded, RunTokenBudgetExceeded
from ..events import Problem
from ..health import REQUIREMENT_ID
from ..ledger import composition_bands, medians as pool_medians, shown_measurements
from ..projection import Block
from ..schemas import name_schema, object_schema
from ..state import spend
from . import Extension, Phase, event, facts, folds
from .analyst import ANALYST_INSTRUCTIONS, analyst_schema, run_analyst
from .edit_class import LEVER_CLASSES, EditClass, edit_classes

MAX_CLAIMS = 2
DIRECTIONS = ("up", "down", "flat")
MAGNITUDES = ("none", "small", "medium", "large")
VERSUS = ("parent", "initial")
CONFIDENCES = (0.5, 0.65, 0.75, 0.85, 0.95)
FAILURE_MODES = ("none", "unqualified", "collapse", "exploit")
# A contrast of fewer standard errors than this is flat; small, medium and large start at these.
SE_SMALL, SE_MEDIUM, SE_LARGE = 2.0, 4.0, 10.0
COMPOSITION_PREFIX = "composition:"
# The pseudo-measurement of a lesson about a condition entering or leaving qualification.
QUALIFIED = "qualified"
# A lesson cites at least this many candidates; a lesson note has at most this many words.
MIN_SUPPORT = 2
NOTE_WORDS = 30
HIT, MISS, UNRESOLVABLE = "hit", "miss", "unresolvable"


# ----- the claim object -------------------------------------------------------------------------------------------

def claim_measurements(shown: Iterable[str], banded: Iterable[str]) -> list[str]:
    """The names a claim may target: every shown scalar, and `composition:<name>` for each measurement with a band."""
    return [*shown, *(COMPOSITION_PREFIX + name for name in sorted(banded))]


def claim_schema(*, conditions: Sequence[str], measurements: Sequence[str], lever_classes: Sequence[str]) -> dict[str, Any]:
    """One claim, in the strict structured-output subset (`schemas.assert_strict_compatible`)."""
    return object_schema({
        "lever_class": name_schema(lever_classes, "the lever this claim attributes the effect to; the candidate must edit it"),
        "target_measurement": name_schema(
            measurements, f"a shown scalar of the pool line, or {COMPOSITION_PREFIX}<name> for that measurement's "
                          "composition flag"),
        "condition": name_schema(conditions, "the condition the claim is about"),
        "direction": {"type": "string", "enum": list(DIRECTIONS),
                      "description": f"up or down: the pool median moves by at least {SE_SMALL:g} standard errors; "
                                     f"flat: it moves by less. For {COMPOSITION_PREFIX}<name>: up is above the "
                                     "initial policy's band, down below it, flat within it"},
        "magnitude": {"type": "string", "enum": list(MAGNITUDES),
                      "description": f"small {SE_SMALL:g} to {SE_MEDIUM:g}, medium {SE_MEDIUM:g} to {SE_LARGE:g}, "
                                     f"large above {SE_LARGE:g} standard errors; none for flat and for a composition flag"},
        "versus": {"type": "string", "enum": list(VERSUS),
                   "description": "the candidate's parent, or the initial policy; a composition flag is always "
                                  "against the initial policy's band"},
        "confidence": {"type": "number", "enum": list(CONFIDENCES),
                       "description": "probability that the direction and magnitude are both right"},
        "failure_mode": {"type": "string", "enum": list(FAILURE_MODES),
                         "description": "how the candidate fails in this condition: unqualified, collapse (library "
                                        "health fails), exploit (a composition flag is raised), or none"},
    })


def claims_fragment(*, conditions: Sequence[str], measurements: Sequence[str],
                    lever_classes: Sequence[str]) -> dict[str, Any]:
    """The property the module adds to each candidate object."""
    return {"claims": {"type": "array", "maxItems": MAX_CLAIMS,
                       "items": claim_schema(conditions=conditions, measurements=measurements,
                                             lever_classes=lever_classes),
                       "description": "one or two claims about what this candidate's pool will measure"}}


def claim_problems(claims: Sequence[Mapping[str, Any]], edits: Sequence[EditClass], *, min_claims: int = 1) -> list[Problem]:
    """Cross-field rules the schema cannot state; each is a typed refusal, never a repair.

    A candidate that edits no recipe field (it continues its parent) may name any lever: the claim then attributes the
    effect to the further training.
    """
    problems: list[Problem] = []
    if len(claims) < min_claims:
        problems.append(Problem(code="claims_missing", path="claims",
                                detail=f"state at least {min_claims} claim about what this candidate will measure"))
    edited = {edit.lever for edit in edits}
    seen: set[tuple[str, str, str]] = set()
    for index, claim in enumerate(claims):
        path = f"claims[{index}]"
        if edited and claim["lever_class"] not in edited:
            problems.append(Problem(code="claim_lever_not_edited", path=path + ".lever_class",
                                    detail=f"{claim['lever_class']!r} is not a lever this candidate edits "
                                           f"(it edits {sorted(edited) or 'nothing'})"))
        composition = claim["target_measurement"].startswith(COMPOSITION_PREFIX)
        if claim["direction"] == "flat" or composition:
            if claim["magnitude"] != "none":
                problems.append(Problem(code="claim_magnitude", path=path + ".magnitude",
                                        detail="magnitude is none for a flat claim and for a composition flag"))
        elif claim["magnitude"] == "none":
            problems.append(Problem(code="claim_magnitude", path=path + ".magnitude",
                                    detail="an up or down claim on a scalar names small, medium or large"))
        if composition and claim["versus"] != "initial":
            problems.append(Problem(code="claim_versus", path=path + ".versus",
                                    detail="a composition flag is a band of the initial policy: versus initial"))
        key = (claim["target_measurement"], claim["condition"], claim["versus"])
        if key in seen:
            problems.append(Problem(code="claim_duplicate", path=path,
                                    detail="two claims name the same measurement, condition and baseline"))
        seen.add(key)
    return problems


# ----- standard errors --------------------------------------------------------------------------------------------

def pool_standard_errors(
    measurements: Mapping[str, Mapping[str, Any]],
    labels: Sequence[str],
    conditions: Sequence[str],
    names: Iterable[str],
    *,
    ratios: Mapping[str, tuple[str, Mapping[str, float]]] | None = None,
) -> dict[str, dict[str, float | None]]:
    """The standard error of each pool median, per condition and measurement, from the pool's per-row values.

    The error is the larger of two: the bootstrap error `experiment_tasks.median_measurement` gives a library (1,000
    resamples, seed 0) and the normal-theory error of a median, sqrt(pi/2) times the standard error of the mean. A
    measurement with a few distinct values has a bootstrap error of zero whenever the median sits on one of them, so
    the second keeps every error positive wherever the pool has any spread. `measurements` maps a name to a record
    with the per-row `values`, `labels` is each row's condition. A name in `ratios` is a measurement the ledger shows
    divided by a per-condition constant (`unique_kmer_ratio`): its error is the error of the named source
    measurement divided by the same constant.
    """
    labels = np.asarray(labels)
    ratios = ratios or {}
    errors: dict[str, dict[str, float | None]] = {condition: {} for condition in conditions}
    for name in names:
        source, scale = ratios.get(name, (name, None))
        values = np.asarray(measurements[source]["values"], dtype=float)
        if values.ndim != 1:
            continue
        for condition in conditions:
            finite = values[labels == condition]
            finite = finite[np.isfinite(finite)]
            error = median_measurement(finite).get("standard_error")
            if error is not None:
                error = max(error, math.sqrt(math.pi / 2.0) * float(finite.std(ddof=1)) / math.sqrt(len(finite)))
            if error is not None and scale is not None:
                error = error / scale[condition]
            errors[condition][name] = error
    return errors


def pairwise_ratio_standard_error(distances: np.ndarray, reference_mean_pairwise: float) -> float:
    """Delete-one-row jackknife of `library_health`'s `pairwise_ratio` from the pool's pairwise distance matrix.

    `distances` is the symmetric normalized edit distance matrix `health._spread` builds; the ratio is the mean
    pairwise distance over `reference_mean_pairwise`.
    """
    n = len(distances)
    row_sums = distances.sum(axis=1)
    total = row_sums.sum()
    leave_one_out = (total - 2.0 * row_sums) / ((n - 1) * (n - 2))
    spread = math.sqrt((n - 1) / n * float(((leave_one_out - leave_one_out.mean()) ** 2).sum()))
    return spread / reference_mean_pairwise


def legal_distances(measurements: Mapping[str, Mapping[str, Any]], pool: Mapping[str, Any], condition: str) -> np.ndarray:
    """The normalized edit distance matrix of one condition's legal pool rows, as `health` builds it."""
    legal = np.asarray(measurements["legal"]["values"], dtype=bool)
    labels = np.asarray(pool["conditions"])
    chosen = [pool["sequences"][i] for i in np.flatnonzero((labels == condition) & legal)]
    if len(chosen) > 512:
        raise ControlError("library_health subsamples a pool above 512 rows; the claims module reads them all")
    return pairwise_normalized(chosen)


def pool_errors(
    measurements: Mapping[str, Mapping[str, Any]],
    pool: Mapping[str, Any],
    conditions: Sequence[str],
    names: Iterable[str],
    reference_mean_pairwise: Mapping[str, float],
    reference_unique_kmers: Mapping[str, float | None],
) -> dict[str, dict[str, float | None]]:
    """The pool standard error of every shown measurement, per condition: what `row_fields` stores in a ledger row.

    `measurements` and `pool` are a pool's `public_measurements.json` (the `measurements` part) and `pool.json`. A
    name the pool has no per-row values for is left out, so the claim vocabulary and the errors come from the same
    names. `pairwise_ratio` is the jackknife of `pairwise_ratio_standard_error`; `unique_kmer_ratio` divides the error
    of the unique k-mer count by the reference's median.
    """
    names = list(names)
    ratios = {}
    if "unique_kmer_count" in measurements and all(reference_unique_kmers.get(c) for c in conditions):
        ratios["unique_kmer_ratio"] = ("unique_kmer_count", dict(reference_unique_kmers))
    plain = [name for name in names if name != "pairwise_ratio" and (name in measurements or name in ratios)]
    errors = pool_standard_errors(measurements, pool["conditions"], conditions, plain, ratios=ratios)
    if "pairwise_ratio" in names:
        for condition in conditions:
            errors[condition]["pairwise_ratio"] = pairwise_ratio_standard_error(
                legal_distances(measurements, pool, condition), reference_mean_pairwise[condition])
    return errors


def row_ext(pool_se: Mapping[str, Mapping[str, float | None]]) -> dict[str, Any]:
    """A ledger row's `ext` as the module leaves it: the module's own fields under `claims`."""
    return {"claims": {"pool_se": {condition: dict(values) for condition, values in pool_se.items()}}}


def initial_reference(pool: Mapping[str, Mapping[str, float | None]],
                      pool_se: Mapping[str, Mapping[str, float | None]]) -> dict[str, Any]:
    """The initial policy's pool as a row-shaped reference for `versus: initial`."""
    return {"id": "initial", "status": "ok", "parent": None, "pool": pool, "composition": {}, "qualified": {},
            "ext": row_ext(pool_se)}


# ----- the resolver -----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Contrast:
    delta: float
    se: float
    z: float


@dataclass(frozen=True)
class Resolution:
    """`verdict` is hit when direction and magnitude are both right; the two parts are kept for the calibration.

    `failure_modes` is every mode the row shows for the condition, and `failure_mode_hit` whether the claimed one is
    among them; it does not enter `verdict`.
    """

    verdict: str
    direction: str | None = None
    magnitude: str | None = None
    z: float | None = None
    direction_hit: bool | None = None
    magnitude_hit: bool | None = None
    failure_modes: tuple[str, ...] = ()
    failure_mode_hit: bool | None = None
    reason: str = ""


def direction_of(z: float) -> str:
    return "up" if z >= SE_SMALL else "down" if z <= -SE_SMALL else "flat"


def magnitude_of(z: float) -> str:
    size = abs(z)
    return ("none" if size < SE_SMALL else "small" if size < SE_MEDIUM else "medium" if size < SE_LARGE else "large")


def _measurement(row: Mapping[str, Any], condition: str, name: str) -> float | None:
    return ((row.get("pool") or {}).get(condition) or {}).get(name)


def _error(row: Mapping[str, Any], condition: str, name: str) -> float | None:
    pool_se = (((row.get("ext") or {}).get("claims") or {}).get("pool_se") or {})
    return (pool_se.get(condition) or {}).get(name)


def contrast(row: Mapping[str, Any], reference: Mapping[str, Any], condition: str, name: str) -> Contrast | str:
    """The change of a pool median against a reference in standard errors, or the reason there is none."""
    now, then = _measurement(row, condition, name), _measurement(reference, condition, name)
    if now is None or then is None:
        return "measurement_missing"
    errors = _error(row, condition, name), _error(reference, condition, name)
    if None in errors:
        return "no_standard_error"
    delta, se = now - then, math.hypot(*errors)
    if se == 0.0:
        return Contrast(delta, 0.0, 0.0 if delta == 0.0 else math.copysign(math.inf, delta))
    return Contrast(delta, se, delta / se)


def failure_modes(row: Mapping[str, Any], condition: str) -> tuple[str, ...]:
    """Every failure the row shows for one condition: `none` only when it shows no other."""
    found = []
    qualified = (row.get("qualified") or {}).get(condition) or {}
    if REQUIREMENT_ID in (qualified.get("failing") or ()):
        found.append("collapse")
    if (row.get("composition") or {}).get(condition):
        found.append("exploit")
    if not qualified.get("qualified", True):
        found.append("unqualified")
    return tuple(found) or ("none",)


def reference_for(row: Mapping[str, Any], versus: str, rows: Mapping[str, Mapping[str, Any]],
                  initial: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The row a claim's `versus` names: the initial policy, or the parent (the initial policy when it has none)."""
    if versus == "initial":
        return initial
    parent = row.get("parent")
    if parent in (None, "initial"):
        return initial
    found = rows.get(parent)
    return found if found is not None and found.get("pool") else None


def resolve(claim: Mapping[str, Any], row: Mapping[str, Any], reference: Mapping[str, Any] | None) -> Resolution:
    """Hit, miss or unresolvable, from the candidate's row and the reference row the claim names."""
    if row.get("status") != "ok" or not row.get("pool"):
        return Resolution(UNRESOLVABLE, reason="no_pool")
    if reference is None:
        return Resolution(UNRESOLVABLE, reason="no_reference")
    condition, name = claim["condition"], claim["target_measurement"]
    modes = failure_modes(row, condition)
    mode_hit = claim["failure_mode"] in modes
    if name.startswith(COMPOSITION_PREFIX):
        if claim["versus"] != "initial":
            return Resolution(UNRESOLVABLE, reason="composition_is_against_the_initial_band")
        flag = ((row.get("composition") or {}).get(condition) or {}).get(name[len(COMPOSITION_PREFIX):])
        direction = "flat" if flag is None else {"above": "up", "below": "down"}[flag["side"]]
        hit = claim["direction"] == direction
        return Resolution(HIT if hit else MISS, direction=direction, magnitude="none", direction_hit=hit,
                          magnitude_hit=True, failure_modes=modes, failure_mode_hit=mode_hit)
    found = contrast(row, reference, condition, name)
    if isinstance(found, str):
        return Resolution(UNRESOLVABLE, reason=found)
    direction, magnitude = direction_of(found.z), magnitude_of(found.z)
    direction_hit, magnitude_hit = claim["direction"] == direction, claim["magnitude"] == magnitude
    return Resolution(HIT if direction_hit and magnitude_hit else MISS, direction=direction, magnitude=magnitude,
                      z=found.z, direction_hit=direction_hit, magnitude_hit=magnitude_hit,
                      failure_modes=modes, failure_mode_hit=mode_hit)


@dataclass(frozen=True)
class ScoredClaim:
    candidate: str
    index: int
    claim: Mapping[str, Any]
    resolution: Resolution

    @property
    def confidence(self) -> float:
        return float(self.claim["confidence"])


def score_claims(claims: Mapping[str, Sequence[Mapping[str, Any]]], rows: Mapping[str, Mapping[str, Any]],
                 initial: Mapping[str, Any]) -> list[ScoredClaim]:
    """Every candidate's claims resolved against its own row, in candidate-id order."""
    scored = []
    for candidate in sorted(claims):
        row = rows[candidate]
        for index, claim in enumerate(claims[candidate]):
            reference = reference_for(row, claim["versus"], rows, initial)
            scored.append(ScoredClaim(candidate, index, claim, resolve(claim, row, reference)))
    return scored


# ----- calibration and the two chance baselines -------------------------------------------------------------------

def _outcome(item: ScoredClaim, use: str) -> bool:
    return item.resolution.verdict == HIT if use == "hit" else bool(item.resolution.direction_hit)


def calibration_table(scored: Sequence[ScoredClaim], *, use: str = "hit") -> dict[str, Any]:
    """Hit rate by confidence bin, and the Brier score of the stated confidences.

    `use` picks the outcome: `hit` (direction and magnitude) or `direction`. Unresolvable claims are counted and
    left out of every rate. `brier_half` is the Brier score of always saying 0.5.
    """
    resolved = [item for item in scored if item.resolution.verdict != UNRESOLVABLE]
    bins = []
    for confidence in CONFIDENCES:
        inside = [item for item in resolved if item.confidence == confidence]
        hits = sum(_outcome(item, use) for item in inside)
        bins.append({"confidence": confidence, "n": len(inside), "hits": hits,
                     "hit_rate": hits / len(inside) if inside else None})
    outcomes = [_outcome(item, use) for item in resolved]
    return {
        "use": use, "n_claims": len(scored), "n_resolved": len(resolved), "n_unresolvable": len(scored) - len(resolved),
        "bins": bins,
        "hit_rate": float(np.mean(outcomes)) if outcomes else None,
        "mean_confidence": float(np.mean([item.confidence for item in resolved])) if resolved else None,
        "brier": float(np.mean([(item.confidence - hit) ** 2 for item, hit in zip(resolved, outcomes)])) if resolved else None,
        "brier_half": float(np.mean([(0.5 - hit) ** 2 for hit in outcomes])) if resolved else None}


def realized_outcome(row: Mapping[str, Any], reference: Mapping[str, Any], condition: str, name: str,
                     ) -> tuple[str, str] | None:
    """(direction, magnitude) a row shows for one claim target, or None when it cannot be told."""
    if row.get("status") != "ok" or not row.get("pool"):
        return None
    if name.startswith(COMPOSITION_PREFIX):
        flag = ((row.get("composition") or {}).get(condition) or {}).get(name[len(COMPOSITION_PREFIX):])
        return ("flat" if flag is None else {"above": "up", "below": "down"}[flag["side"]], "none")
    found = contrast(row, reference, condition, name)
    return None if isinstance(found, str) else (direction_of(found.z), magnitude_of(found.z))


def base_rate(item: ScoredClaim, rows: Mapping[str, Mapping[str, Any]], initial: Mapping[str, Any]) -> float | None:
    """How often the other candidates of the search realize this claim's direction and magnitude on its target.

    This is what a predictor that knows only the marginal outcome of the target would score; None when no other
    candidate shows the target.
    """
    claim = item.claim
    wanted = (claim["direction"], "none" if claim["target_measurement"].startswith(COMPOSITION_PREFIX)
              else claim["magnitude"])
    found = []
    for candidate, row in rows.items():
        if candidate == item.candidate:
            continue
        reference = reference_for(row, claim["versus"], rows, initial)
        outcome = None if reference is None else realized_outcome(
            row, reference, claim["condition"], claim["target_measurement"])
        if outcome is not None:
            found.append(outcome == wanted)
    return float(np.mean(found)) if found else None


def baseline_comparison(claims: Mapping[str, Sequence[Mapping[str, Any]]], rows: Mapping[str, Mapping[str, Any]],
                        initial: Mapping[str, Any], *, draws: int = 1000, seed: int = 0) -> dict[str, Any]:
    """The hit rate against the base rate and against claims shuffled over the search's candidates.

    The shuffled control gives each candidate another candidate's claims and resolves them against its row, `draws`
    times; `shuffled_p` is the share of draws (counting the observed one) whose hit rate reaches the observed one.
    """
    scored = score_claims(claims, rows, initial)
    resolved = [item for item in scored if item.resolution.verdict != UNRESOLVABLE]
    observed = float(np.mean([item.resolution.verdict == HIT for item in resolved])) if resolved else None
    rates = [rate for rate in (base_rate(item, rows, initial) for item in resolved) if rate is not None]
    candidates = sorted(claims)
    rng = random.Random(seed)
    null = []
    for _ in range(draws if resolved else 0):
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        hits = [item.resolution.verdict for item in score_claims(
            {target: claims[source] for target, source in zip(candidates, shuffled)}, rows, initial)
            if item.resolution.verdict != UNRESOLVABLE]
        if hits:
            null.append(float(np.mean([verdict == HIT for verdict in hits])))
    return {
        "n_resolved": len(resolved), "hit_rate": observed,
        "base_rate": float(np.mean(rates)) if rates else None,
        "shuffled_mean": float(np.mean(null)) if null else None,
        "shuffled_p": (1 + sum(value >= observed for value in null)) / (1 + len(null)) if null else None}


# ----- evidence, context and lessons ------------------------------------------------------------------------------

@dataclass(frozen=True)
class Context:
    """The state of the row an edit starts from: a composition flag raised, library health failing."""

    composition: bool
    health_failing: bool

    @property
    def label(self) -> str:
        return (("composition flag" if self.composition else "no composition flag") + ", "
                + ("library health failing" if self.health_failing else "library health ok"))


def context_of(reference: Mapping[str, Any]) -> Context:
    qualified = (reference.get("qualified") or {}).values()
    return Context(
        composition=any(bool(found) for found in (reference.get("composition") or {}).values()),
        health_failing=any(REQUIREMENT_ID in (item.get("failing") or ()) for item in qualified))


@dataclass(frozen=True)
class Evidence:
    """One candidate's realized effect on one measurement of one condition, with the edits and context it came from."""

    candidate: str
    edits: tuple[EditClass, ...]
    context: Context
    measurement: str
    condition: str
    direction: str
    magnitude: str
    z: float


def evidence_from_rows(rows: Mapping[str, Mapping[str, Any]], initial: Mapping[str, Any],
                       edits: Mapping[str, Sequence[EditClass]], measurements: Sequence[str],
                       conditions: Sequence[str], *, anchor: str | None = None) -> list[Evidence]:
    """Every scalar effect of every candidate that edits something, plus whether its qualification flipped.

    The contrast is against the candidate's parent. A candidate that starts from the initial policy is contrasted with
    `anchor`, the candidate that trains the initial recipe unchanged, when there is one: the two differ by the edits
    alone, where the initial policy's pool differs by the training too. A flip of a condition's qualification is the
    measurement `qualified`, up when the candidate qualifies where the reference did not, down for the reverse; its
    size is infinite.
    """
    found = []
    for candidate in sorted(rows):
        row = rows[candidate]
        if row.get("status") != "ok" or not row.get("pool") or not edits.get(candidate) or row.get("anchor"):
            continue
        reference = reference_for(row, "parent", rows, initial)
        base = rows.get(anchor) if anchor is not None and row.get("parent") in (None, "initial") else None
        if base is not None and base.get("pool"):
            reference = base
        if reference is None:
            continue
        context = context_of(reference)
        for condition in conditions:
            for name in measurements:
                if name.startswith(COMPOSITION_PREFIX):
                    continue
                effect = contrast(row, reference, condition, name)
                if not isinstance(effect, str):
                    found.append(Evidence(candidate, tuple(edits[candidate]), context, name, condition,
                                          direction_of(effect.z), magnitude_of(effect.z), effect.z))
            before = ((reference.get("qualified") or {}).get(condition) or {}).get("qualified")
            after = ((row.get("qualified") or {}).get(condition) or {}).get("qualified")
            if before is not None and after is not None:
                flip = int(bool(after)) - int(bool(before))
                found.append(Evidence(candidate, tuple(edits[candidate]), context, QUALIFIED, condition,
                                      "up" if flip > 0 else "down" if flip < 0 else "flat",
                                      "none" if flip == 0 else "large", math.inf * flip if flip else 0.0))
    return found


def class_label(edit: EditClass, *, fine: bool) -> str:
    return edit.key if fine else ":".join(edit.coarse)


def _carries(evidence: Evidence, label: str) -> bool:
    return any(label in (edit.key, ":".join(edit.coarse)) for edit in evidence.edits)


@dataclass(frozen=True)
class Lesson:
    edit_class: str
    context: str
    measurement: str
    condition: str
    direction: str
    cited: tuple[str, ...]
    n_carrying: int
    n_against: int

    def as_dict(self) -> dict[str, Any]:
        return {"edit_class": self.edit_class, "context": self.context, "measurement": self.measurement,
                "condition": self.condition, "direction": self.direction, "cited": list(self.cited),
                "n_carrying": self.n_carrying, "n_against": self.n_against}


def build_lessons(evidence: Sequence[Evidence], *, fine: bool = False, min_support: int = MIN_SUPPORT,
                  max_entries: int = 12, priority: Collection[str] = ()) -> list[Lesson]:
    """The lessons table: edit class x context, per measurement and condition, from the realized effects.

    A cell of the table holds the candidates that carry the class from a row in that context. It becomes a lesson when
    at least `min_support` of them moved the measurement the same way by two standard errors or more and more of them
    did than moved it the other way. A candidate with several edit classes counts for each of them; the table does not
    separate them. Entries on a measurement in `priority` come first; within each group they are ordered by support, then
    by the size of the effect, and cut to `max_entries`.
    """
    cells: dict[tuple[str, str, str, str], list[Evidence]] = defaultdict(list)
    for item in evidence:
        for label in sorted({class_label(edit, fine=fine) for edit in item.edits}):
            cells[(label, item.context.label, item.measurement, item.condition)].append(item)
    ranked = []
    for (label, context, name, condition), items in cells.items():
        up = [item for item in items if item.direction == "up"]
        down = [item for item in items if item.direction == "down"]
        winner, loser = (up, down) if len(up) >= len(down) else (down, up)
        if len(winner) < min_support or len(winner) <= len(loser):
            continue
        strength = float(np.mean([min(abs(item.z), 1e9) for item in winner]))
        ranked.append((name not in priority, -len(winner), -strength, (label, context, name, condition), Lesson(
            label, context, name, condition, winner[0].direction, tuple(sorted(item.candidate for item in winner)),
            len(items), len(loser))))
    ranked.sort(key=lambda entry: entry[:4])
    return [entry[4] for entry in ranked[:max_entries]]


def validate_lesson(lesson: Mapping[str, Any], evidence: Sequence[Evidence], *, min_support: int = MIN_SUPPORT,
                    note_words: int = NOTE_WORDS) -> list[str]:
    """Why a lesson does not reproduce from the rows; empty when it does.

    Every cited candidate must exist, carry the edit class, start from the stated context and show the stated
    direction on the stated measurement and condition by two standard errors or more.
    """
    reasons = []
    cited = list(lesson.get("cited") or ())
    if len(set(cited)) != len(cited):
        reasons.append("a candidate is cited twice")
    if len(set(cited)) < min_support:
        reasons.append(f"cites {len(set(cited))} candidates, fewer than {min_support}")
    if len(str(lesson.get("note", "")).split()) > note_words:
        reasons.append(f"note is longer than {note_words} words")
    if lesson.get("direction") not in ("up", "down"):
        reasons.append("direction is not up or down")
    index = {(item.candidate, item.measurement, item.condition): item for item in evidence}
    for candidate in sorted(set(cited)):
        item = index.get((candidate, lesson.get("measurement"), lesson.get("condition")))
        if item is None:
            reasons.append(f"{candidate} has no resolved {lesson.get('measurement')} in {lesson.get('condition')}")
        elif not _carries(item, str(lesson.get("edit_class"))):
            reasons.append(f"{candidate} does not carry edit class {lesson.get('edit_class')}")
        elif item.context.label != lesson.get("context"):
            reasons.append(f"{candidate} starts from '{item.context.label}', not '{lesson.get('context')}'")
        elif item.direction != lesson.get("direction"):
            reasons.append(f"{candidate} moved {lesson.get('measurement')} {item.direction}, not {lesson.get('direction')}")
    return reasons


def verified_lessons(proposed: Sequence[Mapping[str, Any]], evidence: Sequence[Evidence], **rules: Any,
                     ) -> tuple[list[Mapping[str, Any]], list[tuple[Mapping[str, Any], list[str]]]]:
    """The proposed lessons that reproduce, and the others with the reasons they were dropped."""
    kept, dropped = [], []
    for lesson in proposed:
        reasons = validate_lesson(lesson, evidence, **rules)
        (dropped.append((lesson, reasons)) if reasons else kept.append(lesson))
    return kept, dropped


def dead_ends(scored: Sequence[ScoredClaim], edits: Mapping[str, Sequence[EditClass]], *, min_refuted: int = 2,
              ) -> list[dict[str, Any]]:
    """Edit classes whose claims were refuted at least `min_refuted` times and never confirmed.

    A claim speaks for the classes its candidate carries on the claim's `lever_class`.
    """
    refuted: dict[str, set[str]] = defaultdict(set)
    confirmed: dict[str, set[str]] = defaultdict(set)
    for item in scored:
        if item.resolution.verdict == UNRESOLVABLE:
            continue
        target = refuted if item.resolution.verdict == MISS else confirmed
        for edit in edits.get(item.candidate, ()):
            if edit.lever == item.claim["lever_class"]:
                target[":".join(edit.coarse)].add(item.candidate)
    return [{"edit_class": label, "candidates": sorted(found)} for label, found in sorted(refuted.items())
            if len(found) >= min_refuted and label not in confirmed]


# ----- the text the agent reads (built here, never by the model) --------------------------------------------------

def render_verdict(item_claims: Sequence[Mapping[str, Any]], resolutions: Sequence[Resolution]) -> str:
    """The line beside a ledger row: each claim and what the row showed."""
    parts = []
    for claim, found in zip(item_claims, resolutions):
        said = (f"{claim['target_measurement']} {claim['condition']} {claim['direction']}"
                + (f" {claim['magnitude']}" if claim["magnitude"] != "none" else "")
                + f" vs {claim['versus']} at {claim['confidence']:g}")
        if found.verdict == UNRESOLVABLE:
            parts.append(f"{said}: unresolvable ({found.reason})")
        else:
            seen = found.direction + (f" {found.magnitude}" if found.magnitude != "none" else "")
            parts.append(f"{said}: {found.verdict} (row: {seen}"
                         + (f", {found.z:+.1f} SE" if found.z is not None and math.isfinite(found.z) else "") + ")")
    return " claims: " + " | ".join(parts) if parts else ""


def render_calibration(table: Mapping[str, Any]) -> str:
    if not table["n_resolved"]:
        return "Calibration: no claim has resolved yet."
    cells = "; ".join(f"{item['confidence']:g}: {item['hits']}/{item['n']}" for item in table["bins"] if item["n"])
    return (f"Calibration of your claims ({table['n_resolved']} resolved, hit = direction and magnitude both right): "
            f"hit rate {table['hit_rate']:.2f} at mean stated confidence {table['mean_confidence']:.2f}; "
            f"by stated confidence {cells}; Brier {table['brier']:.3f} (always 0.5: {table['brier_half']:.3f}).")


def render_lessons(lessons: Sequence[Lesson]) -> str:
    """The lessons table as text. Entries that cite the same candidates on the same measurement are one line: the
    classes co-occurred on those candidates, so the table states that it cannot tell them apart."""
    if not lessons:
        return "Lessons: none reproduced from the ledger yet."
    groups: dict[tuple, list[Lesson]] = {}
    for lesson in lessons:
        groups.setdefault((lesson.context, lesson.measurement, lesson.condition, lesson.direction, lesson.cited),
                          []).append(lesson)
    lines = ["Lessons (each computed from the ledger rows it cites; a candidate with several edits counts for each):"]
    for (context, name, condition, direction, cited), found in groups.items():
        together = len(found) > 1
        classes = " + ".join(lesson.edit_class for lesson in found)
        counts = ", ".join(f"{lesson.edit_class}: {lesson.n_against} of {lesson.n_carrying} the other way"
                           for lesson in found) if together else (
            f"{found[0].n_against} of {found[0].n_carrying} the other way")
        lines.append(f"- {classes}{' (same candidates, not separable)' if together else ''} from a row with "
                     f"{context}: {name} in {condition} {direction} in {len(cited)} ({', '.join(cited)}); {counts}")
    return "\n".join(lines)


def render_dead_ends(entries: Sequence[Mapping[str, Any]]) -> str:
    if not entries:
        return "Dead ends: none."
    return "Dead ends (claims about the class refuted at least twice, never confirmed): " + "; ".join(
        f"{entry['edit_class']} ({', '.join(entry['candidates'])})" for entry in entries)


# ----- the module: events, hooks and phases -----------------------------------------------------------------------

CLAIMS_INSTRUCTIONS = (
    "Claims. Each candidate you propose carries one or two claims about what its pool will measure, written before the "
    "result is known and scored by code from the ledger row. A claim names the lever it attributes the effect to (one "
    "the candidate edits), a shown measurement, a condition, a direction (up, down or flat), for a scalar a magnitude "
    "in standard errors of the pool median (small 2 to 4, medium 4 to 10, large above 10; a flat change is under 2), "
    "the baseline (the candidate's parent or the initial policy), your confidence that direction and magnitude are "
    "both right (0.5 to 0.95), and the failure mode you expect in that condition (unqualified, collapse when library "
    "health fails, exploit when a composition flag is raised, or none). `composition:<name>` claims whether that "
    "measurement's pool median lies above, below or within the initial policy's band; its magnitude is none. The ledger "
    "shows the verdict beside each row, a calibration of your confidences, the lessons the rows support (each one "
    "recomputed from the rows it cites) and the edit classes your claims have been refuted on at least twice.")
SETTINGS = {"min_claims": 1, "lessons_max_entries": 12, "min_support": MIN_SUPPORT, "dead_end_min_refuted": 2,
            "analyst": False}
ANALYST_SHOWN = 6   # kept Analyst lessons the ledger shows: the latest


@event
class InitialSet(ev._Event):
    """The initial policy's pool medians and their errors, the reference of every claim `versus: initial`."""

    type: Literal["claims.initial_set"] = "claims.initial_set"
    pool: dict[str, Any]
    pool_se: dict[str, Any]


@event
class Resolved(ev._Event):
    """One candidate's claims as written and what its frozen row showed (`Resolution` as plain data)."""

    type: Literal["claims.resolved"] = "claims.resolved"
    candidate: str
    claims: list[dict[str, Any]]
    resolutions: list[dict[str, Any]]


@event
class LessonsPublished(ev._Event):
    """The lessons table, dead ends and calibration the ledger shows from now on, after `frozen` rows."""

    type: Literal["claims.lessons_published"] = "claims.lessons_published"
    round: int
    frozen: int
    lessons: list[dict[str, Any]]
    dead_ends: list[dict[str, Any]]
    calibration: dict[str, Any]


@event
class AnalystNote(ev._Event, ev.LlmCall):
    """The Analyst's lessons of one round: those the rows reproduce, and those dropped with the reasons."""

    type: Literal["claims.analyst_note"] = "claims.analyst_note"
    round: int
    kept: list[dict[str, Any]]
    dropped: list[dict[str, Any]]


@event
class AnalystUnavailable(ev._Event):
    type: Literal["claims.analyst_unavailable"] = "claims.analyst_unavailable"
    round: int
    cause: str


@folds(InitialSet)
def _fold_initial(state, event):
    facts(state, "claims")["initial"] = {"pool": event.pool, "pool_se": event.pool_se}


@folds(Resolved)
def _fold_resolved(state, event):
    facts(state, "claims").setdefault("resolved", {})[event.candidate] = {
        "claims": event.claims, "resolutions": event.resolutions}


@folds(LessonsPublished)
def _fold_published(state, event):
    facts(state, "claims")["published"] = {
        "round": event.round, "frozen": event.frozen, "lessons": event.lessons, "dead_ends": event.dead_ends,
        "calibration": event.calibration}


@folds(AnalystNote)
def _fold_note(state, event):
    spend(state, event)
    facts(state, "claims").setdefault("analyst", {})[str(event.round)] = {
        "kept": event.kept, "dropped": len(event.dropped)}


@folds(AnalystUnavailable)
def _fold_unavailable(state, event):
    facts(state, "claims").setdefault("analyst", {})[str(event.round)] = {"kept": [], "dropped": 0, "unavailable": True}


def resolution_record(found: Resolution) -> dict[str, Any]:
    record = asdict(found)
    record["z"] = record["z"] if record["z"] is not None and math.isfinite(record["z"]) else None
    record["failure_modes"] = list(record["failure_modes"])
    return record


def resolution_from(record: Mapping[str, Any]) -> Resolution:
    return Resolution(**{**record, "failure_modes": tuple(record["failure_modes"])})


def lesson_from(record: Mapping[str, Any]) -> Lesson:
    return Lesson(**{**record, "cited": tuple(record["cited"])})


class Claims(Extension):
    """Module M1: typed claims in each candidate, resolved from the row, with calibration, lessons and dead ends.

    Every hook reads `state.ext` without creating anything in it, so a run that has not reached the module's first
    event has the state, and the state hash, it would have without the module.
    """

    name = "claims"

    def __init__(self, config: Mapping[str, Any]):
        unknown = sorted(set(config) - set(SETTINGS))
        if unknown:
            raise ControlError(f"control.modules.claims has the keys {unknown}; it reads {sorted(SETTINGS)}")
        self.settings = {**SETTINGS, **config}
        self._vocabulary: list[str] | None = None

    # ----- what the Strategist is told and asked ---------------------------------------------------------------

    def instructions(self) -> str:
        return CLAIMS_INSTRUCTIONS + (
            " The ledger also carries the Analyst's lessons, each kept only where the rows it cites reproduce it."
            if self.settings["analyst"] else "")

    def candidate_fields(self, engine) -> Mapping[str, Mapping[str, Any]]:
        if self._vocabulary is None:
            raw = engine._baseline_result().pool["raw"]
            conditions = list(engine.s.task.conditions)
            reference = engine._delivery_rule()[1].reference
            shown = {*shown_measurements(raw, engine.headline), "pairwise_ratio"}
            if any(item.unique_kmer_median for item in reference.values()):
                shown.add("unique_kmer_ratio")
            banded = {name for bands in composition_bands(raw, conditions).values() for name in bands}
            self._vocabulary = claim_measurements(sorted(shown), banded)
        return claims_fragment(conditions=list(engine.s.task.conditions), measurements=self._vocabulary,
                               lever_classes=LEVER_CLASSES)

    def validate_candidate(self, engine, state, rnd, index, proposal, resolved) -> Sequence[Problem]:
        if resolved is None:
            return ()
        return claim_problems(proposal.get("claims") or [], edit_classes(self._base(engine, state, resolved.parent),
                                                                         resolved.recipe),
                              min_claims=self.settings["min_claims"])

    @staticmethod
    def _base(engine, state, parent: str | None) -> Mapping[str, Any]:
        if parent in (None, "initial"):
            return engine.s.recipe_ctx.initial_recipe
        return engine.blobs.get_json(state.candidates[parent].recipe_sha)["recipe"]

    # ----- what a frozen row carries ------------------------------------------------------------------------------

    def _reference(self, engine) -> tuple[dict[str, float], dict[str, float | None]]:
        reference = engine._delivery_rule()[1].reference
        return ({c: item.mean_pairwise for c, item in reference.items()},
                {c: item.unique_kmer_median for c, item in reference.items()})

    def _errors(self, engine, directory, names: Sequence[str]) -> dict[str, dict[str, float | None]]:
        measurements = json.loads((directory / "public_measurements.json").read_text())["measurements"]
        pool = json.loads((directory / "pool.json").read_text())
        mean_pairwise, kmers = self._reference(engine)
        return pool_errors(measurements, pool, list(engine.s.task.conditions), names, mean_pairwise, kmers)

    def row_fields(self, engine, state, candidate, spec, result) -> Mapping[str, Any] | None:
        if result is None:
            return None
        names = shown_measurements(result["pool"]["raw"], engine.headline)
        return row_ext(self._errors(engine, engine.candidate_dir(candidate), names))["claims"]

    # ----- what the ledger shows -----------------------------------------------------------------------------------

    def round_blocks(self, state, store, arm, rnd) -> Sequence[Block]:
        resolved = (state.ext.get("claims") or {}).get("resolved") or {}
        lines = [f"[{name}]" + render_verdict(resolved[name]["claims"],
                                              [resolution_from(r) for r in resolved[name]["resolutions"]])
                 for name in rnd.candidates if name in resolved]
        text = "\n".join(lines)
        return [Block(text, text)] if lines else []

    def ledger_blocks(self, state, store, arm) -> Sequence[Block]:
        mine = state.ext.get("claims") or {}
        published = mine.get("published")
        if published is None:
            return []
        parts = [render_calibration(published["calibration"]),
                 render_lessons([lesson_from(item) for item in published["lessons"]]),
                 render_dead_ends(published["dead_ends"])]
        kept = [lesson for _, entry in sorted((mine.get("analyst") or {}).items(), key=lambda item: int(item[0]))
                for lesson in entry["kept"]][-ANALYST_SHOWN:]
        system = list(parts)
        if kept:
            scrub = getattr(arm, "scrub", lambda text: text)
            lines = [f"- {lesson['edit_class']} from a row with {lesson['context']}: {lesson['measurement']} in "
                     f"{lesson['condition']} {lesson['direction']} ({', '.join(lesson['cited'])})" for lesson in kept]
            head = "Analyst lessons (each reproduced from the rows it cites):"
            parts.append("\n".join([head, *(f"{line}: {scrub(lesson['note'])}" for line, lesson in zip(lines, kept))]))
            system.append("\n".join([head, *lines]))
        return [Block("\n".join(parts), "\n".join(system))]

    # ----- the phases ----------------------------------------------------------------------------------------------

    def phases(self) -> Sequence[Phase]:
        phases = [Phase("claims_initial", "baseline", self._set_initial),
                  Phase("claims_resolve", "rows", self._resolve),
                  Phase("claims_publish", "rows", self._publish)]
        return phases + ([Phase("claims_analyst", "critic", self._analyst)] if self.settings["analyst"] else [])

    def _set_initial(self, engine, state) -> bool:
        if "initial" in (state.ext.get("claims") or {}) or state.baseline.status != "ingested":
            return False
        raw = engine._baseline_result().pool["raw"]
        conditions = list(engine.s.task.conditions)
        names = sorted(shown_measurements(raw, engine.headline))
        mean_pairwise, kmers = self._reference(engine)
        pool = pool_medians(raw, conditions, names)
        health = ["pairwise_ratio", *(["unique_kmer_ratio"] if all(kmers.values()) else [])]
        for values in pool.values():
            values.update(dict.fromkeys(health, 1.0))
        engine.emit(InitialSet(pool=pool, pool_se=self._errors(engine, engine.s.run_dir / "baseline", [*names, *health])))
        return True

    def _rows(self, engine, state) -> dict[str, dict[str, Any]]:
        return {c.id: engine.blobs.get_json(c.ledger_row_sha) for c in state.ordered_candidates() if c.ledger_row_sha}

    def _claims_of(self, engine, state, candidate: str) -> list[dict[str, Any]]:
        spec = engine.blobs.get_json(state.candidates[candidate].recipe_sha)
        return list((spec.get("proposal") or {}).get("claims") or [])

    def _resolve(self, engine, state) -> bool:
        mine = state.ext.get("claims") or {}
        if "initial" not in mine:
            return False
        done = mine.get("resolved") or {}
        pending = [c.id for c in state.ordered_candidates()
                   if c.ledger_row_sha and c.id not in done and self._claims_of(engine, state, c.id)]
        if not pending:
            return False
        rows = self._rows(engine, state)
        initial = initial_reference(mine["initial"]["pool"], mine["initial"]["pool_se"])
        for name in pending:
            claims = self._claims_of(engine, state, name)
            found = [resolve(claim, rows[name], reference_for(rows[name], claim["versus"], rows, initial))
                     for claim in claims]
            engine.emit(Resolved(candidate=name, claims=claims, resolutions=[resolution_record(r) for r in found]))
        return True

    def _evidence(self, engine, state, rows) -> tuple[list[Evidence], dict[str, tuple[EditClass, ...]]]:
        """Every candidate's effects against its base, and the edit classes they carry."""
        mine = state.ext["claims"]
        initial = initial_reference(mine["initial"]["pool"], mine["initial"]["pool_se"])
        edits, anchor = {}, None
        for c in state.ordered_candidates():
            if c.ledger_row_sha is None:
                continue
            anchor = c.id if c.anchor else anchor
            if not c.anchor:
                recipe = engine.blobs.get_json(c.recipe_sha)["recipe"]
                edits[c.id] = tuple(edit_classes(self._base(engine, state, c.parent), recipe))
        names = sorted(initial["pool"][next(iter(initial["pool"]))])
        return evidence_from_rows(rows, initial, edits, names, list(engine.s.task.conditions), anchor=anchor), edits

    def _tables(self, engine, state, rows) -> tuple[list[Lesson], list[dict[str, Any]], dict[str, Any]]:
        evidence, edits = self._evidence(engine, state, rows)
        lessons = build_lessons(evidence, min_support=self.settings["min_support"],
                                max_entries=self.settings["lessons_max_entries"],
                                priority=[*engine.headline, "pairwise_ratio"])
        scored = [ScoredClaim(name, index, claim, resolution_from(record))
                  for name, entry in state.ext["claims"].get("resolved", {}).items()
                  for index, (claim, record) in enumerate(zip(entry["claims"], entry["resolutions"]))]
        return (lessons, dead_ends(scored, edits, min_refuted=self.settings["dead_end_min_refuted"]),
                calibration_table(scored))

    def _publish(self, engine, state) -> bool:
        mine = state.ext.get("claims") or {}
        candidates = [c for c in state.ordered_candidates() if c.status != "forfeited"]
        if "initial" not in mine or not mine.get("resolved") or any(c.ledger_row_sha is None for c in candidates):
            return False
        if (mine.get("published") or {}).get("frozen") == len(candidates):
            return False
        lessons, dead, calibration = self._tables(engine, state, self._rows(engine, state))
        engine.emit(LessonsPublished(round=state.rounds_done, frozen=len(candidates),
                                     lessons=[lesson.as_dict() for lesson in lessons], dead_ends=dead,
                                     calibration=calibration))
        return True

    def _analyst(self, engine, state) -> bool:
        """One Analyst call per round, after the critic and before the next decision."""
        from ..engine import llm_fields
        from ..projection import render_with_tail

        mine = state.ext.get("claims") or {}
        if state.stop.accepted is not None or state.uningested or state.unscored or state.unfrozen:
            return False
        allocated = [r for r in state.rounds if r.status == "allocated"]
        if not allocated or str(allocated[-1].index) in (mine.get("analyst") or {}):
            return False
        rnd = allocated[-1]
        members = [c for c in rnd.candidates if state.candidates[c].status != "forfeited"]
        published = mine.get("published")
        if (rnd.critic_row_sha is None and not rnd.critic_unavailable) or engine._b_effective(state) == 0 \
                or published is None or published["frozen"] != len([c for c in state.candidates.values()
                                                                      if c.status != "forfeited"]):
            return False
        rows = self._rows(engine, state)
        evidence, _ = self._evidence(engine, state, rows)
        if not evidence:
            return False
        schema = analyst_schema(
            edit_classes=sorted({class_label(edit, fine=False) for item in evidence for edit in item.edits}),
            contexts=sorted({item.context.label for item in evidence}),
            measurements=sorted({item.measurement for item in evidence}),
            conditions=list(engine.s.task.conditions), candidates=sorted({item.candidate for item in evidence}),
            note_words=NOTE_WORDS)
        tail = (f"## Analysis\nWrite lessons from the ledger through round {rnd.index} (the candidates of that round: "
                f"{', '.join(members)}).")
        failure: Exception | None = None
        for _ in range(2):
            try:
                rendered = render_with_tail(state, engine.blobs, arm=engine.arm, header=engine.s.header_text,
                                            instructions=ANALYST_INSTRUCTIONS, tail=tail,
                                            limit=engine.s.control.llm.max_prompt_tokens, schema=schema, tools=(),
                                            extensions=engine.extensions)
                turn = run_analyst(engine.transport, instructions=rendered.instructions,
                                   input_items=rendered.input_items, schema=schema)
            except (BlindingViolation, PromptBudgetExceeded, RunTokenBudgetExceeded, LedgerCorruption):
                raise
            except Exception as exc:  # an advisory call must not stop the search
                failure = exc
                continue
            kept, dropped = verified_lessons(turn.parsed["lessons"], evidence, min_support=self.settings["min_support"])
            engine.emit(AnalystNote(
                round=rnd.index, kept=[dict(item) for item in kept],
                dropped=[{"lesson": dict(item), "reasons": reasons} for item, reasons in dropped],
                **llm_fields(turn, rendered, engine.blobs.put_json(schema), engine.blobs.put_json(turn.parsed), None)))
            engine.count(turn)
            return True
        engine.emit(AnalystUnavailable(round=rnd.index, cause=f"{type(failure).__name__}: {failure}"[:200]))
        return True


def build(config: Mapping[str, Any]) -> Claims:
    return Claims(config)
