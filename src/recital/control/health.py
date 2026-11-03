"""Library health: how spread out a delivered library is, measured against the initial policy's own pool.

Every quantity is relative to the baseline pool, so no alphabet, sequence length or task enters. Distance is
whole-sequence Levenshtein over the longer length (`recital.sequence_distance`), which compares sequences of
unequal length; a threshold is a fraction of what the initial policy's pool measures, not a number chosen for one
alphabet. Nothing here reads an oracle score.

A `HealthRule` is the contract's two fractions applied to one run's baseline reference. The selection plan's
minimum distance and the `library_health` requirement are both derived from it, so a run has one definition.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from recital.sequence_distance import pairwise_normalized

REQUIREMENT_ID = "library_health"
UNIQUE_KMER_MEASUREMENT = "unique_kmer_count"  # the task measurement behind `unique_kmer_ratio`
MIN_BASELINE_ROWS = 30
MAX_ROWS = 512


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ConditionReference(_Model):
    """What the initial policy's pool measures for one condition, over at most `MAX_ROWS` legal rows."""

    n: int
    nn_median: float
    mean_pairwise: float
    unique_kmer_median: float | None


class HealthRule(_Model):
    min_pairwise_ratio: float
    reference: dict[str, ConditionReference]


def _subsample(count: int) -> np.ndarray:
    """Row positions to use: all of them up to `MAX_ROWS`, else evenly spaced ones, so the choice is deterministic."""
    if count <= MAX_ROWS:
        return np.arange(count)
    return np.unique(np.linspace(0, count - 1, MAX_ROWS).round().astype(int))


def _spread(sequences: Sequence[str]) -> tuple[float | None, float | None]:
    """(median nearest-neighbour distance, mean pairwise distance); None below two sequences."""
    n = len(sequences)
    if n < 2:
        return None, None
    distances = pairwise_normalized(list(sequences))
    mean_pairwise = float(distances.sum() / (n * (n - 1)))
    np.fill_diagonal(distances, np.inf)
    return float(np.median(distances.min(axis=1))), mean_pairwise


def _median(values: Sequence[float] | None) -> float | None:
    return None if values is None or len(values) == 0 else float(np.median(values))


def baseline_reference(
    sequences_by_condition: Mapping[str, Sequence[str]],
    unique_kmers_by_condition: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, ConditionReference]:
    """The reference each condition's libraries are judged against, from the baseline pool's legal rows.

    `unique_kmers_by_condition` holds the task's per-row `unique_kmer_count`, aligned with the sequences; it is None
    for a task that does not measure one. A baseline with fewer than `MIN_BASELINE_ROWS` rows in a condition, or
    with no spread, raises: a reference with no spread would turn both thresholds off.
    """
    result = {}
    for condition, sequences in sequences_by_condition.items():
        if len(sequences) < MIN_BASELINE_ROWS:
            raise ValueError(f"the baseline pool has {len(sequences)} usable rows for condition {condition!r}; "
                             f"a library-health reference needs at least {MIN_BASELINE_ROWS}")
        rows = _subsample(len(sequences))
        chosen = [sequences[i] for i in rows]
        nn_median, mean_pairwise = _spread(chosen)
        if not nn_median or not mean_pairwise:
            raise ValueError(f"the baseline pool for condition {condition!r} has no spread (median nearest-neighbour "
                             f"distance {nn_median}, mean pairwise distance {mean_pairwise}); a library cannot be "
                             "judged against it")
        kmers = None if unique_kmers_by_condition is None else unique_kmers_by_condition[condition]
        if kmers is not None and len(kmers) != len(sequences):
            raise ValueError(f"condition {condition!r} has {len(kmers)} unique k-mer counts for {len(sequences)} rows")
        result[condition] = ConditionReference(
            n=len(chosen), nn_median=nn_median, mean_pairwise=mean_pairwise,
            unique_kmer_median=_median(None if kmers is None else [kmers[i] for i in rows]))
    return result


def plan_with_floor(plan: Mapping[str, Any], floor: float) -> dict[str, Any]:
    """The declared selection plan with the derived diversity rule: greedy in rank order, within each condition."""
    return {**plan, "minimum_distance": floor, "distance_metric": "normalized_edit",
            "diversity_scope": "within_stratum"}


def diversity_floor(reference: Mapping[str, ConditionReference], fraction: float) -> float:
    """The minimum normalized edit distance between two rows chosen for one condition.

    `fraction` of the smallest condition's median nearest-neighbour distance: a selection plan carries one number,
    and the smallest keeps every condition's floor at or below its own.
    """
    return fraction * min(item.nn_median for item in reference.values())


def library_health(
    sequences: Sequence[str],
    reference: ConditionReference,
    unique_kmers: Sequence[float] | None = None,
) -> dict[str, Any]:
    """One condition's sequences against its baseline reference.

    `pairwise_ratio` is the mean pairwise distance over the reference's; `unique_kmer_ratio` is the median
    per-sequence unique k-mer count over the reference's, and is None, with `unique_kmer_status` unavailable, when
    the task emitted no count.
    """
    rows = _subsample(len(sequences))
    chosen = [sequences[i] for i in rows]
    nn_median, mean_pairwise = _spread(chosen)
    kmer_median = _median(unique_kmers)
    measured = kmer_median is not None and reference.unique_kmer_median not in (None, 0.0)
    return {
        "n": len(sequences), "nn_median": nn_median, "mean_pairwise": mean_pairwise,
        "pairwise_ratio": None if mean_pairwise is None else mean_pairwise / reference.mean_pairwise,
        "unique_kmer_ratio": kmer_median / reference.unique_kmer_median if measured else None,
        "unique_kmer_status": "measured" if measured else "unavailable"}


def _requirement(condition: str, health: Mapping[str, Any], *, minimum: float, required: int, skipped: int,
                 floor: float) -> dict[str, Any]:
    """The `library_health` row for one condition, in the shape the success spec's requirement rows have.

    It fails when the library is less spread out than `minimum` allows, or when selection skipped rows for being
    closer than `floor` and so delivered fewer than `required`. It is unknown when the library has no pair to measure.
    """
    ratio, selected = health["pairwise_ratio"], health["n"]
    reasons, violation = [], 0.0
    if ratio is not None and ratio < minimum:
        violation = (minimum - ratio) / minimum
        reasons.append(f"mean pairwise distance is {ratio:.3f} of the initial pool's, below {minimum:.3f}")
    if selected < required and skipped > 0:
        violation = max(violation, (required - selected) / required)
        reasons.append(f"only {selected} of {required} rows could be chosen at normalized edit distance "
                       f">= {floor:.4f} from each other; the floor is not lowered to fill the count")
    if reasons:
        status = "fail"
    elif ratio is None:
        status, reasons = "unknown", ["fewer than two sequences were selected"]
    else:
        status = "pass"
    return {"requirement_id": REQUIREMENT_ID, "metric": "pairwise_ratio", "stratum": condition, "status": status,
            "value": ratio, "count": selected, "minimum": minimum, "maximum": None,
            "reason": "; ".join(reasons), "normalized_violation": violation}


def assess(
    rule: HealthRule,
    *,
    pool: Mapping[str, tuple[Sequence[str], Sequence[float] | None]],
    library: Mapping[str, tuple[Sequence[str], Sequence[float] | None]],
    required: Mapping[str, int],
    skipped: Counter,
    minimum_distance: float,
) -> dict[str, Any]:
    """Health of every condition's pool and selected library, and the requirement each library has to meet.

    `pool` and `library` map a condition to its legal sequences and their per-row unique k-mer counts (None when the
    task has none); `skipped` counts the rows selection passed over for being closer than `minimum_distance` to a
    row already chosen.
    """
    by_condition = {}
    for condition, reference in rule.reference.items():
        in_pool = library_health(pool[condition][0], reference, pool[condition][1])
        chosen = library_health(library[condition][0], reference, library[condition][1])
        by_condition[condition] = {
            "pool": in_pool, "library": chosen,
            "requirement": _requirement(condition, chosen, minimum=rule.min_pairwise_ratio,
                                        required=required[condition], skipped=skipped[condition],
                                        floor=minimum_distance)}
    return {"min_pairwise_ratio": rule.min_pairwise_ratio, "minimum_distance": minimum_distance,
            "reference": {condition: item.model_dump() for condition, item in rule.reference.items()},
            "by_condition": by_condition}


def with_requirements(qualification: Mapping[str, Any], assessment: Mapping[str, Any]) -> dict[str, Any]:
    """`qualification` plus one `library_health` row per condition; a row that is not a pass leaves it unqualified."""
    if "requirements" not in qualification:
        raise ValueError("the task's qualification carries no requirement list to add library_health to")
    rows = [item["requirement"] for item in assessment["by_condition"].values()]
    statuses = {qualification.get("status"), *(row["status"] for row in rows)}
    return {**qualification, "requirements": [*qualification["requirements"], *rows],
            "qualified": bool(qualification["qualified"]) and all(row["status"] == "pass" for row in rows),
            "status": "fail" if "fail" in statuses else "unknown" if "unknown" in statuses else "pass"}


def summary_entries(assessment: Mapping[str, Any], scope: str) -> dict[str, dict[str, Any]]:
    """`pairwise_ratio` and `unique_kmer_ratio` of `scope` ("pool" or "library") as measurement summaries.

    They have the shape `summarize_measurements` gives a per-row measurement, with the library-level value as the
    median, so the ledger and the export carry them with the other source-free columns.
    """
    entries = {}
    for name in ("pairwise_ratio", "unique_kmer_ratio"):
        cells = {condition: item[scope] for condition, item in assessment["by_condition"].items()}
        measured = all(cell[name] is not None for cell in cells.values())
        entries[name] = {
            "sources": [], "status": "measured" if measured else "unavailable",
            "metadata": {"scope": scope, "relative_to": "baseline pool",
                         "definition": ("mean pairwise normalized edit distance over the baseline pool's"
                                        if name == "pairwise_ratio" else
                                        "median unique k-mer count over the baseline pool's")},
            "by_condition": {condition: {
                "count": cell["n"], "mean": cell[name], "median": cell[name], "standard_error": None,
                "measured_count": 0 if cell[name] is None else cell["n"]} for condition, cell in cells.items()}}
    return entries
