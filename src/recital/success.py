"""Task-independent eligibility and selection from declared measurements.

Adapters measure candidates. This module never evaluates biological sequences,
infers requirements from flags, or consults a blind-evaluation predictor. Missing
evidence is unknown, and a search parent is distinct from a qualified deliverable.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Annotated, Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class Requirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    strata: tuple[str, ...] = Field(min_length=1)
    minimum: float | None = None
    maximum: float | None = None
    min_count: int = Field(default=1, ge=1)
    # Used only to compare unmet requirements while choosing a search parent.
    # It does not change eligibility or allow one requirement to offset another.
    scale: float = Field(default=1.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_bounds(self) -> "Requirement":
        if self.minimum is None and self.maximum is None:
            raise ValueError("a requirement needs a minimum or maximum")
        for bound in (self.minimum, self.maximum):
            if bound is not None and not math.isfinite(bound):
                raise ValueError("requirement bounds must be finite")
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError("requirement minimum exceeds maximum")
        if any(not stratum for stratum in self.strata):
            raise ValueError("requirement strata must be non-empty")
        if len(set(self.strata)) != len(self.strata):
            raise ValueError("requirement strata must be unique")
        return self


class RankingCriterion(BaseModel):
    """One ranking step. Values within `tolerance` of the best are tied and fall to the next step.

    A number is an absolute band in the metric's units. "auto" is the standard error of the difference between a
    candidate's value and the best value, from the `standard_error` each measurement carries, so two candidates are
    tied exactly when their measured gap is inside the sampling noise of the two values.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str = Field(min_length=1)
    strata: tuple[str, ...] = Field(min_length=1)
    direction: Literal["maximize", "minimize"] = "maximize"
    reduction: Literal["min", "max", "mean"] = "min"
    tolerance: Annotated[float, Field(ge=0, allow_inf_nan=False)] | Literal["auto"] = 0.0

    @model_validator(mode="after")
    def validate_strata(self) -> "RankingCriterion":
        if any(not value for value in self.strata) or len(set(self.strata)) != len(self.strata):
            raise ValueError("ranking strata must be non-empty and unique")
        return self


class SuccessSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requirements: tuple[Requirement, ...] = Field(min_length=1)
    ranking: tuple[RankingCriterion, ...] = ()
    stop_when_qualified: bool = True
    # Explicit provenance for derived metrics, including joint/multi-oracle ones.
    # A sequence-only metric has an empty source tuple.
    metric_sources: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_requirements(self) -> "SuccessSpec":
        ids = [requirement.id for requirement in self.requirements]
        if len(set(ids)) != len(ids):
            raise ValueError("success requirement ids must be unique")
        return self


    def validate_information_access(
        self, *, known_sources: set[str], audit_sources: set[str], private_sources: set[str]
    ) -> None:
        metrics = {item.metric for item in (*self.requirements, *self.ranking)}
        missing = metrics - set(self.metric_sources)
        if missing:
            raise ValueError(f"success metrics need explicit source provenance: {sorted(missing)}")
        sources = {source for metric in metrics for source in self.metric_sources[metric]}
        if sources - known_sources:
            raise ValueError(f"unknown sources in success metrics: {sorted(sources - known_sources)}")
        forbidden = sources & (audit_sources | private_sources)
        if forbidden:
            raise ValueError(
                "online qualification and selection cannot consume audit/private sources: "
                f"{sorted(forbidden)}; use a separately declared terminal evaluation"
            )


class Measurement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float | None = None
    # The sampling error of `value`, which a ranking with tolerance "auto" reads.
    standard_error: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    count: int = Field(default=0, ge=0, strict=True)
    status: Literal["measured", "unavailable", "error"] = "measured"
    reason: str = ""


@dataclass(frozen=True)
class RequirementEvidence:
    requirement_id: str
    metric: str
    stratum: str
    status: Literal["pass", "fail", "unknown"]
    value: float | None
    count: int
    minimum: float | None
    maximum: float | None
    reason: str = ""
    normalized_violation: float = 0.0


@dataclass(frozen=True)
class Qualification:
    status: Literal["pass", "fail", "unknown"]
    evidence: tuple[RequirementEvidence, ...]

    @property
    def qualified(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SelectionDecision:
    selected_iteration: int | None
    search_parent_iteration: int | None
    status: Literal["qualified", "no_qualified_candidate"]
    assessments: dict[int, Qualification] = field(default_factory=dict)
    rule: str = "contract_eligibility_then_declared_ranking_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_iter": self.selected_iteration,
            "search_parent_iter": self.search_parent_iteration,
            "status": self.status,
            "rule": self.rule,
            "assessments": {str(key): value.to_dict() for key, value in self.assessments.items()},
        }


def _measurement(
    measurements: Mapping[str, Any], metric: str, stratum: str
) -> Measurement:
    group = measurements.get(metric)
    if not isinstance(group, Mapping) or stratum not in group:
        return Measurement(status="unavailable", reason="required measurement is missing")
    raw = group[stratum]
    try:
        value = raw if isinstance(raw, Measurement) else Measurement.model_validate(raw)
    except ValidationError as exc:
        return Measurement(status="error", reason=f"invalid measurement: {exc}")
    if value.value is not None and not math.isfinite(value.value):
        return Measurement(status="error", count=value.count, reason="nonfinite measurement")
    return value


def assess_candidate(
    spec: SuccessSpec, measurements: Mapping[str, Any]
) -> Qualification:
    evidence: list[RequirementEvidence] = []
    for requirement in spec.requirements:
        for stratum in requirement.strata:
            value = _measurement(measurements, requirement.metric, stratum)
            status: Literal["pass", "fail", "unknown"] = "unknown"
            reason = value.reason
            violation = 0.0
            if value.status != "measured" or value.value is None:
                reason = reason or "required measurement is unavailable"
            elif value.count < requirement.min_count:
                reason = f"requires {requirement.min_count} observations; received {value.count}"
            else:
                lower = max(0.0, requirement.minimum - value.value) if requirement.minimum is not None else 0.0
                upper = max(0.0, value.value - requirement.maximum) if requirement.maximum is not None else 0.0
                violation = (lower + upper) / requirement.scale
                status = "fail" if violation > 0 else "pass"
                reason = "outside required bounds" if status == "fail" else ""
            evidence.append(RequirementEvidence(
                requirement_id=requirement.id,
                metric=requirement.metric,
                stratum=stratum,
                status=status,
                value=value.value,
                count=value.count,
                minimum=requirement.minimum,
                maximum=requirement.maximum,
                reason=reason,
                normalized_violation=violation,
            ))
    statuses = {item.status for item in evidence}
    overall = "fail" if "fail" in statuses else "unknown" if "unknown" in statuses else "pass"
    return Qualification(status=overall, evidence=tuple(evidence))


def ranking_value(criterion: RankingCriterion, measurements: Mapping[str, Any]) -> float | None:
    observed = [_measurement(measurements, criterion.metric, group) for group in criterion.strata]
    if any(item.status != "measured" or item.value is None or item.count < 1 for item in observed):
        return None
    values = [item.value for item in observed]
    if criterion.reduction == "mean":
        return sum(values) / len(values)
    return min(values) if criterion.reduction == "min" else max(values)


def ranking_error(criterion: RankingCriterion, measurements: Mapping[str, Any]) -> float | None:
    """The standard error of `ranking_value`, or None when any of its strata carries none.

    A min or max reduction takes the error of the stratum that attains the extreme; a mean combines the strata's
    errors as independent.
    """
    observed = [_measurement(measurements, criterion.metric, group) for group in criterion.strata]
    errors = [item.standard_error for item in observed]
    if any(error is None for error in errors):
        return None
    if criterion.reduction == "mean":
        return math.sqrt(sum(error * error for error in errors)) / len(errors)
    values = [item.value for item in observed]
    return errors[values.index(min(values) if criterion.reduction == "min" else max(values))]


def tied_with_best(criterion: RankingCriterion, value: float, error: float | None,
                   best: float, best_error: float | None) -> bool:
    """Whether `value` is inside the criterion's tolerance of `best`; `best_error` is the error of the best value."""
    if criterion.tolerance != "auto":
        return abs(value - best) <= criterion.tolerance
    if error is None or best_error is None:
        raise ValueError(f"ranking tolerance 'auto' needs a standard_error on every {criterion.metric} measurement "
                         f"over {list(criterion.strata)}")
    return abs(value - best) <= math.hypot(error, best_error)


def best_value_error(best: float, scored: Sequence[tuple[float, float | None]]) -> float | None:
    """The error to compare against: the largest among candidates sitting exactly on the best value.

    Taking the largest, rather than whichever candidate is met first, keeps the tied set independent of order.
    """
    errors = [error for value, error in scored if value == best]
    return None if any(error is None for error in errors) else max(errors)


def ranking_available(spec: SuccessSpec, measurements: Mapping[str, Any]) -> bool:
    return all(ranking_value(criterion, measurements) is not None for criterion in spec.ranking)


def select_candidate(records: Sequence[Any], spec: SuccessSpec) -> SelectionDecision:
    """Select only qualified candidates with every declared ranking measurement.

    Ranking tolerances are applied against the best value at each stage, making
    selection independent of record traversal order. The final tie is the earlier
    iteration. No flags or undeclared biological metric affect eligibility.
    """
    from recital.artifacts import NONTRAINING_OUTCOME_STATUSES

    trained = [record for record in records if record.candidate_status not in NONTRAINING_OUTCOME_STATUSES]
    assessments = {
        record.iter: assess_candidate(spec, record.statistics.get("candidate_measurements") or {})
        for record in trained
    }
    pool = [record for record in trained if assessments[record.iter].qualified
            and ranking_available(spec, record.statistics.get("candidate_measurements") or {})]
    if not pool:
        return SelectionDecision(None, None, "no_qualified_candidate", assessments)

    for criterion in spec.ranking:
        scored = [(record, ranking_value(criterion, record.statistics.get("candidate_measurements") or {}),
                   ranking_error(criterion, record.statistics.get("candidate_measurements") or {}))
                  for record in pool]
        measured = [(record, value, error) for record, value, error in scored if value is not None]
        if not measured:
            return SelectionDecision(None, None, "no_qualified_candidate", assessments)
        # A declared ranking criterion is part of the frozen selection
        # contract. Candidates without its complete strata are unknown and
        # cannot silently win on the remaining criteria.
        best = (max if criterion.direction == "maximize" else min)(value for _, value, _ in measured)
        best_error = best_value_error(best, [(value, error) for _, value, error in measured])
        pool = [record for record, value, error in measured
                if tied_with_best(criterion, value, error, best, best_error)]
    selected = min(pool, key=lambda record: record.iter)
    return SelectionDecision(selected.iter, selected.iter, "qualified", assessments)
