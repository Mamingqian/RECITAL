"""Shared evidence qualification, independent of task scores and blind evaluations.

A measured negative check is different from an absent alarm. Self-validating
selection requires every declared group and required sentinel to be measured and
clean. Empty eligibility is an ordinary result; no weaker partition is selected.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


ADVISORY_RULE = "complete_evidence_advisory_flags_v2"

CLEAN, UNKNOWN, FLAGGED = "clean", "unavailable", "flagged"


@dataclass(frozen=True)
class EvidenceEligibility:
    """Task-declared checks in addition to the protocol's required sentinels."""

    required_monitors: tuple[str, ...] = ()
    required_strata: tuple[str, ...] = ()
    rule_id: str = "complete_clean_evidence_v1"

    def __post_init__(self) -> None:
        if self.rule_id not in {"complete_clean_evidence_v1", ADVISORY_RULE}:
            raise ValueError("unknown evidence eligibility rule")
        for name in ("required_monitors", "required_strata"):
            values = getattr(self, name)
            if isinstance(values, str) or any(not isinstance(v, str) or not v.strip() for v in values):
                raise ValueError(f"{name} must contain nonempty names")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} contains duplicate names")
            object.__setattr__(self, name, tuple(values))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def monitor_state(record: Any, monitor: str) -> str:
    """Use explicit monitor outcomes; never infer clean from absent flags."""
    statistics = getattr(record, "statistics", None) or {}
    payload = statistics.get(monitor) or {}
    if monitor in (getattr(record, "flags", None) or ()):
        return FLAGGED
    if isinstance(payload, Mapping) and (payload.get("fired") is True or payload.get("fired_strata")):
        return FLAGGED
    state = ((statistics.get("monitor_coverage") or {}).get("states") or {}).get(monitor)
    if state == "fired":
        return FLAGGED
    if state == "clean" and isinstance(statistics.get(monitor), Mapping) and not payload.get("unavailable_strata"):
        return CLEAN
    return UNKNOWN


def _measured(entry: Any) -> bool:
    if not isinstance(entry, Mapping) or entry.get("status") in {"unavailable", "error", "missing"}:
        return False
    try:
        return math.isfinite(float(entry["value"]))
    except (KeyError, TypeError, ValueError):
        return False


def assess_evidence(
    record: Any,
    policy: EvidenceEligibility | None = None,
    *,
    require_heldout: bool = False,
    required_held_out_oracles: Sequence[str] = (),
    required_strata: Sequence[str] = (),
) -> dict[str, Any]:
    """Assess exactly the declared evidence, with no task-name branches.

    The required panel comes from the frozen source-role policy, not the
    candidate's offered subset. Per-group monitor outcomes and finite source
    measurements must both exist; a global clean label cannot hide a missing
    group or oracle. Task adapters own the statistical test and its calibration.
    """
    policy = policy or EvidenceEligibility()
    strata = tuple(policy.required_strata or required_strata)
    sources = tuple(sorted(set(required_held_out_oracles)))
    monitors = tuple(dict.fromkeys((*policy.required_monitors,
                                  *(("heldout_over_opt",) if require_heldout else ()))))
    states = {name: monitor_state(record, name) for name in monitors}
    accepted_states = {CLEAN, FLAGGED} if policy.rule_id == ADVISORY_RULE else {CLEAN}
    reasons = [f"{name}:{state}" for name, state in states.items() if state not in accepted_states]
    cells: dict[str, dict[str, str]] = {}
    statistics = getattr(record, "statistics", None) or {}
    if require_heldout:
        summary = statistics.get("self_validation") or {}
        observed = summary.get("strata") or {}
        monitored = (statistics.get("heldout_over_opt") or {}).get("strata") or {}
        if not sources:
            reasons.append("required_held_out_oracles:undeclared")
        if not strata:
            reasons.append("required_strata:undeclared")
        if summary.get("error") or "sentinel_unavailable" in (getattr(record, "flags", None) or ()):
            reasons.append("required_sentinel:unavailable")
        for group in strata:
            cell = observed.get(group) or {}
            declared = set(cell.get("required_held_out_oracles") or summary.get("required_held_out_oracles") or ())
            verdict = monitored.get(group) or {}
            if verdict.get("fired") or verdict.get("evidence_state") == "fired":
                state = FLAGGED
            elif verdict.get("evidence_state") == "clean":
                state = CLEAN
            else:
                state = UNKNOWN
            cells[group] = {name: state if name in declared and _measured((cell.get("held_out") or {}).get(name))
                            else UNKNOWN for name in sources}
            for name, offered in cells[group].items():
                if offered not in accepted_states:
                    reasons.append(f"{group}/{name}:{offered}")
    eligible = not reasons
    state = (FLAGGED if FLAGGED in states.values() or any(FLAGGED in c.values() for c in cells.values())
             else UNKNOWN if reasons else CLEAN if require_heldout else "unassessed")
    return {"eligible": eligible, "state": state, "required_monitors": list(monitors),
            "required_strata": list(strata), "required_held_out_oracles": list(sources),
            "monitors": states, "cells": cells, "reasons": reasons}
