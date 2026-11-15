#!/usr/bin/env python3
"""Decision-trace audit: what the agent does in its decisions, and what the delivered libraries show beyond the score.

    decision_trace_audit.py --selftest
    decision_trace_audit.py --gate [--scratch DIR]
    decision_trace_audit.py --run --declared-tag TAG [--scratch DIR] [--out DIR]

Offline, CPU only, no model call, no blind call. It reads the event logs, the frozen ledger rows and the terminal audits of the cells
listed in `experiments/recital_decision_trace_audit_20261111/declared.json`, extracts the metrics defined in
`docs/reports/decision_trace_audit_declaration_2026-11-11.md` by field and string rules only, and writes per-cell tables, per-arm
summaries with cell-clustered intervals, and a results file. The unit is the cell: arm figures are means over cells with a percentile
bootstrap over cells, never over decisions. Nothing here is a test or a pass rule.

`--gate` recomputes, for every listed cell, the final value of the best-qualified-library curve, the committed updates, the candidate
count and the token total, and compares them with `analysis.module_endpoints` (the committed endpoint code); it prints counts and the
largest difference, no metric of this audit. `--run` refuses unless the tag exists, is more than 60 s old and holds the declaration, the
declared file, this script and its tests byte for byte. Run with PYTHONPATH including src, scripts and figures of the tree that holds
`analysis.module_endpoints` and `recital.control.ext.edit_class`.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
DECLARED = REPO / "experiments/recital_decision_trace_audit_20261111/declared.json"
DECLARATION = REPO / "docs/reports/decision_trace_audit_declaration_2026-11-11.md"
TESTS = REPO / "tests/test_analysis/test_decision_trace_audit.py"
SCRATCH = Path("runs")

REWARD_LEVERS = ("oracle_set", "weights", "aggregator", "utility", "penalty", "gate")
TRAINING_LEVERS = ("lr", "kl", "group_size", "steps")
LEVERS = REWARD_LEVERS + TRAINING_LEVERS + ("other",)
OPPOSITE = {"up": "down", "down": "up", "added": "removed", "removed": "added"}
RESPONSES = ("no_edit", "repeat", "reverse", "other_lever", "same_lever_other_detail")
DIVERSITY_TOKENS = ("diversity", "pairwise", "redundan", "library_health", "library health", "k-mer", "kmer", "collapse")
P = {"eps": 1e-9, "near": 1.25, "material_drop": 0.01, "link": 0.3, "bootstrap": 10000, "seed": 20261111, "share_of_gain": 0.9}


# ----- records -----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Req:
    id: str
    metric: str | None
    status: str
    value: float | None
    minimum: float | None
    maximum: float | None
    stratum: str = ""


@dataclass
class Cand:
    id: str
    ordinal: int
    round: int
    parent: str | None
    start: str | None
    steps: int
    anchor: bool
    status: str = "ok"
    score: float | None = None
    qualified: bool | None = None
    classes: tuple[str, ...] | None = None
    reqs: tuple[Req, ...] = ()
    hypothesis: str = ""
    wall_h: float | None = None
    names: frozenset = frozenset()
    values: Mapping | None = None   # per condition, the declared ranking values of the row (None when the row has no measurements)
    qual: Mapping | None = None     # per condition, whether the row qualifies


@dataclass
class CellRec:
    label: str
    block: str
    task: str
    name: str
    cands: list[Cand]
    oracles: frozenset = frozenset()
    conditions: tuple = ()
    index: int = 0
    levers_offered: str = ""
    synthesis_enabled: bool | None = None
    roles_offered: int = 0
    components: list[dict] = field(default_factory=list)
    tokens_total: float | None = None
    n_decisions: int = 0
    committed_updates: int = 0
    delivered: dict = field(default_factory=dict)
    claims: list[dict] = field(default_factory=list)
    capability: dict = field(default_factory=dict)


# ----- class keys ----------------------------------------------------------------------------------------------------

def parts(key: str) -> tuple[str, str, str]:
    lever, direction, detail = (key.split(":", 2) + ["", ""])[:3]
    return lever, direction, detail


def levers_of(classes: Iterable[str]) -> set[str]:
    return {parts(key)[0] for key in classes}


# ----- structure -----------------------------------------------------------------------------------------------------

def anchor_of(cell: CellRec) -> Cand | None:
    return next((c for c in cell.cands if c.anchor), None)


def by_id(cell: CellRec) -> dict[str, Cand]:
    return {c.id: c for c in cell.cands}


def base_of(cell: CellRec, cand: Cand) -> Cand | None:
    """The candidate this one was built from: its parent, or the anchor for a candidate built from the initial policy."""
    ids = by_id(cell)
    if cand.parent is not None:
        return ids.get(cand.parent)
    anchor = anchor_of(cell)
    return anchor if anchor is not None and anchor.id != cand.id else None


def children(cell: CellRec) -> list[Cand]:
    """Agent-proposed candidates with a recipe: not the anchor, edit classes known."""
    return [c for c in cell.cands if not c.anchor and c.classes is not None]


def rounds_of(cell: CellRec) -> list[int]:
    return sorted({c.round for c in cell.cands})


def incumbent_before(cell: CellRec, round_: int) -> Cand | None:
    """The qualified candidate with the highest score among the rounds before `round_` (the earliest on a tie)."""
    pool = [c for c in cell.cands if c.round < round_ and c.qualified and c.score is not None]
    return max(pool, key=lambda c: (c.score, -c.ordinal)) if pool else None


def _condition_data(cand: Cand) -> tuple[Mapping, Mapping]:
    if cand.values is not None:
        return cand.values, cand.qual or {}
    if cand.score is None:
        return {}, {}
    return {"_": (cand.score,)}, {"_": bool(cand.qualified)}


def best_library(cands: Sequence[Cand], conditions: Sequence[str], index: int) -> float | None:
    """The minimum over conditions of the best qualified candidate's endpoint value (the largest tuple of ranking values), as
    `module_endpoints.best_library`; None when a condition has no qualified candidate."""
    per_condition = []
    for condition in conditions:
        scored = []
        for cand in cands:
            values, qual = _condition_data(cand)
            if qual.get(condition) and condition in values and all(v is not None for v in values[condition]):
                scored.append(tuple(values[condition]))
        if not scored:
            return None
        per_condition.append(max(scored)[index])
    return min(per_condition) if per_condition else None


def curve(cell: CellRec) -> list[tuple[int, int, float | None]]:
    """(round, cumulative committed updates, best-library value) at the end of every round, as `module_endpoints.anytime_curve`."""
    conditions = list(cell.conditions) or sorted({k for c in cell.cands for k in _condition_data(c)[0]})
    return [(r, sum(c.steps for c in cell.cands if c.round <= r), best_library([c for c in cell.cands if c.round <= r], conditions, cell.index))
            for r in rounds_of(cell)]


# ----- (1) uptake ----------------------------------------------------------------------------------------------------

def uptake(cell: CellRec) -> dict[str, float]:
    kids = children(cell)
    if not kids:
        return {}
    sets = [levers_of(c.classes) for c in kids]
    out = {f"uptake_{lever}": float(np.mean([lever in s for s in sets])) for lever in LEVERS}
    out["uptake_reward_family"] = float(np.mean([bool(s & set(REWARD_LEVERS)) for s in sets]))
    out["uptake_training_family"] = float(np.mean([bool(s & set(TRAINING_LEVERS)) for s in sets]))
    out["uptake_no_edit"] = float(np.mean([not s for s in sets]))
    out["uptake_exactly_one_lever"] = float(np.mean([len(s) == 1 for s in sets]))
    out["uptake_levers_per_candidate"] = float(np.mean([len(s) for s in sets]))
    out["n_children"] = float(len(kids))
    return out


def components(cell: CellRec) -> dict[str, float]:
    """Offered (the config's roles and request cap) against used (requests, published components, candidates whose recipe names one)."""
    asked = cell.components
    published = {item["name"] for item in asked if item.get("status") == "published"}
    kids = children(cell)
    return {
        "coder_roles_offered": float(cell.roles_offered), "coder_requests": float(len(asked)),
        "coder_published": float(sum(item.get("status") == "published" for item in asked)),
        "coder_rejected": float(sum(item.get("status") == "rejected" for item in asked)),
        "coder_candidates_naming_a_published_component": float(sum(bool(c.names & published) for c in kids)) if published else 0.0,
        "coder_share_candidates_naming_a_published_component":
            float(np.mean([bool(c.names & published) for c in kids])) if kids and published else (0.0 if kids else math.nan)}


# ----- (2) response to a regression ------------------------------------------------------------------------------------

def response_class(failed: Sequence[str], follow: Sequence[str]) -> str:
    """How a later candidate's edit classes relate to a candidate's: the first of the five that holds."""
    if not follow:
        return "no_edit"
    if set(follow) & set(failed):
        return "repeat"
    failed_parts = [parts(key) for key in failed]
    for key in follow:
        lever, direction, detail = parts(key)
        if any(lever == fl and detail == fd and OPPOSITE.get(direction) == fdir for fl, fdir, fd in failed_parts):
            return "reverse"
    if not (levers_of(follow) & levers_of(failed)):
        return "other_lever"
    return "same_lever_other_detail"


def regression_pairs(cell: CellRec) -> list[dict]:
    """One row per (candidate c of round r with a score, a base score and an edit; candidate n of round r+1 with an edit)."""
    ids = by_id(cell)
    out = []
    for c in children(cell):
        base = base_of(cell, c)
        if c.score is None or base is None or base.score is None or not c.classes:
            continue
        drop = base.score - c.score
        for n in children(cell):
            if n.round != c.round + 1:
                continue
            out.append({"cell": cell.name, "c": c.id, "n": n.id, "round_c": c.round, "drop": drop,
                        "regressed": drop > P["eps"], "improved_or_equal": drop <= P["eps"],
                        "material": drop > P["material_drop"] + P["eps"], "response": response_class(c.classes, n.classes),
                        "n_builds_on_c": n.parent == c.id,
                        "n_base_is_c_base": (base_of(cell, n) is not None and base_of(cell, n).id == base.id)})
    return out


def response_summary(pairs: Sequence[dict], prefix: str, select) -> dict[str, float]:
    chosen = [p for p in pairs if select(p)]
    out = {f"{prefix}_pairs": float(len(chosen)), f"{prefix}_candidates": float(len({p['c'] for p in chosen}))}
    for klass in RESPONSES:
        out[f"{prefix}_{klass}"] = float(np.mean([p["response"] == klass for p in chosen])) if chosen else math.nan
    out[f"{prefix}_builds_on_c"] = float(np.mean([p["n_builds_on_c"] for p in chosen])) if chosen else math.nan
    return out


def response_table(cell: CellRec) -> dict[str, float]:
    pairs = regression_pairs(cell)
    out: dict[str, float] = {}
    out.update(response_summary(pairs, "after_regression", lambda p: p["regressed"]))
    out.update(response_summary(pairs, "after_no_regression", lambda p: p["improved_or_equal"]))
    out.update(response_summary(pairs, "after_material_regression", lambda p: p["material"]))
    out["n_regressed_candidates"] = float(len({p["c"] for p in pairs if p["regressed"]}))
    return out


# ----- (3) response to health and qualification flags --------------------------------------------------------------------

def oracle_root(metric: str | None) -> str | None:
    if not metric:
        return None
    for suffix in ("_mean", "_ratio"):
        if metric.endswith(suffix):
            return metric[: -len(suffix)]
    return None


def flags_of(cand: Cand, oracles: Iterable[str]) -> dict[tuple[str, str], str]:
    """{(requirement id, stratum): 'fail' | 'near'} for the requirements of a candidate's row that the declaration maps to a rule."""
    out = {}
    for req in cand.reqs:
        mapped = req.id == "library_health" or oracle_root(req.metric) in set(oracles)
        if not mapped or req.value is None:
            continue
        if req.status != "pass":
            out[(req.id, req.stratum)] = "fail"
        elif req.minimum is not None and req.minimum > 0 and req.value <= P["near"] * req.minimum:
            out[(req.id, req.stratum)] = "near"
        elif req.maximum is not None and req.maximum > 0 and req.value >= req.maximum / P["near"]:
            out[(req.id, req.stratum)] = "near"
    return out


def rule_field(follow: Sequence[str], requirement_id: str, metric: str | None) -> bool:
    """The candidate edits what the requirement is about: a lever acting on the oracle behind it, or for library health a penalty or a gate."""
    if requirement_id == "library_health":
        return bool(levers_of(follow) & {"penalty", "gate"})
    root = oracle_root(metric)
    return any(parts(key)[0] in ("weights", "oracle_set", "utility") and parts(key)[2] == root for key in follow)


def rule_text(text: str, requirement_id: str, metric: str | None) -> bool:
    low = (text or "").lower()
    if requirement_id == "library_health":
        return any(token in low for token in DIVERSITY_TOKENS)
    root = oracle_root(metric)
    return bool(root) and (root in low or root.replace("_", " ") in low)


def flag_pairs(cell: CellRec) -> list[dict]:
    """One row per (candidate c, requirement mapped to a rule, candidate n of the next round with an edit)."""
    out = []
    kids = children(cell)
    for c in cell.cands:
        if c.status != "ok" or not c.reqs:
            continue
        flags = flags_of(c, cell.oracles)
        for req in c.reqs:
            if req.value is None or (req.id != "library_health" and oracle_root(req.metric) not in set(cell.oracles)):
                continue
            state = flags.get((req.id, req.stratum), "none")
            for n in kids:
                if n.round != c.round + 1:
                    continue
                out.append({"cell": cell.name, "c": c.id, "n": n.id, "requirement": req.id, "state": state,
                            "field_rule": rule_field(n.classes, req.id, req.metric),
                            "text_rule": rule_text(n.hypothesis, req.id, req.metric)})
    return out


def flag_table(cell: CellRec) -> dict[str, float]:
    pairs = flag_pairs(cell)
    out: dict[str, float] = {}
    for state in ("fail", "near", "none"):
        for group, select in (("diversity", lambda p: p["requirement"] == "library_health"),
                              ("oracle", lambda p: p["requirement"] != "library_health")):
            chosen = [p for p in pairs if p["state"] == state and select(p)]
            tag = f"flag_{state}_{group}"
            out[f"{tag}_pairs"] = float(len(chosen))
            out[f"{tag}_field_rule"] = float(np.mean([p["field_rule"] for p in chosen])) if chosen else math.nan
            out[f"{tag}_text_rule"] = float(np.mean([p["text_rule"] for p in chosen])) if chosen else math.nan
    return out


# ----- (4) parent choice ----------------------------------------------------------------------------------------------

def parent_rows(cell: CellRec) -> list[dict]:
    anchor = anchor_of(cell)
    out = []
    for n in children(cell):
        if n.round < 2:
            continue
        incumbent = incumbent_before(cell, n.round)
        base = base_of(cell, n)
        if incumbent is None or base is None:
            continue
        klass = "incumbent" if base.id == incumbent.id else ("anchor" if anchor is not None and base.id == anchor.id else "other")
        beats = bool(n.qualified and n.score is not None and n.score > incumbent.score + P["eps"])
        out.append({"cell": cell.name, "n": n.id, "round": n.round, "parent_class": klass, "start": n.start, "beats": beats,
                    "vs_incumbent": None if n.score is None else n.score - incumbent.score})
    return out


def parent_table(cell: CellRec) -> dict[str, float]:
    rows = parent_rows(cell)
    out: dict[str, float] = {"parent_rows": float(len(rows))}
    for klass in ("incumbent", "anchor", "other"):
        chosen = [r for r in rows if r["parent_class"] == klass]
        out[f"parent_share_{klass}"] = float(len(chosen) / len(rows)) if rows else math.nan
        out[f"parent_{klass}_n"] = float(len(chosen))
        out[f"parent_{klass}_beats_incumbent"] = float(np.mean([r["beats"] for r in chosen])) if chosen else math.nan
        scores = [r["vs_incumbent"] for r in chosen if r["vs_incumbent"] is not None]
        out[f"parent_{klass}_mean_vs_incumbent"] = float(np.mean(scores)) if scores else math.nan
    starts = [r["start"] for r in rows]
    for mode in ("from_initial", "from_parent", "continue_parent"):
        out[f"start_share_{mode}"] = float(np.mean([s == mode for s in starts])) if starts else math.nan
    return out


# ----- (5) learning curve ----------------------------------------------------------------------------------------------

def round_rows(cell: CellRec) -> list[dict]:
    points = {r: (u, v) for r, u, v in curve(cell)}
    out = []
    for r in rounds_of(cell):
        kids = [c for c in children(cell) if c.round == r]
        gains = []
        for c in kids:
            base = base_of(cell, c)
            if c.score is not None and base is not None and base.score is not None:
                gains.append(c.score - base.score)
        incumbent = incumbent_before(cell, r) if r >= 2 else None
        beat = [bool(c.qualified and c.score is not None and c.score > incumbent.score + P["eps"]) for c in kids] if incumbent else []
        out.append({"cell": cell.name, "round": r, "children": len(kids), "scored_children": len(gains),
                    "mean_gain_vs_base": float(np.mean(gains)) if gains else math.nan,
                    "share_improved_vs_base": float(np.mean([g > P["eps"] for g in gains])) if gains else math.nan,
                    "share_beat_incumbent": float(np.mean(beat)) if beat else math.nan,
                    "curve_updates": points[r][0], "curve_value": math.nan if points[r][1] is None else points[r][1]})
    return out


# ----- (6) efficiency ---------------------------------------------------------------------------------------------------

def efficiency(cell: CellRec) -> dict[str, float]:
    points = curve(cell)
    anchor = anchor_of(cell)
    final = points[-1][2] if points else None
    nan = math.nan
    out = {"committed_updates": float(cell.committed_updates), "rounds": float(len(points)), "candidates": float(len(cell.cands)),
           "tokens_total": nan if cell.tokens_total is None else float(cell.tokens_total), "decisions": float(cell.n_decisions),
           "wall_h_candidates": float(sum(c.wall_h or 0.0 for c in cell.cands)),
           "final_value": nan if final is None else final, "anchor_score": nan if anchor is None or anchor.score is None else anchor.score,
           "anchor_qualified": nan if anchor is None or anchor.qualified is None else float(anchor.qualified)}
    gain = nan
    if final is not None and anchor is not None and anchor.score is not None and anchor.qualified:
        gain = final - anchor.score
    out["gain_over_anchor"] = gain
    out["rounds_to_90pct_of_gain"] = nan
    if gain > 0:
        target = anchor.score + P["share_of_gain"] * gain
        out["rounds_to_90pct_of_gain"] = float(next(i + 1 for i, (_, _, v) in enumerate(points) if v is not None and v >= target - 1e-12))
    values = [v for _, _, v in points if v is not None]
    out["rounds_to_90pct_of_final"] = nan
    if final is not None and final > 0 and all(v >= 0 for v in values):
        out["rounds_to_90pct_of_final"] = float(next(i + 1 for i, (_, _, v) in enumerate(points)
                                                     if v is not None and v >= P["share_of_gain"] * final - 1e-12))
    last = 0
    best = None
    for i, (_, _, v) in enumerate(points):
        if v is not None and (best is None or v > best + P["eps"]):
            best, last = v, i + 1
    out["last_improvement_round"] = float(last) if best is not None else nan
    return out


# ----- (7) claims calibration -----------------------------------------------------------------------------------------

CONFIDENCES = (0.5, 0.65, 0.75, 0.85, 0.95)


def calibration(claims: Sequence[Mapping]) -> dict[str, Any]:
    """Hit rate by stated confidence and the Brier score, over the claims the cell's own `claims.resolved` events hold.

    `claims` are rows {confidence, verdict, direction_hit}. An unresolvable claim is counted and left out of every rate."""
    resolved = [c for c in claims if c["verdict"] != "unresolvable"]
    hits = [c["verdict"] == "hit" for c in resolved]
    out: dict[str, Any] = {"claims": len(claims), "resolved": len(resolved), "unresolvable": len(claims) - len(resolved)}
    for confidence in CONFIDENCES:
        inside = [h for c, h in zip(resolved, hits) if c["confidence"] == confidence]
        out[f"n_{confidence}"] = len(inside)
        out[f"hit_rate_{confidence}"] = float(np.mean(inside)) if inside else math.nan
    out["hit_rate"] = float(np.mean(hits)) if hits else math.nan
    out["mean_confidence"] = float(np.mean([c["confidence"] for c in resolved])) if resolved else math.nan
    out["confidence_minus_hit_rate"] = out["mean_confidence"] - out["hit_rate"] if hits else math.nan
    out["brier"] = float(np.mean([(c["confidence"] - h) ** 2 for c, h in zip(resolved, hits)])) if hits else math.nan
    out["brier_always_half"] = float(np.mean([(0.5 - h) ** 2 for h in hits])) if hits else math.nan
    out["direction_hit_rate"] = float(np.mean([bool(c.get("direction_hit")) for c in resolved])) if resolved else math.nan
    return out


# ----- non-score axes ----------------------------------------------------------------------------------------------------

def basins(values: Sequence[float], link: float) -> list[int]:
    """Single-linkage groups of a one-dimensional sample: a new group starts where the sorted gap exceeds `link`; labelled from 1
    in order of increasing value, returned in the input order. NaN has label 0."""
    order = sorted((v, i) for i, v in enumerate(values) if not math.isnan(v))
    labels = [0] * len(values)
    label = 0
    previous = None
    for v, i in order:
        if previous is None or v - previous > link + 1e-12:
            label += 1
        labels[i] = label
        previous = v
    return labels


# ----- intervals -----------------------------------------------------------------------------------------------------------

def finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def boot_mean(values: Iterable[float], reps: int, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile 95% interval of the mean over cells (resampling cells); NaN when fewer than two cells have a value."""
    array = finite(values)
    if len(array) < 2:
        return math.nan, math.nan
    means = array[rng.integers(0, len(array), size=(reps, len(array)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def boot_difference(a: Iterable[float], b: Iterable[float], reps: int, rng: np.random.Generator) -> tuple[float, float]:
    x, y = finite(a), finite(b)
    if len(x) < 2 or len(y) < 2:
        return math.nan, math.nan
    diff = (x[rng.integers(0, len(x), size=(reps, len(x)))].mean(axis=1) - y[rng.integers(0, len(y), size=(reps, len(y)))].mean(axis=1))
    return float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))


def boot_ratio(numerator: Iterable[float], denominator: Iterable[float], reps: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Ratio of arm means (numerator over denominator) with a percentile interval from resampling whole cells; cells missing either value are dropped."""
    pairs = [(n, d) for n, d in zip(numerator, denominator) if math.isfinite(n) and math.isfinite(d)]
    if len(pairs) < 2:
        return math.nan, math.nan, math.nan
    array = np.asarray(pairs)
    idx = rng.integers(0, len(array), size=(reps, len(array)))
    num, den = array[:, 0][idx].mean(axis=1), array[:, 1][idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(den > 0, num / den, np.nan)
    ratios = ratios[np.isfinite(ratios)]
    point = array[:, 0].mean() / array[:, 1].mean() if array[:, 1].mean() > 0 else math.nan
    if len(ratios) < reps * 0.9:
        return float(point), math.nan, math.nan
    return float(point), float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


# ----- reading cells -------------------------------------------------------------------------------------------------------

def check_declaration(tag: str) -> None:
    """The tag exists, is more than 60 s old, and holds the declared files byte for byte."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout.strip()
    commit = git("rev-parse", f"{tag}^{{commit}}")
    created = int(git("for-each-ref", f"refs/tags/{tag}", "--format=%(creatordate:unix)"))
    if time.time() - created < 60:
        raise SystemExit(f"tag {tag} is less than 60 s old")
    for path in (DECLARATION, DECLARED, HERE, TESTS):
        relative = path.relative_to(REPO).as_posix()
        blob = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{relative}"], capture_output=True).stdout
        if blob != path.read_bytes():
            raise SystemExit(f"{relative} differs from the file at tag {tag}")
    print(f"declaration verified at tag {tag} ({commit[:12]})")


def tree_commit() -> str:
    """The commit of the tree that supplies `analysis.module_endpoints`, whose code reads the cells."""
    import analysis.module_endpoints as me
    root = Path(me.__file__).resolve().parents[2]
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() + f" ({root})"


def events_of(path: Path) -> list[dict]:
    with open(path / "events" / "log.jsonl") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def req_of(item: Mapping) -> Req:
    return Req(id=item["requirement_id"], metric=item.get("metric"), status=item["status"], value=item.get("value"),
               minimum=item.get("minimum"), maximum=item.get("maximum"), stratum=item.get("stratum", ""))


def load_cell_rec(path: Path, label: str, block: str, task_override: Mapping[str, str]) -> tuple[CellRec, Any]:
    """The cell as a `CellRec`, mirroring `module_endpoints.facts_of` for candidate scores and qualification."""
    import analysis.module_endpoints as me
    from common.control_cells import load_cell
    from recital.control.ext.edit_class import edit_classes
    from recital.control.selection import condition_status, condition_values

    cell = load_cell(path)
    view = cell.view
    task = me.task_of(cell)
    success = view.success_spec()
    conditions = view.conditions
    index = me.endpoint_index(success, task, task_override)
    states = [c for c in view.state.ordered_candidates() if c.status != "forfeited"]
    recipes = {c.id: view.spec(c.id)["recipe"] for c in states if c.recipe_sha}
    anchor = next((c for c in states if c.anchor and c.recipe_sha), None)
    initial = recipes[anchor.id] if anchor else None
    cands, oracles = [], set()
    for state in states:
        row = view.ledger_row(state.id) if state.ledger_row_sha else None
        qualification = (row or {}).get("qualification")
        base = recipes.get(state.parent) if state.parent else initial
        recipe = recipes.get(state.id) or {}
        classes = None if state.anchor or base is None or state.id not in recipes else tuple(
            item.key for item in edit_classes(base, recipes[state.id]))
        oracles |= set(recipe.get("enabled_oracles") or ())
        score = qualified = values_by = qual_by = None
        if qualification is not None:
            values_by = {c: tuple(condition_values(success, qualification, c)) for c in conditions}
            qual_by = {c: bool(condition_status(qualification, c).qualified) for c in conditions}
            ends = [values_by[c][index] for c in conditions]
            score = None if any(v is None for v in ends) else float(min(ends))
            qualified = all(qual_by.values())
        spec = view.spec(state.id) if state.recipe_sha else {}
        names = {p["name"] for p in recipe.get("penalties") or ()} | {g["name"] for g in recipe.get("gates") or ()}
        names.add((recipe.get("aggregator") or {}).get("name"))
        wall = ((row or {}).get("training") or {}).get("wall_h")
        cands.append(Cand(
            id=state.id, ordinal=state.ordinal, round=state.round, parent=state.parent, start=state.start, steps=state.steps,
            anchor=state.anchor, status=state.status, score=score, qualified=qualified, classes=classes,
            reqs=tuple(req_of(r) for r in ((qualification or {}).get("requirements") or ())),
            hypothesis=str((spec or {}).get("hypothesis") or ""), wall_h=wall, names=frozenset(n for n in names if n),
            values=values_by, qual=qual_by))
    control = view.config().get("control") or {}
    synthesis = control.get("synthesis") or {}
    asked, claims, decisions = {}, [], 0
    for event in events_of(path):
        data = event.get("data") or {}
        if event["type"] == "component_requested":
            asked[data["request_id"]] = {"name": data["name"], "role": data["role"], "status": None}
        elif event["type"] == "component_result" and data["request_id"] in asked:
            asked[data["request_id"]]["status"] = data["status"]
        elif event["type"] == "decision_recorded" and data.get("kind") == "propose":
            decisions += 1
        elif event["type"] == "claims.resolved":
            for claim, found in zip(data["claims"], data["resolutions"]):
                claims.append({"confidence": float(claim["confidence"]), "verdict": found["verdict"],
                               "direction_hit": found.get("direction_hit")})
    rec = CellRec(label=label, block=block, task=task, name=path.name, cands=cands, oracles=frozenset(oracles),
                  conditions=tuple(conditions), index=index,
                  levers_offered=",".join(control.get("levers") or ["reward", "training"]),
                  synthesis_enabled=synthesis.get("enabled"), roles_offered=len(synthesis.get("roles") or ()) if synthesis.get("enabled") else 0,
                  components=list(asked.values()), tokens_total=cell.summary["tokens"]["total"], n_decisions=decisions,
                  committed_updates=view.state.committed_updates, claims=claims)
    rec.delivered = delivered_of(cell, task, index, success)
    rec.capability = me.capability_row(cell)
    return rec, cell


def delivered_of(cell, task: str, index: int, success) -> dict[str, float]:
    """What the frozen delivered library carries: the stored ledger row of each condition's delivered candidate (the minimum over
    conditions of each measurement, as `module_endpoints.delivered_means`), and the terminal audit."""
    import analysis.module_endpoints as me
    assignment = cell.summary.get("assignment") or {}
    out: dict[str, float] = {}
    ranking = cell.summary.get("delivered_ranking") or {}
    metric = success.ranking[index].metric
    out["deliv_score"] = float(min(v[metric] for v in ranking.values())) if ranking else math.nan
    rows = {c: cell.view.ledger_row(candidate) or {} for c, candidate in assignment.items()}
    measurements = {c: (row.get("qualification") or {}).get("measurements") or {} for c, row in rows.items()}
    for name in sorted({n for found in measurements.values() for n in found}):
        vals = [(found.get(name, {}).get(c) or {}).get("value") for c, found in measurements.items()]
        out[f"deliv_{name}"] = math.nan if not vals or any(v is None for v in vals) else float(min(vals))
    pr = [(row.get("library") or {}).get(c, {}).get("pairwise_ratio") for c, row in rows.items()]
    out["deliv_pairwise_ratio"] = math.nan if not pr or any(v is None for v in pr) else float(min(pr))
    q = [((row.get("qualified") or {}).get(c) or {}).get("qualified") for c, row in rows.items()]
    out["deliv_qualified"] = float(all(q)) if q and all(v is not None for v in q) else math.nan
    primary, secondary = me.audit_values(cell, task)
    out["audit_primary"], out["audit_secondary"] = float(primary), float(secondary)
    return out


# ----- tables ------------------------------------------------------------------------------------------------------------------

def cell_metrics(rec: CellRec) -> dict[str, Any]:
    row: dict[str, Any] = {"label": rec.label, "block": rec.block, "task": rec.task, "cell": rec.name,
                           "levers_offered": rec.levers_offered, "synthesis_enabled": rec.synthesis_enabled,
                           "agent_cell": len(children(rec)) > 0}
    for part in (uptake(rec), components(rec), response_table(rec), flag_table(rec), parent_table(rec), efficiency(rec), rec.delivered,
                 {f"cap_{key}": value for key, value in rec.capability.items() if key not in ("cell", "campaign")}):
        row.update(part)
    return row


def summarize(table, group: str, columns: Sequence[str], rng) -> list[dict]:
    out = []
    for key, frame in table.groupby(group, sort=False):
        for column in columns:
            values = finite(frame[column])
            lo, hi = boot_mean(frame[column], P["bootstrap"], rng)
            out.append({"group": key, "metric": column, "n_cells": len(values), "mean": float(values.mean()) if len(values) else math.nan,
                        "ci_low": lo, "ci_high": hi, "min": float(values.min()) if len(values) else math.nan,
                        "max": float(values.max()) if len(values) else math.nan})
    return out


# ----- gate and run ---------------------------------------------------------------------------------------------------------------

def cell_paths(spec: Mapping, scratch: Path) -> list[tuple[str, str, Path]]:
    out = []
    for group in spec["groups"]:
        for root in group["roots"]:
            for seed in root["seeds"]:
                out.append((group["label"], group["block"], scratch / root["root"] / root["cell"].format(seed=seed)))
    return out


def gate(scratch: Path) -> None:
    import analysis.module_endpoints as me
    declared = json.loads(DECLARED.read_text())
    worst = {"final": 0.0, "committed": 0, "candidates": 0, "tokens": 0.0}
    n = 0
    for label, block, path in cell_paths(declared, scratch):
        rec, cell = load_cell_rec(path, label, block, {})
        row, _ = me.cell_row(cell, {})
        mine = curve(rec)[-1][2]
        theirs = None if row["failed"] else row["final_best"]
        if (mine is None) != (theirs is None):
            raise SystemExit(f"{path.name}: the curve ends {'without' if mine is None else 'with'} a value and module_endpoints "
                             f"{'with' if mine is None else 'without'}")
        if mine is not None:
            worst["final"] = max(worst["final"], abs(mine - theirs))
        worst["committed"] = max(worst["committed"], abs(rec.committed_updates - row["committed_updates"]))
        worst["candidates"] = max(worst["candidates"], abs(len(rec.cands) - row["candidates"]))
        worst["tokens"] = max(worst["tokens"], abs((rec.tokens_total or 0) - (row["tokens_total"] or 0)))
        n += 1
    print(f"gate: reading code from tree {tree_commit()}")
    print(f"gate: {n} cells read; largest differences against analysis.module_endpoints: final value {worst['final']:.3g}, "
          f"committed updates {worst['committed']}, candidates {worst['candidates']}, tokens {worst['tokens']:.3g}")
    if worst["final"] > 1e-9 or worst["committed"] or worst["candidates"] or worst["tokens"]:
        raise SystemExit("gate failed: the loader does not reproduce the committed endpoint code")
    print("gate passed")


def to_md(frame, floatfmt: str = ".4g") -> str:
    def fmt(value) -> str:
        if isinstance(value, float):
            return "" if math.isnan(value) else format(value, floatfmt)
        return str(value)
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def run(tag: str, scratch: Path, out: Path) -> None:
    import pandas as pd
    check_declaration(tag)
    declared = json.loads(DECLARED.read_text())
    for key, value in declared["parameters"].items():
        if P[key] != value:
            raise SystemExit(f"parameter {key}: the script holds {P[key]}, the declaration {value}")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(P["seed"])
    recs, statuses = [], []
    for label, block, path in cell_paths(declared, scratch):
        rec, cell = load_cell_rec(path, label, block, {})
        recs.append(rec)
        statuses.append({"label": label, "cell": rec.name, "status": cell.status, "audited": cell.audited})
    if any(s["status"] == "running" or not s["audited"] for s in statuses):
        raise SystemExit("a listed cell is not frozen with a complete terminal audit: " + json.dumps([s for s in statuses if s["status"] == "running" or not s["audited"]]))
    table = pd.DataFrame([cell_metrics(r) for r in recs])
    table.to_csv(out / "cells.csv", index=False)
    pd.DataFrame([{"label": r.label, **p} for r in recs for p in regression_pairs(r)]).to_csv(out / "pairs_regression.csv", index=False)
    pd.DataFrame([{"label": r.label, **p} for r in recs for p in flag_pairs(r)]).to_csv(out / "pairs_flags.csv", index=False)
    pd.DataFrame([{"label": r.label, **p} for r in recs for p in parent_rows(r)]).to_csv(out / "parent_rows.csv", index=False)
    rounds = pd.DataFrame([{"label": r.label, **p} for r in recs for p in round_rows(r)])
    rounds.to_csv(out / "rounds.csv", index=False)
    numeric = [c for c in table.columns if c not in ("label", "block", "task", "cell", "levers_offered", "synthesis_enabled", "agent_cell")
               and table[c].dtype.kind in "fiub"]
    summary = pd.DataFrame(summarize(table, "label", numeric, rng))
    summary.to_csv(out / "summary_by_arm.csv", index=False)
    round_summary = pd.DataFrame([
        {"label": label, "round": r, "n_cells": int(frame[c].notna().sum()), "metric": c, "mean": float(frame[c].mean())}
        for (label, r), frame in rounds.groupby(["label", "round"], sort=False)
        for c in ("mean_gain_vs_base", "share_improved_vs_base", "share_beat_incumbent", "curve_value", "children")])
    round_summary.to_csv(out / "rounds_by_arm.csv", index=False)

    contrasts = []
    for item in declared["contrasts"]:
        a, b = table[table.label == item["a"]], table[table.label == item["b"]]
        for column in declared["headline_columns"]:
            lo, hi = boot_difference(a[column], b[column], P["bootstrap"], rng)
            contrasts.append({"contrast": item["name"], "metric": column, "n_a": int(finite(a[column]).size), "n_b": int(finite(b[column]).size),
                              "mean_a": float(finite(a[column]).mean()) if finite(a[column]).size else math.nan,
                              "mean_b": float(finite(b[column]).mean()) if finite(b[column]).size else math.nan,
                              "difference": float(finite(a[column]).mean() - finite(b[column]).mean()) if finite(a[column]).size and finite(b[column]).size else math.nan,
                              "ci_low": lo, "ci_high": hi})
    pd.DataFrame(contrasts).to_csv(out / "contrasts.csv", index=False)

    ratios = []
    for label, frame in table[table.agent_cell].groupby("label", sort=False):
        for column in ("committed_updates", "tokens_total", "wall_h_candidates"):
            point, lo, hi = boot_ratio(frame[column], frame["gain_over_anchor"], P["bootstrap"], rng)
            ratios.append({"label": label, "numerator": column, "per_0.01_gain": point * 0.01, "ci_low": lo * 0.01, "ci_high": hi * 0.01,
                           "cells_with_gain": int(sum(math.isfinite(g) for g in frame["gain_over_anchor"]))})
    pd.DataFrame(ratios).to_csv(out / "efficiency_ratios.csv", index=False)

    reliability, basin_rows = [], []
    floor = declared["reliability"]
    fixed = table[table.label == floor["floor_arm"]]
    floors = {"deliv_score": float(fixed["deliv_score"].mean()), "audit_primary": float(fixed["audit_primary"].mean())}
    for block in sorted({r.block for r in recs if r.task == declared["reliability"]["task"]}):
        frame = table[(table.block == block) & (table.task == floor["task"])].copy()
        frame["basin"] = basins(list(frame["audit_primary"]), P["link"])
        for label, arm in frame.groupby("label", sort=False):
            reliability.append({"label": label, "n_cells": len(arm),
                                **{f"share_at_or_above_{k}_floor": float((arm[k] >= v - 1e-12).mean()) for k, v in floors.items()},
                                **{f"{k}_sd": float(arm[k].std(ddof=1)) if len(arm) > 1 else math.nan for k in floors},
                                **{f"{k}_range": float(arm[k].max() - arm[k].min()) for k in floors},
                                "basins_occupied": int(arm["basin"].nunique())})
            for basin, cells in arm.groupby("basin"):
                basin_rows.append({"block": block, "label": label, "basin": int(basin), "cells": len(cells)})
    pd.DataFrame(reliability).to_csv(out / "reliability.csv", index=False)
    pd.DataFrame(basin_rows).to_csv(out / "basins.csv", index=False)

    claim_rows = []
    claim_cells = [r for r in recs if r.claims]
    for r in claim_cells:
        claim_rows.append({"label": r.label, "cell": r.name, **calibration(r.claims)})
    if claim_cells:
        claim_rows.append({"label": "pooled", "cell": "all", **calibration([c for r in claim_cells for c in r.claims])})
    pd.DataFrame(claim_rows).to_csv(out / "claims.csv", index=False)

    headline = declared["headline_columns"]
    capability = table[["label", "cell"] + [c for c in table.columns if c.startswith("cap_")]].rename(columns=lambda c: c[4:] if c.startswith("cap_") else c)
    parts_md = ["# Decision-trace audit: results", "", f"Tag {tag}; {len(recs)} cells; parameters {json.dumps(P)}; reading code from tree {tree_commit()}.", "",
                "## Capability state of every cell (read from the cell's own config and events; before any number below is cited)", "",
                to_md(capability), "",
                "## Cells", "", to_md(pd.DataFrame(statuses).groupby("label", sort=False).size().reset_index(name="cells")), "",
                "## Headline metrics by arm (mean over cells, 95% interval from resampling cells, range)", "",
                to_md(summary[summary.metric.isin(headline)].round(4)), "",
                "## Contrasts (difference of arm means; interval from resampling cells in each arm; descriptive)", "",
                to_md(pd.DataFrame(contrasts).round(4)), "",
                "## Efficiency: arm mean of the quantity per 0.01 of visible gain over the anchor (ratio of arm means)", "",
                to_md(pd.DataFrame(ratios).round(3)), "",
                "## Reliability and basins (AMP)", "", to_md(pd.DataFrame(reliability).round(4)), "", to_md(pd.DataFrame(basin_rows)), "",
                f"Floors from the fixed-recipe arm: {json.dumps(floors)}.", ""]
    if claim_rows:
        parts_md += ["## Claims calibration (cells with structured claims)", "", to_md(pd.DataFrame(claim_rows).round(4)), ""]
    (out / "results.md").write_text("\n".join(parts_md) + "\n")
    print(f"wrote {len(list(out.iterdir()))} files to {out}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--declared-tag")
    parser.add_argument("--scratch", type=Path, default=SCRATCH)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.selftest:
        import pytest
        return int(pytest.main(["-q", str(TESTS)]))
    if args.gate:
        gate(args.scratch)
        return 0
    if args.run:
        if not args.declared_tag or not args.out:
            parser.error("--run needs --declared-tag and --out")
        run(args.declared_tag, args.scratch, args.out)
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
