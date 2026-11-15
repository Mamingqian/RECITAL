#!/usr/bin/env python3
"""The tables of a control-layer campaign, built from what its cells left on disk.

    aggregate_control_campaign.py ROOT --task {cre,loopseq,cm} --out OUTDIR
                                  [--round1-root ROOT1] [--expected-seeds N]

ROOT is a campaign output root (a directory of cells) or one cell. The campaign is only read: OUTDIR must lie
outside ROOT, and what is read is `events/log.jsonl` through `load_run`, its blobs, `delivery/`, `baseline/`,
`final_sequences.parquet`, `terminal_audit/*` and the config snapshots. A cell with no `terminal_audit/` yields
`no audit yet`, not an error; a field an older cell never recorded reads `not recorded`.

Written to OUTDIR as CSV, plus `summary.md`:

  capability.csv      per cell, what CLAUDE.md requires before any run is cited: control.budget.steps,
                      synthesis (enabled, roles, max_requests), control.health, candidates allocated, rounds,
                      resumes, resolved model ids, stop round and source, job counts, blind header digest
                      (and, for cm, whether the PoET-2 sentinel's scores are finite)
  estimand_by_condition.csv, estimand_by_cell.csv, arm_summary.csv
                      the estimand each declaration names (E1):
      cre      the blind ChromBPNet raw target-minus-off-target gap over the delivered rows (`panel_gap` on the
               audit manifest's declared panel), cal = raw - DELTA[cell line], the worst condition (minimum of
               the calibrated medians), the clusters at 10 percent Hamming (single linkage, first 200 bases),
               the cluster-weighted median and its cluster-bootstrap 95 percent interval (2000 draws, seed 0;
               none for 4 or fewer clusters), and the delivered library's mean pairwise normalized edit
               distance relative to the initial policy's pool (`pairwise_ratio`)
      loopseq  min over conditions of the mean in-band fraction (primary) and of the median signed margin
               (secondary), from terminal_audit/summary.json
      cm       audit_2020_t5 median (primary), audit_2020_esm median (secondary), and the delivered library's
               natural-distance bins (cm_distance_bins.csv)
  contrasts.csv, contrast_summary.csv
                      E2 at matched seed: self_validating - detect_only, self_validating - no_self_validation,
                      and detect_only - no_self_validation as the noise floor, with the preregistered pass rule
                      (`preregistered` false marks an estimand read beside the declared one). One seed supports
                      no contrast and none is written.
  descriptives.csv    D1/E4 per cell: candidates failing library_health, delivered pairwise_ratio and
                      unique_kmer_ratio, invalid decisions and their problem codes, restarts from_initial,
                      kl and lr chosen, candidates enabling a source the initial recipe does not, Coder
                      requests, flag counts
  flag_history.csv    the flag of every comparison of every candidate, in the arms that compute the table:
                      shown to the agent in self_validating, silent in detect_only
  keyword_share.csv   propose/stop decisions whose rationale or hypotheses mention range, artifact, exploit,
                      extrapolation or overoptimization, per cell and per arm
  round1_comparison.csv, round1_arm_summary.csv, round1_contrasts.csv
                      (cre, with --round1-root) the round-1 campaign beside this one, never pooled

In silico, computational estimands only.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.sparse.csgraph import connected_components

from analysis.campaign_status import capability, read_json
from analysis.cm_published_comparators import BINS as DISTANCE_BINS
from analysis.cre_movement_vs_blind import DELTA
from recital.control import events as ev
from recital.control.errors import ControlError
from recital.control.export import LOG_PATH, RunView, load_run, selfval_table, summary_of
from recital.control.health import REQUIREMENT_ID, ConditionReference, baseline_reference, library_health
from tasks.promoter.score_contract import panel_gap

TASKS = ("cre", "loopseq", "cm")
ARMS = ("self_validating", "detect_only", "no_self_validation")
BLIND_ARMS = ("detect_only", "no_self_validation")
# (minuend, subtrahend). The last is the noise floor: the blind arms are prompt-identical.
CONTRASTS = (("self_validating", "detect_only"), ("self_validating", "no_self_validation"),
             ("detect_only", "no_self_validation"))
NOISE_FLOOR = CONTRASTS[-1]
EXPECTED_SEEDS = {"cre": 3, "loopseq": 3, "cm": 1}

NOT_RECORDED = "not recorded"
NO_AUDIT = "no audit yet"

CLUSTER_FRACTION = 0.10
CLUSTER_BASES = 200
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 0
MAX_CLUSTERS_WITHOUT_INTERVAL = 4
# A decision counts once when its rationale or any candidate's hypothesis matches.
KEYWORDS = re.compile(r"\b(range|artifact|exploit|extrapolat|over-?optimi[sz])", re.IGNORECASE)
CM_SECONDARY = "audit_2020_esm"
CM_SENTINEL = "cm_poet"

CAVEATS = {
    "cre": "ChromBPNet audited round 1 and informed the round-2 design, so for any later CRE campaign it is the "
           "same estimator as in round 1, not an independent audit.",
    "loopseq": "The blind DNAcycP2 estimand historically saturates; an arm difference of one to three of 128 "
               "sequences is inside the noise floor.",
    "cm": "One seed supports no between-arm claim. audit_2020_t5 shares ProtT5 with the function_2026_t5 "
          "sentinel and is nearly uninformative at natural distance 0.6 or more.",
}

# Columns of the by-condition table, fixed so a campaign with no audit yet still has them.
BY_CONDITION_COLUMNS = {
    "cre": ["condition", "audit", "n_rows", "raw_median", "summary_raw_median", "matches_summary", "delta",
            "cal_median", "n_clusters", "cluster_weighted_cal_median", "ci95_low", "ci95_high", "interval",
            "pairwise_ratio", "unique_kmer_ratio", "pairwise_ratio_recorded"],
    "loopseq": ["condition", "audit", "n_rows", "primary_in_band", "secondary_median_margin", "pairwise_ratio",
                "unique_kmer_ratio", "pairwise_ratio_recorded"],
    "cm": ["condition", "audit", "n_rows", "primary_source", "primary_median", "secondary_source",
           "secondary_median", "natural_distance_median", "natural_distance_min", "natural_distance_max",
           "pairwise_ratio", "unique_kmer_ratio", "pairwise_ratio_recorded"],
}
# `worst_per_cell` columns (output: source), and the estimands E2 is read on.
BY_CELL_COLUMNS = {
    "cre": {"worst_cal": "cal_median", "worst_cal_cluster_weighted": "cluster_weighted_cal_median",
            "worst_raw_summary": "summary_raw_median", "min_n_clusters": "n_clusters",
            "min_pairwise_ratio": "pairwise_ratio"},
    "loopseq": {"primary_worst": "primary_in_band", "secondary_worst": "secondary_median_margin",
                "min_pairwise_ratio": "pairwise_ratio"},
    "cm": {"primary_median": "primary_median", "secondary_median": "secondary_median",
           "min_pairwise_ratio": "pairwise_ratio"},
}
CONTRAST_ESTIMANDS = {"cre": ["worst_cal", "worst_cal_cluster_weighted"],
                      "loopseq": ["primary_worst", "secondary_worst"],
                      "cm": ["primary_median", "secondary_median"]}
# E2 names the E1 primary; the cluster-weighted worst condition is read beside it, not preregistered as a contrast.
SUPPLEMENTARY = {"worst_cal_cluster_weighted"}


@dataclass
class Cell:
    path: Path
    view: RunView

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def arm(self) -> str:
        return self.view.header.arm

    @property
    def seed(self) -> int:
        return self.view.header.seed

    @property
    def conditions(self) -> list[str]:
        return self.view.conditions

    def ident(self) -> dict[str, Any]:
        return {"cell": self.name, "arm": self.arm, "seed": self.seed}

    @cached_property
    def summary(self) -> dict[str, Any]:
        return summary_of(self.view)

    @cached_property
    def audit_summary(self) -> dict[str, Any] | None:
        return read_json(self.path / "terminal_audit" / "summary.json")

    @cached_property
    def audit(self) -> str:
        """The terminal audit's status, or why there is none."""
        if self.audit_summary is not None:
            return str(self.audit_summary.get("status", NOT_RECORDED))
        return "partial (no summary yet)" if (self.path / "terminal_audit").is_dir() else NO_AUDIT

    @cached_property
    def audit_complete(self) -> bool:
        return self.audit == "complete"

    @cached_property
    def library(self) -> pd.DataFrame | None:
        """The rows `final_sequences.parquet` froze (row_id, sequence, condition); None until the cell has them."""
        path = self.path / "final_sequences.parquet"
        if not path.is_file():
            return None
        try:
            return pd.read_parquet(path, columns=["row_id", "sequence", "condition"])
        except (OSError, ValueError):
            return None

    @cached_property
    def health(self) -> dict[str, dict[str, Any]]:
        return delivered_health(self)


# ----- reading cells ---------------------------------------------------------------------------

def load_cells(root: Path, task: str) -> tuple[list[Cell], list[dict[str, str]]]:
    """The cells of `task` under `root` (or `root` itself), and the directories that could not be read."""
    found = [root] if (root / LOG_PATH).is_file() else sorted(
        path for path in root.iterdir() if path.is_dir() and (path / LOG_PATH).is_file())
    cells, skipped = [], []
    for path in found:
        try:
            view = load_run(path)
        except (ControlError, OSError, ValueError) as exc:
            skipped.append({"cell": path.name, "reason": f"{type(exc).__name__}: {str(exc)[:160]}"})
            continue
        if view.header.task != task:
            skipped.append({"cell": path.name, "reason": f"task {view.header.task!r}, not {task!r}"})
            continue
        cells.append(Cell(path, view))
    return cells, skipped


def library_health_outcomes(cell: Cell) -> dict[str, tuple[bool, bool]]:
    """Per trained candidate: whether its qualification carries a `library_health` requirement, and whether it
    fails it in some condition. Result files are read one at a time and not cached, because each is large;
    a file that is absent or still being written leaves its candidate out."""
    outcomes = {}
    for candidate in cell.view.state.ordered_candidates():
        result = read_json(cell.path / "candidates" / candidate.id / "candidate_result.json")
        if candidate.status != "ok" or result is None:
            continue
        requirements = result["delivery"]["qualification"].get("requirements", ())
        outcomes[candidate.id] = (
            any(item["requirement_id"] == REQUIREMENT_ID for item in requirements),
            any(REQUIREMENT_ID in info["failing_requirements"] for info in result["by_condition"].values()))
    return outcomes


def delivery_measurement(cell: Cell, name: str, row_ids: list[str]) -> np.ndarray | None:
    """A per-row measurement of the delivery pool looked up by row id; None when the pool does not hold it."""
    pool = read_json(cell.path / "delivery" / "pool.json")
    measured = read_json(cell.path / "delivery" / "public_measurements.json")
    entry = ((measured or {}).get("measurements") or {}).get(name)
    if not pool or "row_ids" not in pool or entry is None:
        return None
    by_row = dict(zip(pool["row_ids"], entry["values"]))
    return np.asarray([by_row.get(row_id) for row_id in row_ids], dtype=float)


def _nan(value: Any) -> float:
    return np.nan if value is None else float(value)


# ----- capability state ------------------------------------------------------------------------

def _config_state(config: dict[str, Any]) -> dict[str, Any]:
    """The capability fields of one config snapshot; a field the config does not declare reads not recorded."""
    control = config.get("control") or {}
    steps = (control.get("budget") or {}).get("steps") or {}
    synthesis = control.get("synthesis") or {}
    health = control.get("health") or {}
    bounds = config.get("hyperparameter_bounds") or {}

    def bound(name: str, key: str) -> Any:
        return (bounds.get(name) or {}).get(key, NOT_RECORDED)

    return {
        **{f"steps_{key}": steps.get(key, NOT_RECORDED) for key in ("min", "default", "max")},
        "synthesis_enabled": synthesis.get("enabled", NOT_RECORDED),
        "synthesis_roles": ";".join(synthesis["roles"]) if "roles" in synthesis else NOT_RECORDED,
        "synthesis_max_requests": synthesis.get("max_requests", NOT_RECORDED),
        "health_min_distance_fraction": health.get("min_distance_fraction", NOT_RECORDED),
        "health_min_pairwise_ratio": health.get("min_pairwise_ratio", NOT_RECORDED),
        "kl_bound_min": bound("kl", "minimum"), "kl_bound_max": bound("kl", "maximum"),
        "lr_bound_min": bound("lr", "minimum"), "lr_bound_max": bound("lr", "maximum"),
        "frozen_levers": ";".join(capability(config)[0]) or "none"}


def _yaml(path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}


def blind_headers(cell: Cell) -> list[str]:
    """The distinct digests of the static instructions-plus-header blob over the cell's propose/stop calls.

    `prompt_manifest[0]` is that blob. A cell has one; the two blind arms must share it at the same seed.
    """
    return sorted({recorded.prompt_manifest[0][:10] for _, recorded in cell.view.of_type(ev.DecisionRecorded)
                   if recorded.kind in ("propose", "stop") and recorded.prompt_manifest})


def capability_row(cell: Cell) -> dict[str, Any]:
    """The capability state of one cell. Fields come from the newest config snapshot, which is the one a
    resumed controller ran with; `capability_changed_on_resume` says when it differs from the first."""
    state, summary = cell.view.state, cell.summary
    newest = cell.view.config()
    snapshots = sorted(cell.path.glob("config.resume_*.yaml"))
    stop = summary["stop"] or {}
    headers = blind_headers(cell)
    return {
        **cell.ident(), "status": summary["status"], "library_status": summary["library_status"],
        **_config_state(newest),
        "capability_changed_on_resume": bool(snapshots) and _config_state(_yaml(cell.path / "config.yaml"))
        != _config_state(newest),
        "candidates_allocated": summary["candidates"]["allocated"],
        "candidate_budget": summary["candidates"]["budget"], "candidates_trained": summary["candidates_trained"],
        "rounds_completed": summary["rounds"]["completed"], "rounds_min": summary["rounds"]["minimum"],
        "rounds_max": summary["rounds"]["maximum"], "resumes": summary["resumes"],
        "config_snapshots_on_resume": len(snapshots),
        "model": summary["model"], "resolved_models": ";".join(summary["resolved_models"]) or NOT_RECORDED,
        "n_resolved_models": len(summary["resolved_models"]),
        "stop_round": stop.get("round", "not stopped"), "stop_source": stop.get("source", "not stopped"),
        "stop_outcome": stop.get("outcome", "not stopped"), "stop_rejections": stop.get("rejections", 0),
        "decisions_propose_stop": sum(d.kind in ("propose", "stop") for d in state.decisions.values()),
        "tokens_total": summary["tokens"]["total"],
        "jobs_submitted": len(cell.view.of_type(ev.JobSubmitted)),
        "jobs_infra_failed": len(cell.view.of_type(ev.JobAttemptFailed)),
        "blind_header_sha": ";".join(headers) if headers else NOT_RECORDED,
        "needs_attention": (summary["needs_attention"] or {}).get("type", ""),
        "terminal_audit": cell.audit}


def sentinel_finite(cell: Cell, name: str = CM_SENTINEL) -> str:
    """CM's O1: whether the PoET-2 sentinel's scores are finite in every candidate pool the arm measured it in,
    from the visible measurements in self_validating and the shadow file in detect_only."""
    if cell.arm == "no_self_validation":
        return "not measured by this arm"
    folder = "" if cell.arm == "self_validating" else "shadow"
    checked, bad = 0, []
    for candidate in cell.view.state.ordered_candidates():
        found = read_json(cell.path / "candidates" / candidate.id / folder / "public_measurements.json")
        entry = ((found or {}).get("measurements") or {}).get(name)
        if candidate.status == "ok" and entry is not None:
            checked += 1
            if not np.isfinite(np.asarray(entry["values"], dtype=float)).all():
                bad.append(candidate.id)
    if bad:
        return "NOT finite in " + ";".join(bad)
    return f"finite in {checked} pools" if checked else "no pool yet"


def operational_check(capability_table: pd.DataFrame) -> dict[str, Any]:
    """O1: cells frozen with a complete audit, the infrastructure-failure share, blind headers per seed."""
    frame = capability_table
    submitted, failed = int(frame.jobs_submitted.sum()), int(frame.jobs_infra_failed.sum())
    pairs = []
    for seed, group in frame.groupby("seed"):
        shas = {row.arm: row.blind_header_sha for row in group.itertuples() if row.arm in BLIND_ARMS}
        if set(shas) == set(BLIND_ARMS):
            same = shas["detect_only"] == shas["no_self_validation"] and ";" not in shas["detect_only"]
            pairs.append({"seed": int(seed), "identical": same and shas["detect_only"] != NOT_RECORDED})
    return {"cells": len(frame), "frozen_with_audit": int(((frame.status == "frozen")
                                                          & (frame.terminal_audit == "complete")).sum()),
            "jobs_submitted": submitted, "jobs_infra_failed": failed,
            "infra_share": failed / submitted if submitted else np.nan, "blind_header_pairs": pairs}


# ----- numeric helpers -------------------------------------------------------------------------

def hamming_clusters(sequences: list[str], fraction: float = CLUSTER_FRACTION,
                     bases: int = CLUSTER_BASES) -> np.ndarray:
    """Cluster label per sequence by single linkage: two sequences are linked when at most `fraction` of the
    first `bases` positions differ (normalized Hamming). Sequences are cut to their shortest common length."""
    length = min(bases, min(len(sequence) for sequence in sequences))
    codes = np.frombuffer("".join(sequence[:length] for sequence in sequences).encode(),
                          dtype=np.uint8).reshape(len(sequences), length)
    mismatches = np.stack([(codes != row).sum(axis=1) for row in codes])
    _, labels = connected_components(mismatches <= fraction * length + 1e-9, directed=False)
    return labels


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """The weighted median. Rows of zero weight are ignored, and a split exactly at half the total weight
    averages the two values it falls between, so equal weights give `np.median`."""
    keep = weights > 0
    values, weights = values[keep], weights[keep]
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cumulative, tolerance = np.cumsum(weights), 1e-9
    half = cumulative[-1] / 2.0
    index = int(np.searchsorted(cumulative, half - tolerance, side="left"))
    if abs(cumulative[index] - half) <= tolerance and index + 1 < len(values):
        return float((values[index] + values[index + 1]) / 2.0)
    return float(values[index])


def cluster_weighted_median(values: np.ndarray, labels: np.ndarray) -> float:
    """The median with every cluster carrying the same total weight (each row weighs 1 / its cluster's size)."""
    return weighted_median(values, 1.0 / np.bincount(labels)[labels])


def cluster_bootstrap_interval(values: np.ndarray, labels: np.ndarray, *, draws: int = BOOTSTRAP_DRAWS,
                               seed: int = BOOTSTRAP_SEED, level: float = 0.95) -> tuple[float, float] | None:
    """Percentile interval of the cluster-weighted median over resamples of whole clusters, with replacement.

    None for a library of `MAX_CLUSTERS_WITHOUT_INTERVAL` or fewer clusters: its interval would describe that
    many effective sequences, not the number of rows.
    """
    n_clusters = int(labels.max()) + 1
    if n_clusters <= MAX_CLUSTERS_WITHOUT_INTERVAL:
        return None
    rng, sizes = np.random.default_rng(seed), np.bincount(labels)
    estimates = np.empty(draws)
    for draw in range(draws):
        copies = np.bincount(rng.integers(0, n_clusters, n_clusters), minlength=n_clusters)
        estimates[draw] = weighted_median(values, copies[labels] / sizes[labels])
    low, high = np.percentile(estimates, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(low), float(high)


# ----- the delivered library against the initial policy's pool ---------------------------------

def _baseline_rows(cell: Cell) -> tuple[dict[str, list[str]], dict[str, list[float]] | None] | None:
    """Each condition's legal baseline sequences and unique k-mer counts, as `Engine._baseline_rows` builds them."""
    directory = cell.path / "baseline"
    pool = read_json(directory / "pool.json")
    measured = (read_json(directory / "public_measurements.json") or {}).get("measurements") or {}
    if not pool or "legal" not in measured:
        return None
    legal = [bool(value) for value in measured["legal"]["values"]]
    kmers = (measured.get("unique_kmer_count") or {}).get("values")
    sequences: dict[str, list[str]] = {condition: [] for condition in cell.conditions}
    counts: dict[str, list[float]] = {condition: [] for condition in cell.conditions}
    for index, (sequence, condition) in enumerate(zip(pool["sequences"], pool["conditions"])):
        if legal[index] and condition in sequences:
            sequences[condition].append(sequence)
            if kmers is not None:
                counts[condition].append(float(kmers[index]))
    return sequences, None if kmers is None else counts


def health_reference(cell: Cell) -> tuple[dict[str, ConditionReference] | None, str]:
    """The reference the delivered library is compared with: the one the engine recorded, else the one the
    same `baseline_reference` builds from the cell's baseline pool."""
    recorded = (read_json(cell.path / "delivery" / "delivery_artifact.json") or {}).get("health") or {}
    if recorded.get("reference"):
        return ({condition: ConditionReference.model_validate(item)
                 for condition, item in recorded["reference"].items()}, "recorded")
    rows = _baseline_rows(cell)
    if rows is None:
        return None, "no baseline pool"
    try:
        return baseline_reference(*rows), "rebuilt from baseline pool"
    except ValueError as exc:
        return None, f"not computable: {str(exc)[:100]}"


def delivered_health(cell: Cell) -> dict[str, dict[str, Any]]:
    """Per condition, the delivered library's `library_health` against the initial pool's reference.

    Computed from the frozen rows with `recital.control.health`, so a cell that predates the recorded
    health block gets the same definition; `pairwise_ratio_recorded` is the engine's own value when it has one.
    """
    library = cell.library
    if library is None:
        return {}
    reference, source = health_reference(cell)
    recorded = ((read_json(cell.path / "delivery" / "delivery_artifact.json") or {}).get("health") or {}
                ).get("by_condition") or {}
    counts = delivery_measurement(cell, "unique_kmer_count", library.row_id.tolist())
    out = {}
    for condition in cell.conditions:
        rows = np.flatnonzero((library.condition == condition).to_numpy())
        entry = {"n": len(rows), "reference": source, "pairwise_ratio": np.nan, "unique_kmer_ratio": np.nan,
                 "pairwise_ratio_recorded": ((recorded.get(condition) or {}).get("library") or {}).get(
                     "pairwise_ratio", NOT_RECORDED)}
        if reference is not None and len(rows) >= 2:
            found = library_health(library.sequence.iloc[rows].tolist(), reference[condition],
                                   None if counts is None else counts[rows].tolist())
            entry["pairwise_ratio"] = found["pairwise_ratio"]
            if found["unique_kmer_ratio"] is not None:
                entry["unique_kmer_ratio"] = found["unique_kmer_ratio"]
        out[condition] = entry
    return out


def _health_columns(cell: Cell, condition: str) -> dict[str, Any]:
    found = cell.health.get(condition) or {}
    return {"pairwise_ratio": found.get("pairwise_ratio", np.nan),
            "unique_kmer_ratio": found.get("unique_kmer_ratio", np.nan),
            "pairwise_ratio_recorded": found.get("pairwise_ratio_recorded", NOT_RECORDED)}


# ----- the estimands (E1) ----------------------------------------------------------------------

def cre_rows(cell: Cell) -> list[dict[str, Any]]:
    """Per condition, the blind ChromBPNet gap of the delivered rows, from the audit's own files.

    The gap is `panel_gap` on the manifest's declared panel, as `audit_measure` computes it, and its median
    is checked against `summary.json`. `cal` subtracts the per-cell-line calibration offset.
    """
    audit = cell.path / "terminal_audit"
    manifest = read_json(audit / "manifest.json")
    source = None if manifest is None else manifest["report_definition"]["primary_source"]
    scores = None if source is None else read_json(audit / f"{source}.json")
    if not cell.audit_complete or manifest is None or scores is None:
        return [{**cell.ident(), "condition": condition, "audit": cell.audit, **_health_columns(cell, condition)}
                for condition in cell.conditions]
    if scores["row_ids"] != manifest["row_ids"]:
        raise ValueError(f"{cell.name}: the audit scores belong to different rows than its manifest")
    definition = manifest["report_definition"]
    conditions = np.asarray(manifest["conditions"])
    gap = panel_gap(np.asarray(scores["values"], dtype=float), list(conditions),
                    oracle_columns=definition["oracle_columns"],
                    off_target_cells_by_target=definition["off_target_cells_by_target"], missing="nan").gap
    rows = []
    for condition in cell.conditions:
        chosen = np.flatnonzero(conditions == condition)
        raw = gap[chosen]
        calibrated = raw - DELTA[condition]
        labels = hamming_clusters([manifest["sequences"][i] for i in chosen])
        n_clusters = int(labels.max()) + 1
        interval = cluster_bootstrap_interval(calibrated, labels)
        reported = cell.audit_summary["primary"]["per_condition"][condition]["value"]
        raw_median = float(np.median(raw))
        rows.append({
            **cell.ident(), "condition": condition, "audit": cell.audit, "n_rows": len(chosen),
            "raw_median": raw_median, "summary_raw_median": _nan(reported),
            "matches_summary": reported is not None and bool(np.isclose(raw_median, reported, rtol=0, atol=1e-9)),
            "delta": DELTA[condition], "cal_median": raw_median - DELTA[condition], "n_clusters": n_clusters,
            "cluster_weighted_cal_median": cluster_weighted_median(calibrated, labels),
            "ci95_low": np.nan if interval is None else interval[0],
            "ci95_high": np.nan if interval is None else interval[1],
            "interval": "reported" if interval is not None else
            f"none: {n_clusters} cluster(s), at most {MAX_CLUSTERS_WITHOUT_INTERVAL}",
            **_health_columns(cell, condition)})
    return rows


def loopseq_rows(cell: Cell) -> list[dict[str, Any]]:
    """Per condition, the mean in-band fraction (primary) and the median signed margin (secondary)."""
    summary = cell.audit_summary
    rows = []
    for condition in cell.conditions:
        row = {**cell.ident(), "condition": condition, "audit": cell.audit, **_health_columns(cell, condition)}
        if summary is not None and cell.audit_complete:
            primary = summary["primary"]["per_condition"].get(condition) or {}
            secondary = (((summary.get("secondary") or {}).get("primary") or {}).get("per_condition") or {}
                         ).get(condition) or {}
            row.update(n_rows=primary.get("count"), primary_in_band=_nan(primary.get("value")),
                       secondary_median_margin=_nan(secondary.get("value")))
        rows.append(row)
    return rows


def _distance_label(low: float, high: float) -> str:
    return f"{low:g}-" + ("inf" if high == np.inf else f"{high:g}")


def cm_rows(cell: Cell) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The CM condition's two audit medians, and the delivered library's natural-distance bins with the
    audit endpoints inside each bin."""
    audit, condition = cell.path / "terminal_audit", cell.conditions[0]
    row = {**cell.ident(), "condition": condition, "audit": cell.audit, **_health_columns(cell, condition)}
    library = cell.library
    distance = None if library is None else delivery_measurement(cell, "natural_distance", library.row_id.tolist())
    if distance is not None:
        row.update(natural_distance_median=float(np.median(distance)), natural_distance_min=float(distance.min()),
                   natural_distance_max=float(distance.max()))
    bins: list[dict[str, Any]] = []
    if cell.audit_summary is None or not cell.audit_complete:
        return [row], bins
    definition = cell.audit_summary["definition"]
    names = {"primary": definition["primary_source"], "secondary": definition.get("secondary_source", CM_SECONDARY)}
    row.update(n_rows=cell.audit_summary["count"], primary_source=names["primary"],
               secondary_source=names["secondary"])
    scored: dict[str, np.ndarray | None] = {}
    for key, name in names.items():
        found = ((cell.audit_summary["sources"][name]["per_condition"].get(condition) or {}).get("value"))
        row[f"{key}_median"] = _nan(found)
        values = read_json(audit / f"{name}.json")
        scored[name] = np.asarray(values["values"], dtype=float) if values else None
    if distance is not None:
        position = {row_id: i for i, row_id in enumerate(read_json(audit / "manifest.json")["row_ids"])}
        aligned = [position[row_id] for row_id in library.row_id]
        for low, high in DISTANCE_BINS:
            inside = (distance >= low) & (distance < high)
            entry = {**cell.ident(), "distance_bin": _distance_label(low, high), "n": int(inside.sum()),
                     "share": float(inside.mean())}
            for name, values in scored.items():
                entry[f"{name}_median"] = (float(np.median(values[aligned][inside]))
                                           if values is not None and inside.any() else np.nan)
            bins.append(entry)
    return [row], bins


def worst_per_cell(by_condition: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """One row per cell: the minimum over its conditions of each named column, NaN unless every one is finite."""
    rows = []
    for _, group in by_condition.groupby("cell", sort=False):
        first = group.iloc[0]
        row = {"cell": first.cell, "arm": first.arm, "seed": first.seed, "audit": first.audit}
        for output, source in columns.items():
            values = group[source].astype(float)
            row[output] = float(values.min()) if values.notna().all() else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    if "min_n_clusters" in table:
        table["min_n_clusters"] = table["min_n_clusters"].astype("Int64")
    return table


# ----- matched-seed contrasts (E2) -------------------------------------------------------------

def contrast_tables(values: pd.DataFrame, estimands: list[str], expected: int, campaign: str
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """E2 per estimand. `values` has one row per cell with `arm`, `seed` and one column per estimand.

    The preregistered rule: self_validating minus detect_only is positive in every one of the `expected`
    seeds and its mean exceeds the largest absolute detect_only minus no_self_validation difference. One seed
    supports no contrast and none is written; fewer matched seeds than expected leave the rule unevaluated.
    """
    per_seed, summary = [], []
    for estimand in estimands:
        table = {(row.arm, int(row.seed)): float(row[estimand]) for _, row in values.iterrows()
                 if np.isfinite(row[estimand])}
        diffs = {(plus, minus): {seed: table[(plus, seed)] - table[(minus, seed)]
                                 for seed in sorted(s for arm, s in table if arm == plus and (minus, s) in table)}
                 for plus, minus in CONTRASTS}
        primary, floor = diffs[CONTRASTS[0]], diffs[NOISE_FLOOR]
        row = {"campaign": campaign, "estimand": estimand, "preregistered": estimand not in SUPPLEMENTARY,
               "expected_seeds": expected,
               "matched_seeds_sv_minus_det": len(primary), "matched_seeds_noise_floor": len(floor), "holds": None}
        seeds = {seed for _, seed in table}
        if not table:
            row["reading"] = f"{NO_AUDIT}: nothing to contrast"
        elif len(seeds) <= 1:
            row["reading"] = "not supported: one seed supports no contrast"
        else:
            for (plus, minus), found in diffs.items():
                per_seed += [{"campaign": campaign, "estimand": estimand, "contrast": f"{plus} - {minus}",
                              "role": "noise floor" if (plus, minus) == NOISE_FLOOR else "contrast",
                              "seed": seed, "difference": diff} for seed, diff in found.items()]
            row.update(
                mean_sv_minus_det=float(np.mean(list(primary.values()))) if primary else np.nan,
                n_positive=sum(diff > 0 for diff in primary.values()),
                mean_sv_minus_nsv=(float(np.mean(list(diffs[CONTRASTS[1]].values())))
                                   if diffs[CONTRASTS[1]] else np.nan),
                noise_floor_max_abs=float(max(abs(diff) for diff in floor.values())) if floor else np.nan)
            if len(primary) < expected or len(floor) < expected:
                row["reading"] = (f"incomplete: {len(primary)} (sv - det) and {len(floor)} (noise floor) of "
                                  f"{expected} matched seeds; rule not evaluated")
            else:
                row["holds"] = bool(row["n_positive"] == len(primary)
                                    and row["mean_sv_minus_det"] > row["noise_floor_max_abs"])
                row["reading"] = ("pass rule holds" if row["holds"] else
                                  f"not separated from decision variance at n={expected}; no claim of a "
                                  "self-validation benefit")
        summary.append(row)
    return pd.DataFrame(per_seed, columns=["campaign", "estimand", "contrast", "role", "seed", "difference"]), \
        pd.DataFrame(summary)


def arm_summary(by_cell: pd.DataFrame, by_condition: pd.DataFrame, task: str, campaign: str) -> pd.DataFrame:
    """Per arm, over the seeds that have an audit: the mean and median of each estimand and, for CRE, each cell
    line's median over seeds of the per-library calibrated median. Descriptive; it tests nothing."""
    rows = []
    for arm in ARMS:
        cells = by_cell[by_cell.arm == arm]
        row = {"campaign": campaign, "arm": arm, "cells": len(cells),
               "audited": int(cells[CONTRAST_ESTIMANDS[task][0]].notna().sum())}
        for estimand in CONTRAST_ESTIMANDS[task]:
            found = cells[estimand].dropna().astype(float)
            row[f"{estimand}_mean"] = found.mean() if len(found) else np.nan
            row[f"{estimand}_median"] = found.median() if len(found) else np.nan
        if task == "cre":
            for condition in ("K562", "HepG2"):
                found = by_condition[(by_condition.arm == arm) & (by_condition.condition == condition)]
                row[f"{condition}_cal_median_over_seeds"] = found.cal_median.dropna().astype(float).median()
        rows.append(row)
    return pd.DataFrame(rows)


# ----- descriptives (D1 / E4) ------------------------------------------------------------------

def keyword_row(cell: Cell) -> dict[str, Any]:
    """How many of the cell's propose/stop decisions, refused ones included, mention a keyword."""
    decisions = [d for d in cell.view.state.decisions.values() if d.kind in ("propose", "stop")]
    hits = 0
    for decision in decisions:
        response = cell.view.decision_response(decision.id)
        text = " ".join([response.get("rationale") or "",
                         *[item.get("hypothesis") or "" for item in response.get("candidates") or []]])
        hits += bool(KEYWORDS.search(text))
    return {**cell.ident(), "decisions": len(decisions), "keyword_hits": hits}


def flag_history(cell: Cell) -> list[dict[str, Any]]:
    """Every flag of the cell's candidates: empty for the arm that computes no table, and for a table frozen
    before the flag existed."""
    _, rows = selfval_table(cell.view)
    keys = ("candidate", "round", "condition", "sentinel", "reward_source", "status", "effect", "lower", "upper",
            "reward_lower", "n")
    return [{**cell.ident(), "visible_to_agent": cell.arm == "self_validating", **{key: row.get(key) for key in keys}}
            for row in rows if row["kind"] == "flag"]


def _flag_counts(cell: Cell, flags: list[dict[str, Any]]) -> dict[str, Any]:
    if cell.arm == "no_self_validation":
        return {f"flags_{status}": "not computed" for status in ("flagged", "ok", "uncalibrated")}
    if not flags:
        return {f"flags_{status}": NOT_RECORDED for status in ("flagged", "ok", "uncalibrated")}
    return {f"flags_{status}": sum(flag["status"] == status for flag in flags)
            for status in ("flagged", "ok", "uncalibrated")}


def descriptives_row(cell: Cell, flags: list[dict[str, Any]]) -> dict[str, Any]:
    """D1 and E4 for one cell. `candidates_failed_library_health` reads not recorded for a cell whose
    candidates carry no `library_health` requirement."""
    view, state = cell.view, cell.view.state
    candidates = state.ordered_candidates()
    trained = [c for c in candidates if c.status == "ok"]
    outcomes = library_health_outcomes(cell)
    recorded = any(has for has, _ in outcomes.values())
    failed = sorted(cid for cid, (_, fails) in outcomes.items() if fails)
    decisions = [d for d in state.decisions.values() if d.kind in ("propose", "stop")]
    invalid = [d for d in decisions if d.status == "invalid"]
    codes: dict[str, int] = {}
    for decision in invalid:
        for code in {problem.code for problem in decision.problems}:
            codes[code] = codes.get(code, 0) + 1
    # The agent's proposals: every allocated candidate but the anchor, which is the declared initial recipe.
    proposals = [c for c in candidates if not c.anchor and c.status != "forfeited"]
    later = [c for c in proposals if c.round >= 2]
    recipes = {c.id: view.spec(c.id)["recipe"] for c in proposals if c.recipe_sha}

    def chosen(name: str) -> list[float]:
        return [float(recipe["hyperparameters"][name]) for recipe in recipes.values()
                if name in (recipe.get("hyperparameters") or {})]

    anchor = next((c for c in candidates if c.anchor and c.recipe_sha), None)
    initial = None if anchor is None else set(view.spec(anchor.id)["recipe"]["enabled_oracles"])
    promotions: dict[str, int] = {}
    for recipe in recipes.values():
        for source in set(recipe.get("enabled_oracles") or []) - (initial or set()):
            promotions[source] = promotions.get(source, 0) + 1
    kl, lr = chosen("kl"), chosen("lr")
    stop = cell.summary["stop"] or {}
    from_initial = sum(c.start == "from_initial" for c in proposals)
    row = {
        **cell.ident(), "status": state.finished_status or "running",
        "candidates_allocated": cell.summary["candidates"]["allocated"], "candidates_trained": len(trained),
        "candidates_failed_library_health": len(failed) if recorded else NOT_RECORDED,
        "failed_library_health_ids": ";".join(failed) if recorded else NOT_RECORDED,
        "invalid_decisions": len(invalid), "decisions_propose_stop": len(decisions),
        "invalid_problem_codes": ";".join(f"{code}:{n}" for code, n in sorted(codes.items())) or "none",
        "proposals": len(proposals), "from_initial": from_initial,
        "from_initial_share": from_initial / len(proposals) if proposals else np.nan,
        "proposals_after_round1": len(later),
        "from_initial_after_round1": sum(c.start == "from_initial" for c in later),
        "kl_values": ";".join(f"{value:g}" for value in kl) or NOT_RECORDED,
        "lr_values": ";".join(f"{value:g}" for value in lr) or NOT_RECORDED,
        "kl_min": min(kl) if kl else np.nan, "kl_max": max(kl) if kl else np.nan,
        "source_promotions": ";".join(f"{source}:{n}" for source, n in sorted(promotions.items()))
        or ("none" if initial is not None else NOT_RECORDED),
        "coder_requests": len(state.components),
        "coder_published": sum(c.status == "published" for c in state.components.values()),
        "stop_round": stop.get("round", "not stopped"), "stop_source": stop.get("source", "not stopped"),
        **_flag_counts(cell, flags), "health_recorded": recorded}
    for condition in cell.conditions:
        row[f"delivered_pairwise_ratio__{condition}"] = _health_columns(cell, condition)["pairwise_ratio"]
        row[f"delivered_unique_kmer_ratio__{condition}"] = _health_columns(cell, condition)["unique_kmer_ratio"]
    return row


def keyword_summary(keywords: pd.DataFrame) -> pd.DataFrame:
    """Per-cell rows, then per-arm totals: the share of decisions mentioning a keyword."""
    per_arm = keywords.groupby("arm", sort=False)[["decisions", "keyword_hits"]].sum().reset_index()
    out = pd.concat([keywords.assign(scope="cell"), per_arm.assign(scope="arm")], ignore_index=True)
    out["share"] = out.keyword_hits / out.decisions.where(out.decisions > 0)
    return out[["scope", "arm", "seed", "cell", "decisions", "keyword_hits", "share"]]


# ----- one campaign ----------------------------------------------------------------------------

@dataclass
class Analysis:
    root: Path
    task: str
    expected_seeds: int
    skipped: list[dict[str, str]]
    capability: pd.DataFrame
    by_condition: pd.DataFrame
    by_cell: pd.DataFrame
    arms: pd.DataFrame
    contrasts: pd.DataFrame
    contrast_summary: pd.DataFrame
    descriptives: pd.DataFrame
    flags: pd.DataFrame
    keywords: pd.DataFrame
    bins: pd.DataFrame
    operational: dict[str, Any]


def analyse(root: Path, task: str, expected_seeds: int) -> Analysis:
    cells, skipped = load_cells(root, task)
    if not cells:
        raise SystemExit(f"no {task} control cell under {root}"
                         + "".join(f"; {item['cell']}: {item['reason']}" for item in skipped))
    capability_table = pd.DataFrame([capability_row(cell) for cell in cells])
    if task == "cm":
        capability_table["poet2_scores"] = [sentinel_finite(cell) for cell in cells]
    rows: list[dict[str, Any]] = []
    bins: list[dict[str, Any]] = []
    for cell in cells:
        if task == "cre":
            rows += cre_rows(cell)
        elif task == "loopseq":
            rows += loopseq_rows(cell)
        else:
            cell_rows, cell_bins = cm_rows(cell)
            rows, bins = rows + cell_rows, bins + cell_bins
    by_condition = pd.DataFrame(rows).reindex(columns=["cell", "arm", "seed", *BY_CONDITION_COLUMNS[task]])
    by_cell = worst_per_cell(by_condition, BY_CELL_COLUMNS[task])
    contrasts, contrast_summary = contrast_tables(by_cell, CONTRAST_ESTIMANDS[task], expected_seeds, root.name)
    flags = {cell.name: flag_history(cell) for cell in cells}
    return Analysis(
        root=root, task=task, expected_seeds=expected_seeds, skipped=skipped, capability=capability_table,
        by_condition=by_condition, by_cell=by_cell, arms=arm_summary(by_cell, by_condition, task, root.name),
        contrasts=contrasts, contrast_summary=contrast_summary,
        descriptives=pd.DataFrame([descriptives_row(cell, flags[cell.name]) for cell in cells]),
        flags=pd.DataFrame([flag for found in flags.values() for flag in found]),
        keywords=pd.DataFrame([keyword_row(cell) for cell in cells]), bins=pd.DataFrame(bins),
        operational=operational_check(capability_table))


# ----- round 1 beside this campaign ------------------------------------------------------------

def round1_comparison(this: Analysis, round1: Analysis) -> pd.DataFrame:
    """One row per cell of each campaign, by arm and seed; the two campaigns are never pooled."""
    frames = []
    for label, analysis in ((f"round1 {round1.root.name}", round1), (f"this {this.root.name}", this)):
        frame = analysis.by_cell[["arm", "seed", "cell", "worst_cal", "worst_cal_cluster_weighted",
                                  "min_n_clusters", "min_pairwise_ratio"]].copy()
        for condition in ("K562", "HepG2"):
            subset = analysis.by_condition[analysis.by_condition.condition == condition].set_index("cell")
            frame[f"{condition}_cal"] = frame.cell.map(subset.cal_median)
            frame[f"{condition}_clusters"] = frame.cell.map(subset.n_clusters).astype("Int64")
        frames.append(frame.assign(campaign=label))
    table = pd.concat(frames, ignore_index=True)
    table["_arm"] = table.arm.map({arm: i for i, arm in enumerate(ARMS)})
    table = table.sort_values(["_arm", "seed", "campaign"], kind="stable")
    return table[["campaign", "arm", "seed", "cell", "K562_cal", "HepG2_cal", "worst_cal", "K562_clusters",
                  "HepG2_clusters", "worst_cal_cluster_weighted", "min_pairwise_ratio"]].reset_index(drop=True)


# ----- writing ---------------------------------------------------------------------------------

def _fmt(value: Any, digits: int) -> str:
    if value is None or value is pd.NA or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    return str(value)


def md_table(frame: pd.DataFrame, digits: int = 2) -> str:
    if frame.empty:
        return "(none)\n"
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(_fmt(value, digits) for value in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule, *body]) + "\n"


def _contrast_wide(contrasts: pd.DataFrame) -> pd.DataFrame:
    """The per-seed differences as one row per estimand and contrast, one column per seed."""
    if contrasts.empty:
        return contrasts
    wide = contrasts.pivot_table(index=["estimand", "contrast", "role"], columns="seed", values="difference",
                                 sort=False)
    wide.columns = [f"seed {seed}" for seed in wide.columns]
    return wide.reset_index()


def write_summary(path: Path, this: Analysis, round1: Analysis | None, comparison: pd.DataFrame | None) -> None:
    cap, o1 = this.capability, this.operational
    pinned = [f"{row.cell}: {row.frozen_levers}" for row in cap.itertuples() if row.frozen_levers != "none"]
    changed = list(cap[cap.capability_changed_on_resume].cell)
    arm_keywords = keyword_summary(this.keywords)
    arm_keywords = arm_keywords[arm_keywords.scope == "arm"].drop(columns=["scope", "seed", "cell"])
    lines = [
        f"# {this.task} control campaign: {this.root.name}", "",
        f"Source `{this.root}`, read only. In silico, computational estimands; not wet-lab validation. "
        "Campaigns are never pooled. " + CAVEATS[this.task], "",
        "## Capability state (read before citing any cell)", "",
        md_table(cap[["cell", "status", "steps_min", "steps_default", "steps_max", "synthesis_enabled",
                      "synthesis_max_requests", "health_min_pairwise_ratio", "candidates_allocated",
                      "candidates_trained", "rounds_completed", "resumes", "resolved_models", "stop_round",
                      "stop_source", "terminal_audit"]]),
        "Training levers pinned to one value: " + ("; ".join(pinned) or "none") + ".",
        *(["Capability changed on resume (newest snapshot shown): " + ", ".join(changed) + "."] if changed else []),
        "", "## O1 operational", "",
        f"- {o1['frozen_with_audit']} of {o1['cells']} cells frozen with a complete terminal audit",
        f"- {o1['jobs_infra_failed']} of {o1['jobs_submitted']} submitted jobs ended in an infrastructure failure "
        f"({_fmt(100 * o1['infra_share'], 1)} percent; limit 10 percent)",
        "- blind static headers identical at the same seed: "
        + (", ".join(f"seed {p['seed']} {'yes' if p['identical'] else 'NO'}" for p in o1["blind_header_pairs"])
           or "no seed has both blind arms yet"),
        *(["- PoET-2 scores: " + "; ".join(f"{row.cell} {row.poet2_scores}" for row in cap.itertuples())]
          if "poet2_scores" in cap else []),
        *(["- skipped: " + "; ".join(f"{s['cell']} ({s['reason']})" for s in this.skipped)] if this.skipped else []),
        "", "## Estimand (E1)", ""]
    shown = {"cre": ["cell", "condition", "audit", "raw_median", "cal_median", "n_clusters",
                     "cluster_weighted_cal_median", "ci95_low", "ci95_high", "pairwise_ratio"],
             "loopseq": ["cell", "condition", "audit", "primary_in_band", "secondary_median_margin",
                         "pairwise_ratio"],
             "cm": ["cell", "audit", "primary_median", "secondary_median", "natural_distance_median",
                    "pairwise_ratio"]}[this.task]
    if this.task == "cre":
        lines += ["`cal = raw - delta[cell line]` on the delivered rows of each condition; clusters at 10 percent "
                  "Hamming; an interval needs more than 4 clusters.", ""]
    lines += [md_table(this.by_condition[shown], 3), md_table(this.by_cell.drop(columns=["audit"]), 3),
              "Per arm over seeds (descriptive; tests nothing):", "", md_table(this.arms.drop(columns=["campaign"]), 2)]
    if len(this.bins):
        lines += ["Natural-distance bins of the delivered library:", "", md_table(this.bins, 3)]
    lines += ["## Matched-seed contrasts (E2)", "", md_table(this.contrast_summary, 3)]
    if len(this.contrasts):
        lines += [md_table(_contrast_wide(this.contrasts), 3)]
    lines += ["## Descriptives (D1/E4)", "",
              md_table(this.descriptives[["cell", "candidates_trained", "candidates_failed_library_health",
                                          "invalid_decisions", "invalid_problem_codes", "from_initial", "proposals",
                                          "source_promotions", "coder_requests", "flags_flagged", "flags_ok",
                                          "flags_uncalibrated"]], 3),
              "Propose/stop decisions mentioning range, artifact, exploit, extrapolation or overoptimization "
              "(round 1: 1/21, 0/20, 0/22):", "", md_table(arm_keywords, 3)]
    if comparison is not None and round1 is not None:
        lines += ["## Round 1 beside this campaign (not pooled)", "", md_table(comparison, 2),
                  "Round 1, per arm:", "", md_table(round1.arms, 2),
                  "Round-1 contrasts (E2):", "", md_table(round1.contrast_summary, 3)]
    path.write_text("\n".join(lines))


def write_tables(out: Path, this: Analysis, round1: Analysis | None) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    files = {"capability.csv": this.capability, "estimand_by_condition.csv": this.by_condition,
             "estimand_by_cell.csv": this.by_cell, "arm_summary.csv": this.arms, "contrasts.csv": this.contrasts,
             "contrast_summary.csv": this.contrast_summary, "descriptives.csv": this.descriptives,
             "flag_history.csv": this.flags, "keyword_share.csv": keyword_summary(this.keywords)}
    if len(this.bins):
        files["cm_distance_bins.csv"] = this.bins
    comparison = None
    if round1 is not None:
        comparison = round1_comparison(this, round1)
        files["round1_comparison.csv"] = comparison
        files["round1_arm_summary.csv"] = round1.arms
        files["round1_contrasts.csv"] = round1.contrast_summary
    for name, frame in files.items():
        frame.to_csv(out / name, index=False)
    write_summary(out / "summary.md", this, round1, comparison)
    return [*files, "summary.md"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="campaign output root, or one cell")
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--out", type=Path, required=True, help="directory for the tables; must be outside ROOT")
    parser.add_argument("--round1-root", type=Path, default=None,
                        help="cre only: the round-1 campaign to show beside ROOT")
    parser.add_argument("--expected-seeds", type=int, default=None,
                        help="seeds the pass rule needs (default: cre 3, loopseq 3, cm 1)")
    args = parser.parse_args(argv)
    if args.round1_root is not None and args.task != "cre":
        parser.error("--round1-root is the CRE round-1 comparison")
    root, out = args.root.resolve(), args.out.resolve()
    for guarded in filter(None, (root, args.round1_root and args.round1_root.resolve())):
        if out == guarded or guarded in out.parents:
            parser.error(f"--out {out} lies inside the campaign {guarded}, which is read only")
    expected = args.expected_seeds or EXPECTED_SEEDS[args.task]
    this = analyse(root, args.task, expected)
    round1 = analyse(args.round1_root.resolve(), "cre", expected) if args.round1_root else None
    written = write_tables(out, this, round1)
    print(f"{this.task} {root.name}: {len(this.capability)} cells; wrote {', '.join(written)} to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
