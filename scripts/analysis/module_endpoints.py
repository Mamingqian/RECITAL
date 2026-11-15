#!/usr/bin/env python3
"""The endpoint table every RECITAL module is judged by, computed from a cell's own event log and files.

    module_endpoints.py [CELL_OR_ROOT ...] [--control CELL_OR_ROOT ...] [--out DIR] [--metric TASK=NAME ...]

With no cell given it reads the nine stored searches (`figures/common/control_cells.py`: `MAIN_ROOTS`, `MAIN_SEEDS`,
arm `no_self_validation`); those are also the default controls. Nothing under a cell is written. The capability
state of every cell (CLAUDE.md: `control.budget.steps`, `control.synthesis`, `control.modules`, `control.levers`,
candidates allocated, resumes, resolved model ids) is printed first, then one row of endpoints per cell, then the
control table per task.

Endpoints (one value per cell, the search is the unit)

  curve        the best qualified library after each round against the cumulative committed updates (`steps` of
               every non-forfeited candidate of that round and the rounds before). The library is one qualified
               candidate per condition; a candidate's value is its ranking value on that condition, taken from the
               declared ranking chain: the best candidate is the largest tuple of the declared criteria
               (lexicographic, `maximize` only), and the value reported is that of the endpoint criterion. The
               endpoint criterion is the last declared one (Loop-seq: the margin that follows a hit rate pinned at
               1.0; CRE and CM declare one), unless `--metric TASK=NAME` names another. The library's value is the
               minimum over conditions. It is 0 until every condition has a qualified candidate.
  aubc         area under the curve over the declared total updates divided by that total: a round's value counts
               from the end of the round, and the last value is held to the end of the budget, so a search that
               stops early keeps its library. The CSV carries `aubc` and `final_best`; the normalization by the controls
               of a contrast (`aubc_norm`) is made in `module_contrast.py`, which has the controls. For chorismate mutase the
               stored searches commit 48 to 55 percent of the declared total (3,712, 4,224 and 3,840 of 7,680), so the held
               tail is 54 to 62 percent of their AUBC and AUBC is mostly the final value there (AUBC over final 0.84 to 0.87);
               Loop-seq and CRE commit all of it. A search with no qualified library has aubc 0.
  aubc_committed   the same area over the updates the search committed instead of the declared total; a secondary,
               non-gating column, equal to `aubc` where the whole total is committed.
  final_best   the curve's last value; `failed` is true when no library qualified (also while a cell is running and has none yet).
               A library is the best qualified candidate per condition by the largest tuple of the declared ranking values; the
               controller's tolerance rule and its earlier-candidate tie-break (`selection.rank_condition`) are not applied
               (on 28 other CRE cells 5 of 176 pools differ, by at most 0.3 percent). A qualified library with a negative
               endpoint value would count below no library (0); no stored value is negative.
  delivered    the endpoint value of the library the run delivered, when it delivered one.
  updates_unqualified_share    steps of trained candidates that end unqualified in at least one condition, over the
               steps of all non-forfeited candidates; `updates_failed_share` is the steps of candidates whose job
               did not finish ok.
  first_qualified_round, first_qualified_updates     where the curve first leaves 0.
  edit_repeat_rate    over the candidates of round 2 and later, the share of their edit classes
               (`recital.control.ext.edit_class`: lever, direction and detail, e.g. `penalty:added:repeated_kmer`,
               counted once per candidate that carries it) that a candidate of an earlier round already carried; the
               ratio of the two sums over the search (`edit_class.repeat_counts`).
  edit_attribution_share   over the candidates that are not the anchor, the share whose recipe differs from its base
               (the parent's recipe, or the anchor's for a restart from the initial policy) in exactly one class.
  amp (docs/reports/amp_task_design_2026-11-04.md, sections 5 and 6)   the endpoint criterion is `amp_selection_score`, the delivered
               30 peptides' mean E. coli activity; `audit_primary` is minus log10 of the median APEX 1.1 E. coli MIC in micromolar, so
               that a higher value is better as for every task (the design's tolerance of 0.1 log10 units is a difference of 0.1);
               `audit_secondary` is NaN; the delivered library's `amp_likelihood_mean`, `nontoxic_mean` and `nonhemolytic_mean` are
               written beside the endpoints as `delivered_*`.
  audit_primary, audit_secondary   the terminal blind result of the delivered library, read from
               `terminal_audit/summary.json` once the audit is complete: CRE the worst condition of the audit gap
               minus the per-cell-line offset `DELTA` (the calibrated MinGap), Loop-seq the worst-group in-band
               fraction and the worst-group median margin, CM the audit_2020_t5 and audit_2020_esm medians.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for _entry in (REPO / "src", REPO / "scripts", REPO):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from analysis.cre_movement_vs_blind import DELTA  # noqa: E402
from common.control_cells import (  # noqa: E402
    MAIN_ARM, MAIN_ROOTS, MAIN_SEEDS, SCRATCH_ROOT, Cell, cell_dirs, load_cell)
from recital.control.ext.edit_class import edit_classes, repeat_counts  # noqa: E402
from recital.control.selection import condition_status, condition_values  # noqa: E402
from recital.success import SuccessSpec  # noqa: E402

OUT_DEFAULT = SCRATCH_ROOT / "outputs" / "streams" / "module_endpoints"
ENDPOINT_COLUMNS = (
    "cell", "campaign", "task", "seed", "arm", "modules", "models", "endpoint_metric", "status", "audited", "failed",
    "first_event_ts", "finished_ts",
    "total_updates", "committed_updates", "rounds", "candidates",
    "final_best", "aubc", "aubc_committed", "delivered", "updates_unqualified_share", "updates_failed_share",
    "first_qualified_round", "first_qualified_updates", "edit_repeat_rate", "edit_attribution_share",
    "audit_primary", "audit_secondary", "tokens_total")
# Where a task's endpoint criterion is named rather than taken as the last declared ranking criterion, and what is read
# beside it from the delivered library's qualification measurements.
ENDPOINT_METRIC = {"amp": "amp_selection_score"}
DELIVERED_MEANS = {"amp": ("amp_likelihood_mean", "nontoxic_mean", "nonhemolytic_mean")}
DELIVERED_COLUMNS = tuple(f"delivered_{name}" for names in DELIVERED_MEANS.values() for name in names)
TASKS = (*MAIN_ROOTS, "amp")
CONTROL_COLUMNS = ("final_best", "aubc", "updates_unqualified_share", "first_qualified_round", "edit_repeat_rate",
                   "edit_attribution_share", "audit_primary", "audit_secondary", "tokens_total")


@dataclass(frozen=True)
class Facts:
    """What the endpoints need of one non-forfeited candidate."""

    id: str
    round: int
    steps: int
    status: str
    anchor: bool
    qualified: Mapping[str, bool] | None  # per condition, None when the candidate has no measured row
    values: Mapping[str, tuple[float | None, ...]] | None  # the declared ranking values per condition
    classes: tuple[str, ...] | None  # edit-class keys against the base, None for the anchor


# ----- pure endpoint arithmetic ----------------------------------------------------------------------------------

def best_library(facts: Sequence[Facts], conditions: Sequence[str], sign: Sequence[float], index: int) -> float | None:
    """The minimum over conditions of the best qualified candidate's endpoint value; None if a condition has none."""
    per_condition = []
    for condition in conditions:
        scored = [tuple(s * v for s, v in zip(sign, item.values[condition]))
                  for item in facts
                  if item.qualified and item.qualified.get(condition) and item.values
                  and all(v is not None for v in item.values[condition])]
        if not scored:
            return None
        per_condition.append(max(scored)[index] * sign[index])
    return min(per_condition)


def anytime_curve(facts: Sequence[Facts], conditions: Sequence[str], sign: Sequence[float], index: int
                  ) -> list[tuple[int, int, float | None]]:
    """`(round, cumulative committed updates, library value)` at the end of every round that has candidates."""
    curve = []
    for round_ in sorted({item.round for item in facts}):
        through = [item for item in facts if item.round <= round_]
        curve.append((round_, sum(item.steps for item in through), best_library(through, conditions, sign, index)))
    return curve


def area_under(curve: Sequence[tuple[int, int, float | None]], total_updates: int) -> float:
    """The mean of the curve over `[0, total_updates]`; a round's value counts from its end, None counts as 0."""
    area, previous_x, previous_y = 0.0, 0, 0.0
    for _, updates, value in curve:
        area += previous_y * (min(updates, total_updates) - previous_x)
        previous_x, previous_y = min(updates, total_updates), 0.0 if value is None else value
    area += previous_y * (total_updates - previous_x)
    return area / total_updates


def update_shares(facts: Sequence[Facts], conditions: Sequence[str]) -> tuple[float, float]:
    """The shares of committed updates on trained candidates that end unqualified, and on jobs that did not finish."""
    total = sum(item.steps for item in facts)
    unqualified = sum(item.steps for item in facts
                      if item.status == "ok" and not all((item.qualified or {}).get(c) for c in conditions))
    failed = sum(item.steps for item in facts if item.status != "ok")
    return unqualified / total, failed / total


def edit_endpoints(facts: Sequence[Facts]) -> tuple[float, float]:
    """The edit-class repeat rate and the edit-attribution share; NaN when the denominator is empty."""
    seen: set[str] = set()
    repeated = occurrences = 0
    for round_ in sorted({item.round for item in facts}):
        current = [item for item in facts if item.round == round_ and item.classes]
        if round_ > 1:
            more, total = repeat_counts((item.classes for item in current), seen)
            repeated, occurrences = repeated + more, occurrences + total
        for item in current:
            seen.update(item.classes)
    children = [item for item in facts if not item.anchor and item.classes is not None]
    return (repeated / occurrences if occurrences else math.nan,
            sum(1 for item in children if len(item.classes) == 1) / len(children) if children else math.nan)


# ----- reading a cell --------------------------------------------------------------------------------------------

def endpoint_index(success: SuccessSpec, task: str, override: Mapping[str, str]) -> int:
    metrics = [criterion.metric for criterion in success.ranking]
    if not metrics:
        raise SystemExit(f"{task}: the declaration has no ranking criterion")
    if any(criterion.direction != "maximize" for criterion in success.ranking):
        raise SystemExit(f"{task}: a minimized ranking criterion needs a declared sign; only maximize is handled")
    chosen = override.get(task) or ENDPOINT_METRIC.get(task)
    if chosen is not None:
        if chosen not in metrics:
            raise SystemExit(f"{task}: the endpoint criterion {chosen} is not a ranking criterion {metrics}")
        return metrics.index(chosen)
    return len(metrics) - 1


def facts_of(cell: Cell, success: SuccessSpec) -> list[Facts]:
    view = cell.view
    conditions = view.conditions
    candidates = [c for c in view.state.ordered_candidates() if c.status != "forfeited"]
    anchor = next((c for c in candidates if c.anchor and c.recipe_sha), None)
    initial = view.spec(anchor.id)["recipe"] if anchor else None
    recipes = {c.id: view.spec(c.id)["recipe"] for c in candidates if c.recipe_sha}
    facts = []
    for candidate in candidates:
        row = view.ledger_row(candidate.id) if candidate.ledger_row_sha else None
        qualification = (row or {}).get("qualification")
        base = recipes.get(candidate.parent) if candidate.parent else initial
        classes = None if candidate.anchor or base is None else tuple(
            item.key for item in edit_classes(base, recipes[candidate.id]))
        facts.append(Facts(
            id=candidate.id, round=candidate.round, steps=candidate.steps, status=candidate.status,
            anchor=candidate.anchor,
            qualified=None if qualification is None else {
                c: condition_status(qualification, c).qualified for c in conditions},
            values=None if qualification is None else {
                c: condition_values(success, qualification, c) for c in conditions},
            classes=classes))
    return facts


def audit_values(cell: Cell, task: str) -> tuple[float, float]:
    """The terminal blind result of the delivered library (primary, secondary); NaN without a complete audit."""
    if cell.audit is None:
        return math.nan, math.nan
    return audit_summary_values(task, cell.audit)


def audit_summary_values(task: str, audit: Mapping[str, Any]) -> tuple[float, float]:
    """(primary, secondary) of a terminal audit's `summary.json`: the delivered library's, or a candidate's would-be library's."""
    if task == "cre":
        per = audit["primary"]["per_condition"]
        return min(per[c]["value"] - DELTA[c] for c in per), math.nan
    if task == "loopseq":
        return audit["primary"]["worst_condition"], audit["secondary"]["primary"]["worst_condition"]
    if task == "cm":
        sources = audit["sources"]
        return (sources["audit_2020_t5"]["per_condition"]["CM"]["value"],
                sources["audit_2020_esm"]["per_condition"]["CM"]["value"])
    if task == "amp":
        mic = audit["sources"]["apex_ecoli"]["per_condition"]["AMP"]["value"]
        return (-math.log10(mic) if mic is not None and mic > 0 else math.nan), math.nan
    return math.nan, math.nan


def task_of(cell: Cell) -> str:
    name = cell.name.split("_", 1)[0]
    return name if name in TASKS else str(cell.view.header.task)


def finished_ts(cell: Cell) -> str | None:
    """When the first result of the cell exists: the earlier of its first `run_finished` event and its terminal audit file; None while it runs."""
    times = [event.ts for event in cell.view.events if event.type == "run_finished"][:1]
    audit = cell.path / "terminal_audit" / "summary.json"
    if audit.is_file():
        times.append(datetime.fromtimestamp(audit.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
    return min(times) if times else None


def delivered_means(task: str, assignment: Mapping[str, str], ledger_row) -> dict[str, float]:
    """The delivered library's qualification measurements the task names, as `delivered_<name>`: the minimum over the conditions
    of the candidates the library was assembled from (one condition for amp); NaN where a row or a measurement is missing."""
    found: dict[str, float] = {}
    for name in DELIVERED_MEANS.get(task, ()):
        values = []
        for condition, candidate in assignment.items():
            row = ledger_row(candidate) or {}
            entry = ((row.get("qualification") or {}).get("measurements") or {}).get(name, {}).get(condition) or {}
            values.append(entry.get("value"))
        found[f"delivered_{name}"] = math.nan if not values or any(v is None for v in values) else min(values)
    return found


def cell_row(cell: Cell, override: Mapping[str, str]) -> tuple[dict[str, Any], list[tuple[int, int, float | None]]]:
    view, task = cell.view, task_of(cell)
    success = view.success_spec()
    if success is None:
        raise SystemExit(f"{cell.name}: no declaration.json, so no declared ranking")
    conditions = view.conditions
    index = endpoint_index(success, task, override)
    sign = [1.0] * len(success.ranking)
    facts = facts_of(cell, success)
    curve = anytime_curve(facts, conditions, sign, index)
    total = view.header.budgets.total_updates
    final = curve[-1][2] if curve else None
    first = next(((r, u) for r, u, v in curve if v is not None), (None, None))
    unqualified, failed = update_shares(facts, conditions)
    repeat, attribution = edit_endpoints(facts)
    delivered = (cell.summary["delivered_ranking"] or {})
    delivered_value = (min(v[success.ranking[index].metric] for v in delivered.values()) if delivered else math.nan)
    primary, secondary = audit_values(cell, task)
    capability = capability_row(cell)
    row = {
        "cell": cell.name, "campaign": cell.campaign, "task": task, "seed": cell.seed, "arm": cell.arm,
        "modules": capability["modules"], "models": capability["models"],
        "endpoint_metric": success.ranking[index].metric, "audited": cell.audited,
        "status": cell.status, "failed": final is None, "first_event_ts": cell.summary["first_event_ts"],
        "finished_ts": finished_ts(cell),
        "total_updates": total, "committed_updates": view.state.committed_updates,
        "rounds": cell.summary["rounds"]["completed"], "candidates": len(facts),
        "final_best": math.nan if final is None else final, "aubc": area_under(curve, total),
        "aubc_committed": area_under(curve, view.state.committed_updates) if view.state.committed_updates else math.nan,
        "delivered": delivered_value, "updates_unqualified_share": unqualified, "updates_failed_share": failed,
        "first_qualified_round": first[0], "first_qualified_updates": first[1],
        "edit_repeat_rate": repeat, "edit_attribution_share": attribution,
        "audit_primary": primary, "audit_secondary": secondary, "tokens_total": cell.summary["tokens"]["total"]}
    row.update(delivered_means(task, cell.summary.get("assignment") or {}, view.ledger_row))
    return row, curve


def capability_row(cell: Cell) -> dict[str, Any]:
    """The state CLAUDE.md asks to see before a cell is cited, plus the module and lever blocks."""
    base = cell.capability()
    control = cell.view.config().get("control") or {}
    modules = control.get("modules") or {}
    levers = control.get("levers")
    steps = base["budget_steps_min_default_max"]
    return {
        "cell": base["cell"], "campaign": base["campaign"], "status": base["status"], "audited": base["audited"],
        "steps_min_default_max": "/".join(str(s) for s in steps) if steps else "not recorded",
        "synthesis": ("off" if base["synthesis_enabled"] is False else
                      f"{len(base['synthesis_roles'] or [])} roles, max {base['synthesis_max_requests']}"
                      if base["synthesis_enabled"] else "not recorded"),
        "modules": ",".join(sorted(modules)) if modules else "none",
        "levers": ",".join(levers) if levers else "reward,training",
        "candidates_allocated": base["candidates_allocated"], "candidates_trained": base["candidates_trained"],
        "rounds": base["rounds_completed"], "resumes": base["resumes"],
        "models": ",".join(base["resolved_models"]) or "not recorded"}


# ----- tables ----------------------------------------------------------------------------------------------------

def control_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Per task: the number of searches and the mean, minimum and maximum of each control endpoint."""
    out = []
    for task, group in rows.groupby("task", sort=False):
        record: dict[str, Any] = {"task": task, "n": len(group), "failed": int(group["failed"].sum())}
        for name in CONTROL_COLUMNS:
            values = group[name].astype(float)
            record[f"{name}_mean"] = values.mean()
            record[f"{name}_min"] = values.min()
            record[f"{name}_max"] = values.max()
        out.append(record)
    return pd.DataFrame(out)


def stored_cells() -> list[Path]:
    paths = []
    for task, roots in MAIN_ROOTS.items():
        for root in roots:
            for path in cell_dirs(SCRATCH_ROOT / root):
                if path.name.startswith(f"{task}_{MAIN_ARM}_seed") and int(path.name.rsplit("seed", 1)[1]) in MAIN_SEEDS[task]:
                    paths.append(path)
    return paths


def expand(arguments: Sequence[str]) -> list[Path]:
    return [path for argument in arguments for path in cell_dirs(Path(argument))]


def parse_metric(items: Sequence[str]) -> dict[str, str]:
    return dict(item.split("=", 1) for item in items)


def build(paths: Sequence[Path], override: Mapping[str, str]
          ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[tuple[int, int, float | None]]]]:
    capability, rows, curves = [], [], {}
    for path in paths:
        cell = load_cell(path)
        capability.append(capability_row(cell))
        row, curve = cell_row(cell, override)
        rows.append(row)
        curves[cell.name] = curve
    return pd.DataFrame(capability), pd.DataFrame(rows), curves


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cells", nargs="*", help="cell directories or campaign roots (default: the nine stored searches)")
    parser.add_argument("--control", nargs="*", default=None, help="control cells or roots (default: the nine stored searches)")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--metric", action="append", default=[], metavar="TASK=NAME",
                        help="the endpoint criterion of a task, when it is not its last declared ranking criterion")
    args = parser.parse_args(argv)
    override = parse_metric(args.metric)
    paths = expand(args.cells) if args.cells else stored_cells()
    control_paths = expand(args.control) if args.control else stored_cells()

    capability, rows, curves = build(paths, override)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60)
    print("capability state\n" + capability.to_string(index=False))
    control_rows = rows if control_paths == paths else build(control_paths, override)[1]
    controls = control_table(control_rows)
    columns = [*ENDPOINT_COLUMNS, *(name for name in DELIVERED_COLUMNS if name in rows)]
    print("\nendpoints per cell\n" + rows[columns].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\ncontrol table (the control cells' mean, min, max per task)\n"
          + controls.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    mismatched = rows.groupby("task")["total_updates"].nunique()
    for task in mismatched[mismatched > 1].index:
        print(f"warning: {task} cells declare different total updates; aubc is not comparable across them")

    args.out.mkdir(parents=True, exist_ok=True)
    capability.to_csv(args.out / "capability.csv", index=False)
    rows[columns].to_csv(args.out / "module_endpoints.csv", index=False)
    controls.to_csv(args.out / "control_table.csv", index=False)
    (args.out / "anytime_curves.json").write_text(json.dumps(curves, indent=1))
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
