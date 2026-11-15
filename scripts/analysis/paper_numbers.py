#!/usr/bin/env python3
"""Every number the paper quotes from the campaign cells and the baseline files, recomputed on demand.

    PYTHONPATH=src:scripts python scripts/analysis/paper_numbers.py            # or: make paper-numbers
        [--scratch DIR] [--tex manuscript/numbers.tex] [--csv-dir runs/paper_numbers]
        [--facts docs/reports/PAPER_FACTS_20261008.md]

Reads, and never writes into, the six campaign roots under the scratch directory (events/log.jsonl through
`recital.control.export.load_run`, which takes no lock; the ledger rows and recipes in the blob store;
`terminal_audit/summary.json`; `delivery/`) and the baseline files under `outputs/baselines` and
`outputs/calibration`. A cell contributes to a number only when it is finished: the run is `frozen` and its
terminal audit is `complete`. A macro with no finished cell behind it is written as `\\pending{--}`.

Writes
  manuscript/numbers.tex          one `\\def` per quantity (letters only in the names), `\\pending` while unfinished
  runs/paper_numbers/*.csv        cells.csv, cre_audit.csv, loopseq_audit.csv, cm_audit.csv, candidates.csv,
                                  mechanism_cells.csv, mechanism_rounds.csv, cre_baselines_calibrated.csv,
                                  reglm_taco_libraries.csv, dnadiff_matched_budget_curve.csv, loopseq_baselines.csv,
                                  decision_refusals.csv, lineage_updates.csv, loopseq_selected_baselines.csv,
                                  loopseq_replica_checks.csv, macros.csv
  docs/reports/PAPER_FACTS_20261008.md   the same numbers with the file each one comes from

Estimands. CRE: ChromBPNet raw target-minus-off-target MinGap, median of the 128 delivered rows per cell line,
calibrated as cal = raw - delta (delta = +2.9069 for K562, -2.9069 for HepG2). Loop-seq: DNAcycP2 in-band
fraction of the delivered rows per group, and the worst group. CM: audit_2020_t5 median (primary), audit_2020_esm
median (secondary). In silico, computational estimands; not wet-lab validation.

Macro kinds. A value macro is the plain number once a finished cell stands behind it and `\\pending{--}` before. A
provisional macro (the mechanism counts: GateCells, GateRoundMed, SpendCells, OutShare, Start*, HyperOnly, StepsMax,
KlMax) is always computed from every cell of its block, finished or running, and is `\\pending{<value>}` until the
whole block is finished. The facts file defines each one.

The main line is the `no_self_validation` arm, three seeds per task, with the roots and seeds fixed in
`figures/common/control_cells.py` (`MAIN_ROOTS`, `MAIN_SEEDS`): CRE seeds 0-2 of cre_control_round3_20261006,
Loop-seq loopseq_control_20261008, CM cm_control_20261008. The earlier run of Loop-seq (loopseq_control_20261007)
and CM (cm_control_20261007) gets the `Earlier` macros.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable

REPO = Path(__file__).resolve().parents[2]
for _entry in (REPO / "src", REPO / "scripts", REPO / "scripts"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from analysis.aggregate_control_campaign import Cell, capability_row, delivery_measurement  # noqa: E402
from analysis.campaign_status import read_json  # noqa: E402
from analysis.cre_movement_vs_blind import DELTA  # noqa: E402
from common import anchor_audit  # noqa: E402
from common.control_cells import EARLIER_ROOTS as EARLIER_DIRS, MAIN_ROOTS as MAIN_DIRS, MAIN_SEEDS  # noqa: E402
from recital.control import events as ev  # noqa: E402
from recital.control.errors import ControlError  # noqa: E402
from recital.control.export import LOG_PATH, load_run  # noqa: E402

SCRATCH_DEFAULT = Path(os.environ.get("RECITAL_RUNS_ROOT", "runs"))

# (label, task, directory under the scratch root)
ROOTS = (
    ("cre_round3", "cre", "cre_control_round3_20261006"),
    ("loopseq_20261007", "loopseq", "loopseq_control_20261007"),
    ("loopseq_20261008", "loopseq", "loopseq_control_20261008"),
    ("cm_20261007", "cm", "cm_control_20261007"),
    ("cm_20261008", "cm", "cm_control_20261008"),
)
MAIN_ARM = "no_self_validation"
_LABEL = {directory: label for label, _task, directory in ROOTS}
MAIN_ROOTS = {task: tuple(_LABEL[d] for d in dirs) for task, dirs in MAIN_DIRS.items()}
EARLIER_ROOTS = {task: tuple(_LABEL[d] for d in dirs) for task, dirs in EARLIER_DIRS.items()}
LETTERS = "ABCDEF"
# Loop-seq in-band hit rates are fractions of 128 delivered rows (one sequence is 0.8 points) or of 512 baseline sequences,
# so they keep this many decimals; every other percentage follows `pct_text`.
HIT_RATE_DECIMALS = 1
# The replica scores the pools on the CPU, the cells on a GPU; the six main-line and earlier-run cells differ from the stored CNN scores by 2e-06 to 1.6e-04.
CNN_REPLICA_TOLERANCE = 1e-3
CONDITIONS = {"cre": ("K562", "HepG2"), "loopseq": ("low", "mid", "high"), "cm": ("CM",)}
HYPER_KEYS = {"lr", "kl", "group_size", "steps"}

BASELINES = Path("outputs/baselines")
CALIBRATION = Path("outputs/calibration")
PENDING = r"\pending{--}"
MISSING = "n/a"

# The two fixed hand-tuned CRE recipes rerun without ChromBPNet in the reward (three training seeds each): macro tag, directory.
FIXED_RECIPES = BASELINES / "cre_manual_no_cbp_20260913"
# (macro tag, run folder): the fixed hand-tuned GRPO recipes. "Multi" is the reward with Malinois and Agarwal; "Hand" is the same
# recipe with Malinois as the only predictor, the one the RECOMB paper reports.
FIXED_VARIANTS = (("Multi", "multi"), ("Hand", "malinois"))
FIXED_SEEDS = (0, 1, 2)


# ----- small helpers ---------------------------------------------------------------------------------

def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(handle, "w") as stream:
        stream.write(text)
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(handle)
    frame.to_csv(temporary, index=False)
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)


def cal(condition: str, raw: float) -> float:
    return raw - DELTA[condition]


def finite(values: Iterable[Any]) -> list[float]:
    return [float(v) for v in values if v is not None and np.isfinite(v)]


def mean_or_none(values: Iterable[Any]) -> float | None:
    found = finite(values)
    return float(np.mean(found)) if found else None


def f(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return MISSING
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], digits: int = 3) -> list[str]:
    """Rows are dicts; `columns` is [(key, header)]. Values are formatted by `f`."""
    if not rows:
        return ["(no rows)"]
    lines = ["| " + " | ".join(header for _, header in columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(f(row.get(key), digits)).replace("|", "/") for key, _ in columns) + " |")
    return lines


def counter_text(values: Iterable[Any]) -> str:
    counts = Counter(values)
    if not counts:
        return "none"
    try:
        ordered = sorted(counts.items(), key=lambda item: (float(item[0]), 0))
    except (TypeError, ValueError):
        ordered = sorted(counts.items(), key=lambda item: str(item[0]))
    return ", ".join(f"{value}x{count}" for value, count in ordered)


def texnum(text: str) -> str:
    """A formatted number as TeX text: a leading minus sign becomes `$-$`, a negative zero loses its sign, and an integer
    part of four digits or more gets thousands separators written `{,}` (`7{,}000`)."""
    sign = ""
    if text.startswith("-"):
        if float(text) == 0:
            return texnum(text[1:])
        sign, text = "$-$", text[1:]
    whole, dot, fraction = text.partition(".")
    if whole.isdigit() and len(whole) >= 4:
        whole = re.sub(r"(?<=\d)(?=(\d{3})+$)", "{,}", whole)
    return sign + whole + dot + fraction


def compact(value: float, digits: int = 1) -> str:
    """At most `digits` decimals and no trailing zero: 16 -> '16', 6.8 -> '6.8', 3.0 -> '3'."""
    text = f"{value:.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def pct_text(value: float) -> str:
    """A percentage without decimals from 10 up, with at most one decimal (no trailing zero) below 10."""
    return compact(value, 1) if abs(value) < 10 else f"{value:.0f}"


def shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " [cut]"


def pct(numerator: int, denominator: int) -> str:
    return "n/a" if not denominator else f"{numerator}/{denominator} ({100 * numerator / denominator:.1f}%)"


# ----- reading the cells -----------------------------------------------------------------------------

def audit_cre(cell: Cell) -> dict[str, Any]:
    out: dict[str, Any] = {}
    per = (cell.audit_summary or {})["primary"]["per_condition"]
    for condition in CONDITIONS["cre"]:
        raw = per[condition]["value"]
        out[f"raw_{condition}"] = raw
        out[f"cal_{condition}"] = cal(condition, raw)
        out[f"n_{condition}"] = per[condition]["measured_count"]
    out["worst_cal"] = min(out["cal_K562"], out["cal_HepG2"])
    return out


def loopseq_band_hits(cell: Cell, boundaries: tuple[float, float]) -> dict[str, float]:
    """The in-band fraction recomputed from the audit's raw DNAcycP2 values with the baselines' cut points."""
    scores = read_json(cell.path / "terminal_audit" / "dnacycp2.json")
    manifest = read_json(cell.path / "terminal_audit" / "manifest.json")
    if scores is None or manifest is None:
        return {}
    low_mid, mid_high = boundaries
    values, conditions = np.asarray(scores["values"], dtype=float), np.asarray(manifest["conditions"])
    inside = {"low": values <= low_mid, "mid": (values >= low_mid) & (values <= mid_high), "high": values >= mid_high}
    return {group: float(inside[group][conditions == group].mean()) for group in inside}


def audit_loopseq(cell: Cell, boundaries: tuple[float, float]) -> dict[str, Any]:
    summary = cell.audit_summary or {}
    primary = summary["primary"]
    margin = summary["secondary"]["primary"]
    out: dict[str, Any] = {"worst_hit": primary["worst_condition"], "worst_margin": margin["worst_condition"]}
    recomputed = loopseq_band_hits(cell, boundaries)
    for group in CONDITIONS["loopseq"]:
        out[f"hit_{group}"] = primary["per_condition"][group]["value"]
        out[f"margin_{group}"] = margin["per_condition"][group]["value"]
        out[f"n_{group}"] = primary["per_condition"][group]["measured_count"]
    out["matches_baseline_cutpoints"] = bool(recomputed) and all(
        np.isclose(recomputed[group], out[f"hit_{group}"], rtol=0, atol=1e-9) for group in recomputed)
    return out


def audit_cm(cell: Cell) -> dict[str, Any]:
    summary = cell.audit_summary or {}
    sources = summary["sources"]
    out: dict[str, Any] = {
        "primary_source": summary["definition"]["primary_source"],
        "t5_median": sources["audit_2020_t5"]["per_condition"]["CM"]["value"],
        "esm_median": sources["audit_2020_esm"]["per_condition"]["CM"]["value"],
        "n": sources["audit_2020_t5"]["per_condition"]["CM"]["measured_count"]}
    library = cell.library
    if library is not None:
        row_ids = library.row_id.tolist()
        for name in ("natural_distance", "function_2026_esm", "calibrated/function_2026_esm", "cm_novelty",
                     "cm_max", "cm_bm"):
            values = delivery_measurement(cell, name, row_ids)
            if values is not None and np.isfinite(values).all():
                key = name.replace("calibrated/", "cal_")
                out[f"{key}_median"] = float(np.median(values))
                if name == "natural_distance":
                    out["natural_distance_min"] = float(values.min())
                    out["natural_distance_max"] = float(values.max())
                    out["share_distance_above_0p6"] = float((values > 0.6).mean())
    return out


def wall_hours(cell: Cell) -> tuple[float, int]:
    """Training wall hours: the sum of `training.wall_h` over the ledger rows of the candidates that trained."""
    total, missing = 0.0, 0
    for candidate in cell.view.state.ordered_candidates():
        if candidate.status != "ok":
            continue
        row = cell.view.ledger_row(candidate.id) or {}
        hours = (row.get("training") or {}).get("wall_h")
        if hours is None:
            missing += 1
        else:
            total += float(hours)
    return total, missing


def read_cells(scratch: Path, boundaries: tuple[float, float]) -> list[SimpleNamespace]:
    records = []
    for label, task, directory in ROOTS:
        root = scratch / directory
        if not root.is_dir():
            continue
        for path in sorted(p for p in root.iterdir() if p.is_dir() and (p / LOG_PATH).is_file()):
            try:
                cell = Cell(path, load_run(path))
                summary = cell.summary
            except (ControlError, OSError, ValueError, KeyError) as exc:
                records.append(SimpleNamespace(root=label, task=task, dirname=directory, name=path.name,
                                               arm="?", seed=-1, finished=False, status="unreadable",
                                               note=f"{type(exc).__name__}: {str(exc)[:120]}", cell=None))
                continue
            record = SimpleNamespace(root=label, task=task, dirname=directory, name=path.name, arm=cell.arm,
                                     seed=cell.seed, cell=cell, summary=summary, cap=capability_row(cell),
                                     status=summary["status"], note="")
            record.finished = summary["status"] == "frozen" and cell.audit_complete
            record.hours, record.hours_missing = wall_hours(cell)
            record.audit = {}
            if record.finished:
                if task == "cre":
                    record.audit = audit_cre(cell)
                elif task == "loopseq":
                    record.audit = audit_loopseq(cell, boundaries)
                else:
                    record.audit = audit_cm(cell)
            records.append(record)
    return records


def cell_id(record: SimpleNamespace) -> str:
    """`<root label>/<cell directory>`: a cell directory name alone repeats across roots (seed 0 of two campaigns)."""
    return f"{record.root}/{record.name}"


def select(records: list[SimpleNamespace], *, task: str, roots: Iterable[str], arm: str) -> list[SimpleNamespace]:
    keep = [r for r in records if r.task == task and r.root in tuple(roots) and r.arm == arm]
    return sorted(keep, key=lambda r: r.seed)


def finished(records: Iterable[SimpleNamespace]) -> list[SimpleNamespace]:
    return [r for r in records if r.finished]


# ----- macros ----------------------------------------------------------------------------------------

class Macros:
    """Registry of the LaTeX macros: name, TeX value, raw value, what it means and its kind.

    kind `value`: one finished cell (or a baseline file) stands behind it. `aggregate-final` / `aggregate-partial`: a statistic
    over the finished cells of a block, `\\pending{<value>}` while fewer cells are finished than the block expects.
    `provisional` / `provisional-complete`: a descriptive count over every cell of the block that has candidates, finished or
    running, `\\pending{<value>}` until the whole block is finished.
    """

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def _add(self, name: str, tex: str, raw: Any, description: str, kind: str = "value") -> None:
        assert name.isalpha(), name
        assert all(item["name"] != name for item in self.items), f"duplicate macro {name}"
        self.items.append({"name": name, "tex": tex, "raw": raw, "description": description, "kind": kind})

    def _numeric(self, name: str, shown: float | None, text: str | None, description: str, complete: bool | None,
                 final_kind: str, partial_kind: str) -> None:
        if shown is None or not np.isfinite(shown):
            self._add(name, PENDING, None, description, partial_kind if complete is False else final_kind)
        elif complete is False:
            self._add(name, "\\pending{" + str(text) + "}", shown, description, partial_kind)
        else:
            self._add(name, str(text), shown, description, final_kind)

    def provisional(self, name: str, value: float | None, fmt: Callable[[float], str], description: str,
                    complete: bool) -> None:
        """A descriptive macro computed from every cell of its block, finished or not: plain when the whole block is
        finished, `\\pending{<value>}` while any cell is unfinished, `\\pending{--}` with no data."""
        text = None if value is None or not np.isfinite(value) else texnum(fmt(value))
        self._numeric(name, value, text, description, complete, "provisional-complete", "provisional")

    def provisional_text(self, name: str, value: str | None, description: str, complete: bool) -> None:
        if value is None:
            self._add(name, PENDING, None, description, "provisional")
        else:
            self._add(name, value if complete else "\\pending{" + value + "}", value, description,
                      "provisional-complete" if complete else "provisional")

    def number(self, name: str, value: float | None, digits: int, description: str, scale: float = 1.0,
               complete: bool | None = None, fmt: Callable[[float], str] | None = None) -> None:
        """`complete` None: a plain value (pending only when there is no data). True/False: an aggregate over the finished
        cells of a block, final or partial. `fmt` replaces the fixed `digits`."""
        shown = None if value is None or not np.isfinite(value) else value * scale
        text = None if shown is None else texnum(fmt(shown) if fmt else f"{shown:.{digits}f}")
        self._numeric(name, shown, text, description, complete, "value" if complete is None else "aggregate-final",
                      "aggregate-partial")

    def integer(self, name: str, value: int, description: str, complete: bool | None = None) -> None:
        text = texnum(str(int(value)))
        if complete is False:
            self._add(name, "\\pending{" + text + "}", int(value), description, "aggregate-partial")
        else:
            self._add(name, text, int(value), description, "value" if complete is None else "aggregate-final")

    def text(self, name: str, value: str, description: str) -> None:
        self._add(name, value, value, description)


def is_complete(cells: list[SimpleNamespace], expected: int) -> bool:
    """A block is complete when as many of its cells are finished as it expects."""
    return len(finished(cells)) >= expected


def stat_macros(macros: Macros, prefix: str, values: list[float], digits: int, what: str, complete: bool,
                scale: float = 1.0, with_range: bool = True) -> None:
    macros.number(f"{prefix}mean", float(np.mean(values)) if values else None, digits, f"mean of {what}", scale,
                  complete)
    if with_range:
        macros.number(f"{prefix}min", float(np.min(values)) if values else None, digits, f"minimum of {what}", scale,
                      complete)
        macros.number(f"{prefix}max", float(np.max(values)) if values else None, digits, f"maximum of {what}", scale,
                      complete)


def seed_macros(macros: Macros, prefix: str, cells: list[SimpleNamespace], seeds: tuple[int, ...],
                getter: Callable[[SimpleNamespace], float], digits: int, what: str, scale: float = 1.0) -> None:
    by_seed = {r.seed: r for r in cells}
    for letter, seed in zip(LETTERS, seeds):
        cell = by_seed.get(seed)
        value = getter(cell) if cell is not None and cell.finished else None
        macros.number(f"{prefix}{letter}", value, digits, f"{what}, seed {seed}", scale)


def cost_macros(macros: Macros, prefix: str, cells: list[SimpleNamespace], label: str, complete: bool) -> None:
    done = finished(cells)
    macros.number(f"{prefix}Cand", mean_or_none(r.summary["candidates_trained"] for r in done), 1,
                  f"mean candidates trained per finished {label} cell", complete=complete, fmt=compact)
    macros.number(f"{prefix}Rounds", mean_or_none(r.summary["rounds"]["allocated"] for r in done), 1,
                  f"mean rounds that trained candidates per finished {label} cell (the stop decision's own round "
                  "is not counted)", complete=complete, fmt=compact)
    macros.number(f"{prefix}RoundsOpened", mean_or_none(r.summary["rounds"]["opened"] for r in done), 1,
                  f"mean rounds opened per finished {label} cell (includes the round in which the agent stopped)",
                  complete=complete, fmt=compact)
    macros.number(f"{prefix}GPUh", mean_or_none(r.hours for r in done), 1,
                  f"mean training wall hours per finished {label} cell: sum of training.wall_h over ledger rows "
                  "(one GPU per job)", complete=complete, fmt=compact)
    macros.number(f"{prefix}GPUhSum", float(sum(r.hours for r in done)) if done else None, 1,
                  f"training wall hours summed over the finished {label} cells", complete=complete, fmt=compact)
    macros.number(f"{prefix}Tokens", mean_or_none(r.summary["tokens"]["total"] for r in done), 0,
                  f"mean LLM tokens (prompt + completion, decision + critic + component calls) per finished {label} cell",
                  complete=complete)
    macros.number(f"{prefix}TokensSum", float(sum(r.summary["tokens"]["total"] for r in done)) if done else None, 0,
                  f"LLM tokens summed over the finished {label} cells", complete=complete)


def int_text(value: float) -> str:
    return str(int(round(value)))


def median_text(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def loopseq_search_macros(macros: Macros) -> None:
    """The descriptives of Fig. 2c,d, read from the candidate and band tables that figure draws (`make fig2-data`).

    A pool's entropy is the mean over the three groups of the group's pool median. `RoundOne*` are the candidates of
    round 1 without a gate, the initial recipe included; `Gated*` are the candidates that carry a gate in any round."""
    folder = REPO / "scripts" / "fig2" / "data" / "main_line"
    cands_file, bands_file = folder / "ls_candidates.csv", folder / "ls_composition_bands.csv"
    cands = pd.read_csv(cands_file) if cands_file.is_file() else None
    bands = pd.read_csv(bands_file) if bands_file.is_file() else None
    groups = ("low", "mid", "high")
    if cands is not None:
        cands = cands.assign(entropy=cands[[f"pool_dinucleotide_entropy_{g}" for g in groups]].mean(axis=1),
                             diversity=cands[[f"pool_pairwise_ratio_{g}" for g in groups]].mean(axis=1),
                             gated=cands["n_gates"].fillna(0) > 0)
        first = cands[cands["round"] == 1]
        round_one = first[~first.gated]
        gated = cands[cands.gated]
        failed = gated[~gated.qualified]
        first_gate = gated.groupby("seed")["round"].min()
        kl_initial = float(cands[cands.anchor.astype(bool)].kl.iloc[0])
        delivered_kl = cands[cands.delivered_for.notna()].groupby("seed").kl.max().sort_values()
        at_initial = pd.Series(np.isclose(cands.kl, kl_initial), index=cands.index)
        no_gate_initial = cands[~cands.gated & at_initial]
        gate_initial = cands[cands.gated & at_initial]
        no_gate_above = cands[~cands.gated & (cands.kl > kl_initial + 1e-9)]
        gated_low = gated[gated.kl < kl_initial - 1e-9]
        diversity_by_kl = gated.groupby("kl").diversity.median().sort_index()
        delivered_rounds = cands[cands.delivered_for.notna()]["round"]
    pairs = (
        ("RoundOneN", None if cands is None else float(len(round_one)), 0, "round-1 candidates without a gate, initial recipe included"),
        ("RoundOneQual", None if cands is None else float(round_one.qualified.sum()), 0, "of those, the qualified ones"),
        ("RoundOneEntLo", None if cands is None else float(round_one.entropy.min()), 2, "lowest mean pool dinucleotide entropy (bits) among them"),
        ("RoundOneEntHi", None if cands is None else float(round_one.entropy.max()), 2, "highest mean pool dinucleotide entropy (bits) among them"),
        ("PolicyEnt", None if bands is None else float(bands[bands.measurement == "dinucleotide_entropy"]["median"].mean()), 2,
         "initial policy's pool dinucleotide entropy (bits), mean over searches and groups of the median"),
        ("GatedN", None if cands is None else float(len(gated)), 0, "agent candidates (the initial recipe never carries a gate) that carry a gate, over the three searches"),
        ("GatedQual", None if cands is None else float(gated.qualified.sum()), 0, "of those, the qualified ones"),
        ("GatedEntLo", None if cands is None else float(gated.entropy.min()), 2, "lowest mean pool dinucleotide entropy (bits) among the gated candidates"),
        ("GatedEntHi", None if cands is None else float(gated.entropy.max()), 2, "highest mean pool dinucleotide entropy (bits) among the gated candidates"),
        ("GatedUnqual", None if cands is None else float(len(failed)), 0, "gated candidates that did not qualify"),
        ("GatedUnqualKl", None if cands is None else float((failed.kl_dir < 0).sum()), 0, "of those, the ones with a KL coefficient lower than their base's"),
        ("GateRoundMin", None if cands is None else float(first_gate.min()), 0, "earliest first-gate round over the searches"),
        ("GateRoundMax", None if cands is None else float(first_gate.max()), 0, "latest first-gate round over the searches"),
        ("DelKlLow", None if cands is None else float(cands[cands.delivered_for.notna()].kl.min()), 2,
         "lowest KL coefficient among the delivered candidates"),
        ("DelKlMid", None if cands is None else float(delivered_kl.iloc[:2].max()), 3,
         "highest KL coefficient among the delivered candidates of the two searches with the lowest ones"),
        ("DelKlHigh", None if cands is None else float(delivered_kl.iloc[2]), 2,
         "KL coefficient of the delivered candidates of the third search"),
        ("KlInitNoGateN", None if cands is None else float(len(no_gate_initial)), 0,
         "candidates without a gate that kept the initial recipe's KL coefficient"),
        ("KlInitGateN", None if cands is None else float(len(gate_initial)), 0,
         "gated candidates that kept the initial recipe's KL coefficient"),
        ("GatedLoKlN", None if cands is None else float(len(gated_low)), 0,
         "gated candidates whose KL coefficient is below the initial recipe's"),
        ("GatedLoKlQual", None if cands is None else float(gated_low.qualified.sum()), 0, "of those, the qualified ones"),
        ("GatedKlMin", None if cands is None else float(gated.kl.min()), 3, "lowest KL coefficient among the gated candidates"),
        ("GatedKlMax", None if cands is None else float(gated.kl.max()), 2, "highest KL coefficient among the gated candidates"),
        ("DivKlLow", None if cands is None else float(diversity_by_kl.iloc[0]), 2,
         "median pool diversity (mean over the groups of the pairwise-distance ratio to the initial policy's pool) of the gated "
         "candidates at their lowest KL coefficient"),
        ("DivKlHigh", None if cands is None else float(diversity_by_kl.iloc[-1]), 2,
         "the same at their highest KL coefficient"),
        ("DelRoundMin", None if cands is None else float(delivered_rounds.min()), 0, "earliest round of a delivered candidate"),
        ("DelRoundMax", None if cands is None else float(delivered_rounds.max()), 0, "latest round of a delivered candidate"),
    )
    if cands is not None and not diversity_by_kl.is_monotonic_increasing:
        raise SystemExit("Results 2 says the pool diversity of the gated candidates rose with the KL coefficient; the median "
                         f"diversity by coefficient is {diversity_by_kl.round(2).to_dict()}")
    if cands is not None and len(failed) != int((failed.kl_dir < 0).sum()):
        raise SystemExit("Results 2 says every gated candidate that did not qualify had a lower KL coefficient than its base; "
                         "the table says otherwise")
    if cands is not None and not (delivered_kl.iloc[:2].max() < kl_initial <= delivered_kl.iloc[2]):
        raise SystemExit("Results 3 says two searches delivered candidates with a KL coefficient below the initial one and the "
                         "third one above it; the table says otherwise")
    if cands is not None and (no_gate_initial.qualified.any() or not gate_initial.qualified.all() or len(no_gate_above) != 2
                              or not no_gate_above.qualified.all()):
        raise SystemExit("Results 2 says no candidate without a gate qualified at the initial KL coefficient, every gated "
                         "candidate did, and both candidates without a gate above it qualified; the table says otherwise")
    for name, value, digits, description in pairs:
        macros.number(f"ls{name}", value, digits, f"Loop-seq main line: {description}")


def mechanism_macros(macros: Macros, prefix: str, cells: list[SimpleNamespace], expected: int, mech: pd.DataFrame,
                     label: str, parts: Iterable[str], spend: tuple[str, str] | None = None) -> None:
    """The mechanism descriptives of one block of cells (a task's main line, a reference root, an arm).

    `GateCells`, `GateRoundMed`, `SpendCells`, `PenaltyCells`, `OutShare`, `Start*`, `CandProv`, `HyperOnly`, `StepsMax` and
    `KlMax` come from the candidates of every cell of the block that has a mechanism row (finished or running) and are
    provisional until `expected` cells are finished. `CandProv` is the mean number of agent candidates over exactly those
    cells, so it is the denominator of `StartInit + StartParent + StartCont`. `LibPairMean`/`LibPairMin` read the delivered
    libraries of the finished cells only and are an aggregate. `spend` is (name suffix, oracle) for `SpendCells`.
    """
    parts = set(parts)
    by_id = {row["cell_id"]: row for row in mech.to_dict("records")}
    rows = [by_id[cell_id(r)] for r in cells if cell_id(r) in by_id]
    complete = is_complete(cells, expected)

    def per_cell(key: str, scale: float = 1.0) -> list[float]:
        return [float(row[key]) * scale for row in rows if row.get(key) is not None and np.isfinite(row[key])]

    def mean_of(key: str, scale: float = 1.0) -> float | None:
        found = per_cell(key, scale)
        return float(np.mean(found)) if found else None

    one = f"{label} cells that have candidates (finished or running)"
    if "GateCells" in parts:
        gated = [row for row in rows if row["candidates_with_gate"] > 0]
        macros.provisional(f"{prefix}GateCells", float(len(gated)) if rows else None, int_text,
                           f"{label}: cells whose agent candidates include at least one gate, of the {one}", complete)
        if "GateRoundMed" in parts:
            rounds = [float(row["first_gate_round"]) for row in gated]
            if rounds or not complete:
                macros.provisional(f"{prefix}GateRoundMed", float(np.median(rounds)) if rounds else None, compact,
                                   f"{label}: median over the gated cells of the round of the first gate", complete)
            else:
                macros.text(f"{prefix}GateRoundMed", "none", f"{label}: no cell used a gate")
    if "GateCands" in parts:
        macros.provisional(f"{prefix}GateCands", float(sum(row["gate_candidates"] for row in rows)) if rows else None, int_text,
                           f"{label}: agent candidates whose recipe adds a gate or tightens a shared gate against their parent, "
                           f"over the {one}", complete)
        macros.provisional(f"{prefix}GateCited", float(sum(row["gate_cited"] for row in rows)) if rows else None, int_text,
                           f"{label}: of the GateCands, those whose stored hypothesis text mentions the composition line or band "
                           "(case-insensitive keywords composition, entropy, GC, homopolymer, low-complexity, diversity, band, k-mer)",
                           complete)
    if "SpendCells" in parts and spend is not None:
        macros.provisional(f"{prefix}SpendCells",
                           float(sum(spend[1] in row["agent_enabled_union"].split("+") for row in rows)) if rows else None,
                           int_text, f"{label}: cells in which any agent candidate has {spend[1]} among its enabled oracles, "
                           f"of the {one}", complete)
    if "PenaltyCells" in parts:
        macros.provisional(f"{prefix}PenaltyCells", float(sum(row["agent_penalty_edits"] > 0 for row in rows)) if rows else None,
                           int_text, f"{label}: cells in which any agent candidate edits the penalties against its parent, "
                           f"of the {one}", complete)
    if "OutShare" in parts:
        macros.provisional(f"{prefix}OutShare", mean_of("share_candidates_any_outside", 100.0), pct_text,
                           f"{label}: mean over cells of the share (percent, no sign) of trained candidates whose composition "
                           "line lists a measurement outside the initial policy's 5-95% band", complete)
    for suffix, key, text in (("StartInit", "agent_from_initial", "from_initial"), ("StartParent", "agent_from_parent", "from_parent"),
                              ("StartCont", "agent_continue_parent", "continue_parent")):
        if "Start" in parts:
            macros.provisional(f"{prefix}{suffix}", mean_of(key), compact,
                               f"{label}: mean per cell of non-anchor candidates that start {text}, over the {one}", complete)
    if "CandProv" in parts:
        macros.provisional(f"{prefix}CandProv", mean_of("agent_candidates"), compact,
                           f"{label}: mean per cell of non-anchor agent candidates with an allocated recipe, over exactly the "
                           f"{one} (the denominator of the Start macros: StartInit + StartParent + StartCont)", complete)
    if "HyperOnly" in parts:
        macros.provisional(f"{prefix}HyperOnly", mean_of("hyperparameter_only"), compact,
                           f"{label}: mean per cell of non-anchor candidates whose edits against the parent are only "
                           "lr, kl, group_size or steps", complete)
    if "StepsMax" in parts:
        found = per_cell("agent_steps_max")
        macros.provisional(f"{prefix}StepsMax", max(found) if found else None, int_text,
                           f"{label}: largest number of optimizer steps chosen in any non-anchor candidate", complete)
    if "KlMax" in parts:
        found = per_cell("agent_kl_max")
        macros.provisional(f"{prefix}KlMax", max(found) if found else None, lambda v: f"{v:.3g}",
                           f"{label}: largest kl chosen in any non-anchor candidate", complete)
    if {"LibPairMean", "LibPairMin"} & parts:
        done = {cell_id(r) for r in cells if r.finished}
        ratios = [float(value) for row in rows if row["cell_id"] in done for key, value in row.items()
                  if key.startswith("delivered_pairwise_ratio_") and value is not None and np.isfinite(value)]
        what = (f"pairwise ratio (mean pairwise normalized edit distance over the initial policy's) of the delivered "
                f"libraries of the finished {label} cells")
        if "LibPairMean" in parts:
            macros.number(f"{prefix}LibPairMean", float(np.mean(ratios)) if ratios else None, 2, f"mean {what}",
                          complete=complete)
        if "LibPairMin" in parts:
            macros.number(f"{prefix}LibPairMin", float(np.min(ratios)) if ratios else None, 2, f"minimum {what}",
                          complete=complete)


def refusal_row(record: SimpleNamespace) -> dict[str, Any]:
    """Decision answers of one cell and how many carried at least one typed problem, from the event log.

    An answer is a recorded model response to a propose or stop request (one per attempt; a refused answer is asked
    again, up to `max_decision_attempts`). It is refused when its validation found at least one typed problem, whether or
    not other candidates in the same answer were accepted. Critic notes are `CriticAdvisory` events; Coder calls are the
    LLM calls of kind `component`.
    """
    view = record.cell.view
    answers = [(stored, recorded) for stored, recorded in view.of_type(ev.DecisionRecorded)
               if recorded.kind in ("propose", "stop")]
    codes: Counter = Counter()
    refused = 0
    for _, recorded in answers:
        problems = view.state.decisions[recorded.decision_id].problems
        if problems:
            refused += 1
            codes.update({problem.code for problem in problems})
    return {"root": record.root, "cell": record.name, "cell_id": cell_id(record), "arm": record.arm, "seed": record.seed,
            "cell_finished": record.finished, "answers": len(answers), "refused": refused,
            "problem_codes": json.dumps(dict(sorted(codes.items()))),
            "answers_invalid": sum(view.state.decisions[recorded.decision_id].status == "invalid" for _, recorded in answers),
            "critic_notes": len(view.of_type(ev.CriticAdvisory)),
            "coder_calls": record.summary["tokens"]["by_call"]["component"]["calls"]}


def readable_code(code: str) -> str:
    return code.replace("_", " ")


def refusal_macros(macros: Macros, prefix: str, cells: list[SimpleNamespace], expected: int, refusals: pd.DataFrame,
                   label: str) -> None:
    ids = {cell_id(r) for r in cells}
    rows = [row for row in refusals.to_dict("records") if row["cell_id"] in ids]
    complete = is_complete(cells, expected)
    codes: Counter = Counter()
    for row in rows:
        codes.update(json.loads(row["problem_codes"]))
    macros.provisional(f"{prefix}Decisions", float(sum(row["answers"] for row in rows)) if rows else None, int_text,
                       f"{label}: model answers to propose/stop requests, over all cells of the block (finished or running)", complete)
    macros.provisional(f"{prefix}Refused", float(sum(row["refused"] for row in rows)) if rows else None, int_text,
                       f"{label}: model answers whose validation found at least one typed problem", complete)
    top = None if not rows else (min(codes, key=lambda code: (-codes[code], code)) if codes else "none")
    macros.provisional_text(f"{prefix}TopRefusal", None if top is None else readable_code(top),
                            f"{label}: most frequent typed problem code among the refused answers, underscores as spaces "
                            "(ties: alphabetical; none when nothing was refused)", complete)


def lineage_rows(record: SimpleNamespace) -> list[dict[str, Any]]:
    """Cumulative training updates along the delivered checkpoint's lineage, per condition, for one finished cell.

    The count is the engine's `lineage_updates` of the checkpoint named by `delivery.json` `checkpoint_assignment`,
    checked against an independent walk of the candidate chain: the updates the delivered candidate trained up to that
    checkpoint (`train.checkpoint_steps` at the checkpoint's position), plus the `steps_done` of every ancestor reached
    through `from_parent` or `continue_parent` starts, stopping after the first ancestor that started `from_initial` (a
    `from_initial` candidate begins a new lineage from the initial policy, so what its siblings or its own would-be parent
    trained does not count). `from_parent` and `continue_parent` both add the candidate's own updates to its parent's:
    the first keeps the weights and resets the optimizer and schedule, the second also keeps them. The initial policy's
    supervised training is not counted.
    """
    view = record.cell.view
    assignment = (read_json(record.cell.path / "delivery.json") or {}).get("checkpoint_assignment") or {}
    rows = []
    for condition in record.cell.conditions:
        checkpoint = assignment.get(condition)
        if checkpoint is None:
            continue
        cid = checkpoint.split(":")[0]
        result = view.candidate_result(cid)
        ids = [entry["id"] for entry in result["checkpoints"]]
        position = ids.index(checkpoint)
        engine = result["checkpoints"][position]["lineage_updates"]
        updates, current, chain = (view.spec(cid)["train"]["checkpoint_steps"][position], cid, [cid])
        while view.state.candidates[current].start != "from_initial":
            parent = view.state.candidates[current].parent
            updates += view.state.candidates[parent].completed.steps_done
            current = parent
            chain.append(parent)
        rows.append({"root": record.root, "cell": record.name, "cell_id": cell_id(record), "arm": record.arm,
                     "seed": record.seed, "cell_finished": record.finished, "condition": condition,
                     "delivered_checkpoint": checkpoint, "delivered_start": view.state.candidates[cid].start,
                     "delivered_is_end_checkpoint": checkpoint == result["end_checkpoint_id"],
                     "lineage_length": len(chain), "lineage_chain": " < ".join(chain),
                     "lineage_updates_engine": engine, "lineage_updates_chain": updates,
                     "agree": bool(engine == updates)})
    return rows


def lineage_macros(macros: Macros, prefix: str, cells: list[SimpleNamespace], expected: int, lineage: pd.DataFrame,
                   label: str) -> None:
    ids = {cell_id(r) for r in cells if r.finished}
    values = [float(row["lineage_updates_engine"]) for row in lineage.to_dict("records") if row["cell_id"] in ids]
    macros.number(f"{prefix}LineageUpdates", float(np.mean(values)) if values else None, 0,
                  f"{label}: mean over finished cells and conditions of the cumulative training updates along the delivered "
                  "checkpoint's lineage (delivered candidate and its ancestors, restarts from the initial policy begin a new "
                  "lineage)", complete=is_complete(cells, expected))


# ----- Loop-seq baselines under the agents' delivery rule --------------------------------------------------
# Library name, macro suffix, the file holding the 512-per-group pool (sequence, requested_group, the DNAcycP2 raw score).
LOOPSEQ_POOLS = (
    ("random_dna", "Random", "outputs/baselines/loopseq/random/library_scored.parquet"),
    ("sft_only", "Sft", "outputs/baselines/loopseq/sft_only/library_scored.parquet"),
    ("ga_search", "Ga", "outputs/baselines/loopseq/ga_search/library_scored.parquet"),
    ("handtuned_grpo_4k", "GrpoFourK", "outputs/calibration/loopseq_depth/4k/audit/audit_scores.parquet"),
    ("handtuned_grpo_16k", "GrpoSixteenK", "outputs/calibration/loopseq_depth/16k/audit/audit_scores.parquet"),
    ("handtuned_grpo_32k", "GrpoThirtyTwoK", "outputs/baselines/loopseq/handtuned_grpo_32k/audit/audit_scores.parquet"),
    ("handtuned_grpo_64k", "Grpo", "outputs/baselines/loopseq/handtuned_grpo_64k/audit/audit_scores.parquet"),
    ("kl_isolation_0.001", "Kl", "outputs/calibration/loopseq_kl_isolation/kl0.001/audit/audit_scores.parquet"),
)


def measurement_record(value: Any, row_ids: list[str]) -> dict[str, Any]:
    """A task measurement as the backend stores it for the delivery rule (`control.backend._measure_record`)."""
    return {"values": np.asarray(value.values).tolist(), "sources": list(value.sources), "row_ids": list(row_ids),
            "status": value.status, "metadata": dict(value.metadata or {})}


class LoopseqReplica:
    """The Loop-seq task's own runtime, CNN scorer and delivery plan, to apply the agents' delivery rule to any pool.

    Built from a control cell's newest config snapshot with `tasks.loopseq.experiment.build_runtime` on the CPU. The plan is
    the config's `control.selection.plan` (legal rows, per-group dinucleotide-entropy and distinct-3-mer floors, reverse-
    complement duplicates removed within a group, training copies excluded, ranked by `margin/cnn`, 128 per group) run
    through `recital.library_selection.select_library_plan` with the contract's visible and blind sources and the task's
    training sequences, the call the backend makes. The blind DNAcycP2 estimand is the runtime's own `audit_measure`.
    """

    def __init__(self, config: dict[str, Any], repo: Path, scratch_output: Path) -> None:
        from recital.library_selection import SelectionPlan, select_library_plan
        from tasks.loopseq.experiment import build_runtime

        self.runtime = build_runtime(config, repo, scratch_output, "cpu")
        self.contract = self.runtime["contract"]
        self.plan_dict = dict(config["control"]["selection"]["plan"])
        self.health_fraction = config["control"]["health"]["min_distance_fraction"]
        self.plan = SelectionPlan.from_dict(self.plan_dict)
        self._plan_class = SelectionPlan
        self._select = select_library_plan
        self._cnn = self.runtime["oracle_factories"]["cnn"]()
        self.conditions = list(self.contract.conditions)

    def cnn(self, sequences: list[str]) -> np.ndarray:
        return np.asarray(self._cnn(list(sequences)), dtype=float)

    def diversity_floor(self, sequences: list[str], conditions: list[str]) -> float:
        """The plan's minimum distance, derived as the engine derives it (`Engine._delivery_rule`) from an initial-policy
        pool: `min_distance_fraction` of the smallest condition's median nearest-neighbour normalized edit distance."""
        from recital.control.health import baseline_reference, diversity_floor

        legal = {g: [s for s, c in zip(sequences, conditions) if c == g and len(s) == 50 and set(s) <= set("ACGT")]
                 for g in self.conditions}
        return float(diversity_floor(baseline_reference(legal), self.health_fraction))

    def plan_with_floor(self, floor: float):
        from recital.control.health import plan_with_floor

        return self._plan_class.from_dict(plan_with_floor(self.plan_dict, floor))

    def select(self, sequences: list[str], conditions: list[str], cnn_raw: np.ndarray, plan=None):
        """Generation-order selected indices and the selection report for one pool; `plan` replaces the declared one."""
        row_ids = [f"row{i}" for i in range(len(sequences))]
        derived = self.runtime["measure"](sequences, conditions, {"cnn": cnn_raw})
        measurements = {name: measurement_record(value, row_ids) for name, value in derived.items()}
        return self._select(sequences, conditions, measurements, plan or self.plan, visible_sources=self.contract.visible_sources,
                            blind_sources=self.contract.terminal_blind_sources,
                            training_sequences=self.runtime["training_sequences"], row_ids=row_ids)

    def audit(self, sequences: list[str], conditions: list[str], dnacycp2_raw: np.ndarray) -> dict[str, Any]:
        """Per group in-band fraction (primary) and median signed margin (secondary), and their worst groups."""
        found = self.runtime["audit_measure"](list(sequences), list(conditions), {"dnacycp2": np.asarray(dnacycp2_raw, dtype=float)})
        hits, margins = found["primary"], found["secondary"]["primary"]
        return {"hit": {g: hits["per_condition"][g]["value"] for g in self.conditions}, "hit_worst": hits["worst_condition"],
                "margin": {g: margins["per_condition"][g]["value"] for g in self.conditions},
                "margin_worst": margins["worst_condition"], "count": {g: hits["per_condition"][g]["count"] for g in self.conditions}}


def replica_check(replica: LoopseqReplica, record: SimpleNamespace) -> dict[str, Any]:
    """Run the replica on a finished agent cell's delivery pool and compare with what the cell delivered and audited."""
    cell = record.cell.path
    pool = read_json(cell / "delivery" / "pool.json")
    artifact = read_json(cell / "delivery" / "delivery_artifact.json")
    stored = np.asarray(read_json(cell / "delivery" / "scores" / "cnn.json")["values"], dtype=float)
    cnn_raw = replica.cnn(pool["sequences"])
    delivered_plan = replica._plan_class.from_dict(artifact["selection_plan"])
    indices, _ = replica.select(pool["sequences"], pool["conditions"], cnn_raw, plan=delivered_plan)
    chosen = [pool["row_ids"][i] for i in indices]
    terminal = read_json(cell / "terminal_audit" / "dnacycp2.json")
    manifest = read_json(cell / "terminal_audit" / "manifest.json")
    audit = replica.audit(manifest["sequences"], manifest["conditions"], np.asarray(terminal["values"], dtype=float))
    summary = record.cell.audit_summary
    return {"cell": cell_id(record), "pool_rows": len(pool["sequences"]),
            "minimum_distance_in_cell_plan": artifact["selection_plan"].get("minimum_distance"),
            "cnn_max_abs_diff_vs_stored": float(np.max(np.abs(cnn_raw - stored))),
            "selected_rows_equal_delivered": chosen == list(artifact["selected_row_ids"]), "selected": len(chosen),
            "hit_worst_equals_terminal_audit": bool(np.isclose(audit["hit_worst"], summary["primary"]["worst_condition"], rtol=0, atol=1e-12)),
            "margin_worst_equals_terminal_audit": bool(np.isclose(audit["margin_worst"], summary["secondary"]["primary"]["worst_condition"],
                                                                  rtol=0, atol=1e-12))}


def eligible_per_group(report: dict[str, Any], groups: list[str]) -> dict[str, int]:
    """Rows per group left after the plan's filters and duplicate removal (`by_stratum.<group>.eligible_count`)."""
    return {g: int(report["by_stratum"][g]["eligible_count"]) for g in groups}


def selected_baselines(replica: LoopseqReplica, repo: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Per baseline pool: the 128-per-group library the agents' rule selects and its DNAcycP2 in-band fraction and margin."""
    rows, missing = [], []
    sft = pd.read_parquet(repo / "outputs/baselines/loopseq/sft_only/library_scored.parquet")
    floor = replica.diversity_floor(sft.sequence.tolist(), sft.requested_group.tolist())
    plan = replica.plan_with_floor(floor)
    for name, suffix, relative in LOOPSEQ_POOLS:
        path = repo / relative
        if not path.is_file():
            missing.append(f"{name}: {relative}")
            continue
        frame = pd.read_parquet(path)
        sequences, conditions = frame.sequence.tolist(), frame.requested_group.tolist()
        dnacycp2 = frame["audit__dnacycp2__raw"].to_numpy(dtype=float)
        cnn_raw = replica.cnn(sequences)
        indices, report = replica.select(sequences, conditions, cnn_raw, plan=plan)
        picked = np.asarray(indices, dtype=int)
        selected_per_group = {g: int(sum(conditions[i] == g for i in picked)) for g in replica.conditions}
        if min(selected_per_group.values()) == 0:
            chosen = {"count": selected_per_group, "hit_worst": None, "margin_worst": None,
                      "hit": {g: None for g in replica.conditions}, "margin": {g: None for g in replica.conditions}}
            for g in replica.conditions:
                if selected_per_group[g]:
                    rows_g = [i for i in picked if conditions[i] == g]
                    part = replica.audit([sequences[i] for i in rows_g], [g] * len(rows_g), dnacycp2[rows_g])
                    chosen["hit"][g], chosen["margin"][g] = part["hit"][g], part["margin"][g]
        else:
            chosen = replica.audit([sequences[i] for i in picked], [conditions[i] for i in picked], dnacycp2[picked])
        pool = replica.audit(sequences, conditions, dnacycp2)
        row = {"library": name, "suffix": suffix, "source": relative, "pool_rows": len(frame),
               "pool_per_group": json.dumps({g: int(conditions.count(g)) for g in replica.conditions}),
               "cnn_max_abs_diff_vs_stored": (float(np.max(np.abs(cnn_raw - frame["oracle__cnn"].to_numpy(dtype=float))))
                                              if "oracle__cnn" in frame else None),
               "minimum_distance": floor, "selected_per_group": json.dumps(selected_per_group),
               "selected_total": int(sum(selected_per_group.values())),
               "eligible_per_group": json.dumps(eligible_per_group(report, replica.conditions)),
               "pool_hit_worst": pool["hit_worst"], "pool_hit_equals_stored_column": (
                   bool(np.allclose([float(frame.loc[frame.requested_group == g, "audit__dnacycp2__hit"].mean()) for g in replica.conditions],
                                    [pool["hit"][g] for g in replica.conditions], rtol=0, atol=1e-12))),
               "hit_worst": chosen["hit_worst"], "margin_worst": chosen["margin_worst"]}
        for g in replica.conditions:
            row[f"hit_{g}"], row[f"margin_{g}"] = chosen["hit"][g], chosen["margin"][g]
        rows.append(row)
    return rows, missing


# The comparator libraries of the CRE baseline directory, and what each is. The Methods describe regLM as: released
# human-enhancer checkpoint, temperature 1.0, top-k 2, exact duplicates removed within each target cell, the top 128 per
# cell kept by target-minus-other-cell specificity of the released regLM regression models; and TACO as alpha 0.01, five
# design seeds, 256 sequences per cell.
COMPARATOR_ROLES = {
    "reglm_project_unique_128": "MATCHES Methods (regLM): top_k 2, deduplicated within each cell, top 128 per cell by paired regLM specificity",
    "reglm_published": "superseded: published-threshold filter and top specificity without deduplication",
    "reglm": "not the Methods library: unfiltered top_k 2 direct sample, 256 per cell (the row SUMMARY.csv and the external table carry)",
    "reglm_topk_none_broken": "broken: top_k=None draws (near-random sequences), superseded by the top_k 2 run",
    "taco_alpha001_5seed": "MATCHES Methods (TACO): alpha 0.01, five design seeds, 256 per cell per seed; mean over the five seed medians",
    "taco_alpha001_5seed_s0": "one of the five seeds of the Methods row",
    "taco_alpha001_5seed_s1": "one of the five seeds of the Methods row",
    "taco_alpha001_5seed_s2": "one of the five seeds of the Methods row",
    "taco_alpha001_5seed_s3": "one of the five seeds of the Methods row",
    "taco_alpha001_5seed_s4": "one of the five seeds of the Methods row",
    "taco_alpha001_81823": "superseded: single-seed job 81823",
    "taco": "not the Methods library: alpha 0.0 paper reproduction, one seed, 192 per cell (the row SUMMARY.csv and the external table carry)",
    "taco_perffix_crosscheck": "not the Methods library: alpha 0.0 cross-check run, 192 per cell",
}


def read_comparators(repo: Path) -> dict[str, dict[str, Any]]:
    """One record per directory of COMPARATOR_ROLES: calibrated medians (the directory's own `calibrated_summary.json`,
    checked against the audit manifest's raw median minus delta), sizes, duplicates and which directories hold the
    same sequences."""
    import hashlib

    root = repo / BASELINES / "cre"
    out: dict[str, dict[str, Any]] = {}
    for name, role in COMPARATOR_ROLES.items():
        calibrated = read_json(root / name / "calibrated_summary.json")
        if calibrated is None:
            continue
        manifest = read_json(root / name / "baseline_manifest.json") or {}
        diversity = (manifest.get("provenance") or {}).get("diversity") or {}
        entry: dict[str, Any] = {
            "dir": name, "label": manifest.get("label") or calibrated.get("method"), "role": role,
            "rows": diversity.get("n_sequences"), "unique_rows": diversity.get("n_unique_sequences"),
            "generator_top_k2": "top_k=2" in str(((manifest.get("provenance") or {}).get("extra") or {}).get("generator", "")),
            "git_commit": str(((read_json(root / name / "manifest.json") or {}).get("git") or {}).get("commit", ""))[:10]}
        matches = []
        for condition in CONDITIONS["cre"]:
            block = calibrated["per_cell"][condition]
            if "point_estimate" in block:
                entry.update({f"{condition}_cal": block["point_estimate"], f"{condition}_lo": block["ci_low"],
                              f"{condition}_hi": block["ci_high"], f"{condition}_n": block["n"]})
                raw = ((manifest.get("per_cell") or {}).get(condition) or {}).get("bootstrap", {}).get("point_estimate")
                if raw is not None:
                    matches.append(bool(np.isclose(cal(condition, raw), block["point_estimate"], rtol=0, atol=1e-9)))
            else:
                medians = block["seed_medians"]
                entry.update({f"{condition}_cal": block["mean_over_seed_medians"], f"{condition}_sd": block["sd_over_seed_medians"],
                              f"{condition}_min": min(medians), f"{condition}_max": max(medians), f"{condition}_seeds": block["n_seed"]})
        entry["raw_minus_delta_matches_calibrated"] = all(matches) if matches else None
        library = root / name / "library.parquet"
        if library.is_file():
            sequences = sorted(set(pd.read_parquet(library, columns=["sequence"]).sequence))
            entry["sequence_digest"] = hashlib.sha256("\n".join(sequences).encode()).hexdigest()[:12]
        extra = (manifest.get("provenance") or {}).get("extra") or {}
        entry["tfbs_lambda"] = (extra.get("run_metadata") or {}).get("tfbs_lambda")
        entry["qualification"] = (extra.get("run_metadata") or {}).get("qualification")
        entry["motif_source"] = (extra.get("run_metadata") or {}).get("motif_source")
        entry["run_base"] = calibrated.get("run_base")
        if name == "reglm_project_unique_128" and library.is_file():
            # The selection CSV and the candidate pool it was drawn from sit beside each other outside this repository.
            selected = Path(str(extra.get("source_csv", "")))
            pool = selected.parent / "reglm_candidates_seed0.csv"
            if selected.is_file():
                chosen = pd.read_csv(selected)
                entry["library_equals_source_selection"] = set(chosen.sequence) == set(sequences)
                entry["source_selection_dir"] = str(selected.parent)
            if pool.is_file():
                candidates = pd.read_csv(pool)
                entry["pool_rows_per_cell"] = candidates.groupby("target_cell").size().to_dict()
                entry["pool_unique_per_cell"] = candidates.groupby("target_cell").sequence.nunique().to_dict()
        out[name] = entry
    digests: dict[str, list[str]] = {}
    for name, entry in out.items():
        if entry.get("sequence_digest"):
            digests.setdefault(entry["sequence_digest"], []).append(name)
    for entry in out.values():
        twins = [n for n in digests.get(entry.get("sequence_digest"), []) if n != entry["dir"]]
        entry["identical_sequences_to"] = ", ".join(twins)
    seeds = [out[f"taco_alpha001_5seed_s{i}"] for i in range(5) if f"taco_alpha001_5seed_s{i}" in out]
    if "taco_alpha001_5seed" in out and len(seeds) == 5:
        for condition in CONDITIONS["cre"]:
            out["taco_alpha001_5seed"][f"{condition}_per_seed_matches"] = bool(np.allclose(
                [e[f"{condition}_cal"] for e in seeds], calibrated_seed_medians(root, condition), rtol=0, atol=1e-9))
        out["taco_alpha001_5seed"]["n_per_cell_per_seed"] = seeds[0].get("K562_n")
    return out


def calibrated_seed_medians(root: Path, condition: str) -> list[float]:
    return (read_json(root / "taco_alpha001_5seed" / "calibrated_summary.json") or {})["per_cell"][condition]["seed_medians"]


def comparator_macros(macros: Macros, comparators: dict[str, dict[str, Any]]) -> None:
    regl, taco = comparators.get("reglm_project_unique_128", {}), comparators.get("taco_alpha001_5seed", {})
    for condition, letter in (("K562", "K"), ("HepG2", "H")):
        macros.number(f"reglm{letter}", regl.get(f"{condition}_cal"), 2,
                      f"regLM as in the Methods (reglm_project_unique_128), calibrated {condition} median of the 128 retained")
        macros.number(f"reglm{letter}lo", regl.get(f"{condition}_lo"), 2, f"regLM, 95% bootstrap interval, lower, {condition}")
        macros.number(f"reglm{letter}hi", regl.get(f"{condition}_hi"), 2, f"regLM, 95% bootstrap interval, upper, {condition}")
        macros.number(f"taco{letter}", taco.get(f"{condition}_cal"), 2,
                      f"TACO as in the Methods (taco_alpha001_5seed), calibrated {condition} median averaged over the five seeds")
        macros.number(f"taco{letter}sd", taco.get(f"{condition}_sd"), 2,
                      f"TACO, standard deviation of the five seed medians, {condition}")
    macros.number("reglmN", regl.get("K562_n"), 0, "regLM sequences retained per cell line")
    macros.number("tacoN", taco.get("n_per_cell_per_seed"), 0, "TACO sequences per cell line per design seed")
    macros.number("tacoSeeds", taco.get("K562_seeds"), 0, "TACO design seeds")


def read_baseline_values(repo: Path) -> dict[str, Any]:
    """The comparator numbers the macros quote, from the baseline files (None when a file is absent)."""
    out: dict[str, Any] = {}
    summary_path = repo / BASELINES / "cre" / "SUMMARY.csv"
    if summary_path.is_file():
        frame = pd.read_csv(summary_path).set_index("method")
        for key, method in (("coda", "coda_fastseqprop"), ("dd384", "dna_diffusion")):
            if method in frame.index:
                row = frame.loc[method]
                out[f"{key}K"] = cal("K562", row.k562_blind_chrombpnet_point)
                out[f"{key}H"] = cal("HepG2", row.hepg2_blind_chrombpnet_point)
    matched = repo / BASELINES / "cre" / "dna_diffusion_matched_budget" / "matched_budget_summary.csv"
    if matched.is_file():
        frame = pd.read_csv(matched)
        plain = frame[frame.variant == "plain"]
        for key, size in (("ddNine", 9216), ("ddMax", int(plain.N_per_cell_pool.max()))):
            for cell_line, letter in (("K562", "K"), ("HepG2", "H")):
                pick = plain[(plain.N_per_cell_pool == size) & (plain.cell == cell_line)]
                if len(pick) == 1:
                    out[f"{key}{letter}"] = float(pick.cal_median_pooled.iloc[0])
                    out[f"{key}N"] = size
        smallest = int(plain.N_per_cell_pool.min())
        for cell_line, letter in (("K562", "K"), ("HepG2", "H")):
            small = plain[(plain.N_per_cell_pool == smallest) & (plain.cell == cell_line)]
            big = plain[(plain.N_per_cell_pool == int(plain.N_per_cell_pool.max())) & (plain.cell == cell_line)]
            if len(small) == 1:
                out[f"ddSmall{letter}"] = float(small.cal_median_pooled.iloc[0])
                out[f"ddMalSmall{letter}"] = float(small.malinois_median_selected_mean_of_draws.iloc[0])
            if len(big) == 1:
                out[f"ddMax{letter}lo"] = float(big.cal_pooled_ci_low.iloc[0])
                out[f"ddMax{letter}hi"] = float(big.cal_pooled_ci_high.iloc[0])
                out[f"ddMalMax{letter}"] = float(big.malinois_median_selected_mean_of_draws.iloc[0])
    table = repo / BASELINES / "loopseq" / "baseline_table.csv"
    if table.is_file():
        frame = pd.read_csv(table).set_index("method")
        for key, method in (("Sft", "sft_only"), ("Random", "random_dna"), ("Ga", "ga_search"),
                            ("GrpoFourK", "handtuned_grpo_4k"), ("GrpoSixteenK", "handtuned_grpo_16k"),
                            ("GrpoThirtyTwoK", "handtuned_grpo_32k"), ("GrpoSixtyFourK", "handtuned_grpo_64k")):
            if method in frame.index:
                out[f"lsBase{key}"] = float(frame.loc[method, "blind_worst_group"])
    rescored = repo / "outputs" / "cm_baselines_rescored_20260917"
    libraries = []
    for seed in (2301, 2302):
        path = rescored / f"matched_seed{seed}_audited_libraries.csv"
        if path.is_file():
            for name, part in pd.read_csv(path).groupby("library"):
                libraries.append({"seed": seed, "library": name, "t5": float(part.audit_2020_t5.median()),
                                  "esm": float(part.audit_2020_esm.median()), "distance": float(part.natural_distance.median())})
    if libraries:
        out["protrl"] = libraries
    return out


def fixed_recipe_medians(repo: Path) -> pd.DataFrame:
    """One row per fixed-recipe library that finished its terminal evaluation: the calibrated ChromBPNet median of its
    128 sequences of one cell line, from the per-sequence table `scripts/baselines/cre_manual_no_cbp.py score` wrote.
    The stored calibrated column must equal raw minus delta."""
    rows = []
    for tag, variant in FIXED_VARIANTS:
        for seed in FIXED_SEEDS:
            folder = repo / FIXED_RECIPES / variant / f"seed_{seed}"
            if not (folder / "evaluation_complete.json").is_file():
                continue
            frame = pd.read_parquet(folder / "per_sequence_scores.parquet")
            config = (read_json(folder / "run_manifest.json") or {})
            for condition in CONDITIONS["cre"]:
                part = frame[frame.target_cell == condition]
                calibrated = part.chrombpnet_specificity_raw.to_numpy(float) - DELTA[condition]
                if not np.allclose(calibrated, part.chrombpnet_specificity_calibrated.to_numpy(float), atol=1e-6):
                    raise SystemExit(f"{folder}: stored calibrated specificity differs from raw - delta for {condition}")
                rows.append({"recipe": variant, "tag": tag, "seed": seed, "condition": condition, "n": len(part),
                             "cal_median": float(np.median(calibrated)), "steps": config.get("checkpoint_rule"),
                             "source": str((folder / "per_sequence_scores.parquet").relative_to(repo))})
    return pd.DataFrame(rows, columns=["recipe", "tag", "seed", "condition", "n", "cal_median", "steps", "source"])


def anchor_values(records: list[SimpleNamespace]) -> pd.DataFrame:
    """The initial-recipe (anchor) library of every finished main-line cell next to the library the cell delivered, one row
    per task, seed, condition and source. The anchor values are those of `scripts/analysis/anchor_audit.py` (read through
    `common.anchor_audit`); CRE values are calibrated (cal = raw - delta). A cell whose anchor audit is not complete has no row."""
    rows = []
    for task in ("cre", "loopseq", "cm"):
        for cell in finished(select(records, task=task, roots=MAIN_ROOTS[task], arm=MAIN_ARM)):
            anchor, record = anchor_audit.summary(cell.name), anchor_audit.record(cell.name)
            if anchor is None or record is None:
                continue
            delivered = cell.cell.audit_summary
            if task == "loopseq":
                blocks = (("in_band", anchor, delivered), ("margin", anchor["secondary"], delivered["secondary"]))
            else:
                blocks = (("median", anchor, delivered),)
            for statistic, mine, theirs in blocks:
                for source, item in mine["sources"].items():
                    other = theirs["sources"][source]
                    for condition, entry in item["per_condition"].items():
                        a, d = entry["value"], other["per_condition"][condition]["value"]
                        if task == "cre":
                            a, d = cal(condition, a), cal(condition, d)
                        rows.append({"task": task, "cell": cell.name, "seed": cell.seed, "source": source,
                                     "statistic": statistic, "condition": condition, "anchor": a, "delivered": d,
                                     "anchor_rows": entry["count"], "anchor_qualified": bool(record["library_qualified"]),
                                     "anchor_candidate": record["anchor"]})
            if task == "loopseq":
                rows.append({"task": task, "cell": cell.name, "seed": cell.seed, "source": "dnacycp2", "statistic": "in_band",
                             "condition": "worst", "anchor": anchor["primary"]["worst_condition"],
                             "delivered": delivered["primary"]["worst_condition"], "anchor_rows": record["rows_per_condition"],
                             "anchor_qualified": bool(record["library_qualified"]), "anchor_candidate": record["anchor"]})
            if task == "cm":
                distance = anchor_audit.library_measurement(cell.name, "natural_distance")
                rows.append({"task": task, "cell": cell.name, "seed": cell.seed, "source": "natural_distance",
                             "statistic": "median", "condition": "CM", "anchor": float(np.median(distance)),
                             "delivered": cell.audit["natural_distance_median"], "anchor_rows": len(distance),
                             "anchor_qualified": bool(record["library_qualified"]), "anchor_candidate": record["anchor"]})
    return pd.DataFrame(rows)


def build_macros(records: list[SimpleNamespace], baseline: dict[str, Any], kl: dict[str, Any],
                 mech: pd.DataFrame, refusals: pd.DataFrame, lineage: pd.DataFrame,
                 comparators: dict[str, dict[str, Any]], selected: list[dict[str, Any]] | None,
                 cm_initial: dict[str, Any], now: dt.datetime, fixed: pd.DataFrame, anchor: pd.DataFrame) -> Macros:
    macros = Macros()
    macros.text("numbersasof", now.strftime("%Y-%m-%d %H:%M UTC"), "UTC time of the last regeneration")

    # CRE main line
    cre = select(records, task="cre", roots=MAIN_ROOTS["cre"], arm=MAIN_ARM)
    cre_done = finished(cre)
    seeds = MAIN_SEEDS["cre"]
    cre_complete = is_complete(cre, len(seeds))
    macros.integer("creN", len(cre_done), "CRE main-line cells finished (frozen with a complete terminal audit)")
    macros.integer("creExpected", len(seeds), "CRE main-line cells declared (seeds 0-2)")
    macros.integer("lsExpected", len(MAIN_SEEDS["loopseq"]), "Loop-seq main-line cells declared (seeds 0-2)")
    macros.integer("cmExpected", len(MAIN_SEEDS["cm"]), "CM main-line cells declared (seeds 2401-2403)")
    seed_macros(macros, "creK", cre, seeds, lambda r: r.audit["cal_K562"], 2, "CRE calibrated K562 median")
    seed_macros(macros, "creH", cre, seeds, lambda r: r.audit["cal_HepG2"], 2, "CRE calibrated HepG2 median")
    seed_macros(macros, "creW", cre, seeds, lambda r: r.audit["worst_cal"], 2,
                "CRE worst-condition calibrated median (the smaller of K562 and HepG2)")
    stat_macros(macros, "creK", [r.audit["cal_K562"] for r in cre_done], 2, "calibrated K562 medians over finished seeds",
                cre_complete)
    stat_macros(macros, "creH", [r.audit["cal_HepG2"] for r in cre_done], 2,
                "calibrated HepG2 medians over finished seeds", cre_complete)
    stat_macros(macros, "creW", [r.audit["worst_cal"] for r in cre_done], 2,
                "worst-condition calibrated medians over finished seeds", cre_complete)
    cost_macros(macros, "cre", cre, "CRE main-line", cre_complete)
    mechanism_macros(macros, "cre", cre, len(seeds), mech, "CRE main-line",
                     ("GateCells", "GateRoundMed", "GateCands", "SpendCells", "OutShare", "Start", "CandProv", "HyperOnly",
                      "LibPairMean", "LibPairMin", "StepsMax"), spend=("grammar", "grammar_prior"))
    refusal_macros(macros, "cre", cre, len(seeds), refusals, "CRE main-line")
    lineage_macros(macros, "cre", cre, len(seeds), lineage, "CRE main-line")

    # Comparators on the same estimator
    for key, desc in (("dd384K", "DNA-Diffusion, unselected 384 per cell, calibrated K562"),
                      ("dd384H", "DNA-Diffusion, unselected 384 per cell, calibrated HepG2"),
                      ("ddNineK", "DNA-Diffusion, top 128 of 9,216 by Malinois, calibrated K562"),
                      ("ddNineH", "DNA-Diffusion, top 128 of 9,216 by Malinois, calibrated HepG2"),
                      ("ddMaxK", "DNA-Diffusion, top 128 of the largest pool by Malinois, calibrated K562"),
                      ("ddMaxH", "DNA-Diffusion, top 128 of the largest pool by Malinois, calibrated HepG2"),
                      ("codaK", "CODA Fast SeqProp, calibrated K562"), ("codaH", "CODA Fast SeqProp, calibrated HepG2")):
        macros.number(key.replace("dd384", "ddUnsel"), baseline.get(key), 2, desc)
    for letter, name in (("K", "K562"), ("H", "HepG2")):
        macros.number(f"ddSmall{letter}", baseline.get(f"ddSmall{letter}"), 2,
                      f"DNA-Diffusion, top 128 of the smallest pool (512 per cell) by Malinois, calibrated {name}")
        macros.number(f"ddMax{letter}lo", baseline.get(f"ddMax{letter}lo"), 2,
                      f"DNA-Diffusion, top 128 of the largest pool, {name}, lower end of the 95% bootstrap interval")
        macros.number(f"ddMax{letter}hi", baseline.get(f"ddMax{letter}hi"), 2,
                      f"DNA-Diffusion, top 128 of the largest pool, {name}, upper end of the 95% bootstrap interval")
        macros.number(f"ddMalSmall{letter}", baseline.get(f"ddMalSmall{letter}"), 1,
                      f"Malinois median of the selected 128 at the smallest DNA-Diffusion pool, {name} (mean of draws)")
        macros.number(f"ddMalMax{letter}", baseline.get(f"ddMalMax{letter}"), 1,
                      f"Malinois median of the selected 128 at the largest DNA-Diffusion pool, {name}")
    if "ddMaxN" in baseline:
        macros.integer("ddMaxPool", baseline["ddMaxN"], "largest DNA-Diffusion pool per cell in the matched-budget curve")
    else:
        macros._add("ddMaxPool", PENDING, None, "largest DNA-Diffusion pool per cell in the matched-budget curve")
    if "ddNineN" in baseline:
        macros.integer("ddNinePool", baseline["ddNineN"], "smaller of the two DNA-Diffusion pools per cell that Fig. 3b shows")
    else:
        macros._add("ddNinePool", PENDING, None, "smaller of the two DNA-Diffusion pools per cell that Fig. 3b shows")
    for letter, name in (("K", "K562"), ("H", "HepG2")):
        pair = [baseline.get(f"ddNine{letter}"), baseline.get(f"ddMax{letter}")]
        macros.number(f"ddBest{letter}", max(pair) if None not in pair else None, 2,
                      f"DNA-Diffusion, top 128 of the larger of its two pools (the 9,216 and the largest) by Malinois, calibrated {name}")

    comparator_macros(macros, comparators)

    # Loop-seq
    seeds_ls = MAIN_SEEDS["loopseq"]
    for tag, roots, label in (("ls", MAIN_ROOTS["loopseq"], "Loop-seq main-line"),):
        cells = select(records, task="loopseq", roots=roots, arm=MAIN_ARM)
        done = finished(cells)
        complete = is_complete(cells, len(seeds_ls))
        macros.integer(f"{tag}N", len(done), f"{label} cells finished")
        seed_macros(macros, f"{tag}W", cells, seeds_ls, lambda r: r.audit["worst_hit"], HIT_RATE_DECIMALS,
                    f"{label} worst-group in-band hit rate, percent without the sign", 100)
        for letter, group in (("L", "low"), ("M", "mid"), ("H", "high")):
            seed_macros(macros, f"{tag}{letter}", cells, seeds_ls, lambda r, g=group: r.audit[f"hit_{g}"], HIT_RATE_DECIMALS,
                        f"{label} {group}-group in-band hit rate, percent without the sign", 100)
        stat_macros(macros, f"{tag}W", [r.audit["worst_hit"] for r in done], HIT_RATE_DECIMALS,
                    f"{label} worst-group in-band hit rates, percent", complete, 100)
        for letter, group in (("L", "low"), ("M", "mid"), ("H", "high")):
            macros.number(f"{tag}{letter}mean", mean_or_none(r.audit[f"hit_{group}"] for r in done), HIT_RATE_DECIMALS,
                          f"mean {label} {group}-group in-band hit rate over finished seeds, percent", 100, complete)
        cost_macros(macros, tag, cells, label, complete)
        main_block = tag == "ls"
        mechanism_macros(macros, tag, cells, len(seeds_ls), mech, label,
                         ("GateCells", "GateRoundMed", "GateCands", "SpendCells", "PenaltyCells", "OutShare", "Start", "CandProv",
                          "HyperOnly", "LibPairMean", "LibPairMin", "StepsMax", "KlMax") if main_block
                         else ("GateCells", "Start", "LibPairMean"), spend=("mixbend", "mixbend"))
        if main_block:
            loopseq_search_macros(macros)
            refusal_macros(macros, tag, cells, len(seeds_ls), refusals, label)
            lineage_macros(macros, tag, cells, len(seeds_ls), lineage, label)
    for key, desc in (("Sft", "SFT-only"), ("Random", "random DNA"), ("Ga", "genetic-algorithm search"),
                      ("GrpoFourK", "fixed-reward GRPO, 4k steps"), ("GrpoSixteenK", "fixed-reward GRPO, 16k steps"),
                      ("GrpoThirtyTwoK", "fixed-reward GRPO, 32k steps"),
                      ("GrpoSixtyFourK", "fixed-reward GRPO, 64k steps")):
        macros.number(f"lsBase{key}", baseline.get(f"lsBase{key}"), HIT_RATE_DECIMALS,
                      f"Loop-seq baseline worst-group DNAcycP2 in-band rate, {desc}; 512 as-generated sequences per "
                      "group, percent", 100)
    macros.number("lsBaseKl", kl.get("worst"), HIT_RATE_DECIMALS,
                  "Loop-seq KL-0.001 isolation run (outputs/calibration/loopseq_kl_isolation/kl0.001), worst-group "
                  "in-band rate over 512 as-generated sequences per group, percent", 100)

    # Loop-seq margin of the agents' libraries (raw DNAcycP2 coordinates, worst group) and the baselines under the same delivery rule
    for tag, roots, label in (("ls", MAIN_ROOTS["loopseq"], "main-line"),):
        cells = select(records, task="loopseq", roots=roots, arm=MAIN_ARM)
        macros.number(f"{tag}MarginWorst", mean_or_none(r.audit["worst_margin"] for r in finished(cells)), 2,
                      f"Loop-seq {label} agents' libraries: mean over finished cells of the worst-group median signed margin of the delivered "
                      "128 rows to the DNAcycP2 cut points (raw coordinates, positive inside the band)",
                      complete=is_complete(cells, len(seeds_ls)))
    if selected is not None:
        by_suffix = {row["suffix"]: row for row in selected}
        for suffix, row in by_suffix.items():
            what = f"{row['library']} pool of 512 per group under the agents' delivery rule"
            if row["hit_worst"] is None:
                macros.text(f"lsSel{suffix}", "n/a", f"{what}: fewer than 128 rows pass the filters in at least one group, so no worst-group "
                            f"fraction exists ({row['selected_total']} rows selected of 384)")
                macros.integer(f"lsSel{suffix}Rows", row["selected_total"], f"{what}: rows the rule selects (384 needed)")
            else:
                macros.number(f"lsSel{suffix}", row["hit_worst"], HIT_RATE_DECIMALS,
                              f"DNAcycP2 in-band fraction, worst group, of the 128 per group selected from the {what}, percent without the sign", 100)
        if "Grpo" in by_suffix and by_suffix["Grpo"]["margin_worst"] is not None:
            macros.number("lsMarginGrpo", by_suffix["Grpo"]["margin_worst"], 2,
                          "worst-group median signed DNAcycP2 margin of the 128 per group the agents' rule selects from the handtuned_grpo_64k pool")

    # CM
    seeds_cm = MAIN_SEEDS["cm"]
    for tag, roots, label in (("cm", MAIN_ROOTS["cm"], "CM main-line"),
                              ("cmEarlier", EARLIER_ROOTS["cm"], "CM earlier run (Coder not offered, no distance ceiling)")):
        cells = select(records, task="cm", roots=roots, arm=MAIN_ARM)
        done = finished(cells)
        expected = len(seeds_cm) if tag == "cm" else 1  # the earlier run is one search (seed 2401)
        complete = is_complete(cells, expected)
        macros.integer(f"{tag}N", len(done), f"{label} cells finished")
        seed_macros(macros, f"{tag}T", cells, seeds_cm, lambda r: r.audit["t5_median"], 3,
                    f"{label} audit_2020_t5 median of the delivered 128 (primary)")
        seed_macros(macros, f"{tag}E", cells, seeds_cm, lambda r: r.audit["esm_median"], 3,
                    f"{label} audit_2020_esm median of the delivered 128 (secondary)")
        seed_macros(macros, f"{tag}Novelty", cells, seeds_cm,
                    lambda r: r.audit.get("natural_distance_median", float("nan")), 3,
                    f"{label} median normalized edit distance to the nearest natural family member, delivered 128")
        for letter, key, name in (("T", "t5_median", "audit_2020_t5"), ("E", "esm_median", "audit_2020_esm")):
            stat_macros(macros, f"{tag}{letter}", [r.audit[key] for r in done], 3, f"{label} {name} medians over finished cells",
                        complete)
        stat_macros(macros, f"{tag}Novelty", finite(r.audit.get("natural_distance_median") for r in done), 3,
                    f"{label} median normalized edit distance to the family over finished cells", complete)
        cost_macros(macros, tag, cells, label, complete)
        main_block = tag == "cm"
        mechanism_macros(macros, tag, cells, expected, mech, label,
                         ("GateCells", "SpendCells", "OutShare", "Start", "CandProv", "LibPairMean") if main_block
                         else ("GateCells", "Start", "LibPairMean"), spend=("max", "cm_max"))
        if main_block:
            refusal_macros(macros, tag, cells, expected, refusals, label)
            lineage_macros(macros, tag, cells, expected, lineage, label)
            macros.number("cmFuncMed", mean_or_none(r.audit.get("cal_function_2026_esm_median") for r in done), 3,
                          f"{label} median calibrated function_2026_esm of the delivered library (mean of the per-cell medians)",
                          complete=complete)
        macros.number(f"{tag}ShareFar", mean_or_none(r.audit.get("share_distance_above_0p6") for r in done), 1,
                      f"{label} share of delivered sequences with natural distance above 0.6, percent without the sign "
                      "(mean over finished cells)", 100, complete, pct_text)
    bounded = cm_initial.get("bounded") or []
    for key, values, what in (("T", [b["selected_128"]["audit_2020_t5"] for b in bounded], "audit_2020_t5 median (primary)"),
                              ("E", [b["selected_128"]["audit_2020_esm"] for b in bounded], "audit_2020_esm median (secondary)"),
                              ("Dist", [b["selected_128_natural_distance"] for b in bounded], "median normalized edit distance to the family")):
        macros.number(f"cmInitBound{key}", mean_or_none(values), 2,
                      f"CM initial-policy library selected under each main-line cell's own plan (distance 0.20-0.60), mean over the "
                      f"{len(bounded)} cells, {what}")
    protrl = baseline.get("protrl") or []
    for key, field, what in (("T", "t5", "audit_2020_t5 median (primary)"), ("E", "esm", "audit_2020_esm median (secondary)"),
                             ("Dist", "distance", "median normalized edit distance to the nearest natural family member")):
        for bound, function in (("min", min), ("max", max)):
            macros.number(f"protrl{key}{bound}", function(r[field] for r in protrl) if protrl else None, 2,
                          f"ProtRL (official offline GRPO, two seeds, run here), {bound}imum over its {len(protrl)} rescored 128-row "
                          f"libraries (2 seeds x 3 ways of choosing 128) of the {what}")
    macros.integer("protrlN", len(protrl), "ProtRL rescored 128-row libraries (2 seeds x 3 delivery rules)")
    for key, value, what in (("T", cm_initial.get("t5"), "audit_2020_t5 median (primary)"), ("E", cm_initial.get("esm"), "audit_2020_esm median (secondary)"),
                             ("Dist", cm_initial.get("distance"), "median normalized edit distance to the nearest natural family member")):
        macros.number(f"cmInit{key}", value, 3, f"CM initial-policy library (the cell's selection plan replayed on its baseline pool of initial-policy samples, "
                                              f"128 rows), {what}")

    # Critic notes and Coder calls over every main-line cell of the three tasks
    main_cells = [r for task in MAIN_ROOTS for r in select(records, task=task, roots=MAIN_ROOTS[task], arm=MAIN_ARM)]
    ids = {cell_id(r) for r in main_cells}
    rows = [row for row in refusals.to_dict("records") if row["cell_id"] in ids]
    all_complete = is_complete(main_cells, sum(len(v) for v in MAIN_SEEDS.values()))
    macros.provisional("criticNotes", float(sum(row["critic_notes"] for row in rows)) if rows else None, int_text,
                       "Critic advisory events over the main-line cells of all three tasks (finished or running)", all_complete)
    for task, tag in (("cre", "cre"), ("loopseq", "ls"), ("cm", "cm")):
        task_cells = select(records, task=task, roots=MAIN_ROOTS[task], arm=MAIN_ARM)
        task_ids = {cell_id(r) for r in task_cells}
        task_rows = [row for row in refusals.to_dict("records") if row["cell_id"] in task_ids]
        macros.provisional(f"{tag}CoderCalls", float(sum(row["coder_calls"] for row in task_rows)) if task_rows else None, int_text,
                           f"Coder (component synthesis) LLM calls over the {task} main-line cells (finished or running)",
                           is_complete(task_cells, len(MAIN_SEEDS[task])))
    macros.provisional("coderCalls", float(sum(row["coder_calls"] for row in rows)) if rows else None, int_text,
                       "Coder (component synthesis) LLM calls over the main-line cells of all three tasks",
                       all_complete)
    fixed_recipe_macros(macros, fixed)
    anchor_macros(macros, anchor)
    frontier_macros(macros)
    edit_macros(macros)
    endpoint_macros(macros)
    nosearch_macros(macros)
    enhancer_search_macros(macros)
    chorismate_search_macros(macros)
    return macros


def fixed_recipe_macros(macros: Macros, fixed: pd.DataFrame) -> None:
    """Per-library, mean and range of the calibrated medians of the fixed recipe (K562 and HepG2), their size and updates."""
    for tag, variant in FIXED_VARIANTS:
        for condition, letter in (("K562", "K"), ("HepG2", "H")):
            part = fixed[(fixed.recipe == variant) & (fixed.condition == condition)].sort_values("seed")
            medians = list(part.cal_median)
            what = f"calibrated {condition} medians of the fixed {variant} recipe's libraries (one per training seed)"
            for seed_letter, seed in zip(LETTERS, FIXED_SEEDS):
                value = part[part.seed == seed].cal_median
                macros.number(f"fix{tag}{letter}{seed_letter}", float(value.iloc[0]) if len(value) else None, 2,
                              f"fixed {variant} recipe, calibrated {condition} median of the library of training seed {seed}")
            stat_macros(macros, f"fix{tag}{letter}", medians, 2, what, len(medians) == len(FIXED_SEEDS))
    macros.integer("fixSeeds", len(FIXED_SEEDS), "training seeds of each fixed recipe")
    sizes = sorted(fixed.n.unique()) if len(fixed) else []
    macros.number("fixN", float(sizes[0]) if len(sizes) == 1 else None, 0, "sequences per cell line in each fixed-recipe library")
    for tag, variant in FIXED_VARIANTS:
        manifests = [read_json(REPO / FIXED_RECIPES / variant / f"seed_{seed}" / "run_manifest.json") or {} for seed in FIXED_SEEDS]
        steps = {m.get("checkpoint_rule") for m in manifests}
        rule = steps.pop() if len(steps) == 1 else None
        macros.number("fixUpdates" if tag == "Multi" else f"fix{tag}Updates",
                      float(rule.rsplit("_", 1)[1]) if rule and rule.startswith("fixed_final_step_") else None, 0,
                      f"optimizer updates of each run of the fixed {variant} recipe (the fixed final step of run_manifest.json)")


def anchor_macros(macros: Macros, anchor: pd.DataFrame) -> None:
    """The library of the declared initial recipe (the anchor candidate), per seed, mean and range, and the gain of the
    delivered library over it, for the three tasks. A seed whose anchor audit is incomplete is `\\pending`."""
    def pick(task: str, source: str, statistic: str, condition: str) -> pd.DataFrame:
        part = anchor[(anchor.task == task) & (anchor.source == source) & (anchor.statistic == statistic)
                      & (anchor.condition == condition)]
        return part.sort_values("seed")

    def per_seed(prefix: str, task: str, part: pd.DataFrame, what: str, digits: int, scale: float = 1.0,
                 column: str = "anchor", fmt: Callable[[float], str] | None = None) -> None:
        expected = MAIN_SEEDS[task]
        for letter, seed in zip(LETTERS, expected):
            value = part[part.seed == seed][column]
            macros.number(f"{prefix}{letter}", float(value.iloc[0]) if len(value) else None, digits,
                          f"{what}, seed {seed}", scale, fmt=fmt)
        stat_macros(macros, prefix, [float(v) for v in part[column]], digits, f"{what} over seeds", len(part) == len(expected),
                    scale)

    for condition, letter in (("K562", "K"), ("HepG2", "H")):
        part = pick("cre", "chrombpnet", "median", condition)
        per_seed(f"creAnch{letter}", "cre", part, f"CRE initial-recipe library, calibrated {condition} median", 2)
        gain = part.assign(gain=part.delivered - part.anchor)
        macros.number(f"creAnch{letter}gain", float(gain.gain.mean()) if len(gain) else None, 2,
                      f"mean over seeds of the delivered library's calibrated {condition} median minus the initial recipe's",
                      complete=len(part) == len(MAIN_SEEDS["cre"]))
    for letter, group in (("W", "worst"), ("L", "low"), ("M", "mid"), ("H", "high")):
        part = pick("loopseq", "dnacycp2", "in_band", group)
        per_seed(f"lsAnch{letter}", "loopseq", part, f"Loop-seq initial-recipe library, {group}-group DNAcycP2 in-band fraction, percent",
                 HIT_RATE_DECIMALS, 100)
    rows = anchor[(anchor.task == "loopseq") & (anchor.source == "dnacycp2") & (anchor.statistic == "in_band")
                  & (anchor.condition == "high")].sort_values("seed")
    for letter, seed in zip(LETTERS, MAIN_SEEDS["loopseq"]):
        value = rows[rows.seed == seed].anchor_rows
        macros.number(f"lsAnchRows{letter}", float(value.iloc[0]) if len(value) else None, 0,
                      f"sequences the delivery plan selects for the high group from the initial recipe's pool (of 128), seed {seed}")
    for letter, source, what in (("T", "audit_2020_t5", "ProtT5"), ("E", "audit_2020_esm", "ESM2")):
        part = pick("cm", source, "median", "CM")
        per_seed(f"cmAnch{letter}", "cm", part, f"CM initial-recipe library, reserved {what} median", 2)
    part = pick("cm", "natural_distance", "median", "CM")
    per_seed("cmAnchDist", "cm", part, "CM initial-recipe library, median natural distance", 2)


# ----- CM predicted score at matched distance --------------------------------------------------------

FRONTIER_METHODS = {"RECITAL": "RD", "Initial recipe": "Anch", "Initial policy": "Pol", "ProtRL": "Prl",
                    "Russ 2020": "Russ", "Netti 2026": "Netti"}
FRONTIER_BINS = {"A": "0.30-0.40", "B": "0.40-0.50", "C": "0.50-0.60"}


def frontier_macros(macros: Macros) -> None:
    """Median reserved ProtT5 of each method's sequences in each distance bin with the 95% bootstrap interval over
    sequences and the number of sequences (`cmFront<method><bin>`, `...lo`, `...hi`, `...n`), the median reserved ESM2 of the
    sequences of the four methods that have it in the bins B and C (`cmFrontEsm<method><bin>`), and per search the delivered
    library's median minus the initial recipe's in the same bin (`cmFrontDiff<bin><seed letter>`). A method with fewer
    than `cm_frontier.MIN_ROWS` sequences in a bin is `\\pending`."""
    from analysis import cm_frontier

    macros.integer("cmFrontMinRows", cm_frontier.MIN_ROWS,
                   "sequences a library needs in a natural-distance bin to be summarized there")
    macros.integer("cmFrontBoot", cm_frontier.N_BOOT, "bootstrap resamples of the sequences of a library in a bin")
    frame = cm_frontier.sequences()
    table = cm_frontier.binned(frame)
    pooled = table[table.unit == -1]
    esm = cm_frontier.binned(frame, score=cm_frontier.ESM)  # the published libraries have no ESM2 score
    esm = esm[esm.unit == -1]
    for method in ("RECITAL", "Initial recipe", "Initial policy", "ProtRL"):
        for letter in ("B", "C"):
            row = esm[(esm.method == method) & (esm.bin == FRONTIER_BINS[letter])]
            macros.number(f"cmFrontEsm{FRONTIER_METHODS[method]}{letter}", float(row.iloc[0]["median"]) if len(row) else None, 2,
                          f"median reserved ESM2 of {method} sequences in the {FRONTIER_BINS[letter]} natural-distance bin")
    dist = cm_frontier.distribution(frame, cm_frontier.binned(frame, open_ends=True))
    held = dist[(dist.method == "ProtRL") & (dist.bin == FRONTIER_BINS["C"])]
    macros.number("cmFrontPrlShareC", 100 * float(held.iloc[0].share), 0,
                  f"percent of the distinct ProtRL sequences in the {FRONTIER_BINS['C']} natural-distance bin")
    for method, tag in FRONTIER_METHODS.items():
        for letter, label in FRONTIER_BINS.items():
            row = pooled[(pooled.method == method) & (pooled.bin == label)]
            what = f"{method} sequences in the {label} natural-distance bin"
            for suffix, column, digits in (("", "median", 2), ("lo", "lo", 2), ("hi", "hi", 2), ("n", "n", 0)):
                value = float(row.iloc[0][column]) if len(row) else None
                macros.number(f"cmFront{tag}{letter}{suffix}", value, digits,
                              f"median reserved ProtT5 of {what}" if not suffix else
                              {"lo": f"lower 95% bootstrap bound of the median reserved ProtT5 of {what}",
                               "hi": f"upper 95% bootstrap bound of the median reserved ProtT5 of {what}",
                               "n": f"number of {what}"}[suffix])
    units = table[table.unit != -1]
    for letter, label in FRONTIER_BINS.items():
        for seed_letter, seed in zip(LETTERS, MAIN_SEEDS["cm"]):
            def median(method: str) -> float | None:
                row = units[(units.method == method) & (units.unit == seed) & (units.bin == label) & units.enough]
                return float(row.iloc[0]["median"]) if len(row) else None

            delivered, anchor = median("RECITAL"), median("Initial recipe")
            macros.number(f"cmFrontDiff{letter}{seed_letter}", None if delivered is None or anchor is None
                          else delivered - anchor, 2,
                          f"delivered library's median reserved ProtT5 minus the initial recipe's, sequences in the {label} "
                          f"natural-distance bin, seed {seed}")


# ----- what the agent changed ------------------------------------------------------------------------

EDIT_TASKS = {"cre": "Cre", "loopseq": "Ls", "cm": "Cm"}
EDIT_CLASS_TAGS = {"source_on": "SrcOn", "source_off": "SrcOff", "weights": "Weights", "penalty": "Penalty", "gate": "Gate",
                   "kl": "Kl", "lr": "Lr", "steps": "Steps", "group_size": "Group", "start": "Start"}
EDIT_CLASS_WORDS = {"source_on": "enabled a reward source", "source_off": "disabled a reward source",
                    "weights": "changed aggregator weights or a per-source utility", "penalty": "added or changed a penalty",
                    "gate": "added or changed a gate", "kl": "changed the KL coefficient", "lr": "changed the learning rate",
                    "steps": "changed the training length", "group_size": "changed the group size",
                    "start": "restarted from a parent's checkpoint"}


def edit_macros(macros: Macros) -> None:
    """The recipe edits of the nine main-line searches (`scripts/analysis/agent_edits.py`): candidates proposed by the agent per
    task, the share of them carrying each edit class, the round and searches of the first gate, the change in the number of
    composition measures outside the initial policy's band against the base candidate with and without a gate, the
    qualification of candidates against their base, and the chorismate-mutase delivered distance against the initial recipe's."""
    from analysis import agent_edits

    candidates, _ = agent_edits.build()
    agent = agent_edits.agent_rows(candidates)
    macros.integer("edNAgentAll", len(agent), "candidates the Strategist proposed beyond the declared initial recipe, nine searches")
    for task, tag in EDIT_TASKS.items():
        part = agent[agent.task == task]
        macros.integer(f"edNAgent{tag}", len(part), f"candidates the Strategist proposed in the {task} searches")
        for name, short in EDIT_CLASS_TAGS.items():
            macros.number(f"ed{short}{tag}", float(part[name].mean()), 0, f"share (percent) of the {task} agent-proposed candidates that "
                          f"{EDIT_CLASS_WORDS[name]} against their base", 100)
        gated = part[part.gate]
        macros.integer(f"edGateFirst{tag}", int(gated["round"].min()) if len(gated) else 0, f"first round with a gate edit, {task} searches")
        macros.integer(f"edGateSearches{tag}", int(gated.seed.nunique()), f"{task} searches with at least one gate edit (of three)")
        macros.integer(f"edGateN{tag}", len(gated), f"{task} agent-proposed candidates with a gate edit")
        if task != "cm":
            measured = part[part.comp_outside_delta.notna()]
            for flag, word in ((True, ""), (False, "No")):
                delta = measured[measured.gate == flag].comp_outside_delta
                macros.number(f"ed{word}GateBand{tag}", float(delta.median()) if len(delta) else None, 0,
                              f"median change in composition measures outside the initial policy's band, candidate minus base, "
                              f"{task} candidates {'with' if flag else 'without'} a gate edit", fmt=lambda v: f"{v:+.0f}".replace("+", "").replace("-", "$-$"))
        anchors = candidates[(candidates.task == task) & candidates.anchor.astype(bool)]
        macros.integer(f"edAnchQ{tag}", int((anchors.qualified == True).sum()), f"{task} searches whose initial recipe's library qualified (of three)")  # noqa: E712
        macros.integer(f"edEntered{tag}", int(part.entered.sum()), f"{task} candidates that qualified when their base did not")
        macros.integer(f"edEnteredGate{tag}", int((part.entered & part.gate).sum()), f"{task} candidates that entered qualification and carry a gate edit")
        macros.integer(f"edLeft{tag}", int(part.left.sum()), f"{task} candidates that did not qualify when their base did")
        macros.integer(f"edLeftStart{tag}", int((part.left & part.start).sum()), f"{task} candidates that left qualification and restarted from a parent's checkpoint")
        macros.integer(f"edUnq{tag}", int((part.qualified == False).sum()), f"{task} agent-proposed candidates whose library did not qualify")  # noqa: E712
    cm = agent[(agent.task == "cm") & (agent.qualified == False)]  # noqa: E712
    macros.integer("edCmNoLib", int(cm.failing.str.contains("complete_delivery").sum()),
                   "chorismate mutase candidates whose pool held no 128 qualifying sequences within the distance bound")
    macros.integer("edCmHealth", int((~cm.failing.str.contains("complete_delivery")).sum()),
                   "chorismate mutase candidates that failed only the library-health floor")
    cm_all = candidates[candidates.task == "cm"]
    for letter, seed in zip(LETTERS, MAIN_SEEDS["cm"]):
        part = cm_all[cm_all.seed == seed]
        anchor, delivered = part[part.anchor.astype(bool)].iloc[0], part[part.delivered.astype(bool)].iloc[0]
        macros.number(f"edCmAnchDist{letter}", float(anchor.distance_library), 2, f"median natural distance of the initial recipe's library, seed {seed}")
        macros.number(f"edCmDelDist{letter}", float(delivered.distance_library), 2, f"median natural distance of the delivered library, seed {seed}")
        macros.number(f"edCmOff{letter}", float(agent[(agent.task == "cm") & (agent.seed == seed)].source_off.mean()), 0,
                      f"share (percent) of the agent-proposed candidates of search {seed} that disabled a reward source", 100)


# ----- CM endpoints outside every search's reward, ranking and delivery ------------------------------

ENDPOINT_TAGS = {"poet2": ("Poet", 2, "PoET-2 log-likelihood per residue"), "plddt": ("Plddt", 1, "ESMFold pLDDT"),
                 "tm_experimental": ("Tm", 2, "TM-score to experimental AroQ structures"),
                 "retention": ("Cons", 2, "fraction of the conserved family positions that carry the consensus residue")}
ENDPOINT_METHODS = {"RECITAL": "RD", "Initial recipe": "Anch", "Initial policy": "Pol", "ProtRL": "Prl",
                    "Russ 2020": "Russ", "Netti 2026": "Netti", "Natural family": "Nat"}


def endpoint_macros(macros: Macros) -> None:
    """Median of each endpoint of `scripts/analysis/cm_endpoints.py` by method and distance bin (`cmEp<endpoint><method><bin>`,
    bins A, B, C as for `cmFront*`), the bootstrap interval of the delivered libraries and the number of sequences in each cell
    (`cmEpN<method><bin>`). A quantity whose scorer has not run is `\\pending`."""
    from analysis import cm_endpoints

    folder = cm_endpoints.ENDPOINT_DIR
    if not (folder / "library_rows.csv").is_file():
        return
    frame = cm_endpoints.library_endpoints(folder)
    table = cm_endpoints.binned(frame)
    pooled = table[table.unit == -1]
    for endpoint, (tag, digits, what) in ENDPOINT_TAGS.items():
        part = pooled[pooled.endpoint == endpoint] if len(pooled) else pooled
        for method, mtag in ENDPOINT_METHODS.items():
            for letter, label in FRONTIER_BINS.items():
                row = part[(part.method == method) & (part.bin == label)] if len(part) else part
                where = f"{method} sequences in the {label} natural-distance bin"
                macros.number(f"cmEp{tag}{mtag}{letter}", float(row.iloc[0]["median"]) if len(row) else None, digits,
                              f"median {what} of {where}")
                if method in ("RECITAL", "Natural family"):
                    for suffix, column in (("lo", "lo"), ("hi", "hi")):
                        macros.number(f"cmEp{tag}{mtag}{letter}{suffix}", float(row.iloc[0][column]) if len(row) else None, digits,
                                      f"{'lower' if suffix == 'lo' else 'upper'} 95% bootstrap bound of the median {what} of {where}")
    reference = pooled[pooled.endpoint == "retention"] if len(pooled) else pooled
    for method, mtag in ENDPOINT_METHODS.items():
        for letter, label in FRONTIER_BINS.items():
            row = reference[(reference.method == method) & (reference.bin == label)] if len(reference) else reference
            macros.number(f"cmEpN{mtag}{letter}", float(row.iloc[0]["n"]) if len(row) else None, 0,
                          f"sequences of {method} in the {label} natural-distance bin that enter the endpoint medians")
    natural = frame[frame.method == "Natural family"]
    macros.integer("cmEpNatN", len(natural), "natural family sequences scored as a reference")
    macros.number("cmEpNatDist", float(natural.natural_distance.median()), 2,
                  "median distance of the reference natural sequences to the nearest other family sequence")
    macros.integer("cmEpConsCols", int(frame.n_conserved.dropna().iloc[0]) if "n_conserved" in frame and frame.n_conserved.notna().any() else 0,
                   "strongly conserved alignment columns (consensus residue in at least half of the family sequences)")
    macros.integer("cmEpConsStrict", int(frame.n_conserved_strict.dropna().iloc[0]) if "n_conserved_strict" in frame and frame.n_conserved_strict.notna().any() else 0,
                   "alignment columns with the consensus residue in at least 70% of the family sequences")
    references = folder / "reference_structures.csv"
    macros.integer("cmEpRefStructures", len(pd.read_csv(references)) if references.is_file() else 0,
                   "experimental AroQ chorismate-mutase structures used as TM-score references")
    request = folder / "poet2_request.json"
    macros.integer("cmEpPrompt", len(json.loads(request.read_text())["context"]) if request.is_file() else 0,
                   "natural sequences in the PoET-2 prompt")


def enhancer_search_macros(macros: Macros) -> None:
    """Descriptives of the enhancer searches read from the candidate table of Fig. 3 (`figures/fig3/data/cre/trajectory.csv`, main
    arm): the ratio of the delivered libraries' Malinois gap to the initial recipe's, per cell line, mean over the searches."""
    path = REPO / "scripts" / "fig3" / "data" / "cre" / "trajectory.csv"
    frame = pd.read_csv(path) if path.is_file() else None
    ratios: dict[str, float | None] = {"K": None, "H": None}
    if frame is not None:
        frame = frame[frame.arm == "no_self_validation"]
        for letter, cell_line in (("K", "K562"), ("H", "HepG2")):
            rows = frame[frame.condition == cell_line]
            ratios[letter] = float(rows[rows.delivered].lib_malinois_gap.mean() / rows[rows.anchor].lib_malinois_gap.mean())
    if frame is not None and not all(1.8 <= ratio <= 2.3 for ratio in ratios.values()):
        raise SystemExit("Results 2 says the delivered libraries had about twice the Malinois gap of the initial recipe's; the table says otherwise")
    init_steps = None if frame is None else float(frame[frame.anchor].steps.mean())
    macros.number("creInitSteps", init_steps, 0, "CRE main line: training updates of the declared initial recipe (one candidate per search)")
    for letter, cell_line in (("K", "K562"), ("H", "HepG2")):
        macros.number(f"creMalRatio{letter}", ratios[letter], 1,
                      f"CRE main line: mean Malinois gap of the delivered libraries over that of the initial recipes' libraries, {cell_line}")
    chains = None if frame is None else delivered_chains(frame)
    macros.number("creChainMin", None if chains is None else float(chains.candidates.min()), 0,
                  "CRE main line: fewest candidates in the chain of a delivered checkpoint (the candidate and its parents back to "
                  "the last restart from the initial policy)")
    macros.number("creChainMax", None if chains is None else float(chains.candidates.max()), 0,
                  "CRE main line: most candidates in the chain of a delivered checkpoint")
    macros.number("creLineageMin", None if chains is None else float(chains.updates.min()), 0,
                  "CRE main line: fewest training updates along the chain of a delivered checkpoint")
    macros.number("creLineageMax", None if chains is None else float(chains.updates.max()), 0,
                  "CRE main line: most training updates along the chain of a delivered checkpoint")
    per_search = None if frame is None else frame.drop_duplicates(["seed", "candidate"]).groupby("seed").steps.sum()
    macros.number("creSearchUpdates", None if per_search is None else float(per_search.iloc[0]), 0,
                  "CRE main line: training updates summed over all candidates of one search (equal in the three searches)")


def delivered_chains(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per delivered candidate of the main enhancer arm: the candidates on its chain of parents (back to the last restart
    from the initial policy), their updates and the updates with the motif score in the reward. Also checks the statements of
    Results 2 that the table has to bear out: the chain updates are the engine's lineage updates, the three searches trained
    the same number of updates, the motif score was in the reward for at least half of every chain, two searches delivered a
    recipe with a gate, and every candidate had a composition measure outside the initial policy's range in the pool of a cell."""
    unique = frame.drop_duplicates(["seed", "candidate"]).set_index(["seed", "candidate"])
    if unique.groupby("seed").steps.sum().nunique() != 1:
        raise SystemExit("Results 2 says the three enhancer searches trained the same number of updates; they did not")
    rows = []
    for seed, name in frame[frame.delivered].drop_duplicates(["seed", "candidate"])[["seed", "candidate"]].itertuples(index=False):
        count = updates = motif = 0
        current = name
        while isinstance(current, str):
            row = unique.loc[(seed, current)]
            count += 1
            updates += int(row.steps)
            motif += int(row.steps) * bool(row.grammar_prior_enabled)
            current = None if row.start == "from_initial" else row.parent
        if updates != int(unique.loc[(seed, name)].lineage_updates):
            raise SystemExit(f"seed {seed} {name}: the chain's updates ({updates}) differ from the lineage updates of the table")
        rows.append({"seed": seed, "candidate": name, "candidates": count, "updates": updates, "motif_updates": motif,
                     "gate": bool(unique.loc[(seed, name)].has_gate)})
    chains = pd.DataFrame(rows)
    if not (chains.motif_updates >= chains.updates / 2).all():
        raise SystemExit("Results 2 says the motif score was in the reward for at least half of the updates of every delivered chain")
    if chains[chains.gate].seed.nunique() != 2:
        raise SystemExit("Results 2 says two searches added gates on composition")
    if not (frame.groupby(["seed", "candidate"]).n_outside_band.max() >= 1).all():
        raise SystemExit("Results 2 says every trained candidate had a composition measure outside the initial policy's range")
    return chains


def chorismate_search_macros(macros: Macros) -> None:
    """Descriptives of the chorismate mutase searches read from the candidate table of Fig. 4 (`figures/fig4/data/cm_candidates.csv`):
    how many candidates qualified, how many had no library inside the distance bound or a library below the diversity floor,
    the visible function score of the two groups, and the edit counts of the Strategist's candidates."""
    path = REPO / "scripts" / "fig4" / "data" / "cm_candidates.csv"
    if not path.is_file():
        frame = None
    else:
        frame = pd.read_csv(path)
        failing = frame.failing.fillna("")
        frame = frame.assign(
            no_library=~frame.qualified & failing.str.contains("complete_delivery"),
            collapsed=~frame.qualified & ~failing.str.contains("complete_delivery"))
        agent = frame[frame.anchor != True]  # noqa: E712
        qualified, collapsed, no_library = frame[frame.qualified], frame[frame.collapsed], frame[frame.no_library]
        dca_off = sum(1 for _, rows in agent.groupby("seed") if (~rows.enabled.str.contains("cm_max")).mean() > 0.5)
        if not collapsed.library_function_2026_esm.mean() > qualified.library_function_2026_esm.mean():
            raise SystemExit("Results 3 says the candidates below the diversity floor had a higher visible function score than the "
                             "qualified ones; the table says otherwise")
    for name, value, digits, description in (
            ("cmCandN", None if frame is None else float(len(frame)), 0, "candidates of the three CM searches"),
            ("cmQualN", None if frame is None else float(frame.qualified.sum()), 0, "of those, the qualified ones"),
            ("cmNoLibN", None if frame is None else float(len(no_library)), 0, "candidates whose pool held fewer than 128 sequences inside the distance bound"),
            ("cmNoLibPoolLo", None if frame is None else float(no_library.pool_natural_distance.min()), 2,
             "lowest median natural distance of the pool among them"),
            ("cmNoLibPoolHi", None if frame is None else float(no_library.pool_natural_distance.max()), 2,
             "highest median natural distance of the pool among them"),
            ("cmCollapsedN", None if frame is None else float(len(collapsed)), 0,
             "candidates with a library below the diversity floor and no other failing check"),
            ("cmFnCollapsed", None if frame is None else float(collapsed.library_function_2026_esm.mean()), 2,
             "mean visible function score of the libraries of those candidates"),
            ("cmFnQualified", None if frame is None else float(qualified.library_function_2026_esm.mean()), 2,
             "mean visible function score of the libraries of the qualified candidates"),
            ("cmNonAnchorN", None if frame is None else float(len(agent)), 0, "candidates the Strategist proposed in the CM searches"),
            ("cmAggN", None if frame is None else float(agent.aggregator_changed.sum()), 0,
             "of those, the ones whose reward weights differ from the initial recipe's"),
            ("cmKlRaisedN", None if frame is None else float((agent.kl_dir > 0).sum()), 0,
             "of those, the ones whose KL coefficient is above the initial recipe's"),
            ("cmDcaOffSearches", None if frame is None else float(dca_off), 0,
             "searches in which most of the Strategist's candidates dropped the DCA prior")):
        macros.number(name, value, digits, f"CM main line: {description}")


def nosearch_macros(macros: Macros) -> None:
    """The CM no-search control (`scripts/analysis/cm_nosearch_table.py`): its cells, how many delivered a qualified library, the
    cells whose library failed the health floor with the range of its pairwise-distance ratio and of the median distance of the
    library, and the cells with no row inside the distance bound with the median distance of their pool. A quantity whose cells
    have not finished is `\\pending`."""
    from analysis import cm_nosearch_table
    from common.control_cells import NOSEARCH_ROOT, SCRATCH_ROOT

    table = cm_nosearch_table.build()
    done = table[table.status != "running"] if len(table) else table
    library = done[done.library_rows > 0] if len(done) else done
    empty = done[done.library_rows == 0] if len(done) else done
    macros.integer("cmNosCells", len(table), "cells of the CM no-search control (the declared initial recipe trained once)")
    macros.integer("cmNosQualified", int(table.qualified.sum()) if len(table) else 0,
                   "no-search CM cells whose candidate qualified")
    macros.integer("cmNosHealthCells", len(library), "no-search CM cells that selected 128 rows and failed the library-health floor")
    macros.integer("cmNosEmptyCells", len(empty), "no-search CM cells with no row inside the delivery bound")
    floor = None
    if len(table):
        floor = yaml.safe_load((SCRATCH_ROOT / NOSEARCH_ROOT / table.cell.iloc[0] / "config.yaml").read_text())["control"]["health"]["min_pairwise_ratio"]
    macros.number("cmNosFloor", floor, 2, "library-health floor on the pairwise-distance ratio of a library to the initial policy's pool")
    for name, column, frame, what in (
            ("cmNosRatio", "library_pairwise_ratio", library, "pairwise-distance ratio of the 128-row library of a no-search CM cell"),
            ("cmNosDist", "library_natural_distance", library, "median natural distance of the 128-row library of a no-search CM cell"),
            ("cmNosPool", "pool_natural_distance", empty, "median natural distance of the pool of a no-search CM cell with no row inside the bound")):
        for suffix, reduce in (("Min", "min"), ("Max", "max")):
            macros.number(f"{name}{suffix}", float(getattr(frame[column], reduce)()) if len(frame) else None, 2,
                          f"{'smallest' if reduce == 'min' else 'largest'} {what}")


def write_numbers_tex(path: Path, macros: Macros, now: dt.datetime) -> None:
    lines = ["% Generated by scripts/analysis/paper_numbers.py at " + now.strftime("%Y-%m-%d %H:%M UTC") + ". Do not edit.",
             "% Every value is recomputed from the campaign cells and baseline files; a quantity with no finished",
             "% cell behind it is \\pending{--} (define \\pending in the preamble). Meanings: docs/reports/PAPER_FACTS_20261008.md."]
    for item in macros.items:
        lines.append(f"\\def\\{item['name']}{{{item['tex']}}}% {item['description']}")
    atomic_write(path, "\n".join(lines) + "\n")


# ----- mechanism descriptives ------------------------------------------------------------------------

# Does a hypothesis cite the composition checks? Case-insensitive; one pattern per keyword so each count can be audited.
# `gc`, `band` and `k-mer` must stand as words (letters and digits may not touch them; `_`, `-` and spaces may), so
# `gc_bounds_gate`, `GC-content` and `k-mer` match while `bandwidth`, `gcn` and `ckmer` do not.
COMPOSITION_KEYWORDS = {
    "composition": re.compile(r"composition", re.I),
    "entropy": re.compile(r"entropy", re.I),
    "gc": re.compile(r"(?<![a-z0-9])gc(?![a-z0-9])", re.I),
    "homopolymer": re.compile(r"homopolymer", re.I),
    "low-complexity": re.compile(r"low[\s_-]*complexity", re.I),
    "diversity": re.compile(r"diversity", re.I),
    "band": re.compile(r"(?<![a-z0-9])bands?(?![a-z0-9])", re.I),
    "k-mer": re.compile(r"(?<![a-z0-9])k-?mers?(?![a-z0-9])", re.I),
}


def cited_keywords(text: str | None) -> list[str]:
    return [name for name, pattern in COMPOSITION_KEYWORDS.items() if pattern.search(text or "")]


def gate_change(parent_gates: list[dict[str, Any]], gates: list[dict[str, Any]]) -> tuple[bool, bool]:
    """(adds a gate, tightens a gate) of a recipe against its parent's gates.

    A gate is added when its name is not among the parent's. A gate present in both is tightened when one of its `min*`
    parameters is raised or one of its `max*` parameters is lowered; a gate that is only relaxed or dropped counts for
    neither. The parent of a candidate that starts `from_initial` with no parent is the declared initial recipe, which
    carries no gate, the same base the ledger row's `edits` are written against.
    """
    before = {gate["name"]: gate for gate in parent_gates or []}
    added = tightened = False
    for gate in gates or []:
        old = before.get(gate["name"])
        if old is None:
            added = True
            continue
        for key, value in gate.items():
            if key == "name" or key not in old or not isinstance(value, (int, float)):
                continue
            if (key.startswith("min") and value > old[key]) or (key.startswith("max") and value < old[key]):
                tightened = True
    return added, tightened


def candidate_rows(record: SimpleNamespace) -> list[dict[str, Any]]:
    view, conditions = record.cell.view, record.cell.conditions
    accepted = view.state.stop.accepted
    delivered = set(accepted.assignment.values()) if accepted else set()
    initial_gates = (((view.declaration() or {}).get("contract") or {}).get("initial_recipe") or {}).get("gates") or []
    rows = []
    for candidate in view.state.ordered_candidates():
        row: dict[str, Any] = {
            "root": record.root, "cell": record.name, "cell_id": cell_id(record), "arm": record.arm,
            "seed": record.seed, "cell_finished": record.finished, "candidate": candidate.id, "ordinal": candidate.ordinal,
            "round": candidate.round, "slot": candidate.slot, "parent": candidate.parent, "start": candidate.start,
            "anchor": candidate.anchor, "status": candidate.status, "delivered": candidate.id in delivered}
        spec = view.spec(candidate.id) if candidate.recipe_sha else None
        ledger = view.ledger_row(candidate.id) if candidate.ledger_row_sha else None
        row["has_spec"] = spec is not None
        if spec is not None:
            recipe = spec["recipe"]
            hyper = recipe.get("hyperparameters") or {}
            aggregator = recipe.get("aggregator")
            row.update(
                hypothesis=spec.get("hypothesis"), lr=hyper.get("lr"), kl=hyper.get("kl"),
                group_size=hyper.get("group_size"), steps=(spec.get("train") or {}).get("additional_updates"),
                enabled_oracles="+".join(recipe.get("enabled_oracles") or []),
                held_out_oracles="+".join(recipe.get("held_out_oracles") or []),
                aggregator=(aggregator or {}).get("name") if isinstance(aggregator, dict) else aggregator,
                n_gates=len(recipe.get("gates") or []),
                gates="+".join(g.get("name", "?") for g in recipe.get("gates") or []),
                penalties="+".join(p.get("name", "?") for p in recipe.get("penalties") or []),
                utilities="+".join(f"{k}:{(v or {}).get('name')}" for k, v in
                                   sorted((recipe.get("oracle_utilities") or {}).items())))
            parent = view.state.candidates.get(candidate.parent) if candidate.parent not in (None, "initial") else None
            parent_gates = (view.spec(parent.id)["recipe"].get("gates") or []) if parent is not None and parent.recipe_sha \
                else initial_gates
            added, tightened = gate_change(parent_gates, recipe.get("gates") or [])
            row.update(gate_added=added, gate_tightened=tightened, gate_change=added or tightened)
            keywords = cited_keywords(spec.get("hypothesis"))
            row.update(hypothesis_keywords=",".join(keywords), hypothesis_cites_composition=bool(keywords))
        if ledger is not None:
            edits = list(ledger.get("edits") or [])
            row["edits"] = " | ".join(edits)
            row["edit_keys"] = ",".join(sorted({edit.split(":", 1)[0] for edit in edits}))
            composition = ledger.get("composition")
            row["composition_recorded"] = composition is not None
            if composition is not None:
                counts = {c: len(composition.get(c) or {}) for c in conditions}
                row["comp_outside"] = "|".join(f"{c}:{n}" for c, n in counts.items())
                row["comp_pairs_total"] = len(counts)
                row["comp_pairs_outside"] = sum(n > 0 for n in counts.values())
                row["comp_measurements_outside"] = sum(counts.values())
                row["comp_any_outside"] = any(n > 0 for n in counts.values())
            library = ledger.get("library") or {}
            ratios = [(library.get(c) or {}).get("pairwise_ratio") for c in conditions]
            row["library_pairwise_ratio_min"] = min(finite(ratios)) if finite(ratios) else None
            training = ledger.get("training") or {}
            row["wall_h"] = training.get("wall_h")
            qualified = ledger.get("qualified")
            row["qualified"] = None if not qualified else all(v["qualified"] for v in qualified.values())
        rows.append(row)
    return rows


def mechanism_cell(record: SimpleNamespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    allocated = [r for r in rows if r["has_spec"]]
    agent = [r for r in allocated if not r["anchor"]]
    ok = [r for r in allocated if r["status"] == "ok"]
    gated = [r for r in agent if r.get("n_gates")]
    with_comp = [r for r in ok if r.get("composition_recorded")]
    keys = Counter(key for r in agent for key in (r.get("edit_keys") or "").split(",") if key)
    hyper_only = [r for r in agent if r.get("edit_keys") and set(r["edit_keys"].split(",")) <= HYPER_KEYS]
    unchanged = [r for r in agent if not r.get("edit_keys")]
    anchor = next((r for r in allocated if r["anchor"]), None)
    out: dict[str, Any] = {
        "root": record.root, "cell": record.name, "cell_id": cell_id(record), "arm": record.arm,
        "seed": record.seed, "cell_finished": record.finished, "status": record.status,
        "candidates_allocated": len(allocated), "candidates_trained": len(ok), "agent_candidates": len(agent),
        "rounds_with_candidates": record.summary["rounds"]["allocated"],
        "first_gate_round": min((r["round"] for r in gated), default=None),
        "candidates_with_gate": len(gated),
        "gate_candidates": sum(bool(r.get("gate_change")) for r in agent),
        "gate_cited": sum(bool(r.get("gate_change")) and bool(r.get("hypothesis_cites_composition")) for r in agent),
        "gate_names": counter_text(g for r in gated for g in r["gates"].split("+")),
        "composition_recorded": len(with_comp),
        "share_candidates_any_outside": (sum(r["comp_any_outside"] for r in with_comp) / len(with_comp)
                                         if with_comp else None),
        "share_candidate_conditions_outside": (sum(r["comp_pairs_outside"] for r in with_comp)
                                               / sum(r["comp_pairs_total"] for r in with_comp) if with_comp else None),
        "mean_measurements_outside_per_candidate_condition": (
            sum(r["comp_measurements_outside"] for r in with_comp) / sum(r["comp_pairs_total"] for r in with_comp)
            if with_comp else None),
        "anchor_start": anchor["start"] if anchor else None,
        "agent_enabled_union": "+".join(sorted({o for r in agent for o in (r.get("enabled_oracles") or "").split("+") if o})),
        "agent_steps_max": max(finite(r.get("steps") for r in agent), default=None),
        "agent_kl_max": max(finite(r.get("kl") for r in agent), default=None),
        "agent_from_initial": sum(r["start"] == "from_initial" for r in agent),
        "agent_from_parent": sum(r["start"] == "from_parent" for r in agent),
        "agent_continue_parent": sum(r["start"] == "continue_parent" for r in agent),
        "hyperparameter_only": len(hyper_only), "no_edit_repeats": len(unchanged),
        "agent_penalty_edits": sum("penalties" in (r.get("edit_keys") or "").split(",") for r in agent),
        "edit_key_counts": counter_text(sorted(keys.elements())),
        "enabled_oracle_sets": counter_text(r["enabled_oracles"] for r in allocated),
        "aggregators": counter_text(r.get("aggregator") or "none" for r in allocated),
        "penalty_sets": counter_text(r.get("penalties") or "none" for r in allocated),
        "anchor_lr": anchor.get("lr") if anchor else None, "anchor_kl": anchor.get("kl") if anchor else None,
        "anchor_group_size": anchor.get("group_size") if anchor else None,
        "anchor_steps": anchor.get("steps") if anchor else None,
        "agent_lr_values": counter_text(r.get("lr") for r in agent), "agent_kl_values": counter_text(r.get("kl") for r in agent),
        "agent_group_size_values": counter_text(r.get("group_size") for r in agent),
        "agent_steps_values": counter_text(r.get("steps") for r in agent)}
    accepted = record.cell.view.state.stop.accepted
    assignment = dict(accepted.assignment) if accepted else {}
    out["delivered_candidates"] = ", ".join(f"{c}:{cid[-3:]}" for c, cid in assignment.items()) or None
    by_id = {r["candidate"]: r for r in rows}
    out["delivered_start"] = ", ".join(
        f"{c}:{by_id[cid]['start']}(r{by_id[cid]['round']})" for c, cid in assignment.items() if cid in by_id) or None
    try:
        health = record.cell.health
    except (ControlError, OSError, ValueError, KeyError):
        health = {}
    for condition, entry in health.items():
        out[f"delivered_pairwise_ratio_{condition}"] = entry.get("pairwise_ratio")
    return out


def mechanism_rounds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_cell.setdefault(r["cell_id"], []).append(r)
    for cell, members in by_cell.items():
        for round_ in sorted({r["round"] for r in members}):
            group = [r for r in members if r["round"] == round_ and r["has_spec"]]
            if not group:
                continue
            keys = Counter(key for r in group if not r["anchor"] for key in (r.get("edit_keys") or "").split(",") if key)
            outside = [r for r in group if r.get("composition_recorded")]
            out.append({
                "root": group[0]["root"], "cell": group[0]["cell"], "cell_id": cell, "arm": group[0]["arm"],
                "seed": group[0]["seed"], "round": round_,
                "candidates": len(group), "anchor_in_round": any(r["anchor"] for r in group),
                "from_initial": sum(r["start"] == "from_initial" and not r["anchor"] for r in group),
                "from_parent": sum(r["start"] == "from_parent" for r in group),
                "continue_parent": sum(r["start"] == "continue_parent" for r in group),
                "edit_key_counts": counter_text(sorted(keys.elements())),
                "enabled_oracle_sets": counter_text(r["enabled_oracles"] for r in group),
                "candidates_with_gate": sum(bool(r.get("n_gates")) for r in group),
                "lr_values": counter_text(r.get("lr") for r in group), "kl_values": counter_text(r.get("kl") for r in group),
                "group_size_values": counter_text(r.get("group_size") for r in group),
                "steps_values": counter_text(r.get("steps") for r in group),
                "composition_any_outside": sum(bool(r.get("comp_any_outside")) for r in outside),
                "composition_recorded": len(outside)})
    return out


EXAMPLE_RULES = (
    ("first revision that adds a gate", lambda r: "gates" in (r.get("edit_keys") or "").split(",")),
    ("first restart from the initial policy that changes the reward recipe",
     lambda r: r["start"] == "from_initial" and not r["anchor"]
     and bool({"enabled_oracles", "aggregator", "penalties", "gates", "oracle_utilities"}
              & set((r.get("edit_keys") or "").split(",")))),
    ("first revision whose hypothesis names the composition or the band",
     lambda r: not r["anchor"] and any(w in (r.get("hypothesis") or "").lower()
                                       for w in ("composition", "band", "within the"))),
)


def examples(candidates: pd.DataFrame, cell_ids: list[str]) -> list[dict[str, Any]]:
    """One candidate per rule. Rule i looks first in the i-th cell (in the order given, wrapping round) and then in the
    others, takes the lowest-ordinal candidate that satisfies it, and never repeats a candidate an earlier rule took."""
    found, taken = [], set()
    for index, (rule, predicate) in enumerate(EXAMPLE_RULES):
        hit = None
        order = cell_ids[index % len(cell_ids):] + cell_ids[:index % len(cell_ids)] if cell_ids else []
        for cell in order:
            members = candidates[(candidates.cell_id == cell) & candidates.has_spec]
            for row in members.sort_values("ordinal").to_dict("records"):
                row = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}
                if (cell, row["candidate"]) not in taken and predicate(row):
                    hit = row
                    taken.add((cell, row["candidate"]))
                    break
            if hit:
                break
        found.append({"rule": rule, "row": hit})
    return found


# ----- baselines -------------------------------------------------------------------------------------

def cre_baseline_table(repo: Path) -> pd.DataFrame:
    frame = pd.read_csv(repo / BASELINES / "cre" / "SUMMARY.csv")
    rows = []
    for row in frame.itertuples():
        out = {"method": row.method, "label": row.label, "status": row.status, "n_k562": row.n_k562,
               "n_hepg2": row.n_hepg2}
        for condition, prefix in (("K562", "k562"), ("HepG2", "hepg2")):
            for part in ("point", "ci_low", "ci_high"):
                raw = getattr(row, f"{prefix}_blind_chrombpnet_{part}")
                out[f"{condition}_raw_{part}"] = raw
                out[f"{condition}_cal_{part}"] = cal(condition, raw)
        out["source_summary"] = row.summary_path
        rows.append(out)
    return pd.DataFrame(rows)


def dnadiff_curve(repo: Path) -> pd.DataFrame:
    frame = pd.read_csv(repo / BASELINES / "cre" / "dna_diffusion_matched_budget" / "matched_budget_summary.csv")
    keep = ["variant", "N_per_cell_pool", "cell", "n_draws", "n_sequences_audited", "cal_median_pooled",
            "cal_pooled_ci_low", "cal_pooled_ci_high", "cal_median_mean_of_draws", "cal_median_min_draw",
            "cal_median_max_draw", "cal_median_sd_draws", "raw_median_pooled", "malinois_median_selected_mean_of_draws",
            "pairwise_ratio_vs_initial_policy_mean", "distinct_8mer_fraction_mean"]
    return frame[keep].sort_values(["variant", "cell", "N_per_cell_pool"]).reset_index(drop=True)


def loopseq_baseline_table(repo: Path) -> pd.DataFrame:
    frame = pd.read_csv(repo / BASELINES / "loopseq" / "baseline_table.csv")
    frame["n_per_group"] = frame["n"] / 3
    return frame


def kl_isolation(repo: Path) -> dict[str, Any]:
    """The KL-0.001 isolation run: hyperparameters from control_result.json, hit rates from its blind audit."""
    root = repo / CALIBRATION / "loopseq_kl_isolation" / "kl0.001"
    result, scores = read_json(root / "control_result.json"), root / "audit" / "audit_scores.parquet"
    if result is None or not scores.is_file():
        return {}
    frame = pd.read_parquet(scores)
    hits = frame.groupby("requested_group")["audit__dnacycp2__hit"].mean().to_dict()
    spec = result["spec"]
    return {"path": str(root.relative_to(repo)), "n": len(frame), "n_per_group": int(frame.groupby("requested_group").size().min()),
            "hit_low": hits["low"], "hit_mid": hits["mid"], "hit_high": hits["high"], "worst": min(hits.values()),
            "dinucleotide_entropy_mean": float(frame["complexity__dinucleotide_entropy"].mean()),
            "unique_kmer_count_mean": float(frame["complexity__unique_kmer_count"].mean()),
            "hyperparameters": spec["hyperparameters"], "enabled_oracles": spec["enabled_oracles"],
            "held_out_oracles": spec["held_out_oracles"], "flags": result.get("flags")}


def run_loopseq_selected(records: list[SimpleNamespace], scratch: Path) -> dict[str, Any]:
    """Build the replica, prove it on the finished main-line and earlier-run cells, and select from every baseline pool.

    Any failure, or a replica that does not reproduce a cell's delivered rows, its stored CNN scores or its terminal audit,
    returns no rows (so no `lsSel...` macro is defined) and says why.
    """
    done = [r for r in records if r.task == "loopseq" and r.arm == MAIN_ARM and r.finished]
    if not done:
        return {"rows": [], "reason": "no finished Loop-seq no_self_validation cell to take the delivery plan from"}
    try:
        with tempfile.TemporaryDirectory(prefix="paper_numbers_ls_") as scratch_output:
            replica = LoopseqReplica(done[0].cell.view.config(), REPO, Path(scratch_output))
            checks = [replica_check(replica, r) for r in done if r.root in EARLIER_ROOTS["loopseq"] + MAIN_ROOTS["loopseq"]]
            rows, missing = selected_baselines(replica, REPO)
            floor = float(rows[0]["minimum_distance"]) if rows else float("nan")
            plan, fraction = replica.plan_dict, replica.health_fraction
    except (ControlError, OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        return {"rows": [], "reason": f"the replica could not be built or run ({type(exc).__name__}: {str(exc)[:160]})"}
    bad = [c["cell"] for c in checks if not (c["selected_rows_equal_delivered"] and c["hit_worst_equals_terminal_audit"]
                                             and c["margin_worst_equals_terminal_audit"] and c["cnn_max_abs_diff_vs_stored"] < CNN_REPLICA_TOLERANCE)]
    if not checks or bad:
        return {"rows": [], "checks": checks, "reason": "the replica does not reproduce the delivered library of " + (", ".join(bad) or "any cell")}
    if missing:
        return {"rows": [], "checks": checks, "reason": "baseline pool files are missing: " + "; ".join(missing)}
    return {"rows": rows, "checks": checks, "plan": plan, "floor": floor, "health_fraction": fraction}


def read_cm_initial(repo: Path) -> dict[str, Any]:
    """The CM initial-policy comparator: the cell's own library-selection plan replayed on its baseline pool of initial-policy
    samples and scored with the current audit heads (`figures/fig4/data`, written by another helper; read only)."""
    folder = repo / "figures" / "fig4" / "data"
    audit, medians = read_json(folder / "cm_initial_policy_audit.json"), read_json(folder / "cm_initial_policy_medians.json")
    if audit is None or medians is None:
        return {}
    check = audit["validation_on_delivered_library"]
    return {
        "audit_file": "figures/fig4/data/cm_initial_policy_audit.json", "medians_file": "figures/fig4/data/cm_initial_policy_medians.json",
        "cell": audit["cell"], "pool_rows": audit["pool_rows"], "device": audit["device"], "selection": audit["selection"],
        "n_selected": len(audit["selected_row_ids"]),
        "t5": audit["selected_128"]["audit_2020_t5"]["median"], "esm": audit["selected_128"]["audit_2020_esm"]["median"],
        "pool_t5": audit["pool_512"]["audit_2020_t5"]["median"], "pool_esm": audit["pool_512"]["audit_2020_esm"]["median"],
        "distance": medians["selected_128_natural_distance"], "pool_distance": medians["pool_512_natural_distance"],
        "function": medians["selected_128_function_2026_esm"], "pool_function": medians["pool_512_function_2026_esm"],
        "same_pool_in_all_cells": medians.get("same_initial_pool_in_all_cells"),
        "bounded": list((medians.get("bounded_by_seed") or {}).values()),
        "check_cell": check["cell"], "check_rows": check["rows"],
        "check_t5_cpu": check["audit_2020_t5"]["cpu_median"], "check_t5_audit": check["audit_2020_t5"]["audit_median"],
        "check_t5_diff": check["audit_2020_t5"]["max_abs_difference"],
        "check_esm_cpu": check["audit_2020_esm"]["cpu_median"], "check_esm_audit": check["audit_2020_esm"]["audit_median"],
        "check_esm_diff": check["audit_2020_esm"]["max_abs_difference"]}


def cm_policy(record: SimpleNamespace) -> dict[str, Any]:
    declaration = record.cell.view.declaration() or {}
    contract = declaration.get("contract") or {}
    checkpoint = contract.get("initial_checkpoint")
    config = read_json(Path(checkpoint) / "config.json") if checkpoint else None
    return {"initial_checkpoint": checkpoint, "source_model": (config or {}).get("source_model"),
            "adapter": (config or {}).get("adapter"), "initial_recipe": contract.get("initial_recipe"),
            "bounds": contract.get("hyperparameter_bounds")}


def cm_initial_policy_feasibility(record: SimpleNamespace, scratch: Path) -> dict[str, Any]:
    """How many of the initial policy's own 512 sequences in this cell's baseline pool pass the delivery filters."""
    path = record.cell.path
    pool, measured = read_json(path / "baseline" / "pool.json"), (read_json(path / "baseline" / "public_measurements.json") or {})
    measurements = measured.get("measurements") or {}
    if not pool or "legal" not in measurements:
        return {}
    plan = (record.cell.view.config().get("control") or {}).get("selection", {}).get("plan", {})
    minimum_distance = next((item["minimum"] for item in plan.get("filters", []) if item.get("measurement") == "natural_distance"), None)
    sequences = pool["sequences"]
    legal = np.asarray(measurements["legal"]["values"], dtype=float) >= 1
    distance = np.asarray(measurements["natural_distance"]["values"], dtype=float)
    copy = np.asarray(measurements["training_copy"]["values"], dtype=float) > 0
    length_ok = np.asarray([plan.get("min_length", 0) <= len(s) <= plan.get("max_length", 10**9) for s in sequences])
    eligible = legal & ~copy & length_ok & (distance >= (minimum_distance if minimum_distance is not None else -np.inf))
    unique_eligible = len({s for s, ok in zip(sequences, eligible) if ok})
    required = int((plan.get("count") or {}).get("CM", 128))
    qualification = scratch / "cm_qualification_20260917" / "oracles"
    return {"pool_rows": len(sequences), "eligible_rows": int(eligible.sum()), "eligible_unique": unique_eligible,
            "required": required, "distance_minimum": minimum_distance,
            "audit_heads_present": {name: (qualification / f"{name}.joblib").is_file()
                                    for name in ("audit_2020_t5", "audit_2020_esm")}}


# ----- the facts file --------------------------------------------------------------------------------

def src(record: SimpleNamespace, scratch: Path, tail: str) -> str:
    return f"`{record.dirname}/{record.name}/{tail}`"


def capability_rows(records: list[SimpleNamespace]) -> list[dict[str, Any]]:
    rows = []
    for r in records:
        if r.cell is None:
            rows.append({"root": r.root, "cell": r.name, "status": r.status, "note": r.note})
            continue
        cap, s = r.cap, r.summary
        rows.append({
            "root": r.root, "cell": r.name, "cell_id": cell_id(r), "arm": r.arm, "seed": r.seed, "status": r.status,
            "audit": cap["terminal_audit"],
            "finished": r.finished, "source_commit": (getattr(r.cell.view.header, "source_commit", None) or "")[:10],
            "steps_min": cap["steps_min"], "steps_default": cap["steps_default"], "steps_max": cap["steps_max"],
            "synthesis_enabled": cap["synthesis_enabled"], "synthesis_max_requests": cap["synthesis_max_requests"],
            "frozen_levers": cap["frozen_levers"], "candidates_allocated": cap["candidates_allocated"],
            "candidate_budget": cap["candidate_budget"], "candidates_trained": cap["candidates_trained"],
            "rounds_completed": cap["rounds_completed"], "rounds_allocated": s["rounds"]["allocated"],
            "rounds_opened": s["rounds"]["opened"], "resumes": cap["resumes"],
            "resolved_models": cap["resolved_models"], "jobs_submitted": cap["jobs_submitted"],
            "jobs_infra_failed": cap["jobs_infra_failed"], "tokens_total": s["tokens"]["total"],
            "tokens_prompt": s["tokens"]["prompt"], "tokens_completion": s["tokens"]["completion"],
            "tokens_reasoning": s["tokens"]["reasoning"], "tokens_cached": s["tokens"]["cached"],
            "llm_calls": s["tokens"]["calls"], "decision_calls": s["tokens"]["by_call"]["decision"]["calls"],
            "critic_calls": s["tokens"]["by_call"]["critic"]["calls"], "coder_calls": s["tokens"]["by_call"]["component"]["calls"], "training_wall_h": r.hours, "ok_candidates_without_wall_h": r.hours_missing,
            "stop_round": cap["stop_round"], "stop_source": cap["stop_source"], "needs_attention": cap["needs_attention"],
            "first_event": s["first_event_ts"], "last_event": s["last_event_ts"]})
    return rows


def cost_rows(records: list[SimpleNamespace]) -> list[dict[str, Any]]:
    """Per task, root and arm: the means over the finished cells."""
    rows = []
    keys = sorted({(r.task, r.root, r.arm) for r in records if r.finished}, key=lambda k: (k[0], k[1], k[2]))
    for task, root, arm in keys:
        group = [r for r in records if r.finished and (r.task, r.root, r.arm) == (task, root, arm)]
        tokens = [r.summary["tokens"] for r in group]
        rows.append({
            "task": task, "root": root, "arm": arm, "cells": len(group),
            "candidates": np.mean([r.summary["candidates_trained"] for r in group]),
            "rounds_trained": np.mean([r.summary["rounds"]["allocated"] for r in group]),
            "gpu_h": np.mean([r.hours for r in group]), "gpu_h_sum": sum(r.hours for r in group),
            "tokens_total": np.mean([t["total"] for t in tokens]), "tokens_prompt": np.mean([t["prompt"] for t in tokens]),
            "tokens_cached": np.mean([t["cached"] for t in tokens]),
            "tokens_completion": np.mean([t["completion"] for t in tokens]),
            "tokens_reasoning": np.mean([t["reasoning"] for t in tokens]),
            "llm_calls": np.mean([t["calls"] for t in tokens]),
            "decision_calls": np.mean([t["by_call"]["decision"]["calls"] for t in tokens]),
            "critic_calls": np.mean([t["by_call"]["critic"]["calls"] for t in tokens])})
    return rows


def status_rows(records: list[SimpleNamespace], now: dt.datetime) -> list[dict[str, Any]]:
    rows = []
    for r in records:
        if r.cell is None:
            continue
        main = r.arm == MAIN_ARM and r.root in MAIN_ROOTS[r.task]
        if not (main or r.status == "running"):
            continue
        s = r.summary
        last = dt.datetime.fromisoformat(s["last_event_ts"].replace("Z", "+00:00"))
        first = dt.datetime.fromisoformat(s["first_event_ts"].replace("Z", "+00:00"))
        rows.append({
            "task": r.task, "root": r.root, "cell": r.name, "cell_id": cell_id(r), "state": r.status,
            "finished": r.finished, "training_wall_h": r.hours, "rounds_completed": s["rounds"]["completed"], "rounds_allocated": s["rounds"]["allocated"],
            "rounds_max": s["rounds"]["maximum"], "candidates_allocated": s["candidates"]["allocated"],
            "candidates_budget": s["candidates"]["budget"], "candidates_by_status": json.dumps(s["candidates"]["by_status"]),
            "candidates_trained": s["candidates_trained"],
            "elapsed_h": ((now if r.status == "running" else last) - first).total_seconds() / 3600,
            "since_last_event_h": (now - last).total_seconds() / 3600 if r.status == "running" else None, "last_event": s["last_event_ts"],
            "needs_attention": (s["needs_attention"] or {}).get("type", ""), "terminal_audit": r.cell.audit})
    return rows


# What the previous manuscript text printed for these two baselines (relayed in the coordinator's message of 2026-10-08).
OLD_MANUSCRIPT = {"reglmK": 0.36, "reglmH": 0.99, "tacoK": -0.36, "tacoH": 0.26}


def pool_text(entry: dict[str, Any]) -> str:
    rows, unique = entry.get("pool_rows_per_cell"), entry.get("pool_unique_per_cell")
    if not rows:
        return ""
    return ("a candidate pool of " + ", ".join(f"{rows[c]:,} rows ({unique[c]:,} distinct) for {c}" for c in CONDITIONS["cre"] if c in rows)
            + " (`reglm_candidates_seed0.csv` beside the selection file), ")


def dup_text(entry: dict[str, Any] | None) -> str:
    if not entry or not entry.get("rows"):
        return "not found"
    return f"{entry['unique_rows']} distinct sequences in {entry['rows']} rows"


def comparator_section(comparators: dict[str, dict[str, Any]], repo: Path) -> list[str]:
    """Which library of outputs/baselines/cre matches the Methods description of regLM and of TACO."""
    def ci(entry: dict[str, Any], condition: str) -> str:
        if f"{condition}_lo" in entry:
            return f"{entry[f'{condition}_cal']:.3f} [{entry[f'{condition}_lo']:.3f}, {entry[f'{condition}_hi']:.3f}]"
        if f"{condition}_sd" in entry:
            return (f"{entry[f'{condition}_cal']:.3f} (sd over {entry[f'{condition}_seeds']} seed medians {entry[f'{condition}_sd']:.3f}; "
                    f"seed range {entry[f'{condition}_min']:.3f} to {entry[f'{condition}_max']:.3f})")
        return MISSING

    rows = []
    for name, entry in comparators.items():
        per_cell = entry.get("K562_n") or entry.get("n_per_cell_per_seed")
        rows.append({"dir": f"`{name}`", "role": entry["role"], "n": per_cell if "K562_n" in entry else f"{per_cell} per seed",
                     "rows": f"{entry['unique_rows']}/{entry['rows']}" if entry.get("rows") else None,
                     "k": ci(entry, "K562"), "h": ci(entry, "HepG2"),
                     "same": entry.get("identical_sequences_to") or None,
                     "check": entry.get("raw_minus_delta_matches_calibrated")})
    out = ["Source: `outputs/baselines/cre/<directory>/calibrated_summary.json` (calibrated blind ChromBPNet median, cal = raw - delta, 95% bootstrap interval over the "
           "rows of that cell line, 2000 draws), `baseline_manifest.json` (label, row and unique-row counts, generator text), `library.parquet` (sequence digests). `n` = rows per "
           "cell line; `unique/rows` = distinct sequences over all rows of the library; `check` = the manifest's raw median minus delta equals the calibrated value; "
           "`identical sequences to` = another directory holding exactly the same sequence set.", ""]
    out += md_table(rows, [("dir", "directory"), ("role", "what it is"), ("n", "n per cell line"), ("rows", "unique/rows"), ("k", "K562 cal [95% CI]"),
                           ("h", "HepG2 cal [95% CI]"), ("same", "identical sequences to"), ("check", "check")])
    regl, taco = comparators.get("reglm_project_unique_128"), comparators.get("taco_alpha001_5seed")
    summary_methods = set(pd.read_csv(repo / BASELINES / "cre" / "SUMMARY.csv").method) if (repo / BASELINES / "cre" / "SUMMARY.csv").is_file() else set()
    out += [""]
    if regl:
        out += [f"**regLM, as the Methods describe it: `reglm_project_unique_128`.** Released human-enhancer checkpoint (Zenodo 12668907), temperature 1.0, top_k 2 "
                f"(`generator` text in its `baseline_manifest.json`; top_k=2 found there: {regl['generator_top_k2']}), " + pool_text(regl) + "exact duplicates removed "
                "within each target cell, then the top 128 per cell line by paired regLM regression-model specificity (target prediction minus the other cell line's); "
                "the rule is implemented by `scripts/baselines/cre_reglm_official_filter.py "
                "--selection-rule project_unique --top-n 128`. Result: K562 " + ci(regl, "K562") + ", HepG2 " + ci(regl, "HepG2") +
                f", n = {regl['K562_n']} per cell line, {regl['unique_rows']} distinct sequences in {regl['rows']} rows. "
                f"The stored `library.parquet` is the selection file named in the manifest (same sequences: {regl.get('library_equals_source_selection', 'file not found')}). "
                "The libraries `reglm_published` (" + dup_text(comparators.get("reglm_published")) + ", no deduplication) and `reglm` ("
                + dup_text(comparators.get("reglm")) + ", no specificity selection) are different protocols; "
                "`reglm_topk_none_broken` was sampled with top_k=None and its own manifest calls it superseded."]
    if taco:
        twin = comparators.get("taco_alpha001_81823", {})
        out += ["", "**TACO, as the Methods describe it: `taco_alpha001_5seed`** (alpha 0.01 in the labels of its five seed directories, "
                f"{taco.get('K562_seeds')} design seeds, {taco.get('n_per_cell_per_seed')} sequences per cell line per seed; run base `{taco.get('run_base')}` named in its "
                "`calibrated_summary.json`). Mean over the five per-seed medians: K562 " + ci(taco, "K562") +
                ", HepG2 " + ci(taco, "HepG2") + ". The per-seed medians equal the `calibrated_summary.json` of `taco_alpha001_5seed_s0` to `_s4` "
                f"(K562 {taco.get('K562_per_seed_matches')}, HepG2 {taco.get('HepG2_per_seed_matches')}). The aggregate directory stores no pooled interval; the per-seed 95% "
                "intervals are in the `..._s0` to `..._s4` rows above. "
                "`taco_alpha001_81823` holds the same sequences as seed 0 (see the `identical sequences to` column) and records tfbs_lambda "
                f"{twin.get('tfbs_lambda')}; `taco` and `taco_perffix_crosscheck` are alpha 0.0 with 192 sequences per cell line, one seed. "
                f"Caveat recorded in the `run_metadata` of job 81823 (the seed-0 twin; the five-seed directories carry no such field): \"{twin.get('qualification')}\", motifs: "
                f"\"{twin.get('motif_source')}\"."]
    if regl and taco:
        found = {"reglmK": regl["K562_cal"], "reglmH": regl["HepG2_cal"], "tacoK": taco["K562_cal"], "tacoH": taco["HepG2_cal"]}
        agree = {key: bool(np.isclose(round(found[key], 2), OLD_MANUSCRIPT[key])) for key in found}
        out += ["", "The values the previous manuscript text gave (regLM 0.36 K562 / 0.99 HepG2, TACO -0.36 / 0.26, as relayed by the coordinator) against these libraries rounded to two "
                "decimals: " + ", ".join(f"{key} {round(found[key], 2):.2f} vs {OLD_MANUSCRIPT[key]:.2f} ({'agree' if agree[key] else 'DIFFER'})" for key in found) + "."]
    if summary_methods:
        absent = [name for name in ("reglm_project_unique_128", "taco_alpha001_5seed") if name not in summary_methods]
        out += ["", f"`SUMMARY.csv` and `external_baselines_final_table.json` do **not** contain the Methods libraries (absent from `SUMMARY.csv`: {', '.join(absent) or 'none'}); "
                "they carry the unfiltered `reglm` and the alpha-0.0 `taco` rows. The macros `\\reglmK`, `\\reglmH`, `\\tacoK`, `\\tacoH` read the matching libraries above, "
                "not `SUMMARY.csv`."]
    return out


def loopseq_selected_section(sel: dict[str, Any], records: list[SimpleNamespace]) -> list[str]:
    """b.5: the Loop-seq baseline pools under the agents' delivery rule, or one line saying why there is no such table."""
    out = ["### b.5 Loop-seq baselines under the agents' delivery rule", ""]
    if not sel.get("rows"):
        return out + [f"Not computed: {sel.get('reason', 'no reason recorded')}. No `lsSel...` or `lsMarginGrpo` macro is defined."]
    plan = sel["plan"]
    floors = {f["measurement"]: {} for f in plan["filters"] if "minimum" in f and f["measurement"] != "legal"}
    for f in plan["filters"]:
        if "minimum" in f and f["measurement"] != "legal":
            floors[f["measurement"]].update({st: f["minimum"] for st in f.get("strata") or ["all"]})
    floor_text = "; ".join(f"{name} " + "/".join(f"{st} {value:.4g}" for st, value in values.items()) for name, values in floors.items())
    out += ["Question: how do the baselines look once the agents' own library-delivery rule picks 128 per group from a 512-per-group pool? "
            "The baseline numbers of b.4 are over the whole unselected pool, the agents' numbers over the 128 rows per group their rule selected.", "",
            "**Rule applied** (`control.selection.plan` of the Loop-seq cell config, run through `recital.library_selection.select_library_plan` exactly as "
            "`control/backend.py` calls it): legal rows only; floors " + floor_text + "; training copies excluded; exact reverse-complement duplicates removed within a "
            f"group; rank by `margin/cnn` (the CNN ensemble's signed distance to the group's interval, higher is better); 128 per group; and a minimum normalized edit distance of "
            f"{sel['floor']:.4g} between two selected rows of one group, derived as the engine derives it (`{sel['health_fraction']}` x the smallest group's median nearest-neighbour "
            "distance of the initial policy's pool, here the `sft_only` pool; every cell's delivery plan carries the same 0.044). "
            "**Scores.** The CNN scores of the random, SFT and GA pools are not stored on disk (their files hold only the DNAcycP2 score), so every pool is scored here with "
            "the same `CNNEnsembleOracle` and the checkpoints the cells use (`backends/loopseq_rl/outputs/scorer/member_0..2.pt`) on the CPU; "
            "where a file stores `oracle__cnn` (GRPO and KL pools) the recomputed value agrees to the difference in the table. The DNAcycP2 raw scores are the stored "
            "`audit__dnacycp2__raw`; the in-band fraction and the median signed margin come from the task's own `audit_measure`, the function that writes `terminal_audit/summary.json`.", "",
            "**Check that the replica is the agents' rule** (it is run on the delivery pool of each finished main-line and earlier-run `no_self_validation` cell, with that cell's own stored plan):", ""]
    out += md_table(sel["checks"], [("cell", "cell"), ("pool_rows", "pool rows"), ("minimum_distance_in_cell_plan", "min distance in the cell's plan"),
                                   ("cnn_max_abs_diff_vs_stored", "CNN: max abs diff vs stored"), ("selected", "rows selected"),
                                   ("selected_rows_equal_delivered", "same rows as delivered"), ("hit_worst_equals_terminal_audit", "worst-group in-band = audit"),
                                   ("margin_worst_equals_terminal_audit", "worst-group margin = audit")], digits=6)
    out += ["", "**Results.** `pool` = whole unselected pool (b.4 number, recomputed here); `eligible` = rows per group left by the filters and duplicate removal; `selected` = rows the rule delivers per group (low/mid/high).", ""]
    rows = [dict(r, eligible=" / ".join(str(v) for v in json.loads(r["eligible_per_group"]).values()),
                 selected=" / ".join(str(v) for v in json.loads(r["selected_per_group"]).values())) for r in sel["rows"]]
    out += md_table(rows, [("library", "library"), ("pool_hit_worst", "pool worst-group in-band"), ("eligible", "eligible"), ("selected", "selected"),
                           ("hit_low", "in-band low"), ("hit_mid", "mid"), ("hit_high", "high"), ("hit_worst", "worst group"),
                           ("margin_low", "margin low"), ("margin_mid", "mid"), ("margin_high", "high"), ("margin_worst", "margin worst group"),
                           ("cnn_max_abs_diff_vs_stored", "CNN diff vs stored")], digits=4)
    out += ["", "Sources: " + "; ".join(f"`{r['library']}` = `{r['source']}`" for r in sel["rows"]) + ". `handtuned_grpo_64k` is `lsSelGrpo` and `lsMarginGrpo`; the other GRPO budgets are `lsSelGrpoFourK`, `lsSelGrpoSixteenK`, `lsSelGrpoThirtyTwoK`.", ""]
    short = [r for r in sel["rows"] if r["hit_worst"] is None]
    for r in short:
        pergroup = lambda text: ", ".join(f"{g} {n}" for g, n in json.loads(text).items())  # noqa: E731
        out += [f"`{r['library']}`: the rule delivers {r['selected_total']} of 384 rows (per group {pergroup(r['selected_per_group'])}); rows left after the filters and duplicate removal per group: "
                f"{pergroup(r['eligible_per_group'])}. The pool is a collapsed, low-complexity library (b.4: mean dinucleotide entropy 2.39, mean distinct k-mers 12.7 against floors above 2.8 and 20), so the rule "
                "cannot deliver 128 per group and the worst-group fraction does not exist; `\\lsSel" + r["suffix"] + "` is `n/a` and `\\lsSel" + r["suffix"] + "Rows` is the number of rows delivered."]
    ags = [r for r in records if r.task == "loopseq" and r.arm == MAIN_ARM and r.finished]
    out += ["", "For comparison, the agents' delivered libraries (`no_self_validation`, finished cells; terminal audit, worst group): " + "; ".join(
        f"`{r.root}/{r.name}` in-band {r.audit['worst_hit']:.4f}, margin {r.audit['worst_margin']:.4f}" for r in ags) + "."]
    return out + [""]


def gate_evidence_section(cand_frame: pd.DataFrame) -> list[str]:
    """c.6: how many agent candidates add or tighten a gate, and how many of their hypotheses mention the composition checks."""
    out = ["### c.6 Gate candidates and the hypotheses that cite the composition checks", "",
           "**Counted:** an agent (non-anchor) candidate with an allocated recipe whose gates, against its parent's, add a gate (a name the parent does not carry) or tighten one it carries "
           "(a `min*` parameter raised or a `max*` parameter lowered); the parent of a candidate that starts `from_initial` with no parent is the declared initial recipe, which has no gate. "
           "A gate that is only relaxed or dropped, and a gate inherited unchanged, does not count. Running cells count the candidates allocated so far. "
           "**Cited:** the candidate's stored `hypothesis` (the text the agent wrote with the proposal, in its spec blob) contains, case-insensitively, one of: `composition`, `entropy`, "
           "`gc` (as a word: not inside a longer word or number; `gc_bounds_gate` and `GC-content` match), `homopolymer`, `low-complexity` (also `low complexity`), `diversity`, "
           "`band` or `bands` (as a word), `k-mer` or `kmer` (as a word, also `k-mers`). The rule measures a mention, not that the composition line caused the edit; several keywords are the "
           "names of the gates' own subjects (`entropy` for the dinucleotide-entropy gate), so the second column below gives the stricter count of hypotheses that name the line or the band itself.", ""]
    frame = cand_frame[(~cand_frame.anchor.fillna(False).astype(bool)) & cand_frame.gate_change.eq(True)]
    rows = []
    for (cell, arm), group in frame.groupby(["cell_id", "arm"], sort=False):
        counts = Counter(k for ks in group.hypothesis_keywords.fillna("") for k in ks.split(",") if k)
        rows.append({"cell": cell, "arm": arm, "gate_candidates": len(group), "added": int(group.gate_added.eq(True).sum()),
                     "tightened_only": int((group.gate_tightened.eq(True) & ~group.gate_added.eq(True)).sum()),
                     "cited": int(group.hypothesis_cites_composition.eq(True).sum()),
                     "names_composition_or_band": int(group.hypothesis_keywords.fillna("").apply(lambda k: bool({"composition", "band"} & set(k.split(",")))).sum()),
                     "keywords": ", ".join(f"{k} x{n}" for k, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))})
    out += md_table(rows, [("cell", "cell"), ("arm", "arm"), ("gate_candidates", "gate candidates"), ("added", "adds a gate"), ("tightened_only", "tightens only"),
                           ("cited", "hypothesis cites"), ("names_composition_or_band", "names composition or band"), ("keywords", "keywords matched (candidates)")])
    for label, roots in (("CRE main-line (`creGateCands`, `creGateCited`)", MAIN_ROOTS["cre"]), ("Loop-seq main-line (`lsGateCands`, `lsGateCited`)", MAIN_ROOTS["loopseq"])):
        part = frame[frame.root.isin(roots) & frame.arm.eq(MAIN_ARM)]
        out += ["", f"{label}: {len(part)} gate candidates, {int(part.hypothesis_cites_composition.eq(True).sum())} cite the composition checks by the keyword rule, "
                f"{int(part.hypothesis_keywords.fillna('').apply(lambda k: bool({'composition', 'band'} & set(k.split(',')))).sum())} of them name `composition` or `band` itself."]
    return out + [""]


def build_facts(*, records, macros, scratch, now, baseline, baseline_frames, comparators, kl, cand_frame, mech_frame, round_frame,
                refusal_frame, lineage_frame, loopseq_selected,
                example_blocks, cm_extra, cap_rows, stat_rows, cost_rows, boundaries, repo) -> str:
    L: list[str] = []
    stamp = now.strftime("%Y-%m-%d %H:%M UTC")
    L += ["# Paper facts (generated)", "",
          f"Generated {stamp} by `scripts/analysis/paper_numbers.py` (re-run: `make paper-numbers`). Every number below is "
          "recomputed from the files named beside it; a cell that is not finished (run `frozen` with a complete terminal "
          "audit) contributes to no aggregate. In silico, computational estimands; not wet-lab validation.", "",
          "Conventions. CRE: ChromBPNet raw target-minus-off-target MinGap, median of the 128 delivered rows per cell line, "
          f"cal = raw - delta with delta K562 = {DELTA['K562']:+.4f}, HepG2 = {DELTA['HepG2']:+.4f} "
          "(`scripts/analysis/cre_movement_vs_blind.py`). Loop-seq: DNAcycP2 in-band fraction of the delivered rows per group. "
          "CM: audit_2020_t5 median (primary) and audit_2020_esm median (secondary). Main line = the `no_self_validation` arm "
          "of `cre_control_round3_20261006` (seeds 0-2), `loopseq_control_20261008` (seeds 0-2) and `cm_control_20261008` "
          "(seeds 2401-2403); the earlier runs are `loopseq_control_20261007` and `cm_control_20261007`. "
          "Scratch root: `" + str(scratch) + "`.", "",
          "Estimator caveats that apply to every CRE number: the blind ChromBPNet scored the delivered libraries of every "
          "earlier campaign that shaped this design, so it is not an independent audit of it; the main-line configuration was chosen on "
          "these same three searches (post hoc); the matched DNA-Diffusion libraries use up to 73,728 Malinois-scored samples against about "
          "432,000 Malinois-scored rollouts per run here (matched in selection-pool size, not in training compute).", ""]

    # --- (a) -----------------------------------------------------------------------------------------
    L += ["## (a) Per-cell audited numbers, all arms of all roots", ""]
    L += ["### a.1 Capability state, cost and status of every cell (print before citing any run)", "",
          "Source: `<root>/<cell>/events/log.jsonl` through `recital.control.export.load_run` and "
          "`analysis.aggregate_control_campaign.capability_row`; steps and synthesis from the newest `config.yaml` snapshot. "
          "`finished` = status frozen and terminal audit complete. GPU-h = sum of `training.wall_h` over the ledger rows of "
          "the trained candidates (one GPU per job; failed attempts and the evaluation/audit passes are not in it). "
          "Tokens = prompt + completion over decision, critic and component calls.", ""]
    L += md_table(cap_rows, [
        ("root", "root"), ("cell", "cell"), ("status", "status"), ("audit", "audit"), ("source_commit", "commit"),
        ("steps_min", "steps min"), ("steps_default", "def"), ("steps_max", "max"),
        ("synthesis_enabled", "Coder on"), ("synthesis_max_requests", "max req"), ("frozen_levers", "frozen levers"),
        ("candidates_trained", "cand ok"), ("candidates_allocated", "alloc"), ("candidate_budget", "budget"),
        ("rounds_allocated", "rounds trained"), ("rounds_opened", "opened"), ("resumes", "resumes"),
        ("resolved_models", "model"), ("jobs_submitted", "jobs"), ("jobs_infra_failed", "infra fail"),
        ("training_wall_h", "GPU-h"), ("tokens_total", "tokens")], digits=1)
    coder = sum(row.get("coder_calls") or 0 for row in cap_rows)
    L += ["", "Rounds: `rounds trained` counts rounds that trained candidates; `opened` also counts the round in which the "
          "agent stopped. `Coder on` = `control.synthesis.enabled` in the cell's newest config snapshot (`max req` its `max_requests`). "
          f"Coder (component) LLM calls summed over all {len(cap_rows)} cells: {coder} "
          "(per cell in `runs/paper_numbers/cells.csv`, column `coder_calls`).", ""]

    L += ["Cost per cell, mean over the finished cells of each task, root and arm (`runs/paper_numbers/cells.csv` has every cell): "
          "tokens are prompt + completion (completion includes reasoning; `cached` is a subset of prompt).", ""]
    L += md_table(cost_rows, [("task", "task"), ("root", "root"), ("arm", "arm"), ("cells", "cells"), ("candidates", "candidates trained"),
                              ("rounds_trained", "rounds trained"), ("gpu_h", "GPU-h"), ("gpu_h_sum", "GPU-h summed"),
                              ("tokens_total", "tokens"), ("tokens_prompt", "prompt"), ("tokens_cached", "cached"),
                              ("tokens_completion", "completion"), ("tokens_reasoning", "reasoning"), ("llm_calls", "LLM calls"),
                              ("decision_calls", "decision"), ("critic_calls", "critic")], digits=1)
    L.append("")

    cre_rows = [dict(root=r.root, cell=r.name, arm=r.arm, seed=r.seed, **r.audit) for r in records
                if r.task == "cre" and r.finished]
    L += ["### a.2 CRE: blind ChromBPNet, calibrated median of the 128 delivered rows per cell line", "",
          "Source: `<root>/<cell>/terminal_audit/summary.json` (`primary.per_condition.<cell line>.value` is the raw median; "
          "cal = raw - delta). Worst = the smaller of the two calibrated medians. Unfinished cells are in section (e).", ""]
    L += md_table(cre_rows, [("root", "root"), ("arm", "arm"), ("seed", "seed"), ("raw_K562", "K562 raw"), ("cal_K562", "K562 cal"),
                             ("raw_HepG2", "HepG2 raw"), ("cal_HepG2", "HepG2 cal"), ("worst_cal", "worst cal")])
    L.append("")
    group = finished(select(records, task="cre", roots=MAIN_ROOTS["cre"], arm=MAIN_ARM))
    if group:
        L.append(f"- `{MAIN_ARM}`, {len(group)} finished cells (seeds {', '.join(str(r.seed) for r in group)}): K562 mean "
                 f"{np.mean([r.audit['cal_K562'] for r in group]):.3f}, HepG2 mean {np.mean([r.audit['cal_HepG2'] for r in group]):.3f}, "
                 f"worst-condition mean {np.mean([r.audit['worst_cal'] for r in group]):.3f}.")
    L.append("")

    ls_rows = [dict(root=r.root, cell=r.name, arm=r.arm, seed=r.seed, **r.audit) for r in records
               if r.task == "loopseq" and r.finished]
    L += ["### a.3 Loop-seq: DNAcycP2 in-band fraction of the delivered 128 rows per group (primary) and median signed margin (secondary)", "",
          "Source: `terminal_audit/summary.json` (`primary.per_condition.<group>.value`, `secondary.primary.per_condition.<group>.value`). "
          f"`cut points` = whether the in-band fraction recomputed from `terminal_audit/dnacycp2.json` with the baselines' cut points "
          f"({boundaries[0]:.6f}, {boundaries[1]:.6f}; `outputs/baselines/loopseq/random/manifest.json`) equals the audit's own value in every group.", ""]
    L += md_table(ls_rows, [("root", "root"), ("arm", "arm"), ("seed", "seed"), ("hit_low", "low"), ("hit_mid", "mid"),
                            ("hit_high", "high"), ("worst_hit", "worst"), ("margin_low", "margin low"),
                            ("margin_mid", "margin mid"), ("margin_high", "margin high"), ("worst_margin", "margin worst"),
                            ("matches_baseline_cutpoints", "cut points")], digits=4)
    if ls_rows:
        worst = min(r["worst_hit"] for r in ls_rows)
        full = sum(r["worst_hit"] >= 1.0 for r in ls_rows)
        L += ["", f"Across the {len(ls_rows)} finished Loop-seq cells the smallest worst-group in-band fraction is {worst:.4f}; "
              f"{full} of {len(ls_rows)} cells have worst = 1.0 (128 of 128 in band in every group): the primary estimand is saturated."]
    L.append("")

    cm_rows = [dict(root=r.root, cell=r.name, arm=r.arm, seed=r.seed, **r.audit) for r in records
               if r.task == "cm" and r.finished]
    L += ["### a.4 CM: audit_2020_t5 (primary) and audit_2020_esm (secondary) medians of the delivered 128", "",
          "Source: `terminal_audit/summary.json` (`sources.<head>.per_condition.CM.value`); delivered-library measurement "
          "medians from `delivery/public_measurements.json` at the rows of `final_sequences.parquet`. `natural_distance` = "
          "full-sequence Levenshtein / max length to the nearest natural family member.", ""]
    L += md_table(cm_rows, [("root", "root"), ("arm", "arm"), ("seed", "seed"), ("t5_median", "audit t5"), ("esm_median", "audit esm"),
                            ("natural_distance_median", "natural dist median"), ("natural_distance_min", "min"),
                            ("natural_distance_max", "max"), ("share_distance_above_0p6", "share > 0.6"),
                            ("function_2026_esm_median", "function_2026_esm median"), ("cal_function_2026_esm_median", "calibrated")])
    L.append("")

    # --- (b) -----------------------------------------------------------------------------------------
    L += ["## (b) External baselines on the same estimator", "", "### b.1 CRE: calibrated blind ChromBPNet medians", ""]
    cre_b = baseline_frames["cre"]
    L += ["Source: `outputs/baselines/cre/SUMMARY.csv` (raw medians and 95% bootstrap intervals, columns "
          "`*_blind_chrombpnet_*`); cal = raw - delta. `n` is the library size per cell line as stored there. The agent cells' libraries are the "
          "128 rows per cell line chosen by the delivery rule, so sizes and selection differ from these rows.", ""]
    not_methods = {"reglm", "taco"}
    rows = [dict(method=r.label + (" [not the Methods library, see b.1a]" if r.method in not_methods else ""),
                 n=f"{r.n_k562}/{r.n_hepg2}", k=f"{r.K562_cal_point:.3f} [{r.K562_cal_ci_low:.3f}, {r.K562_cal_ci_high:.3f}]",
                 h=f"{r.HepG2_cal_point:.3f} [{r.HepG2_cal_ci_low:.3f}, {r.HepG2_cal_ci_high:.3f}]",
                 kr=r.K562_raw_point, hr=r.HepG2_raw_point) for r in cre_b.itertuples()]
    L += md_table(rows, [("method", "method"), ("n", "n K562/HepG2"), ("k", "K562 cal [95% CI]"), ("h", "HepG2 cal [95% CI]"),
                         ("kr", "K562 raw"), ("hr", "HepG2 raw")])
    external = read_json(repo / BASELINES / "cre" / "external_baselines_final_table.json") or []
    if external:
        L += ["", "`outputs/baselines/cre/external_baselines_final_table.json` (strings as stored; the same two libraries as the `taco` and `reglm` rows above, "
              "which are **not** the libraries the Methods describe, see b.1a): "
              + "; ".join(f"{e['method']} n={e['n']}, K562 cal {e['K562_cal']}, HepG2 cal {e['HepG2_cal']}" for e in external) + "."]
    L += ["", "### b.1a regLM and TACO: which library is the one the Methods describe", ""]
    L += comparator_section(comparators, repo)
    L += ["", "### b.2 DNA-Diffusion (CFG 7) at a matched Malinois-scoring budget", "",
          "Source: `outputs/baselines/cre/dna_diffusion_matched_budget/matched_budget_summary.csv` (README.md there describes the "
          "protocol: pools of unselected samples, the 128 with the highest Malinois gap per cell line, then the blind ChromBPNet "
          "audit). `plain` = top 128 of a pool of N per cell line; `best_of_k` = best of N/512 candidate-structured pools; draws "
          "are disjoint groups of pool seeds (8 at N <= 9,216, 4 at 18,432, 2 at 36,864, 1 at 73,728). Median is the pooled "
          "median over draws x 128 rows with its 95% interval; the last column is the mean of the per-draw medians.", ""]
    curve = baseline_frames["dnadiff"]
    for variant in ("unselected_control", "plain", "best_of_k_candidate_pools"):
        part = curve[curve.variant == variant]
        if part.empty:
            continue
        L += [f"**{variant}**", ""]
        wide = []
        for size in sorted(part.N_per_cell_pool.unique()):
            row = {"N": int(size)}
            for cell_line in ("K562", "HepG2"):
                pick = part[(part.N_per_cell_pool == size) & (part.cell == cell_line)]
                if len(pick) == 1:
                    p = pick.iloc[0]
                    row[f"{cell_line}_median"] = f"{p.cal_median_pooled:.3f} [{p.cal_pooled_ci_low:.3f}, {p.cal_pooled_ci_high:.3f}]"
                    row[f"{cell_line}_draws"] = f"{p.cal_median_mean_of_draws:.3f}"
                    row["draws"] = int(p.n_draws)
                    row[f"{cell_line}_mal"] = p.malinois_median_selected_mean_of_draws
            wide.append(row)
        L += md_table(wide, [("N", "N per cell line"), ("draws", "draws"), ("K562_median", "K562 cal [95% CI]"),
                             ("K562_draws", "K562 mean of draw medians"), ("HepG2_median", "HepG2 cal [95% CI]"),
                             ("HepG2_draws", "HepG2 mean of draw medians"), ("K562_mal", "K562 Malinois median"),
                             ("HepG2_mal", "HepG2 Malinois median")], digits=2)
        L.append("")
    L += ["### b.3 Main-line CRE cells against the comparators", ""]
    cre_main = select(records, task="cre", roots=MAIN_ROOTS["cre"], arm=MAIN_ARM)
    cre_done = finished(cre_main)
    comparators = [("DNA-Diffusion, unselected (existing 384 per cell line)", baseline.get("dd384K"), baseline.get("dd384H")),
                   ("DNA-Diffusion, top 128 of 9,216 (pooled median)", baseline.get("ddNineK"), baseline.get("ddNineH")),
                   (f"DNA-Diffusion, top 128 of {baseline.get('ddMaxN', 0):,} (pooled median)", baseline.get("ddMaxK"), baseline.get("ddMaxH")),
                   ("CODA Fast SeqProp", baseline.get("codaK"), baseline.get("codaH"))]
    L += md_table([dict(name=n, k=k, h=h) for n, k, h in comparators], [("name", "comparator"), ("k", "K562 cal"), ("h", "HepG2 cal")])
    seed_rows = []
    for r in cre_main:
        base = dict(seed=r.seed, cell=r.name, state="finished" if r.finished else f"{r.status}, not counted")
        if r.finished:
            base.update(K=r.audit["cal_K562"], H=r.audit["cal_HepG2"], W=r.audit["worst_cal"],
                        k_above=r.audit["cal_K562"] > baseline.get("ddMaxK", np.inf),
                        h_above=r.audit["cal_HepG2"] > baseline.get("ddMaxH", np.inf))
        seed_rows.append(base)
    L += [""] + md_table(seed_rows, [("seed", "seed"), ("state", "state"), ("K", "K562 cal"), ("H", "HepG2 cal"), ("W", "worst"),
                                     ("k_above", "K562 > DD 73,728"), ("h_above", "HepG2 > DD 73,728")])
    L += ["", f"Finished main-line CRE cells: {len(cre_done)} of {len(MAIN_SEEDS['cre'])}.", ""]

    L += ["### b.4 Loop-seq baselines", "",
          "**Estimand and library size.** Source: `outputs/baselines/loopseq/baseline_table.csv` (written 2026-09-28) with the per-method "
          "`manifest.json` beside it. `blind_worst_group` = the minimum over the three groups of the fraction of that group's "
          "as-generated sequences that DNAcycP2 places in the requested group (cut points "
          f"{boundaries[0]:.6f} and {boundaries[1]:.6f}), with a 95% bootstrap interval over 2000 draws. The library is **512 sequences per group "
          "(n = 1536), unselected**; the agent cells' primary is the same in-band fraction (section a.3 checks the cut points) but over the "
          "**128 rows per group the delivery rule selected** (by CNN margin, among legal rows above the entropy and k-mer floors) from the "
          "winning candidate's 512-row pool. Same estimator, different library: a baseline number is not a delivered-library number.", ""]
    lb = baseline_frames["loopseq"]
    rows = [dict(method=r.method, n=r.n, per_group=int(r.n_per_group), worst=r.blind_worst_group,
                 ci=f"[{r.ci_low:.3f}, {r.ci_high:.3f}]", low=r.hit_low, mid=r.hit_mid, high=r.hit_high,
                 entropy=r.dinuc_entropy_mean, kmer=r.unique_3mer_mean, commit=str(r.git_commit)[:10]) for r in lb.itertuples()]
    L += md_table(rows, [("method", "method"), ("n", "n"), ("per_group", "per group"), ("worst", "worst group"), ("ci", "95% CI"),
                         ("low", "low"), ("mid", "mid"), ("high", "high"), ("entropy", "dinuc. entropy"), ("kmer", "unique 3-mers"),
                         ("commit", "commit")])
    L += ["", "Rows: `random_dna` uniform random ACGT 50-mers; `sft_only` samples of the group-conditioned SFT policy (no RL); `ga_search` population "
          "search (truncation selection, per-base mutation) maximizing the same target-interval reward the GRPO arms use; "
          "`handtuned_grpo_{4k,16k,32k,64k}` GRPO with the fixed hand-tuned recipe at that many optimizer steps (control, not the agent loop); "
          "`weak_kl_reward_hack_16k` GRPO with lr 5e-4, kl 0.001, group size 32, 16,000 steps (the table's own config column). "
          "Descriptions are from the manifests under `outputs/baselines/loopseq/<method>/manifest.json`."]
    if kl:
        hp = kl["hyperparameters"]
        L += ["", f"**KL-0.001 isolation run** (`{kl['path']}`; fixed-spec control, enabled oracle(s) {kl['enabled_oracles']}, held-out {kl['held_out_oracles']}; "
              f"lr {hp.get('lr')}, kl {hp.get('kl')}, group size {hp.get('group_size')}, steps {hp.get('steps')}): library "
              f"{kl['n']} sequences ({kl['n_per_group']} per group, unselected); DNAcycP2 in-band low {kl['hit_low']:.4f}, mid {kl['hit_mid']:.4f}, high "
              f"{kl['hit_high']:.4f}, worst {kl['worst']:.4f} (`audit/audit_scores.parquet`, column `audit__dnacycp2__hit`); mean dinucleotide entropy "
              f"{kl['dinucleotide_entropy_mean']:.3f}, mean unique k-mer count {kl['unique_kmer_count_mean']:.2f}; engine flags in `control_result.json`: "
              f"{', '.join(kl['flags'] or [])}. This is a different run from the `weak_kl_reward_hack_16k` row above (different lr and group size)."]
    L.append("")

    L += loopseq_selected_section(loopseq_selected, records)

    # --- (c) -----------------------------------------------------------------------------------------
    L += ["## (c) Mechanism descriptives from the ledger rows", "",
          "Source: `<root>/<cell>/events/log.jsonl` blobs (`recipe` and `proposal` of each allocated candidate, the immutable ledger row: "
          "`edits` against the parent recipe, `composition`, `library.pairwise_ratio`, `training.wall_h`). Candidates, rounds, edits and starts "
          "are per cell in `runs/paper_numbers/candidates.csv`, `mechanism_cells.csv`, `mechanism_rounds.csv`. The anchor candidate is the "
          "declared default recipe and is excluded from the edit counts. `start` (`src/recital/control/schemas.py`): from_initial restarts from the SFT policy, from_parent starts a new stage "
          "from the parent's final weights with a fresh optimizer and learning-rate schedule, continue_parent resumes the parent's optimizer and schedule "
          "(keeps the parent's lr). A composition line is **outside** when at least one source-free "
          "measurement's pool median lies outside the central 90% of the initial policy's pool for that condition; the share is over trained candidates. "
          "Live cells are shown with the candidates finished so far and are marked partial.", ""]
    L += ["### c.1 One row per cell", ""]
    cell_view = []
    for row in mech_frame.to_dict("records"):
        row = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}
        row["cell_state"] = "finished" if row["cell_finished"] else f"{row['status']} (partial)"
        row["pairwise"] = ", ".join(f"{k.rsplit('_', 1)[-1]} {v:.2f}" for k, v in row.items()
                                     if k.startswith("delivered_pairwise_ratio_") and v is not None) or None
        row["share_any"] = None if row["share_candidates_any_outside"] is None else f"{row['share_candidates_any_outside']:.2f}"
        row["share_pairs"] = None if row["share_candidate_conditions_outside"] is None else f"{row['share_candidate_conditions_outside']:.2f}"
        row["starts"] = f"{row['agent_from_initial']}/{row['agent_from_parent']}/{row['agent_continue_parent']}"
        row["first_gate_round"] = None if row["first_gate_round"] is None else int(row["first_gate_round"])
        cell_view.append(row)
    L += md_table(cell_view, [("cell_id", "cell"), ("cell_state", "state"), ("candidates_trained", "trained"), ("agent_candidates", "agent cand."),
                              ("first_gate_round", "first gate round"), ("candidates_with_gate", "with gate"),
                              ("share_any", "share outside band (candidates)"), ("share_pairs", "share outside (cand. x condition)"),
                              ("starts", "restarts init/parent/continue"), ("hyperparameter_only", "hyper-only"),
                              ("no_edit_repeats", "no-edit"), ("pairwise", "delivered pairwise ratio"),
                              ("delivered_start", "delivered cand. start (round)")])
    L += ["", "Restarts column = number of non-anchor candidates with start from_initial / from_parent / continue_parent. `hyper-only` = candidates whose edits "
          "against the parent are only lr, kl, group_size or steps; `no-edit` = identical recipe to the parent. Delivered pairwise ratio = the "
          "delivered library's mean pairwise normalized edit distance over the initial policy's, computed by `recital.control.health` from `final_sequences.parquet` "
          "(requirement `library_health` >= 0.2).", ""]
    L += ["### c.2 Which edits, which values, per cell", ""]
    detail = [dict(cell=r["cell_id"], keys=r["edit_key_counts"], oracles=r["enabled_oracle_sets"], aggr=r["aggregators"], pens=r["penalty_sets"],
                   anchor=f"lr {r['anchor_lr']}, kl {r['anchor_kl']}, g {r['anchor_group_size']}, steps {r['anchor_steps']}",
                   lr=r["agent_lr_values"], kl=r["agent_kl_values"], g=r["agent_group_size_values"], steps=r["agent_steps_values"],
                   gates=r["gate_names"]) for r in mech_frame.to_dict("records")]
    L += md_table(detail, [("cell", "cell"), ("keys", "edit keys (count over agent candidates)"), ("oracles", "enabled oracle sets (count)"),
                           ("aggr", "aggregator"), ("pens", "penalties"), ("gates", "gate names"), ("anchor", "anchor values"),
                           ("lr", "lr chosen"), ("kl", "kl chosen"), ("g", "group size chosen"), ("steps", "steps chosen")])
    L += ["", "Per-round counts (which edit keys were used in each round of each cell) are in `runs/paper_numbers/mechanism_rounds.csv`; the "
          "main-line CRE cells are repeated here.", ""]
    main_rounds = round_frame[round_frame.root.isin(MAIN_ROOTS["cre"]) & (round_frame.arm == MAIN_ARM)]
    L += md_table(main_rounds.to_dict("records"), [("cell_id", "cell"), ("round", "round"), ("candidates", "cand."), ("from_initial", "init"),
                                                   ("from_parent", "parent"), ("continue_parent", "cont."), ("edit_key_counts", "edit keys"),
                                                   ("candidates_with_gate", "gates"), ("kl_values", "kl"), ("lr_values", "lr"),
                                                   ("steps_values", "steps")])
    L += ["", "### c.3 Example revisions quoted from the agents' hypotheses", "",
          "Selection is by rule, not by hand: rule i takes the lowest-ordinal candidate that satisfies it, searching the i-th listed cell first and then the others, "
          "and never repeats a candidate an earlier rule took. The text is the candidate's full `hypothesis` as stored in its spec blob; `edits` are the "
          "ledger row's differences against the parent recipe (cut at 600 characters)."]
    for task, blocks in example_blocks.items():
        L += ["", f"**{task}** (cells searched, in order: {', '.join(f'`{c}`' for c in blocks['cells'])}{blocks['note']})", ""]
        for item in blocks["items"]:
            row = item["row"]
            if row is None:
                L.append(f"- {item['rule']}: none found in these cells.")
                continue
            L.append(f"- {item['rule']}: `{row['cell_id']}` `{row['candidate']}` (round {row['round']}, start {row['start']}, parent {row['parent']}); edits: "
                     f"{shorten(row.get('edits') or 'none', 600)}. Hypothesis: \"{row.get('hypothesis')}\"")
    L.append("")

    L += ["### c.4 Decision answers, refusals, Critic notes and Coder calls", "",
          "Source: `events/log.jsonl` of each cell (`decision_recorded`, `decision_validated`, `critic_advisory` events and the `component` LLM calls of the summary). "
          "An *answer* is one recorded model response to a propose or stop request (one per attempt); it is *refused* when its validation recorded at least one typed "
          "problem (`Problem.code`), even when other candidates of the same answer were accepted; a refused answer is asked again, up to `max_decision_attempts`. "
          "Per cell in `runs/paper_numbers/decision_refusals.csv`.", ""]
    rows = []
    for row in refusal_frame.to_dict("records"):
        codes = json.loads(row["problem_codes"])
        rows.append({**row, "codes": ", ".join(f"{readable_code(code)} x{count}" for code, count in codes.items()) or "none",
                     "state": "finished" if row["cell_finished"] else "running"})
    L += md_table(rows, [("cell_id", "cell"), ("state", "state"), ("answers", "answers"), ("refused", "refused"), ("answers_invalid", "answers invalid"),
                         ("codes", "problem codes (answers carrying each)"), ("critic_notes", "Critic notes"), ("coder_calls", "Coder calls")])
    main_ids = {cell_id(r) for task in MAIN_ROOTS for r in select(records, task=task, roots=MAIN_ROOTS[task], arm=MAIN_ARM)}
    main_rows = [row for row in refusal_frame.to_dict("records") if row["cell_id"] in main_ids]
    L += ["", f"Main-line cells together (finished or running): {sum(r['answers'] for r in main_rows)} answers, {sum(r['refused'] for r in main_rows)} refused, "
          f"{sum(r['critic_notes'] for r in main_rows)} Critic notes, {sum(r['coder_calls'] for r in main_rows)} Coder calls. "
          "`Coder on` in section a.1 says which cells could have called the Coder.", ""]

    L += ["### c.5 Training updates along the delivered lineage", "",
          "Source: `delivery.json` `checkpoint_assignment` (the checkpoint each condition's library was delivered from), the candidate result's checkpoint list "
          "(`lineage_updates`, the engine's own count) and the candidate recipes (`start`, `parent`, `train.checkpoint_steps`, `steps_done`). "
          "**How restarts are counted.** Start at the delivered checkpoint: the updates its candidate trained up to it (`train.checkpoint_steps` at its position; the full "
          "`steps_done` when it is the candidate's final checkpoint). If that candidate started `from_initial` the lineage ends there; the initial policy's own supervised "
          "training is not counted. If it started `from_parent` or `continue_parent`, add the parent's full `steps_done` and repeat for the parent: `from_parent` keeps the "
          "parent's weights and resets optimizer and schedule, `continue_parent` keeps those too, and both add the child's updates to the parent's. A candidate that "
          "restarts `from_initial` therefore discards everything trained before it. Candidates that were trained and not delivered, and siblings of ancestors, are not counted. "
          "`chain` is this walk, `engine` is `lineage_updates` of the delivered checkpoint; the macros use `engine`, and the two agree in every row below unless `agree` says no. "
          "The mean per cell is over its conditions, then over cells (equal here because every cell of a task has the same conditions). Per row in `runs/paper_numbers/lineage_updates.csv`.", ""]
    if len(lineage_frame):
        lrows = lineage_frame.to_dict("records")
        L += md_table(lrows, [("cell_id", "cell"), ("condition", "condition"), ("delivered_checkpoint", "delivered checkpoint"), ("delivered_start", "its start"),
                              ("lineage_length", "candidates in lineage"), ("lineage_chain", "lineage (delivered < ... < root)"),
                              ("lineage_updates_engine", "engine updates"), ("lineage_updates_chain", "chain updates"), ("agree", "agree")])
        L += ["", f"Rows where the engine count and the chain walk differ: {sum(not r['agree'] for r in lrows)} of {len(lrows)}. "
              f"Delivered checkpoints that are not their candidate's final checkpoint: {sum(not r['delivered_is_end_checkpoint'] for r in lrows)} of {len(lrows)}."]
    L.append("")

    L += gate_evidence_section(cand_frame)

    # --- (d) -----------------------------------------------------------------------------------------
    L += ["## (d) CM specifics", ""]
    L += cm_extra
    L.append("")

    # --- (e) -----------------------------------------------------------------------------------------
    L += ["## (e) Status of the main-line cells and every running cell", "",
          f"Read at {stamp} from `events/log.jsonl` (no lock, nothing written). `rounds` = completed/allocated (maximum allowed); `candidates` = trained/allocated "
          "of the budget. Elapsed = first to last event for a finished cell, first event to now for a running one.", ""]
    L += md_table(stat_rows, [("task", "task"), ("root", "root"), ("cell", "cell"), ("state", "state"), ("rounds_completed", "rounds done"),
                              ("rounds_allocated", "rounds alloc."), ("rounds_max", "rounds max"), ("candidates_trained", "cand. trained"),
                              ("candidates_allocated", "cand. allocated"), ("candidates_budget", "budget"), ("candidates_by_status", "by status"),
                              ("training_wall_h", "GPU-h so far"), ("elapsed_h", "elapsed h"), ("since_last_event_h", "h since last event"), ("terminal_audit", "audit"),
                              ("needs_attention", "needs attention")], digits=1)
    L += ["", "## Notes and discrepancies found while computing", ""]
    dd_h = baseline.get("dd384H")
    if dd_h is not None:
        L.append(f"- DNA-Diffusion unselected (existing 384 per cell line), HepG2: `SUMMARY.csv` raw median gives calibrated {dd_h:.4f}, i.e. {dd_h:.2f} at two "
                 "decimals; `docs/reports/CRE_ROUND3_RESULTS_20261007_zh.md` prints 1.98 and the matched-budget README prints +1.975. `\\ddUnselH` follows the file value.")
    L.append("- `\\creRounds`/`\\lsRounds`/`\\cmRounds` count rounds that trained candidates; `rounds opened` in the cell summary is one higher in cells whose agent "
             "stopped in an extra round (`\\creRoundsOpened` etc. give that count).")
    L.append("- GPU-h is the sum of `training.wall_h` of the ledger rows (4 significant digits each); it excludes failed attempts, evaluation passes and the terminal audit.")
    L += ["", "## Macros in `manuscript/numbers.tex`", "",
          "Four kinds of macro, told apart by the `kind` column below. A *finished* cell is a run that is `frozen` with a complete terminal audit; a *block* is the set of cells a macro "
          "aggregates (a task's main-line cells, the earlier-run cells for `...Earlier...`) and *expected* is how many cells the block has "
          "(`creExpected` 3, `lsExpected` 3, `cmExpected` 3; 3 for the earlier Loop-seq block, 1 for the earlier CM block).", "",
          "- **value**: one finished cell or one baseline file stands behind it (per-seed macros with suffix A, B, C..., baselines, `creN`/`lsN`/`cmN`, `...Expected`): the plain number, "
          "or `\\pending{--}` while that seed or file has no data.",
          "- **aggregate-final / aggregate-partial**: a statistic across the finished cells of a block (`...mean`, `...min`, `...max`, "
          "the cost macros `...Cand`, `...Rounds`, `...RoundsOpened`, `...GPUh`, `...GPUhSum`, `...Tokens`, `...TokensSum`, `...LibPairMean`, `...LibPairMin`, "
          "`...LineageUpdates`, `lsLmean`/`lsMmean`/`lsHmean`, `cmFuncMed`, `cmShareFar`, `cmEarlierShareFar`). It is printed plain once the block has "
          "as many finished cells as it expects (`aggregate-final`) and as `\\pending{<value over the cells finished so far>}` before that (`aggregate-partial`); with no finished cell "
          "it is `\\pending{--}`.",
          "- **provisional / provisional-complete**: the mechanism counts (`...GateCells`, `...GateRoundMed`, `...GateCands`, `...GateCited`, `...SpendCells`, `...PenaltyCells`, `...OutShare`, `...StartInit`, "
          "`...StartParent`, `...StartCont`, `...CandProv`, `...HyperOnly`, `...StepsMax`, `...KlMax`), the refusal counts (`...Decisions`, `...Refused`, `...TopRefusal`) and "
          "`criticNotes`, `coderCalls`. They are always computed from **every cell of the block that has candidates, finished or still running**, and printed as `\\pending{<value>}` "
          "while any cell of the block is unfinished; once every cell is finished they print plain (`provisional-complete`). `\\pending{--}` means the block has no data at all. "
          "A provisional value changes as candidates arrive; do not quote it as final. `criticNotes` and `coderCalls` pool the nine main-line cells of the three tasks (block = all nine).", "",
          "Formatting. CRE medians 2 decimals, CM audit values 3, `LibPair` 2. Counts and means of candidates, rounds, hours and starts have at most one decimal and no trailing "
          "zero (`16`, `6.8`). Percentages from `OutShare` and `cmShareFar` have no decimals from 10 up and at most one below 10 (`78`, `5.3`); Loop-seq in-band hit rates "
          f"(`lsW...`, `lsL/M/H...`, `lsBase...`) keep {HIT_RATE_DECIMALS} decimal because one delivered sequence of 128 is 0.8 points (constant `HIT_RATE_DECIMALS` in the script). "
          "No percent sign anywhere. A negative number prints its sign as `$-$` (for example `\\tacoK`).", "",
          "Definitions. An *agent candidate* is any candidate except the anchor (the declared default recipe, which has no gate). *GateCands/GateCited*: section c.6 (agent candidates that add or tighten a gate against their parent; of those, the ones whose hypothesis matches the keyword rule); *GateCells*: cells with at least one agent candidate whose "
          "recipe carries a gate; *GateRoundMed*: the median over those cells of the round of their first gated agent candidate. *SpendCells*: cells with at least one agent candidate that has the named source "
          "among its enabled oracles (CRE `grammar_prior`, Loop-seq `mixbend`, CM `cm_max`); `grammar_prior` and `mixbend` are not in the anchor recipe, `cm_max` is (the declared CM initial recipe "
          "enables `function_2026_esm`, `cm_novelty`, `cm_max`), so for CM a cell counts when its agent keeps `cm_max` in at least one candidate and does not count when it drops it everywhere. "
          "*PenaltyCells*: cells with at least one agent candidate whose ledger-row edits against its parent include `penalties`. *OutShare*: per cell, the share of trained "
          "candidates (anchor included) whose ledger-row composition line lists at least one source-free measurement outside the initial policy's 5-95% band in at least one condition, "
          "then the mean over cells. *Start*: per cell, the number of agent candidates with that start mode, then the mean over cells. *CandProv*: per cell, the number of agent candidates "
          "with an allocated recipe, then the mean over exactly the cells behind the other provisional macros of the block, so `StartInit + StartParent + StartCont = CandProv` and a sentence "
          "`a mean of \\creStartParent of \\creCandProv candidates` has one denominator. *HyperOnly*: per cell, the number of agent candidates "
          "whose edits against the parent are only lr, kl, group_size or steps. *StepsMax/KlMax*: the largest steps (`train.additional_updates`) and kl in any agent candidate of the block. "
          "*LibPair*: `pairwise_ratio` of each delivered library (cell x condition) of the finished cells, computed by `recital.control.health` from `final_sequences.parquet`. "
          "*Decisions/Refused/TopRefusal*: section c.4; the problem code is printed with underscores as spaces, ties broken alphabetically. *LineageUpdates*: section c.5. "
          "The agent-candidate counts include candidates still training; `OutShare` and `HyperOnly` need the ledger row and so use only candidates that have finished training, "
          "which is why `StartInit + StartParent + StartCont` can exceed the number of candidates behind `OutShare` while a cell is running.", ""]
    L += md_table([dict(name="\\" + m["name"], kind=m["kind"], value=("pending" if m["raw"] is None else m["tex"]), desc=m["description"])
                   for m in macros.items], [("name", "macro"), ("kind", "kind"), ("value", "value now"), ("desc", "meaning")])
    return "\n".join(L) + "\n"


# ----- CM section ------------------------------------------------------------------------------------

def cm_section(records: list[SimpleNamespace], cand_frame: pd.DataFrame, scratch: Path, repo: Path,
               cm_initial: dict[str, Any]) -> list[str]:
    cm_cells = [r for r in records if r.task == "cm" and r.cell is not None]
    out: list[str] = []
    first = next((r for r in cm_cells if r.arm == MAIN_ARM), None)
    if first is not None:
        policy = cm_policy(first)
        recipe = policy["initial_recipe"] or {}
        out += [f"**Policy.** Initial checkpoint `{policy['initial_checkpoint']}`, a supervised fine-tune of `{policy['source_model']}` "
                f"(adapter `{policy['adapter']}`; `cm_rebuild_20260917/sft/summary.json` holds the SFT record). Source: the cell's `declaration.json` "
                f"(`contract.initial_checkpoint`) and the checkpoint's `config.json`.", "",
                "**Initial recipe / reward terms** (`declaration.json` `contract.initial_recipe`, identical in every CM cell of both roots unless a row below says otherwise): "
                f"enabled oracles {recipe.get('enabled_oracles')}, held-out {recipe.get('held_out_oracles')}, aggregator {json.dumps(recipe.get('aggregator'))}, "
                f"hyperparameters {json.dumps(recipe.get('hyperparameters'))}, invalid penalty {recipe.get('invalid_penalty')}, reward clip low "
                f"{recipe.get('reward_clip_low')}. Agent-editable bounds: {json.dumps(policy['bounds'])}. `cm_novelty` is a reward term only: a two-sided "
                "negative shortfall on the normalized edit distance to the nearest natural family member, zero between 0.20 and 0.40, negative outside "
                "(`declaration.json` `measurement_definitions.cm_novelty`); delivery requires natural_distance >= 0.2 (`control.selection.plan.filters`) and ranks by "
                "`calibrated/function_2026_esm`.", ""]
    cm_cand = cand_frame[cand_frame.cell_id.isin([cell_id(r) for r in cm_cells])]
    out += ["**Reward recipes the agents trained (per cell; anchor excluded from the edit counts).** Source: candidate recipes in the blob store.", ""]
    rows = []
    for r in cm_cells:
        members = cm_cand[(cm_cand.cell_id == cell_id(r)) & cm_cand.has_spec]
        rows.append({"root": r.root, "cell": r.name, "state": "finished" if r.finished else "running (partial)",
                     "oracle_sets": counter_text(members.enabled_oracles), "aggregators": counter_text(members.aggregator.fillna("none")),
                     "penalties": counter_text(members.penalties.fillna("none").replace("", "none")),
                     "kl": counter_text(members[~members.anchor].kl.dropna()), "lr": counter_text(members[~members.anchor].lr.dropna()),
                     "steps": counter_text(members[~members.anchor].steps.dropna())})
    out += md_table(rows, [("root", "root"), ("cell", "cell"), ("state", "state"), ("oracle_sets", "enabled oracle sets"), ("aggregators", "aggregator"),
                           ("penalties", "penalties"), ("kl", "kl"), ("lr", "lr"), ("steps", "steps")])
    out += ["", "**Delivered libraries.** Source: `terminal_audit/summary.json` and `delivery/public_measurements.json` at the rows of `final_sequences.parquet`.", ""]
    rows = [dict(root=r.root, cell=r.name, arm=r.arm, **r.audit) for r in cm_cells if r.finished]
    out += md_table(rows, [("root", "root"), ("arm", "arm"), ("t5_median", "audit_2020_t5"), ("esm_median", "audit_2020_esm"),
                           ("function_2026_esm_median", "function_2026_esm"), ("cal_function_2026_esm_median", "calibrated function_2026_esm"),
                           ("natural_distance_median", "natural dist. median"), ("natural_distance_min", "min"), ("natural_distance_max", "max"),
                           ("share_distance_above_0p6", "share > 0.6"), ("cm_novelty_median", "cm_novelty"), ("cm_max_median", "cm_max")])
    main_cm = [r for r in cm_cells if r.root in MAIN_ROOTS["cm"] and r.arm == MAIN_ARM]
    out += ["", f"Main-line CM cells finished: {sum(r.finished for r in main_cm)} of {len(main_cm)} (`cm_control_20261007`); the main-line CM macros are `\\pending` until a cell is finished."]
    if first is not None:
        feasibility = cm_initial_policy_feasibility(first, scratch)
        out += ["", "**Initial-policy / SFT-only comparator.**", ""]
        if feasibility:
            out += [f"- Inside a control cell: `baseline/pool.json` of `{first.dirname}/{first.name}` holds {feasibility['pool_rows']} sequences sampled from the initial policy, "
                    f"scored on the visible measurements only (`baseline/public_measurements.json`). Under the delivery filters of that cell (legal, not a training copy, length "
                    f"80-96, natural_distance >= {feasibility['distance_minimum']}) {feasibility['eligible_rows']} rows are eligible, {feasibility['eligible_unique']} exact-unique, against "
                    f"{feasibility['required']} required. A terminal audit scores only the delivered library, so the initial-policy library was scored separately (next bullets)."]
        sft = read_json(scratch / "cm_rebuild_20260917" / "baselines" / "sft_summary.json")
        if sft:
            out += [f"- An SFT-only library already exists from the 2026-09-17 rebuild (`cm_rebuild_20260917/baselines/sft_summary.json`, `sft_candidates.csv`): "
                    f"{sft['selected_count']} selected of {sft['candidate_count']} samples, natural_distance median {sft['natural_distance_median']:.3f}, scored "
                    f"by `{sft['audit']['oracle']}` with median {sft['audit']['median']:.4f}. That head and the selection rule (\"{sft['selection_rule']}\") "
                    "predate the current audit heads, so it is **not** on the audit_2020_t5 / audit_2020_esm scale of the control cells."]
        legal = read_json(scratch / "cm_legal_baseline_20260926" / "terminal_audit" / "summary.json")
        if legal:
            out += [f"- `cm_legal_baseline_20260926/terminal_audit/summary.json` has current-head audits (audit_2020_t5 median "
                    f"{legal['primary']['per_condition']['CM']['value']:.4f}, audit_2020_esm "
                    f"{legal['sources']['audit_2020_esm']['per_condition']['CM']['value']:.4f}, 128 rows) but its `config.yaml` describes a `reference` built from a historical "
                    "adaptive recipe with `initial_training_updates: 512`; whether that library is an untrained-policy library was not verified here, so it is not offered as the SFT-only comparator."]
        rescored = read_json(scratch / "outputs" / "cm_baselines_rescored_20260917" / "rescore_manifest.json")
        if rescored:
            out += ["- ProtRL baselines re-scored with the current heads (`outputs/cm_baselines_rescored_20260917/rescore_manifest.json`): "
                    + "; ".join(f"{b['baseline']} audit_2020_t5 {b['audit_2020_t5_median']:.4f}, audit_2020_esm {b['audit_2020_esm_median']:.4f} (n_rows {b['n_rows']}, unique {b['n_unique']})"
                                for b in rescored["baselines"]) + ". These are different libraries (ProtRL, 384 rows over three delivery variants), not an initial-policy library."]
        published = read_json(scratch / "outputs" / "cm_published_comparators" / "summary.json")
        if published:
            out += ["- Published designed libraries (`outputs/cm_published_comparators/summary.json`, same predicted-endpoint quantity, top 128 by predicted endpoint): "
                    + "; ".join(f"{p['library']} {p['top128_by_predicted_endpoint_median']:.4f} (n={p['n']}, natural distance median {p['natural_distance_median']:.3f})"
                                for p in published["published"]) + ". Their own caveat: " + published["caveat"]]
        if cm_initial:
            ci = cm_initial
            out += [f"- **Initial-policy library, audited with the current heads** (`{ci['audit_file']}`, `{ci['medians_file']}`): the file describes it as "
                    f"\"{ci['selection']}\" (cell `{'/'.join(Path(ci['cell']).parts[-2:])}`): {ci['n_selected']} rows selected from its baseline pool of {ci['pool_rows']} initial-policy samples, scored on the "
                    f"{ci['device'].upper()} with the audit heads: **audit_2020_t5 median {ci['t5']:.4f}** (primary), **audit_2020_esm {ci['esm']:.4f}** (secondary), natural distance median "
                    f"{ci['distance']:.4f}, function_2026_esm median {ci['function']:.4f}. The whole {ci['pool_rows']}-sample pool: audit_2020_t5 {ci['pool_t5']:.4f}, audit_2020_esm "
                    f"{ci['pool_esm']:.4f}, natural distance {ci['pool_distance']:.4f}, function_2026_esm {ci['pool_function']:.4f}. Check recorded in the file: the same CPU path "
                    f"reproduces the audit of the {ci['check_rows']} rows delivered by `{'/'.join(Path(ci['check_cell']).parts[-2:])}` (audit_2020_t5 {ci['check_t5_cpu']:.6f} against {ci['check_t5_audit']:.6f}, "
                    f"largest row difference {ci['check_t5_diff']:.1e}; audit_2020_esm {ci['check_esm_cpu']:.6f} against {ci['check_esm_audit']:.6f}, {ci['check_esm_diff']:.1e}). "
                    f"The file states that the initial pool is the same in every CM cell: {ci['same_pool_in_all_cells']}. The macros are `\\cmInitT`, `\\cmInitE`, `\\cmInitDist`.",
                    "- Against it, the delivered libraries of the finished CM cells (section a.4, same heads, same selection rule): "
                    + "; ".join(f"`{r.root}/{r.arm}` audit_2020_t5 {r.audit['t5_median']:.4f}, audit_2020_esm {r.audit['esm_median']:.4f}, natural distance median "
                                f"{r.audit['natural_distance_median']:.4f}" for r in cm_cells if r.finished)
                    + ". In-silico predictions, not measurements."]
        elif feasibility:
            out += ["- No audited initial-policy library was found (`figures/fig4/data/cm_initial_policy_audit.json` is absent); the heads `audit_2020_t5.joblib` and `audit_2020_esm.joblib` "
                    f"exist ({feasibility['audit_heads_present']}) and the scorer is `figures/fig4/code/score_initial_policy.py` (`make fig4-initial-policy`); not run."]
    return out


# ----- main ------------------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scratch", type=Path, default=SCRATCH_DEFAULT)
    parser.add_argument("--tex", type=Path, default=REPO / "manuscript" / "numbers.tex")
    parser.add_argument("--csv-dir", type=Path, default=REPO / "runs" / "paper_numbers")
    parser.add_argument("--facts", type=Path, default=REPO / "docs" / "reports" / "PAPER_FACTS_20261008.md")
    args = parser.parse_args()
    now = dt.datetime.now(dt.timezone.utc)

    random_manifest = read_json(REPO / BASELINES / "loopseq" / "random" / "manifest.json") or {}
    calibration = random_manifest.get("dnacycp2_calibration") or {}
    boundaries = (calibration.get("low_mid_boundary"), calibration.get("mid_high_boundary"))
    if None in boundaries:
        raise SystemExit("outputs/baselines/loopseq/random/manifest.json carries no dnacycp2_calibration cut points")

    records = read_cells(args.scratch, boundaries)
    readable = [r for r in records if r.cell is not None]
    baseline = read_baseline_values(REPO)
    kl = kl_isolation(REPO)
    comparators = read_comparators(REPO)
    cm_initial = read_cm_initial(REPO)
    candidate_list: list[dict[str, Any]] = []
    mech_list: list[dict[str, Any]] = []
    for r in readable:
        rows = candidate_rows(r)
        candidate_list += rows
        mech_list.append(mechanism_cell(r, rows))
    cand_frame, mech_frame = pd.DataFrame(candidate_list), pd.DataFrame(mech_list)
    round_frame = pd.DataFrame(mechanism_rounds(candidate_list))
    refusal_frame = pd.DataFrame([refusal_row(r) for r in readable])
    lineage_frame = pd.DataFrame([row for r in readable if r.finished for row in lineage_rows(r)])
    loopseq_selected = run_loopseq_selected(records, args.scratch)
    fixed = fixed_recipe_medians(REPO)
    anchor = anchor_values(records)
    macros = build_macros(records, baseline, kl, mech_frame, refusal_frame, lineage_frame, comparators,
                          loopseq_selected["rows"] or None, cm_initial, now, fixed, anchor)
    write_numbers_tex(args.tex, macros, now)

    cap_rows = capability_rows(records)
    stat_rows = status_rows(records, now)

    def audit_frame(task: str) -> pd.DataFrame:
        return pd.DataFrame([dict(root=r.root, cell=r.name, arm=r.arm, seed=r.seed, **r.audit)
                             for r in records if r.task == task and r.finished])

    csv_dir = args.csv_dir
    write_csv(csv_dir / "cells.csv", pd.DataFrame(cap_rows))
    write_csv(csv_dir / "cre_audit.csv", audit_frame("cre"))
    write_csv(csv_dir / "loopseq_audit.csv", audit_frame("loopseq"))
    write_csv(csv_dir / "cm_audit.csv", audit_frame("cm"))
    write_csv(csv_dir / "candidates.csv", cand_frame)
    write_csv(csv_dir / "mechanism_cells.csv", mech_frame)
    write_csv(csv_dir / "mechanism_rounds.csv", round_frame)
    write_csv(csv_dir / "decision_refusals.csv", refusal_frame)
    write_csv(csv_dir / "lineage_updates.csv", lineage_frame)
    if loopseq_selected["rows"]:
        write_csv(csv_dir / "loopseq_selected_baselines.csv", pd.DataFrame(loopseq_selected["rows"]))
        write_csv(csv_dir / "loopseq_replica_checks.csv", pd.DataFrame(loopseq_selected["checks"]))
    cre_b, curve, ls_b = cre_baseline_table(REPO), dnadiff_curve(REPO), loopseq_baseline_table(REPO)
    write_csv(csv_dir / "cre_baselines_calibrated.csv", cre_b)
    write_csv(csv_dir / "dnadiff_matched_budget_curve.csv", curve)
    write_csv(csv_dir / "loopseq_baselines.csv", ls_b)
    write_csv(csv_dir / "macros.csv", pd.DataFrame([{k: v for k, v in m.items() if k != "raw"} | {"value_now": m["raw"]} for m in macros.items]))
    write_csv(csv_dir / "reglm_taco_libraries.csv", pd.DataFrame(list(comparators.values())))
    write_csv(csv_dir / "cre_fixed_recipes.csv", fixed)
    write_csv(csv_dir / "anchor_vs_delivered.csv", anchor)

    # Examples: the finished main-line (else earlier-run) no_self_validation cells of each task, in seed order.
    example_blocks: dict[str, Any] = {}
    for task in ("cre", "loopseq", "cm"):
        main_cells = finished(select(records, task=task, roots=MAIN_ROOTS[task], arm=MAIN_ARM))
        note = ""
        if not main_cells:
            main_cells = finished(select(records, task=task, roots=EARLIER_ROOTS.get(task, ()), arm=MAIN_ARM))
            note = "; no main-line cell is finished, these are the earlier-run cells"
        names = [cell_id(r) for r in main_cells]
        example_blocks[task] = {"cells": names, "note": note,
                                "items": examples(cand_frame, names) if names else [{"rule": rule, "row": None} for rule, _ in EXAMPLE_RULES]}
    cm_extra = cm_section(records, cand_frame, args.scratch, REPO, cm_initial)
    facts = build_facts(records=records, macros=macros, scratch=args.scratch, now=now, baseline=baseline,
                        baseline_frames={"cre": cre_b, "dnadiff": curve, "loopseq": ls_b}, comparators=comparators, kl=kl, cand_frame=cand_frame,
                        mech_frame=mech_frame, round_frame=round_frame, refusal_frame=refusal_frame, lineage_frame=lineage_frame,
                        loopseq_selected=loopseq_selected,
                        example_blocks=example_blocks, cm_extra=cm_extra,
                        cap_rows=cap_rows, stat_rows=stat_rows, cost_rows=cost_rows(records), boundaries=boundaries, repo=REPO)
    atomic_write(args.facts, facts)
    pending = sum(m["raw"] is None for m in macros.items)
    provisional = sum(m["kind"] == "provisional" and m["raw"] is not None for m in macros.items)
    print(f"{len(readable)} cells read ({sum(r.finished for r in readable)} finished, "
          f"{sum(r.status == 'running' for r in readable)} running, {len(records) - len(readable)} unreadable); "
          f"{len(macros.items)} macros, {pending} pending, {provisional} provisional")
    print(f"wrote {args.tex}, {args.facts}, {csv_dir}/*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
