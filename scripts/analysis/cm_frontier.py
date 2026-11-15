"""Predicted score against natural distance for every chorismate-mutase library the paper compares.

A reserved-predictor score rises with the distance of a library from the family, so libraries are compared at
matched distance. This module reads, read-only, the per-sequence reserved ProtT5 score (`audit_2020_t5`), the reserved ESM2
score (`audit_2020_esm`; the published libraries have none) and natural distance of

  * the three delivered libraries of the main-line searches (`terminal_audit/` and the cell's `delivery/` pool),
  * the three initial-recipe (anchor) libraries selected under the same plan (`outputs/anchor_audit_20261022`),
  * the initial policy's library selected under the same plan (`figures/fig4/data/cm_initial_policy_audit_bounded_seed*.json`),
  * the six ProtRL libraries (three delivery methods x two seeds, rescored) and
  * the top-128-by-ProtT5 rows of the two published libraries,

and reduces them to one median per library and distance bin with a bootstrap interval over the sequences of the
library. A unit is the library of one seed; the three ProtRL libraries of a seed share rows, so they form one unit of
their distinct sequences. A unit contributes to a bin only with at least `MIN_ROWS` sequences in it.

    python scripts/analysis/cm_frontier.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for _path in (REPO / "src", REPO / "scripts", REPO / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common import anchor_audit  # noqa: E402
from common.control_cells import MAIN_ROOTS, MAIN_SEEDS, SCRATCH_ROOT  # noqa: E402

(MAIN_ROOT,) = MAIN_ROOTS["cm"]
SEEDS = MAIN_SEEDS["cm"]
FIG_DATA = REPO / "scripts" / "fig4" / "data"
PROTRL = REPO / "outputs" / "cm_baselines_rescored_20260917"
PUBLISHED = REPO / "outputs" / "cm_published_comparators" / "rescored_t5.csv"
SCORE = "audit_2020_t5"
ESM = "audit_2020_esm"  # the second reserved predictor; the figure draws ProtT5, the text also reads ESM2

# Methods in the order the figure and the tables list them; the values are the labels the paper uses.
METHODS = ("RECITAL", "Initial recipe", "Initial policy", "ProtRL", "Russ 2020", "Netti 2026")
PUBLISHED_METHODS = {"Russ_2020_designed": "Russ 2020", "Netti_2026_designed": "Netti 2026"}
EDGES = (0.30, 0.40, 0.50, 0.60)  # the bins of the figure and the text
FINE_EDGES = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)  # a finer reading for the data table
MIN_ROWS = 10  # sequences a library needs in a bin to be summarized there
N_BOOT = 2000
BOOT_SEED = 20261022
TOP_PUBLISHED = 128


def delivered_rows() -> list[pd.DataFrame]:
    """The delivered library of each main-line cell: terminal-audit ProtT5 and the natural distance of the same rows."""
    frames = []
    for seed in SEEDS:
        cell = SCRATCH_ROOT / MAIN_ROOT / f"cm_no_self_validation_seed{seed}"
        manifest = json.loads((cell / "terminal_audit" / "manifest.json").read_text())
        audits = {name: json.loads((cell / "terminal_audit" / f"{name}.json").read_text()) for name in (SCORE, ESM)}
        if any(a["status"] != "complete" or a["row_ids"] != manifest["row_ids"] for a in audits.values()):
            raise SystemExit(f"{cell.name}: the terminal audit is incomplete or not in manifest order")
        pool = json.loads((cell / "delivery" / "pool.json").read_text())
        distance = dict(zip(pool["row_ids"], json.loads(
            (cell / "delivery" / "public_measurements.json").read_text())["measurements"]["natural_distance"]["values"]))
        frames.append(pd.DataFrame({
            "method": "RECITAL", "unit": seed, "library": f"seed {seed}", "sequence": manifest["sequences"],
            "natural_distance": [distance[r] for r in manifest["row_ids"]], SCORE: audits[SCORE]["values"],
            ESM: audits[ESM]["values"]}))
    return frames


def anchor_rows() -> list[pd.DataFrame]:
    frames = []
    for seed in SEEDS:
        cell = f"cm_no_self_validation_seed{seed}"
        manifest = anchor_audit.manifest(cell)
        values, esm = anchor_audit.source_values(cell, SCORE), anchor_audit.source_values(cell, ESM)
        if manifest is None or values is None or esm is None:
            raise SystemExit(f"{cell}: the initial-recipe audit has no complete {SCORE} and {ESM}")
        frames.append(pd.DataFrame({
            "method": "Initial recipe", "unit": seed, "library": f"seed {seed}", "sequence": manifest["sequences"],
            "natural_distance": anchor_audit.library_measurement(cell, "natural_distance"), SCORE: values, ESM: esm}))
    return frames


def initial_policy_rows() -> list[pd.DataFrame]:
    """The 128 rows the cell's plan selects from the initial policy's 512-sequence pool, scored by `score_initial_policy.py`."""
    frames = []
    for seed in SEEDS:
        audit = json.loads((FIG_DATA / f"cm_initial_policy_audit_bounded_seed{seed}.json").read_text())
        cell = Path(audit["cell"])
        if cell.name != f"cm_no_self_validation_seed{seed}":
            raise SystemExit(f"cm_initial_policy_audit_bounded_seed{seed}.json was scored from {cell}")
        pool = json.loads((cell / "baseline" / "pool.json").read_text())
        distance = json.loads((cell / "baseline" / "public_measurements.json").read_text())[
            "measurements"]["natural_distance"]["values"]
        position = {row: i for i, row in enumerate(pool["row_ids"])}
        chosen = [position[row] for row in audit["selected_row_ids"]]
        frames.append(pd.DataFrame({
            "method": "Initial policy", "unit": seed, "library": f"seed {seed}",
            "sequence": [pool["sequences"][i] for i in chosen], "natural_distance": [distance[i] for i in chosen],
            SCORE: audit["selected_128"][SCORE]["values"], ESM: audit["selected_128"][ESM]["values"]}))
    return frames


def protrl_rows() -> list[pd.DataFrame]:
    """One unit per training seed: the distinct sequences of its three rescored 128-row libraries."""
    frames = []
    for path in sorted(PROTRL.glob("matched_seed*_audited_libraries.csv")):
        seed = int(path.name.split("seed")[1].split("_")[0])
        table = pd.read_csv(path)
        distinct = table.drop_duplicates("sequence")
        frames.append(pd.DataFrame({
            "method": "ProtRL", "unit": seed, "library": f"seed {seed}", "sequence": distinct.sequence,
            "natural_distance": distinct.natural_distance, SCORE: distinct[SCORE], ESM: distinct[ESM]}))
    return frames


def published_rows() -> list[pd.DataFrame]:
    """The top 128 rows by the current ProtT5 head of each published library, the selection the paper's tables use."""
    table = pd.read_csv(PUBLISHED)
    frames = []
    for library, method in PUBLISHED_METHODS.items():
        part = table[table.library == library].sort_values("current_t5", ascending=False).head(TOP_PUBLISHED)
        frames.append(pd.DataFrame({
            "method": method, "unit": 0, "library": "top 128", "sequence": part.sequence,
            "natural_distance": part.natural_distance, SCORE: part[SCORE], ESM: np.nan}))
    return frames


def sequences() -> pd.DataFrame:
    """One row per sequence of every library: method, unit (seed), library, sequence, natural distance and ProtT5."""
    frame = pd.concat(delivered_rows() + anchor_rows() + initial_policy_rows() + protrl_rows() + published_rows(),
                      ignore_index=True)
    frame["method"] = pd.Categorical(frame.method, METHODS, ordered=True)
    return frame.sort_values(["method", "unit"], kind="stable").reset_index(drop=True)


def bin_labels(edges: tuple[float, ...]) -> list[str]:
    return [f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(edges[:-1], edges[1:])]


def _median_interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    boot = np.median(values[rng.integers(0, len(values), size=(N_BOOT, len(values)))], axis=1)
    return float(np.median(values)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def binned(frame: pd.DataFrame, edges: tuple[float, ...] = EDGES, open_ends: bool = False, score: str = SCORE,
           methods: tuple[str, ...] = METHODS) -> pd.DataFrame:
    """Median ProtT5 of each unit in each distance bin with a 95% bootstrap interval over its sequences, and the
    method's pooled median per bin (the rows of its units that reach `MIN_ROWS`, resampled within each unit).
    `unit` is the seed; the pooled rows carry unit = -1. With `open_ends` the bins below the first edge and above the last
    are summarized too; they are drawn after the inner bins, so the inner bins' bootstrap draws do not depend on them.
    `score` is the column summarized (the reserved ProtT5 `SCORE` by default); a sequence without it is left out. `methods`
    are the rows of `frame.method` summarized."""
    rng = np.random.default_rng(BOOT_SEED)
    inner = bin_labels(edges)
    outer = [f"<{edges[0]:.2f}", f">{edges[-1]:.2f}"] if open_ends else []
    # Bins are closed on the left; the last inner one also holds the delivery bound itself (a sequence at distance 0.60).
    cuts = list(edges[:-1]) + [edges[-1] + 1e-9]
    if open_ends:
        cuts = [-np.inf] + cuts + [np.inf]
    frame = frame.assign(bin=pd.cut(frame.natural_distance, cuts, labels=outer[:1] + inner + outer[1:], right=False))
    frame = frame[frame[score].notna()]
    rows = []
    for labels in (inner, outer):
        rows += _bin_rows(frame, labels, rng, score, methods)
    return pd.DataFrame(rows)


def _bin_rows(frame: pd.DataFrame, labels: list[str], rng: np.random.Generator, score: str, methods: tuple[str, ...]) -> list[dict]:
    rows = []
    for method in methods:
        part = frame[frame.method == method]
        for label in labels:
            units = {}
            for unit, sub in part[part.bin == label].groupby("unit"):
                units[unit] = sub
                n = len(sub)
                enough = n >= MIN_ROWS
                median, lo, hi = _median_interval(sub[score].to_numpy(float), rng) if enough else (np.nan,) * 3
                rows.append({"method": method, "unit": unit, "bin": label, "n": n, "enough": enough, "median": median,
                             "lo": lo, "hi": hi, "distance_median": float(sub.natural_distance.median())})
            kept = [sub for sub in units.values() if len(sub) >= MIN_ROWS]
            if kept:
                pooled = np.concatenate([sub[score].to_numpy(float) for sub in kept])
                draws = [rng.integers(0, len(sub), size=(N_BOOT, len(sub))) for sub in kept]
                values = [sub[score].to_numpy(float) for sub in kept]
                boot = np.median(np.concatenate([v[d] for v, d in zip(values, draws)], axis=1), axis=1)
                rows.append({"method": method, "unit": -1, "bin": label, "n": len(pooled), "enough": True,
                             "median": float(np.median(pooled)), "lo": float(np.percentile(boot, 2.5)),
                             "hi": float(np.percentile(boot, 97.5)),
                             "distance_median": float(np.median(np.concatenate(
                                 [sub.natural_distance.to_numpy(float) for sub in kept])))})
    return rows


def distribution(frame: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """The rows of Fig. 4c: for each method and each of the five bins (the three inner bins and the two open ends) the number
    of its sequences in the bin, their share of all its sequences, and the pooled median ProtT5 with its interval and the
    sequences it rests on (NaN when no library of the method has `MIN_ROWS` sequences in the bin). `table` is
    `binned(frame, open_ends=True)`."""
    labels = bin_labels(EDGES)
    labels = [f"<{EDGES[0]:.2f}"] + labels + [f">{EDGES[-1]:.2f}"]
    cuts = [-np.inf] + list(EDGES[:-1]) + [EDGES[-1] + 1e-9, np.inf]
    frame = frame.assign(bin=pd.cut(frame.natural_distance, cuts, labels=labels, right=False))
    pooled = table[table.unit == -1].set_index(["method", "bin"])
    rows = []
    for method in METHODS:
        part = frame[frame.method == method]
        for label in labels:
            n_all = int((part.bin == label).sum())
            row = {"method": method, "bin": label, "n_all": n_all, "share": n_all / len(part), "n": np.nan,
                   "median": np.nan, "lo": np.nan, "hi": np.nan}
            if (method, label) in pooled.index:
                found = pooled.loc[(method, label)]
                row.update(n=int(found.n), median=found["median"], lo=found.lo, hi=found.hi)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=FIG_DATA)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frame = sequences()
    frame.to_csv(args.out / "cm_frontier_sequences.csv", index=False, float_format="%.6g")
    table = binned(frame)
    table.to_csv(args.out / "cm_frontier_bins.csv", index=False, float_format="%.4g")
    binned(frame, FINE_EDGES).to_csv(args.out / "cm_frontier_bins_fine.csv", index=False, float_format="%.4g")
    distribution(frame, binned(frame, open_ends=True)).to_csv(args.out / "cm_frontier_distribution.csv", index=False,
                                                              float_format="%.4g")
    print(table[table.unit == -1].to_string())


if __name__ == "__main__":
    main()
