"""Per-seed table of the no-search CM control against the searches and the 256-update initial recipe.

One row per cell of the control (`common.control_cells.NOSEARCH_ROOT`: the declared initial recipe trained once for each of
`NOSEARCH_STEPS` updates, per seed) with its status, the reserved ProtT5 and ESM2 medians and the median natural distance of the
delivered library, and, for the same seed, the delivered library of the search (`figures/fig4/data/cm_terminal.csv`) and the
library of the initial recipe after its 256 updates (`cm_anchor.csv`). A cell that ended `completed_no_qualified_candidate` has
no library and no audit: its row says so and gives the median natural distance of the candidate's pool, the requirements it
failed, and the rows its selection took and their pairwise-distance ratio against the initial policy's pool (the health floor
is 0.20). Writes `figures/fig4/data/cm_nosearch.csv` and prints the table.

    python scripts/analysis/cm_nosearch_table.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for _path in (REPO / "src", REPO / "scripts", REPO / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common.control_cells import MAIN_SEEDS, NOSEARCH_ROOT, NOSEARCH_STEPS, SCRATCH_ROOT, load_campaign  # noqa: E402

FIG_DATA = REPO / "scripts" / "fig4" / "data"
AUDITS = ("audit_2020_t5", "audit_2020_esm")


def cell_row(cell) -> dict:
    """What one control cell holds: its status, its one candidate's pool and, when delivered and audited, the library."""
    row = {"cell": cell.name, "seed": cell.seed, "status": cell.status, "audited": cell.audited}
    frame = cell.candidates()
    if frame.empty:
        return row
    candidate = frame.iloc[0]
    row.update(steps=int(candidate.steps), candidate_status=candidate.status, qualified=candidate.qualified,
               pool_natural_distance=candidate.get("pool__CM__natural_distance"),
               library_natural_distance=candidate.get("library__CM__natural_distance"),
               library_function_2026_esm=candidate.get("library__CM__function_2026_esm"),
               wall_h=(candidate.wall_s / 3600.0 if pd.notna(candidate.wall_s) else None),
               failing=",".join(candidate.get("failing__CM") or []))
    row.update(library_rows=None, library_pairwise_ratio=None)
    evaluation = cell.path / "candidates" / str(candidate.candidate) / "evaluation.json"
    if evaluation.is_file():
        health = json.loads(evaluation.read_text())["delivery"]["health"]["by_condition"]["CM"]
        row.update(library_rows=int(health["library"]["n"]), library_pairwise_ratio=health["requirement"]["value"])
    if cell.audited:
        for name in AUDITS:
            row[name] = cell.audit["sources"][name]["per_condition"]["CM"]["value"]
    return row


def build() -> pd.DataFrame:
    root = SCRATCH_ROOT / NOSEARCH_ROOT
    rows = [cell_row(cell) for cell in load_campaign(root)] if root.is_dir() else []
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["steps"] = table.cell.str.extract(r"steps(\d+)")[0].astype(int)  # a cell that has trained nothing yet has no candidate row
    return table.sort_values(["seed", "steps"]).reset_index(drop=True)


def by_seed(control: pd.DataFrame) -> pd.DataFrame:
    """The control cells beside the search's delivered library and the 256-update initial recipe, one row per seed and update count."""
    search = pd.read_csv(FIG_DATA / "cm_terminal.csv").set_index("seed")
    anchor = pd.read_csv(FIG_DATA / "cm_anchor.csv").set_index("seed")
    rows = []
    for seed in MAIN_SEEDS["cm"]:
        rows.append({"seed": seed, "library": "search", "updates": "~4,000 (search)", "status": "frozen",
                     **{k: search.loc[seed, k] for k in (*AUDITS, "library_natural_distance")}})
        rows.append({"seed": seed, "library": "initial recipe", "updates": 256, "status": "frozen",
                     **{k: anchor.loc[seed, k] for k in (*AUDITS, "library_natural_distance")}})
        for steps in NOSEARCH_STEPS:
            part = control[(control.seed == seed) & (control.steps == steps)] if len(control) else control
            if part.empty:
                rows.append({"seed": seed, "library": "declared recipe", "updates": steps, "status": "not started"})
                continue
            r = part.iloc[0]
            rows.append({"seed": seed, "library": "declared recipe", "updates": steps, "status": r.status,
                         **{k: r.get(k) for k in (*AUDITS, "library_natural_distance", "pool_natural_distance", "failing")}})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=FIG_DATA)
    args = parser.parse_args()
    control = build()
    args.out.mkdir(parents=True, exist_ok=True)
    control.to_csv(args.out / "cm_nosearch.csv", index=False, float_format="%.6g")
    print(by_seed(control).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
