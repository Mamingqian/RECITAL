"""Independent endpoints of the chorismate-mutase libraries, by distance bin.

Joins the per-sequence scores the endpoint scorers wrote under the endpoint directory

  poet2_response.json   PoET-2 mean conditional log likelihood per predicted token (`slurm/cm_endpoints_poet2.sbatch`)
  esmfold_scores.csv    ESMFold mean pLDDT (`slurm/cm_endpoints_esmfold.sbatch`)
  tmscore.csv           TM-score of the ESMFold structure to the natural structures (`scripts/analysis/cm_tmscore.py`)
  conservation.csv      retention of the strongly conserved family positions (`scripts/analysis/cm_conservation.py`)

to the library rows of `scripts/analysis/cm_endpoints_inputs.py` and reduces each endpoint with the distance bins and the
bootstrap intervals of `scripts/analysis/cm_frontier.py`. None of these scorers was part of any search's reward, ranking
or delivery; ESMFold shares the ESM2 lineage of the reward's Function predictor and of the reserved ESM2 head, so its scores
are structure-prediction scores and not an independent evaluation. Writes `cm_endpoints_sequences.csv` and
`cm_endpoints_bins.csv` under figures/fig4/data/.

    python scripts/analysis/cm_endpoints.py [--endpoints DIR] [--out DIR]
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

from analysis import cm_frontier  # noqa: E402
from common.control_cells import SCRATCH_ROOT  # noqa: E402

ENDPOINT_DIR = SCRATCH_ROOT / "outputs" / "cm_endpoints_20261022"
OUT = REPO / "scripts" / "fig4" / "data"
NATURAL = "Natural family"
METHODS = cm_frontier.METHODS + (NATURAL,)
# column -> axis label of the figure
ENDPOINTS = {
    "poet2": "PoET-2 log-likelihood per residue",
    "plddt": "ESMFold pLDDT",
    "tm_experimental": "TM-score to natural structures",
    "retention": "Conserved positions retained",
}


def read_scores(folder: Path) -> pd.DataFrame:
    """One row per distinct sequence with every endpoint score the directory holds."""
    sequences = pd.read_csv(folder / "sequences.csv")
    table = sequences[["id", "sequence"]].copy()
    response = folder / "poet2_response.json"
    if response.is_file():
        request = json.loads((folder / "poet2_request.json").read_text())
        values = json.loads(response.read_text())["log_likelihood_per_token"]
        if len(values) != len(request["sequences"]):
            raise SystemExit("poet2_response.json does not hold one value per requested sequence")
        table = table.merge(pd.DataFrame({"sequence": request["sequences"], "poet2": values}), on="sequence", how="left")
    if (folder / "esmfold_scores.csv").is_file():
        esm = pd.read_csv(folder / "esmfold_scores.csv")
        table = table.merge(esm[["id", "mean_plddt"]].rename(columns={"mean_plddt": "plddt"}), on="id", how="left")
    if (folder / "tmscore.csv").is_file():
        table = table.merge(pd.read_csv(folder / "tmscore.csv"), on="id", how="left")
    if (folder / "conservation.csv").is_file():
        cons = pd.read_csv(folder / "conservation.csv")
        table = table.merge(cons[["id", "retention", "retention_strict", "n_conserved", "n_conserved_strict"]], on="id",
                            how="left")
    return table


def library_endpoints(folder: Path) -> pd.DataFrame:
    rows = pd.read_csv(folder / "library_rows.csv")
    merged = rows.merge(read_scores(folder).drop(columns=["id"]), on="sequence", how="left", validate="many_to_one")
    merged["method"] = pd.Categorical(merged.method, METHODS, ordered=True)
    return merged


def binned(frame: pd.DataFrame) -> pd.DataFrame:
    """The bin table of every endpoint present in `frame` (columns: endpoint, method, unit, bin, n, median, lo, hi, ...)."""
    parts = []
    for column in ENDPOINTS:
        if column in frame and frame[column].notna().any():
            parts.append(cm_frontier.binned(frame, score=column, methods=METHODS).assign(endpoint=column))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--endpoints", type=Path, default=ENDPOINT_DIR)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frame = library_endpoints(args.endpoints)
    frame.to_csv(args.out / "cm_endpoints_sequences.csv", index=False, float_format="%.6g")
    table = binned(frame)
    table.to_csv(args.out / "cm_endpoints_bins.csv", index=False, float_format="%.4g")
    pooled = table[table.unit == -1]
    print(pooled.pivot_table(index=["endpoint", "method"], columns="bin", values="median", observed=True).round(3).to_string())


if __name__ == "__main__":
    main()
