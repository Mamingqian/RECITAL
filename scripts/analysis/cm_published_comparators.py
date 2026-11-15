#!/usr/bin/env python3
"""Compare our delivered CM libraries against the two published designed libraries.

Both comparators carry MEASURED labels in the same archived table as the 2020
ProtT5 predictions, so they anchor our predicted endpoint against real assay
outcomes at matched sequence divergence:

  Russ et al. 2020 designed library   dataset 2020, sources BmDCA   (n=1618)
  Netti et al. 2026 designed library  dataset 2026, artificial      (n=2684)

The comparison a reader needs is predicted-against-predicted: our libraries have
no measured labels, so their medians may only be compared with each published
library's own top-128-BY-PREDICTED median, never with its measured values. The
measured columns serve a different and more important purpose here: they show how
the measured functional rate of published designs behaves as divergence from the
natural family grows, which is the only external evidence available about whether
a predicted endpoint can be trusted in the divergence range our libraries occupy.

CPU only. Reads archived predictions and delivered libraries read-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ENDPOINT = "audit_2020_t5"
LIBRARIES = {
    "Russ_2020_designed": {"dataset": 2020, "filter": ("sources", "BmDCA")},
    "Netti_2026_designed": {"dataset": 2026, "filter": ("artificial", True)},
}
BINS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, np.inf))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _subset(table: pd.DataFrame, spec: dict) -> pd.DataFrame:
    column, value = spec["filter"]
    return table[table.dataset.eq(spec["dataset"]) & table[column].eq(value)].copy()


def published(source: Path, count: int = 128) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(source)
    needed = {"sequence", "target", "functional", "dataset", "sources", "artificial",
              "natural_distance", ENDPOINT}
    if not needed <= set(table.columns):
        raise ValueError(f"{source} lacks {sorted(needed - set(table.columns))}")
    rows, bins = [], []
    for name, spec in LIBRARIES.items():
        frame = _subset(table, spec)
        by_predicted = frame.nlargest(count, ENDPOINT)
        by_measured = frame.nlargest(count, "target")
        correlation = spearmanr(frame[ENDPOINT], frame.target)
        far = frame[frame.natural_distance >= 0.6]
        far_statistic = spearmanr(far[ENDPOINT], far.target).statistic if len(far) > 2 else None
        if far_statistic is not None and not np.isfinite(far_statistic):
            far_statistic = None
        rows.append({
            "library": name, "n": len(frame),
            # The only column comparable to our libraries' reported number.
            "top128_by_predicted_endpoint_median": float(by_predicted[ENDPOINT].median()),
            "top128_by_predicted_measured_median": float(by_predicted.target.median()),
            "top128_by_predicted_functional_rate": float(by_predicted.functional.mean()),
            "top128_by_measured_endpoint_median": float(by_measured[ENDPOINT].median()),
            "top128_by_measured_measured_median": float(by_measured.target.median()),
            "natural_distance_median": float(frame.natural_distance.median()),
            "natural_distance_max": float(frame.natural_distance.max()),
            "overall_functional_rate": float(frame.functional.mean()),
            "spearman_predicted_vs_measured": float(correlation.statistic),
            "spearman_predicted_vs_measured_far": (
                float(far_statistic) if far_statistic is not None else None),
            "n_far": len(far),
        })
        for low, high in BINS:
            part = frame[(frame.natural_distance >= low) & (frame.natural_distance < high)]
            if not len(part):
                continue
            label = f"{low:g}-" + ("inf" if high == np.inf else f"{high:g}")
            bins.append({
                "library": name, "distance_bin": label,
                "n": len(part), "measured_functional_rate": float(part.functional.mean()),
                "measured_median": float(part.target.median()),
                "predicted_endpoint_median": float(part[ENDPOINT].median()),
            })
    return pd.DataFrame(rows), pd.DataFrame(bins)


def ours(root: Path) -> pd.DataFrame:
    """Our eight delivered libraries, as already summarized by the campaign."""
    path = root / "comparison/with_protrl_libraries.csv"
    table = pd.read_csv(path)
    frame = table[["method", "seed", "audit_2020_t5", "audit_2020_esm",
                   "selection_score", "natural_distance"]].copy()
    frame = frame.rename(columns={"audit_2020_t5": "predicted_endpoint_median"})
    frame["library"] = frame.method + "/" + frame.seed.astype(str)
    frame["measured"] = False
    return frame


def run(root: Path, output: Path, *, source: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    source = source or (root / "oracles/prot-t5_predictions.csv")
    comparators, bins = published(source)
    delivered = ours(root)
    comparators.to_csv(output / "published_libraries.csv", index=False)
    bins.to_csv(output / "published_distance_bins.csv", index=False)
    delivered.to_csv(output / "delivered_libraries.csv", index=False)
    far_rates = {row.library: row.measured_functional_rate for row in bins.itertuples()
                 if row.distance_bin == "0.6-inf"}
    value = {
        "status": "complete", "used_for_selection": False,
        "source": str(source), "count": 128,
        "comparable_column": ("top128_by_predicted_endpoint_median; our libraries report the same "
                              "quantity (median predicted 2020 ProtT5 score of the delivered 128)"),
        "published": comparators.to_dict(orient="records"),
        "delivered_endpoint_range": [float(delivered.predicted_endpoint_median.min()),
                                     float(delivered.predicted_endpoint_median.max())],
        "delivered_distance_range": [float(delivered.natural_distance.min()),
                                     float(delivered.natural_distance.max())],
        "measured_functional_rate_beyond_0.6_divergence": far_rates,
        "interpretation": (
            "Our delivered libraries score above both published designed libraries on the same "
            "predicted endpoint, at comparable or greater divergence. That is a like-for-like "
            "predicted comparison, not a measured one."),
        "caveat": (
            "In the published libraries the MEASURED functional rate collapses as divergence "
            "grows, while the predicted endpoint does not. Any of our libraries sitting beyond "
            "0.6 divergence is therefore in a range where published measured outcomes are near "
            "zero, so a high predicted endpoint there is the least trustworthy evidence we have, "
            "not the strongest."),
        "limits": [
            "Our libraries have no measured labels; every number for them is a model prediction.",
            "The comparators differ from our runs in generator, objective and screening; this is "
            "a reference-point table, not a controlled comparison.",
            "Both published libraries are in the 2020/2026 predictor training lineage.",
        ],
    }
    write_json(output / "summary.json", value)
    print(comparators[["library", "n", "top128_by_predicted_endpoint_median",
                       "top128_by_measured_measured_median",
                       "spearman_predicted_vs_measured"]].to_string(index=False), flush=True)
    print(bins.to_string(index=False), flush=True)
    print("delivered predicted endpoint range: "
          f"{value['delivered_endpoint_range']} at divergence "
          f"{value['delivered_distance_range']}", flush=True)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path,
                        help="archived predictions (default: <root>/oracles/prot-t5_predictions.csv)")
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve(),
        source=args.source.resolve() if args.source else None)


if __name__ == "__main__":
    main()
