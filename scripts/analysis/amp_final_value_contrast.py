#!/usr/bin/env python3
"""The amp pass rule of design section 6 on the final value of each search instead of its AUBC: the equal-update test of section 8b item 6.

    amp_final_value_contrast.py --arm CSV --comparator CSV --n-arm 6 --n-comparator 6 [--out JSON]

Each CSV is the `module_endpoints.csv` of one group of cells. AUBC credits a library found early and held (a search that commits fewer
updates keeps its value to the end of the declared axis), so this takes `final_best`, the best qualified library at the end of the
search, one value per search, in its place: the same normalization (the comparator's mean final value), floor (the sample standard
deviation of the comparator's normalized values), exact one-sided permutation p, blind allowance and failure handling as
`module_contrast.amp_contrast`, which it calls. The two-sided p of the same values is added for the core-against-closer report.
It is a descriptive report: the sizes are command-line numbers, so it is not part of the pass rule until it reads them from the tagged
`declared.json` and an audit pass reads it (design section 8b item 6).
Run with PYTHONPATH including src and scripts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analysis import module_contrast as mc


def final_value_contrast(arm: pd.DataFrame, comparator: pd.DataFrame, *, n_arm: int, n_comparator: int) -> dict:
    result = mc.amp_contrast(arm.assign(aubc=arm["final_best"]), comparator.assign(aubc=comparator["final_best"]),
                             n_arm=n_arm, n_comparator=n_comparator)
    # `prepare` gives a search with no qualified library the final value 0, as it does for the AUBC.
    arm_final = mc.prepare("arm", arm.assign(aubc=arm["final_best"]), size=n_arm)["final_best"] / result["normalizer"]
    comparator_final = mc.prepare("comparator", comparator.assign(aubc=comparator["final_best"]),
                                  size=n_comparator)["final_best"] / result["normalizer"]
    result["p_two_sided"] = mc.permutation_p(arm_final.tolist(), comparator_final.tolist(), alternative="two-sided")
    result["endpoint"] = "final_best"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--comparator", type=Path, required=True)
    parser.add_argument("--n-arm", type=int, required=True)
    parser.add_argument("--n-comparator", type=int, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = final_value_contrast(pd.read_csv(args.arm), pd.read_csv(args.comparator),
                                  n_arm=args.n_arm, n_comparator=args.n_comparator)
    text = json.dumps(result, indent=2, default=float)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
