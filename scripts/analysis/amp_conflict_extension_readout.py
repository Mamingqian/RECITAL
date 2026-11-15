#!/usr/bin/env python3
"""The 12-against-12 readout of the extension of the AMP conflict objective: reward-only and core on seeds 3101 to 3112, descriptive.

    amp_conflict_extension_readout.py --arm NAME=ROOT[:SIZE][,ROOT[:SIZE] ...] [NAME=...] [--size 6] [--arm-csv CSV --comparator-csv CSV] [--out CSV]

An arm is the cells of one or more roots, here the batch's root (seeds 3101 to 3106) and the extension's (3107 to 3112), read by the batch's readout
(`amp_conflict_readout.py`, unchanged): the cells with a qualified library, their delivered means and what the agent did, and for every ordered pair of arms the
exact one-sided Fisher p of the count of cells with a qualified library (12 against 12 for reward-only and core, 12 against the fixed recipe's 6). A seed must
be in 3101 to 3112 and in one cell of its arm. The counts are also printed by part (the batch's seeds, the extension's).

`--arm-csv` and `--comparator-csv` are the `module_endpoints.csv` of the core's 12 cells and of the reward-only arm's 12 (`module_endpoints.py ROOT ROOT`): the amp
contrast of `module_contrast.py` is computed on them with their own sizes (normalized AUBC of the best qualified library's mean activity_ecoli, the exact one-sided
permutation p, the APEX median -log10 MIC difference) and printed without its pass and rule fields. `module_contrast.py amp` itself cannot read them: it refuses
declared sizes committed after the earliest cell of the arm began, and the batch's cells began before the extension was declared.

Everything here is labelled `extension, descriptive`: the declared P1 and P2 stay the batch's 6-against-6 readout, and no rule or threshold is attached to a number
printed here. Nothing is read from a terminal audit by this script; the APEX values are those `module_endpoints.py` wrote into the CSV.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import pandas as pd

from analysis import amp_conflict_readout as readout
from analysis import module_contrast as contrast

SEEDS = range(3101, 3113)
EXTENSION_FIRST_SEED = 3107
NOT_A_READING = ("passes", "rule_met", "rule", "above_floor", "passes_if_failures_left_out_of_blind", "blind_not_worse", "blind_not_worse_frozen_only")
LABEL = "extension, descriptive"


def read_arms(items: Sequence[str], default_size: int = 6) -> pd.DataFrame:
    """The cells of every arm, one row each, with the seed and the part (batch or extension) of the cell."""
    frames = []
    for item in items:
        name, spec = item.split("=", 1)
        for part in spec.split(","):
            root, _, size = part.partition(":")
            frames.append(readout.read_arm(name, Path(root), int(size) if size else default_size))
    cells = pd.concat(frames, ignore_index=True)
    cells["seed"] = cells["cell"].map(lambda name: int(re.search(r"seed(\d+)$", name).group(1)))
    outside = cells[~cells["seed"].isin(SEEDS)]
    if len(outside):
        raise SystemExit(f"seeds outside 3101 to 3112: {sorted(set(zip(outside['arm'], outside['seed'])))}")
    repeated = cells[cells.duplicated(["arm", "seed"], keep=False)]
    if len(repeated):
        raise SystemExit(f"a seed is in two cells of one arm: {sorted(set(zip(repeated['arm'], repeated['seed'])))}")
    cells["part"] = cells["seed"].map(lambda seed: "extension" if seed >= EXTENSION_FIRST_SEED else "batch")
    return cells


def counts_by_part(cells: pd.DataFrame) -> pd.DataFrame:
    return (cells.groupby(["arm", "part"], sort=False)["qualified"].agg(cells="size", cells_with_a_qualified_library="sum").reset_index()
            .astype({"cells_with_a_qualified_library": int}))


def descriptive_contrast(arm_csv: Path, comparator_csv: Path) -> dict:
    """The amp contrast of the module_endpoints tables of two arms at their own sizes, without the fields that read as a pass or a fail."""
    arm, comparator = pd.read_csv(arm_csv), pd.read_csv(comparator_csv)
    result = contrast.amp_contrast(arm, comparator, n_arm=len(arm), n_comparator=len(comparator))
    return {key: value for key, value in result.items() if key not in NOT_A_READING}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm", nargs="+", required=True, metavar="NAME=ROOT[:SIZE][,ROOT[:SIZE]]", help="the fixed recipe first, then the searches")
    parser.add_argument("--size", type=int, default=6, help="the default number of cells of a root")
    parser.add_argument("--arm-csv", type=Path)
    parser.add_argument("--comparator-csv", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if bool(args.arm_csv) != bool(args.comparator_csv):
        raise SystemExit("--arm-csv and --comparator-csv go together")
    cells = read_arms(args.arm, args.size)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(f"{LABEL}: the declared P1 and P2 are the batch's 6-against-6 readout (amp_conflict_readout.py); nothing below carries a rule\n")
    print(cells.drop(columns=["seed"]).round(3).to_string(index=False))
    print(f"\ncells with a qualified library by part ({LABEL})\n" + counts_by_part(cells).to_string(index=False))
    print(f"\narm table ({LABEL})\n" + readout.arm_table(cells.drop(columns=["seed", "part"])).round(4).to_string(index=False))
    if args.arm_csv:
        print(f"\namp contrast of the module_endpoints tables ({LABEL}; no pass or fail field)")
        print("  module_contrast.py amp is not the reader here: its declared-tag check needs sizes committed before the arm's first cell, and the batch's cells "
              "began before the extension was declared")
        for key, value in descriptive_contrast(args.arm_csv, args.comparator_csv).items():
            print(f"  {key}: {value}")
    if args.out:
        cells.to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
