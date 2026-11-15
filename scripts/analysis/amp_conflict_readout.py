#!/usr/bin/env python3
"""The in-loop readout of the AMP conflict-objective batch: which cells deliver a library that meets all four floors, and what the agent did.

    amp_conflict_readout.py --arm NAME=ROOT[:SIZE] [NAME=ROOT[:SIZE] ...] [--size 6] [--out CSV] [--release-check]

One row per cell of each arm root: its status, whether it delivered a qualified library, the delivered library's mean of the five columns
(activity_ecoli, amp_likelihood, nontoxic, nonhemolytic, plddt_esmfold) and which of the contract's library-mean floors it meets (the floors
are read from the cell's own `declaration.json`), the share of its candidates that qualified, and what the agent did: candidates that enabled
the nonhemolytic or the pLDDT source, candidates that used an aggregator other than the weighted sum, component requests made. The arm table
counts the cells with a qualified library and gives, for every ordered pair of arms, the exact one-sided Fisher p that the row's arm has more
cells with a qualified library than the column's (the count tests P1, each LLM arm against the fixed recipe, and P2, core against reward-only).
`:SIZE` after a root is that arm's declared number of cells (the LLM arms hold 3 cells after stage 1), `--size` the default.

`--release-check` reads the cells of seeds 3101 to 3103 of the given arms (stage 1) and exits 0 when at least one ended with a qualified library,
1 when none did and 2 when a stage-1 cell has not ended; it is the data half of the release of stage 2, which also needs the program manager's word.

Nothing is read from a terminal audit or an APEX head: the delivered library is found through `delivery/assignment.json`, and the blind read
of the batch is `module_endpoints.py` (the audit's primary value) after the operational lines. Descriptive only, beside the declared rule.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

import pandas as pd
from scipy.stats import fisher_exact

COLUMNS = ("activity_ecoli", "amp_likelihood", "nontoxic", "nonhemolytic", "plddt_esmfold")
STAGE_1_SEEDS = (3101, 3102, 3103)
FLOOR_COLUMN = {"nontoxic_mean": "nontoxic", "amp_likelihood_mean": "amp_likelihood", "nonhemolytic_mean": "nonhemolytic",
                "plddt_mean": "plddt_esmfold"}


def cells_of(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if (path / "events" / "log.jsonl").is_file())


def floors_of(cell: Path) -> dict[str, float]:
    declaration = json.loads((cell / "declaration.json").read_text())
    requirements = declaration["contract"]["delivery_requirements"]["success"]["requirements"]
    return {item["metric"]: item["minimum"] for item in requirements if item["metric"] in FLOOR_COLUMN}


def events_of(cell: Path) -> list[dict]:
    return [json.loads(line) for line in (cell / "events" / "log.jsonl").read_text().splitlines() if line]


def delivered_means(cell: Path) -> dict[str, float] | None:
    """The mean of each column over the delivered library, or None when the cell delivered none."""
    directory = cell / "delivery"
    if not (directory / "assignment.json").is_file():
        return None
    ids = json.loads((directory / "delivery_artifact.json").read_text())["selected_row_ids"]
    measurements = json.loads((directory / "public_measurements.json").read_text())["measurements"]
    means = {}
    for name in COLUMNS:
        entry = measurements[name]
        position = {row: i for i, row in enumerate(entry["row_ids"])}
        values = [entry["values"][position[row]] for row in ids]
        means[name] = float("nan") if any(v is None for v in values) else sum(values) / len(values)
    return means


def behaviour(cell: Path) -> dict[str, int]:
    """What the agent did, from the candidates' job specs and the log."""
    by_candidate = {}
    for path in sorted((cell / "jobs").glob("cand_*_a*/spec.json")):
        by_candidate[path.parent.name.rsplit("_a", 1)[0]] = json.loads(path.read_text())["train"]["recipe"]
    recipes = list(by_candidate.values())
    return {"candidates": len(recipes),
            "with_nonhemolytic": sum("nonhemolytic" in r["enabled_oracles"] for r in recipes),
            "with_plddt": sum("plddt_esmfold" in r["enabled_oracles"] for r in recipes),
            "other_aggregator": sum(r["aggregator"]["name"] != "weighted_sum" for r in recipes),
            "component_requests": sum(e["type"] == "component_attempt" for e in events_of(cell))}


def candidates_qualified(cell: Path) -> int:
    return sum(json.loads(path.read_text()).get("qualified", False) is True for path in sorted((cell / "candidates").glob("cand_*/delivery_artifact.json")))


def read_arm(name: str, root: Path, size: int | None) -> pd.DataFrame:
    cells = cells_of(root)
    if size is not None and len(cells) != size:
        raise SystemExit(f"{root}: {len(cells)} cells, the declared size is {size}")
    rows = []
    for cell in cells:
        events = events_of(cell)
        finished = [e for e in events if e["type"] == "run_finished"]
        floors, means = floors_of(cell), delivered_means(cell)
        row = {"arm": name, "cell": cell.name, "status": finished[-1]["data"]["status"] if finished else "running",
               "qualified": means is not None, "candidates_qualified": candidates_qualified(cell), **behaviour(cell)}
        for column in COLUMNS:
            row[column] = float("nan") if means is None else means[column]
        for metric, column in FLOOR_COLUMN.items():
            row[f"meets_{column}"] = None if means is None or metric not in floors else bool(means[column] >= floors[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def arm_table(cells: pd.DataFrame) -> pd.DataFrame:
    counts = {arm: (int(group["qualified"].sum()), len(group)) for arm, group in cells.groupby("arm", sort=False)}
    rows = []
    for arm, group in cells.groupby("arm", sort=False):
        qualified, size = counts[arm]
        p = {f"p_more_qualified_than_{other}": float("nan") if other == arm else fisher_exact(
            [[qualified, size - qualified], [counts[other][0], counts[other][1] - counts[other][0]]], alternative="greater")[1] for other in counts}
        rows.append({"arm": arm, "cells": size, "cells_with_a_qualified_library": qualified, **p, "candidates": int(group["candidates"].sum()),
                     "candidates_qualified": int(group["candidates_qualified"].sum()),
                     **{f"mean_{column}_of_qualified": float(group.loc[group["qualified"], column].mean()) for column in COLUMNS},
                     "candidates_with_nonhemolytic": int(group["with_nonhemolytic"].sum()), "candidates_with_plddt": int(group["with_plddt"].sum()),
                     "candidates_other_aggregator": int(group["other_aggregator"].sum()), "component_requests": int(group["component_requests"].sum())})
    return pd.DataFrame(rows)


def release_check(cells: pd.DataFrame) -> tuple[int, str]:
    """The exit code and the line of the stage-2 release check over the stage-1 cells (seeds 3101 to 3103) of the arms read."""
    seeds = cells["cell"].map(lambda name: int(re.search(r"seed(\d+)$", name).group(1)))
    stage_1 = cells[seeds.isin(STAGE_1_SEEDS)]
    if stage_1.empty or (stage_1["status"] == "running").any():
        return 2, f"{len(stage_1)} stage-1 cells read, {int((stage_1['status'] == 'running').sum())} not ended: the release check needs every stage-1 cell to have ended"
    qualified = int(stage_1["qualified"].sum())
    return (0 if qualified else 1), (f"{qualified} of {len(stage_1)} stage-1 LLM cells ended with a qualified library; "
                                     + ("the data condition for releasing stage 2 is met" if qualified else "none did: no stage 2, report a feasibility negative"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm", nargs="+", required=True, metavar="NAME=ROOT[:SIZE]", help="the fixed recipe first, then the searches")
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--release-check", action="store_true")
    args = parser.parse_args(argv)
    frames = []
    for item in args.arm:
        name, spec = item.split("=", 1)
        root, _, size = spec.partition(":")
        frames.append(read_arm(name, Path(root), int(size) if size else args.size))
    cells = pd.concat(frames, ignore_index=True)
    if args.release_check:
        code, line = release_check(cells)
        print(line)
        return code
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(cells.round(3).to_string(index=False))
    print(arm_table(cells).round(4).to_string(index=False))
    if args.out:
        cells.to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
