#!/usr/bin/env python3
"""The blind endpoint of the AMP broad-spectrum batch: the worst-pathogen APEX MIC of the delivered 30.

    amp_broad_blind_readout.py --selftest
    amp_broad_blind_readout.py --declared-tag TAG --fixed ROOT --reward-only ROOT --core ROOT [--out DIR]
    amp_broad_blind_readout.py --declared-tag TAG CELL_OR_ROOT [CELL_OR_ROOT ...] [--out DIR]

For each cell with a complete terminal audit: for every delivered peptide the largest of its three APEX 1.1 MICs (E. coli,
S. aureus, P. aeruginosa, micromolar; a peptide is only as good as against its weakest pathogen); `B` = minus log10 of the median of
those values over the delivered 30 (higher is better). Secondary columns: the three per-pathogen medians and the minimum over
pathogens of minus log10 of the per-pathogen medians. A cell without a complete terminal audit has no `B`.

With `--fixed`, `--reward-only` and `--core` the script also reads the two blind conditions of the declaration from the median `B` of
each arm: P1, each LLM arm's median `B` minus the fixed recipe's at least +0.1; P2, core's median `B` minus reward-only's at least
-0.1. Each is printed over the cells that have a `B` and on the conservative reading, on which an arm with a cell that has no `B`
does not meet it (the convention of `module_contrast.py`, whose own blind clause reads the E. coli head only and is not used for
this batch). Every arm must hold the declared six cells.

The terminal audit is read once per cell and nothing here enters a prompt, a reward or a delivery choice. It refuses unless the tag
exists, is more than 60 s old and holds `campaign.json`, `declared.json`, the declaration and this script byte for byte. The batch's
declaration is `docs/reports/tier1_item9_amp_broad_spectrum_declaration_2026-11-08.md`.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

HEADS = ("apex_ecoli", "apex_saureus", "apex_paeruginosa")
HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
CAMPAIGN = REPO / "experiments/recital_amp_broad_stage1_20261109"
DECLARATION = REPO / "docs/reports/tier1_item9_amp_broad_spectrum_declaration_2026-11-08.md"
DECLARED_FILES = (CAMPAIGN / "campaign.json", CAMPAIGN / "declared.json", DECLARATION, HERE)
MARGIN = 0.1  # log10 MIC: what P1 asks of a search over the fixed recipe, and what P2 allows core below reward-only
EPS = 1e-9


def worst_pathogen_row(values: dict[str, Sequence[float]]) -> np.ndarray:
    """Per peptide, the largest MIC over the three heads."""
    return np.max(np.vstack([np.asarray(values[h], dtype=float) for h in HEADS]), axis=0)


def endpoint(values: dict[str, Sequence[float]]) -> dict[str, float]:
    worst = worst_pathogen_row(values)
    medians = {h: float(np.median(np.asarray(values[h], dtype=float))) for h in HEADS}
    return {"B": -math.log10(float(np.median(worst))), "worst_pathogen_median_um": float(np.median(worst)),
            **{f"{h}_median_um": m for h, m in medians.items()},
            "min_over_pathogens_neg_log10_median": min(-math.log10(m) for m in medians.values())}


def read_cell(cell: Path) -> dict[str, float | str]:
    audit = cell / "terminal_audit"
    summary = json.loads((audit / "summary.json").read_text())
    if summary.get("status") != "complete":
        raise SystemExit(f"{cell.name}: terminal audit not complete")
    data = {h: json.loads((audit / f"{h}.json").read_text()) for h in HEADS}
    ids = [d["row_ids"] for d in data.values()]
    if not (ids[0] == ids[1] == ids[2]):
        raise SystemExit(f"{cell.name}: the three heads score different rows")
    if any(d["status"] != "complete" or len(d["values"]) != 30 for d in data.values()):
        raise SystemExit(f"{cell.name}: a head has missing values")
    return {"cell": cell.name, **endpoint({h: d["values"] for h, d in data.items()})}


def check_declaration(tag: str) -> None:
    """The tag exists, is more than 60 s old, and holds the declared files byte for byte."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout.strip()
    commit = git("rev-parse", f"{tag}^{{commit}}")
    created = int(git("for-each-ref", f"refs/tags/{tag}", "--format=%(creatordate:unix)"))
    if time.time() - created < 60:
        raise SystemExit(f"tag {tag} is less than 60 s old")
    for path in DECLARED_FILES:
        relative = path.relative_to(REPO).as_posix()
        blob = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{relative}"], capture_output=True).stdout
        if blob != path.read_bytes():
            raise SystemExit(f"{relative} differs from the file at tag {tag}")
    print(f"declaration verified at tag {tag} ({commit[:12]})")


def condition(arm: Sequence[float], reference: Sequence[float], missing: int, *, margin: float) -> dict[str, object]:
    """The arm's median `B` minus the reference's median `B`, and whether it is at least `margin`.

    `margin` is +0.1 for P1 (a search over the fixed recipe) and -0.1 for P2 (core against reward-only). `met_with_b` is over the
    arm's cells that have a `B`; `met` also needs every cell of the arm to have one (`missing` is 0). An arm with no `B` meets nothing."""
    if not len(arm) or not len(reference):
        return {"difference": math.nan, "margin": margin, "met_with_b": False, "met": False, "arm_cells_without_b": missing}
    difference = float(np.median(arm) - np.median(reference))
    met_with_b = bool(difference >= margin - EPS)
    return {"difference": difference, "margin": margin, "met_with_b": met_with_b, "met": bool(met_with_b and missing == 0),
            "arm_cells_without_b": missing}


def read_arm(root: Path, size: int) -> list[dict[str, object]]:
    """One row per cell of `root`: its `B` and secondary columns, or none when its terminal audit is not complete."""
    cells = cells_of(root)
    if len(cells) != size:
        raise SystemExit(f"{root}: {len(cells)} cells, the declared size is {size}")
    rows = []
    for cell in cells:
        summary = cell / "terminal_audit" / "summary.json"
        if summary.is_file() and json.loads(summary.read_text()).get("status") == "complete":
            rows.append({**read_cell(cell), "status": "complete"})
        else:
            rows.append({"cell": cell.name, "B": math.nan, "status": "no complete terminal audit"})
    return rows


def contrast(fixed: Sequence[Mapping], reward_only: Sequence[Mapping], core: Sequence[Mapping]) -> dict[str, dict[str, object]]:
    """The three blind conditions of the declaration from the per-cell rows of the three arms."""
    def b(rows: Sequence[Mapping]) -> list[float]:
        return [float(r["B"]) for r in rows if not math.isnan(float(r["B"]))]
    def missing(rows: Sequence[Mapping]) -> int:
        return sum(math.isnan(float(r["B"])) for r in rows)
    return {"P1_core_vs_fixed": condition(b(core), b(fixed), missing(core), margin=MARGIN),
            "P1_reward_only_vs_fixed": condition(b(reward_only), b(fixed), missing(reward_only), margin=MARGIN),
            "P2_core_vs_reward_only": condition(b(core), b(reward_only), missing(core), margin=-MARGIN)}


def selftest() -> None:
    values = {"apex_ecoli": [10.0, 20.0, 30.0], "apex_saureus": [40.0, 5.0, 25.0], "apex_paeruginosa": [8.0, 50.0, 12.0]}
    assert list(worst_pathogen_row(values)) == [40.0, 50.0, 30.0]
    result = endpoint(values)
    assert abs(result["B"] + math.log10(40.0)) < 1e-12 and result["worst_pathogen_median_um"] == 40.0
    assert abs(result["min_over_pathogens_neg_log10_median"] + math.log10(25.0)) < 1e-12, result
    arm, reference = [1.0, 1.2, 1.4], [1.0, 1.1, 1.3]
    assert abs(condition(arm, reference, 0, margin=MARGIN)["difference"] - 0.1) < 1e-12
    assert condition(arm, reference, 0, margin=MARGIN)["met"], "a median difference of exactly 0.1 meets +0.1"
    assert condition([1.0, 1.05, 1.2], reference, 0, margin=MARGIN)["difference"] < MARGIN
    assert not condition([1.0, 1.05, 1.2], reference, 0, margin=MARGIN)["met"]
    below = condition([0.9, 1.0, 1.15], reference, 0, margin=-MARGIN)
    assert below["met"] and abs(below["difference"] + 0.1) < 1e-12, below
    assert not condition([0.8, 0.9, 1.0], reference, 0, margin=-MARGIN)["met"]
    missing_one = condition([1.2, 1.4], reference, 1, margin=MARGIN)
    assert missing_one["met_with_b"] and not missing_one["met"], missing_one
    assert not condition([], reference, 6, margin=-MARGIN)["met_with_b"]
    rows = lambda values: [{"B": v} for v in values]
    result = contrast(rows(reference), rows(reference), rows([1.0, 1.2, math.nan]))
    assert not result["P1_core_vs_fixed"]["met"] and result["P2_core_vs_reward_only"]["met_with_b"], result
    print("selftest ok")


def cells_of(path: Path) -> list[Path]:
    if (path / "events" / "log.jsonl").is_file():
        return [path]
    return sorted(p for p in path.iterdir() if (p / "events" / "log.jsonl").is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--declared-tag")
    parser.add_argument("--fixed", type=Path)
    parser.add_argument("--reward-only", type=Path)
    parser.add_argument("--core", type=Path)
    parser.add_argument("--out", type=Path, help="a directory for the per-cell table and the conditions")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return 0
    arms = {"fixed": args.fixed, "reward_only": args.reward_only, "core": args.core}
    given = [name for name, root in arms.items() if root is not None]
    if given and len(given) != 3:
        parser.error("--fixed, --reward-only and --core are given together")
    if not given and not args.paths:
        parser.print_help()
        return 1
    if not args.declared_tag:
        parser.error("--declared-tag is required")
    check_declaration(args.declared_tag)
    import pandas as pd
    if given:
        size = json.loads((CAMPAIGN / "declared.json").read_text())["n_arm"]["amp"]
        rows = {name: read_arm(root, size) for name, root in arms.items()}
        table = pd.concat([pd.DataFrame(r).assign(arm=name) for name, r in rows.items()], ignore_index=True)
    else:
        rows = None
        table = pd.DataFrame([read_cell(cell) for path in args.paths for cell in cells_of(path)])
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    result = None
    if rows is not None:
        for name, group in table.groupby("arm", sort=False):
            print(f"\n{name}: median B {group.B.median():.4f} over {int(group.B.notna().sum())} of {len(group)} cells")
        result = contrast(rows["fixed"], rows["reward_only"], rows["core"])
        print()
        for name, c in result.items():
            print(f"{name}: median difference {c['difference']:+.4f} (margin {c['margin']:+.1f}); met over cells with a B: "
                  f"{c['met_with_b']}; met on the conservative reading: {c['met']} ({c['arm_cells_without_b']} arm cells without a B)")
    else:
        print(f"\nmedian B over {int(table.B.notna().sum())} cells: {table.B.median():.4f}")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out / "blind_by_cell.csv", index=False)
        if result is not None:
            (args.out / "blind_conditions.json").write_text(json.dumps(result, indent=1, allow_nan=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
