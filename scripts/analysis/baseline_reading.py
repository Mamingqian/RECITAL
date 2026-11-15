#!/usr/bin/env python
"""Read a literature-baseline arm against the core in the three classes the declarations fix, from the audited contrast's own numbers.

    baseline_reading.py tasks  VERDICT.json                  # the verdict of `module_contrast.py core` with the baseline as the arm
    baseline_reading.py amp --arm ARM.csv --comparator CORE.csv --declared-tag TAG --declared-path PATH

Classes (docs/reports/baselines_2026-11/literature_baselines_2026-11-04.md, section 7):
  1 "baseline matches or exceeds core": the verdict's `passes` (the module pass rule with the baseline as the arm).
  2 "core exceeds baseline": `p_worse` <= 0.05 (exact one-sided permutation p that the controls exceed the arm), and
    `-difference` > `floor`, and the arm's blind result is not better than the controls' by more than the blind floor.
  3 "unresolved": neither.
For the amp task `module_contrast.amp_contrast` gives the arm-higher p only, and running it with the roles swapped is refused
(the core cells predate the baseline's declaration tag), so the reverse p is computed here from the same normalized values, with the
same comparator-mean normalizer, floor and blind tolerance as `amp_contrast`. Run with the recital interpreter, from the repository root.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import module_contrast as mc  # noqa: E402

ALPHA = mc.ALPHA


def classify(*, passes: bool, p_worse: float, difference: float, floor: float, blind_difference: float,
             blind_tolerance: float) -> dict[str, object]:
    """The class of one verdict; `blind_difference` is the arm's blind result minus the controls', higher = better."""
    blind_not_better = (blind_difference != blind_difference) or blind_difference <= blind_tolerance + mc.EPS
    core_exceeds = bool(p_worse <= ALPHA and -difference > floor + mc.EPS and blind_not_better)
    if passes:
        label = "1 baseline matches or exceeds core"
    elif core_exceeds:
        label = "2 core exceeds baseline"
    else:
        label = "3 unresolved"
    return {"class": label, "passes": bool(passes), "core_exceeds": core_exceeds, "p_worse": p_worse,
            "difference": difference, "floor": floor, "blind_difference": blind_difference,
            "blind_tolerance": blind_tolerance}


def from_verdict(verdict: Mapping[str, object]) -> dict[str, object]:
    return classify(passes=bool(verdict["passes"]), p_worse=float(verdict["p_worse"]), difference=float(verdict["difference"]),
                    floor=float(verdict["floor"]), blind_difference=float(verdict["blind_difference"]),
                    blind_tolerance=float(verdict["blind_floor"]))


def amp(arm: pd.DataFrame, comparator: pd.DataFrame, *, n_arm: int, n_comparator: int) -> dict[str, object]:
    """The amp reading: `amp_contrast`'s numbers (arm = the baseline, comparator = the core) plus the reverse p."""
    forward = mc.amp_contrast(arm, comparator, n_arm=n_arm, n_comparator=n_comparator)
    arm, comparator = mc.prepare("arm", arm, size=n_arm), mc.prepare("comparator", comparator, size=n_comparator)
    normalizer = comparator["final_best"].mean()
    arm_norm, comparator_norm = (arm["aubc"] / normalizer).tolist(), (comparator["aubc"] / normalizer).tolist()
    p_worse = mc.permutation_p(comparator_norm, arm_norm, alternative="greater")
    return {**classify(passes=bool(forward["passes"]), p_worse=p_worse, difference=float(forward["difference"]),
                       floor=float(forward["floor"]), blind_difference=float(forward["blind_difference"]),
                       blind_tolerance=float(forward["blind_tolerance"])),
            "p_one_sided_arm_higher": forward["p_one_sided"], "n_arm": n_arm, "n_comparator": n_comparator}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    tasks = sub.add_parser("tasks")
    tasks.add_argument("verdict", type=Path)
    amp_parser = sub.add_parser("amp")
    amp_parser.add_argument("--arm", type=Path, required=True)
    amp_parser.add_argument("--comparator", type=Path, required=True)
    amp_parser.add_argument("--declared-tag", required=True)
    amp_parser.add_argument("--declared-path", required=True)
    args = parser.parse_args(argv)
    if args.command == "tasks":
        result = from_verdict(json.loads(args.verdict.read_text()))
    else:
        declared, _, _ = mc._read_declared(args.declared_tag, args.declared_path)
        result = amp(pd.read_csv(args.arm), pd.read_csv(args.comparator), n_arm=declared["n_arm"]["amp"],
                     n_comparator=declared["n_comparator"]["amp"])
    print(json.dumps(result, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
