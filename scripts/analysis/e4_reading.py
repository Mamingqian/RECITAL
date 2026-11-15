"""E4 reading as declared in experiments/journal_e4_enhancer_baselines/campaign.json.

Per seed and cell line (K562, HepG2): the calibrated ChromBPNet MinGap median of the library each E4 cell delivered, read from the cell's
terminal_audit/summary.json as primary.per_condition.<line>.value - DELTA[line] (the paper's calibration, DELTA from
analysis.cre_movement_vs_blind); the arm mean over the 3 seeds for `motif` and `motif_gate`; RECITAL's three searches (cre_control_round3_20261006) and the
same-seed depth-matched initial recipe (cre_depth_matched_20261023) beside them; the three clauses of the declared rule evaluated in the campaign's words.
A cell that ends without a qualified library counts as no library and is not dropped; the arm mean is then not defined.
The card class beside every number is read from docs/reports/journal_e4_card_class_per_cell.md.

Usage: python3 scripts/analysis/e4_reading.py [--references-only] [--controller-pid PID] [--out PATH]
  --references-only   print RECITAL's and the depth-matched values and check the depth-matched values against the declared ones; reads no E4 file.
Without it the script refuses to run until six E4 summary.json files exist (and the E4 controller, if --controller-pid is given, has exited).
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _entry in (REPO / "src", REPO / "scripts", REPO / "scripts"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from analysis.cre_movement_vs_blind import DELTA  # noqa: E402
from common.control_cells import MAIN_ARM, MAIN_ROOTS, MAIN_SEEDS, SCRATCH_ROOT, Cell, cell_dirs  # noqa: E402
from recital.control.export import load_run  # noqa: E402

LINES = ("K562", "HepG2")
ARMS = ("motif", "motif_gate")
SEEDS = (0, 1, 2)
E4_ROOT = SCRATCH_ROOT / "journal_e4_20261104"
DEPTH_ROOT = SCRATCH_ROOT / "cre_depth_matched_20261023"
# The depth-matched cell of each seed and cell line: the update count of that seed's delivered search lineage (STATE.md, experiments/cre_depth_matched).
DEPTH_CELL = {("K562", 0): "cre_fixedrecipe_steps9000_seed0", ("HepG2", 0): "cre_fixedrecipe_steps9000_seed0",
              ("K562", 1): "cre_fixedrecipe_steps3000_seed1", ("HepG2", 1): "cre_fixedrecipe_steps6000_seed1",
              ("K562", 2): "cre_fixedrecipe_steps7500_seed2", ("HepG2", 2): "cre_fixedrecipe_steps7500_seed2"}
# Declared in experiments/journal_e4_enhancer_baselines/campaign.json (what_this_tests), per seed 0, 1, 2.
DEPTH_DECLARED = {"K562": (2.42, -0.29, -0.49), "HepG2": (0.30, 0.67, 0.54)}
CARD_FILE = REPO / "docs" / "reports" / "journal_e4_card_class_per_cell.md"
QUALIFIED_OUTCOME = "qualified_delivery"


def read_json(path: Path):
    with open(path) as handle:
        return json.load(handle)


def calibrated(summary: dict, line: str) -> tuple[float, dict]:
    entry = summary["primary"]["per_condition"][line]
    return entry["value"] - DELTA[line], entry


def recital_values() -> dict:
    """RECITAL's three searches: calibrated median per seed and cell line, from each cell's own terminal audit."""
    cells = []
    for root in MAIN_ROOTS["cre"]:
        for path in cell_dirs(SCRATCH_ROOT / root):
            view = load_run(path)
            if view.header.arm == MAIN_ARM and int(view.header.seed) in MAIN_SEEDS["cre"]:
                cells.append(Cell(path, view, root))
    cells.sort(key=lambda c: c.seed)
    if [c.seed for c in cells] != sorted(MAIN_SEEDS["cre"]):
        raise SystemExit(f"RECITAL cells found for seeds {[c.seed for c in cells]}, declared {sorted(MAIN_SEEDS['cre'])}")
    return {line: [calibrated(c.audit, line)[0] for c in cells] for line in LINES}


def depth_matched_values() -> dict:
    """The same-seed depth-matched initial recipe, checked against the values the campaign declares (two decimals)."""
    values = {}
    for line in LINES:
        row = []
        for seed in SEEDS:
            summary = read_json(DEPTH_ROOT / DEPTH_CELL[(line, seed)] / "terminal_audit" / "summary.json")
            row.append(calibrated(summary, line)[0])
        values[line] = row
        for seed, (got, declared) in enumerate(zip(row, DEPTH_DECLARED[line])):
            if abs(round(got, 2) - declared) > 0.0051:
                raise SystemExit(f"depth-matched {line} seed {seed}: {got:.4f} is not the declared {declared}")
    return values


def card_classes() -> dict:
    """(arm, seed) -> (baseline card, candidate card) from the committed card-class table."""
    rows = {}
    for line in CARD_FILE.read_text().splitlines():
        match = re.match(r"\|\s*(motif(?:_gate)?)_seed(\d)\s*\|\s*attempt \d+\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|", line)
        if match:
            rows[(match.group(1), int(match.group(2)))] = (match.group(3), match.group(4))
    return rows


def e4_cells() -> dict:
    cells = {}
    for arm in ARMS:
        for seed in SEEDS:
            folder = E4_ROOT / f"cre_fixedrecipe_{arm}_steps9000_seed{seed}"
            summary_path = folder / "terminal_audit" / "summary.json"
            if not summary_path.is_file():
                raise SystemExit(f"{summary_path} does not exist yet")
            cells[(arm, seed)] = {"summary": read_json(summary_path), "outcome": read_json(folder / "delivery.json").get("outcome")}
    return cells


def arm_values(cells: dict) -> dict:
    """(arm, line) -> per-seed list of (calibrated value or None for no library, raw, count, measured_count, status, outcome)."""
    out = {}
    for arm in ARMS:
        for line in LINES:
            row = []
            for seed in SEEDS:
                cell = cells[(arm, seed)]
                value, entry = calibrated(cell["summary"], line)
                has_library = cell["outcome"] == QUALIFIED_OUTCOME and entry["status"] == "complete"
                row.append({"cal": value if has_library else None, "raw": entry["value"], "count": entry["count"],
                            "measured": entry["measured_count"], "status": entry["status"], "outcome": cell["outcome"]})
            out[(arm, line)] = row
    return out


def mean_or_none(values: list) -> float | None:
    return None if any(v is None for v in values) else float(np.mean(values))


def rule(recital: dict, arms: dict) -> dict:
    """The declared rule's three clauses per cell line, as facts about the numbers."""
    facts = {}
    for line in LINES:
        mean_r = float(np.mean(recital[line]))
        per_arm = {}
        for arm in ARMS:
            seeds = [cell["cal"] for cell in arms[(arm, line)]]
            mean_a = mean_or_none(seeds)
            per_arm[arm] = {"mean": mean_a, "seeds": seeds,
                            "recital_mean_higher": None if mean_a is None else mean_r > mean_a,
                            "mean_at_or_above_recital": None if mean_a is None else mean_a >= mean_r,
                            "seed_exceeds_recital_seed_same_seed": [s for s, (a, r) in enumerate(zip(seeds, recital[line])) if a is not None and a > r],
                            "seed_exceeds_any_recital_seed": any(a is not None and a > r for a in seeds for r in recital[line])}
        defined = all(per_arm[arm]["mean"] is not None for arm in ARMS)
        facts[line] = {"recital_mean": mean_r, "arms": per_arm,
                       "plus": None if not defined else all(per_arm[arm]["recital_mean_higher"] for arm in ARMS),
                       "minus": None if not defined else any(per_arm[arm]["mean_at_or_above_recital"] for arm in ARMS)}
    return facts


def fmt(value, places=2):
    return "no library" if value is None else f"{value:.{places}f}"


def yes_no(value) -> str:
    return "not evaluable" if value is None else ("yes" if value else "no")


def report(recital: dict, depth: dict, arms: dict, facts: dict, cards: dict) -> str:
    lines = ["# E4 reading", "",
             "Declared in `experiments/journal_e4_enhancer_baselines/campaign.json`. Each value is the calibrated ChromBPNet MinGap median of the delivered "
             "128-row library of one cell line: `terminal_audit/summary.json` `primary.per_condition.<line>.value` minus DELTA "
             f"(K562 {DELTA['K562']:+.4f}, HepG2 {DELTA['HepG2']:+.4f}; `scripts/analysis/cre_movement_vs_blind.py`). "
             "Computed by `scripts/analysis/e4_reading.py`. The same calibration reproduces the declared depth-matched values (checked to two decimals).", ""]
    for line in LINES:
        lines += [f"## {line}", "", "| seed | arm | baseline card | training card | raw median | calibrated median | count / measured | status | delivery outcome |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for seed in SEEDS:
            for arm in ARMS:
                cell = arms[(arm, line)][seed]
                baseline, candidate = cards[(arm, seed)]
                lines.append(f"| {seed} | {arm} | {baseline} | {candidate} | {cell['raw']:.4f} | {fmt(cell['cal'])} | {cell['count']} / {cell['measured']} | {cell['status']} | {cell['outcome']} |")
        lines += ["", "| reference | seed 0 | seed 1 | seed 2 | mean |", "|---|---|---|---|---|",
                  f"| RECITAL, three searches | {' | '.join(f'{v:.2f}' for v in recital[line])} | {np.mean(recital[line]):.2f} |",
                  f"| depth-matched initial recipe | {' | '.join(f'{v:.2f}' for v in depth[line])} | {np.mean(depth[line]):.2f} |"]
        for arm in ARMS:
            seeds = facts[line]["arms"][arm]["seeds"]
            lines.append(f"| E4 {arm} | {' | '.join(fmt(v) for v in seeds)} | {fmt(facts[line]['arms'][arm]['mean'])} |")
        lines += ["", f"Declared rule, {line}:", ""]
        lines.append(f"- (+) RECITAL's three-search mean ({facts[line]['recital_mean']:.2f}) exceeds the mean of every baseline arm: "
                     + ("not evaluable (an arm has a cell with no library)" if facts[line]["plus"] is None else ("holds" if facts[line]["plus"] else "does not hold")))
        lines.append("- (-) a baseline arm's mean is at or above RECITAL's: "
                     + ("not evaluable (an arm has a cell with no library)" if facts[line]["minus"] is None else ("holds" if facts[line]["minus"] else "does not hold")))
        for arm in ARMS:
            info = facts[line]["arms"][arm]
            same_seed = ", ".join(f"seed {s}" for s in info["seed_exceeds_recital_seed_same_seed"]) or "none"
            lines.append(f"- (0) {arm}: RECITAL's mean is higher than the arm's mean: {yes_no(info['recital_mean_higher'])}; "
                         f"a seed of the arm exceeds the same-seed RECITAL seed: {same_seed}; "
                         f"a seed of the arm exceeds some RECITAL seed: {yes_no(info['seed_exceeds_any_recital_seed'])}")
        lines.append("")
    lines += ["## Fields read", "",
              "- per cell, `journal_e4_20261104/cre_fixedrecipe_<arm>_steps9000_seed<seed>/terminal_audit/summary.json`: `primary.per_condition.K562.value`, "
              "`primary.per_condition.HepG2.value` (raw median), `.count`, `.measured_count`, `.status`; `status` and `primary.status` (all `complete`); "
              "`count` 256 (128 rows per cell line)",
              "- per cell, `delivery.json` `outcome` (a cell whose outcome is not `qualified_delivery` counts as no library)",
              f"- calibrated value = raw value - DELTA[line], DELTA K562 {DELTA['K562']:+.10f}, HepG2 {DELTA['HepG2']:+.10f}",
              "- RECITAL's three searches: the three `cre_no_self_validation_seed<s>` cells of `cre_control_round3_20261006` (the paper's CRE main line), "
              "same field minus DELTA",
              "- depth-matched initial recipe: `cre_depth_matched_20261023` cells `steps9000_seed0`, `steps3000_seed1` (K562), `steps6000_seed1` (HepG2), "
              "`steps7500_seed2`, same field minus DELTA, equal to the declared values to two decimals",
              "- baseline card: the card class of the baseline attempt that was ingested (attempt 3 in every cell); training card: the card class of the candidate "
              "training job; both from `docs/reports/journal_e4_card_class_per_cell.md` (job ids in parentheses)", ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--references-only", action="store_true")
    parser.add_argument("--controller-pid", type=int)
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "reports" / "journal_e4_reading.md")
    args = parser.parse_args()
    recital, depth = recital_values(), depth_matched_values()
    for line in LINES:
        print(f"{line}: RECITAL per seed {[round(v, 4) for v in recital[line]]} mean {np.mean(recital[line]):.4f}; "
              f"depth-matched per seed {[round(v, 4) for v in depth[line]]} (declared {DEPTH_DECLARED[line]} reproduced)")
    if args.references_only:
        return
    if args.controller_pid is not None:
        try:
            os.kill(args.controller_pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise SystemExit(f"the E4 controller (pid {args.controller_pid}) is still running")
    cells = e4_cells()
    arms = arm_values(cells)
    text = report(recital, depth, arms, rule(recital, arms), card_classes())
    args.out.write_text(text)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
