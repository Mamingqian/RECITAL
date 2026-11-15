"""What the Strategist changed: the recipe edits of the nine main-line searches and the visible measurements that followed.

Reads, read-only, the event log and ledger of every main-line cell (`common.control_cells`: `MAIN_ROOTS`, `MAIN_SEEDS`)
and writes under figures/fig5/data/:

  edit_atoms.csv        one row per edit of an agent-proposed candidate against its base (the parent, or the declared initial
                        recipe when it has none): task, seed, round, candidate, class, detail, direction
  edit_candidates.csv   one row per candidate: its edit classes, the base it is compared with, and the visible measurements of
                        the candidate's pool and library with their change against the base
  cm_distance_path.csv  the chorismate-mutase libraries' natural distance by candidate, the declared initial recipe's and the
                        delivered one's marked

An edit class is one of: source_on, source_off (a reward source enabled or disabled), weights (aggregator weights or the
per-source utility of a source that stays enabled), penalty, gate, kl, lr, steps, group_size, start (a restart from a
parent's checkpoint). Direction is `up`/`down` for the numeric levers, `add`/`tighten`/`relax`/`drop` for gates and
penalties. Everything is descriptive: a candidate usually carries several edits, so a change in a measurement is not
attributed to one of them, and the reserved predictors are never read.

    python scripts/analysis/agent_edits.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for _path in (REPO / "src", REPO / "scripts", REPO / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common.control_cells import MAIN_ARM, MAIN_ROOTS, MAIN_SEEDS, SCRATCH_ROOT, Cell, load_campaign  # noqa: E402

OUT = REPO / "scripts" / "fig5" / "data"
TASKS = ("cre", "loopseq", "cm")
# The measurement of a candidate's pool each task's Strategist is shown as the reward source (median over the task's
# conditions), and the second reward source of the pool: CRE Malinois and grammar-prior MinGap, Loop-seq CNN and MIXBend
# target margin, CM Function predictor and DCA family prior.
SCORES = {"cre": ("mingap/malinois", "mingap/grammar_prior"), "loopseq": ("margin/cnn", "margin/mixbend"),
          "cm": ("function_2026_esm", "cm_max")}
CLASSES = ("source_on", "source_off", "weights", "penalty", "gate", "kl", "lr", "steps", "group_size", "start")
LEVERS = ("kl", "lr", "steps", "group_size")


def _load(text: str) -> Any:
    return json.loads(text)


def _named(items: list[dict] | None) -> dict[str, dict]:
    return {item["name"]: item for item in items or []}


def _gate_direction(old: dict, new: dict) -> str:
    """`tighten` when a `min*` parameter rose or a `max*` parameter fell, else `relax`."""
    tighter = looser = False
    for key, value in new.items():
        if key == "name" or key not in old or not isinstance(value, (int, float)):
            continue
        if value == old[key]:
            continue
        if (key.startswith("min") and value > old[key]) or (key.startswith("max") and value < old[key]):
            tighter = True
        else:
            looser = True
    return "tighten" if tighter and not looser else "relax" if looser and not tighter else "change"


def parse_edit(text: str) -> list[dict[str, str]]:
    """The atoms of one ledger edit line `field: old -> new`."""
    field, rest = text.split(": ", 1)
    old, new = (_load(part) for part in rest.split(" -> ", 1))
    atoms: list[dict[str, str]] = []
    if field == "enabled_oracles":
        atoms += [{"class": "source_on", "detail": name, "direction": "enable"} for name in new if name not in old]
        atoms += [{"class": "source_off", "detail": name, "direction": "disable"} for name in old if name not in new]
    elif field == "oracle_utilities":
        old, new = old or {}, new or {}
        atoms += [{"class": "weights", "detail": f"utility:{name}", "direction": "change"}
                  for name in sorted(set(old) & set(new)) if old[name] != new[name]]
    elif field == "aggregator":
        if (old or {}).get("name") != (new or {}).get("name"):
            atoms.append({"class": "weights", "detail": "aggregator", "direction": "change"})
        before, after = (old or {}).get("weights") or {}, (new or {}).get("weights") or {}
        for name in sorted(set(before) & set(after)):
            if before[name] != after[name]:
                atoms.append({"class": "weights", "detail": f"weight:{name}",
                              "direction": "up" if after[name] > before[name] else "down"})
    elif field == "penalties":
        before, after = _named(old), _named(new)
        atoms += [{"class": "penalty", "detail": name, "direction": "add"} for name in after if name not in before]
        atoms += [{"class": "penalty", "detail": name, "direction": "drop"} for name in before if name not in after]
        for name in set(before) & set(after):
            if before[name] != after[name]:
                weight_old, weight_new = before[name].get("weight"), after[name].get("weight")
                same_but_weight = {k: v for k, v in before[name].items() if k != "weight"} == \
                                  {k: v for k, v in after[name].items() if k != "weight"}
                direction = ("up" if weight_new > weight_old else "down") if same_but_weight and weight_old != weight_new \
                    else "change"
                atoms.append({"class": "penalty", "detail": name, "direction": direction})
    elif field == "gates":
        before, after = _named(old), _named(new)
        atoms += [{"class": "gate", "detail": name, "direction": "add"} for name in after if name not in before]
        atoms += [{"class": "gate", "detail": name, "direction": "drop"} for name in before if name not in after]
        for name in set(before) & set(after):
            if before[name] != after[name]:
                atoms.append({"class": "gate", "detail": name, "direction": _gate_direction(before[name], after[name])})
    elif field in LEVERS:
        atoms.append({"class": field, "detail": field, "direction": "up" if new > old else "down"})
    else:
        raise ValueError(f"unknown edit field {field!r}")
    return atoms


def composition_outside(ledger: dict | None) -> int | None:
    """Number of composition measurements whose pool median lies outside the initial policy's band, over the conditions."""
    if ledger is None or ledger.get("composition") is None:
        return None
    return sum(len(v or {}) for v in ledger["composition"].values())


def _median(frame: pd.Series) -> float:
    values = pd.to_numeric(frame, errors="coerce").dropna()
    return float(values.median()) if len(values) else np.nan


def candidate_table(task: str, cell: Cell) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = cell.candidates().reset_index(drop=True)
    first, second = SCORES[task]
    conditions = cell.conditions
    rows, atoms = [], []
    anchor = frame[frame.anchor.astype(bool)].candidate.iloc[0]
    for record in frame.itertuples():
        ledger = cell.view.ledger_row(record.candidate) if cell.view.state.candidates[record.candidate].ledger_row_sha else None
        pool, library = (ledger or {}).get("pool") or {}, (ledger or {}).get("library") or {}
        row = {"task": task, "cell": cell.name, "seed": cell.seed, "candidate": record.candidate, "ordinal": record.ordinal,
               "round": record.round, "slot": record.slot, "anchor": bool(record.anchor), "start": record.start,
               "parent": record.parent, "status": record.status,
               "qualified": None if pd.isna(record.qualified) else bool(record.qualified),
               "delivered": bool(record.delivered_for),
               "failing": ",".join(sorted({str(x) for column in frame.columns if column.startswith("failing__")
                                           for x in (getattr(record, column.replace("/", "_").replace("-", "_"), None) or [])})),
               "comp_outside": composition_outside(ledger),
               "pairwise_ratio_library": min([(library.get(c) or {}).get("pairwise_ratio") for c in conditions
                                              if (library.get(c) or {}).get("pairwise_ratio") is not None], default=np.nan),
               "pairwise_ratio_pool": min([(pool.get(c) or {}).get("pairwise_ratio") for c in conditions
                                           if (pool.get(c) or {}).get("pairwise_ratio") is not None], default=np.nan),
               "n_gates": record.n_gates, "enabled": ",".join(record.enabled or [])}
        for key, source in (("score1", first), ("score2", second)):
            row[f"{key}_pool"] = _median(pd.Series([(pool.get(c) or {}).get(source) for c in conditions]))
            row[f"{key}_library"] = _median(pd.Series([(library.get(c) or {}).get(source) for c in conditions]))
        if task == "cm":
            row["distance_pool"] = (pool.get("CM") or {}).get("natural_distance")
            row["distance_library"] = (library.get("CM") or {}).get("natural_distance")
        classes = []
        if not record.anchor:
            for line in record.edits or []:
                for atom in parse_edit(line):
                    atoms.append({"task": task, "cell": cell.name, "seed": cell.seed, "candidate": record.candidate,
                                  "round": record.round, **atom})
                    classes.append(atom["class"])
            if record.start in ("from_parent", "continue_parent"):
                atoms.append({"task": task, "cell": cell.name, "seed": cell.seed, "candidate": record.candidate,
                              "round": record.round, "class": "start", "detail": record.start, "direction": record.start})
                classes.append("start")
        row["classes"] = ",".join(sorted(set(classes)))
        row["base"] = anchor if record.anchor else (record.parent if isinstance(record.parent, str) and record.parent != "initial"
                                                    else anchor)
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(atoms)


def with_deltas(table: pd.DataFrame) -> pd.DataFrame:
    """Add `<measure>_delta`: the candidate's value minus its base's, for trained candidates with a measured base."""
    measures = [c for c in table.columns if c.startswith(("score", "distance", "pairwise", "comp_outside"))]
    out = table.copy()
    by_key = table.set_index(["cell", "candidate"])
    for measure in measures:
        base = [by_key[measure].get((cell, base_id), np.nan) for cell, base_id in zip(table.cell, table.base)]
        out[f"{measure}_delta"] = pd.to_numeric(table[measure], errors="coerce") - pd.to_numeric(pd.Series(base), errors="coerce")
    out["base_qualified"] = [by_key["qualified"].get((cell, base_id)) for cell, base_id in zip(table.cell, table.base)]
    out.loc[out.anchor, [c for c in out.columns if c.endswith("_delta")]] = np.nan
    return out


def agent_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    """The agent-proposed candidates (not the declared initial recipe) with one boolean column per edit class and whether the
    candidate entered or left qualification against its base."""
    agent = candidates[~candidates.anchor.astype(bool)].copy()
    carried = agent.classes.fillna("").str.split(",")
    for name in CLASSES:
        agent[name] = carried.map(lambda found, name=name: name in found)
    agent["entered"] = (agent.qualified == True) & (agent.base_qualified == False)  # noqa: E712
    agent["left"] = (agent.qualified == False) & (agent.base_qualified == True)  # noqa: E712
    return agent


def lineage(candidates: pd.DataFrame, cell: str, candidate: str) -> list[str]:
    """The candidates from the declared initial recipe to `candidate`, following each candidate's base."""
    by_id = candidates[candidates.cell == cell].set_index("candidate")
    chain = [candidate]
    while not by_id.loc[chain[-1], "anchor"]:
        chain.append(by_id.loc[chain[-1], "base"])
    return chain[::-1]


def main_cells(task: str) -> list[Cell]:
    (root,) = MAIN_ROOTS[task]
    cells = [c for c in load_campaign(SCRATCH_ROOT / root) if c.arm == MAIN_ARM and c.seed in MAIN_SEEDS[task]]
    if sorted(c.seed for c in cells) != sorted(MAIN_SEEDS[task]):
        raise SystemExit(f"{root} holds seeds {sorted(c.seed for c in cells)}, not {list(MAIN_SEEDS[task])}")
    return sorted(cells, key=lambda c: c.seed)


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    tables, atoms = [], []
    for task in TASKS:
        for cell in main_cells(task):
            table, cell_atoms = candidate_table(task, cell)
            tables.append(with_deltas(table))
            atoms.append(cell_atoms)
    return pd.concat(tables, ignore_index=True), pd.concat(atoms, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    candidates, atoms = build()
    candidates.to_csv(args.out / "edit_candidates.csv", index=False, float_format="%.6g")
    atoms.to_csv(args.out / "edit_atoms.csv", index=False)
    print(f"{len(candidates)} candidates ({int((~candidates.anchor).sum())} agent-proposed), {len(atoms)} edits")
    print(atoms.groupby(["task", "class"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
