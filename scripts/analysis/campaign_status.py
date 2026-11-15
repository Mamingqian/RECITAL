#!/usr/bin/env python3
"""One row per control-engine cell: what it spent, what it decided, what it delivered.

Reads a run through `recital.control.export.load_run` and never writes into a run:
  - events/log.jsonl and its blobs       -> status, rounds, committed updates, the candidates
                                            (parent, start, steps, enabled/held-out oracles,
                                            reward utilities), decisions, the stop
  - candidates/*/candidate_result.json   -> each candidate's qualification evidence, from which
                                            the declared ranking orders them
  - declaration.json                     -> the declared success spec (requirements, ranking)
  - delivery/delivery_artifact.json      -> qualification and ranking value of the delivered library
  - terminal_audit/*.json                -> the blind endpoint, when the cell has reached it
  - config.yaml (newest snapshot)        -> which levers the declaration pins, whether synthesis was open

Tolerant of running and partial cells: it reports what is present and says so, which
is what makes it usable as a live readout rather than only a post-mortem.

The capability columns are not decoration. `frozen` non-empty means the declaration pins a
training lever to one value, so no agent could move it, and `coder` false means component
synthesis was closed, so a null result from such a cell says nothing about the framework.
Neither is visible from a directory name, which is why they print on every row.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recital.control.export import LOG_PATH, load_run, summary_of
from recital.control.selection import Evidence, leaderboard


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def ranking_values(delivery: dict, spec: dict) -> dict:
    """The declared ranking criterion's value, reduced exactly as declared."""
    measurements = (delivery.get("qualification") or {}).get("measurements") or {}
    out = {}
    for criterion in spec.get("ranking") or []:
        metric, strata = criterion["metric"], criterion.get("strata") or []
        observed = []
        for stratum in strata:
            entry = (measurements.get(metric) or {}).get(stratum) or {}
            if entry.get("status") == "measured" and entry.get("value") is not None:
                observed.append(float(entry["value"]))
        if len(observed) != len(strata) or not observed:
            out[metric] = None
            continue
        reduction = criterion.get("reduction", "min")
        out[metric] = (min(observed) if reduction == "min" else
                       max(observed) if reduction == "max" else sum(observed) / len(observed))
    return out


def capability(config: dict) -> tuple[list[str], bool | None]:
    """The levers a run's declaration pins to a single value, and whether synthesis was open.

    A lever with no declared bound is free, so only bounds that admit one value count as
    frozen. Synthesis is None when the config declares no `control` block.
    """
    control = config.get("control") or {}
    bounds = dict(config.get("hyperparameter_bounds") or {})
    steps = (control.get("budget") or {}).get("steps")
    if steps:
        bounds["steps"] = steps
    frozen = sorted(name for name, bound in bounds.items()
                    if (bound.get("minimum") is not None and bound.get("minimum") == bound.get("maximum"))
                    or len(bound.get("enum") or ()) == 1)
    enabled = (control.get("synthesis") or {}).get("enabled")
    return frozen, None if enabled is None else bool(enabled)


def cell_report(root: Path) -> dict:
    view = load_run(root)
    summary = summary_of(view)
    state, success = view.state, view.success_spec()
    spec = ((((view.declaration() or {}).get("contract") or {}).get("delivery_requirements") or {})
            .get("success") or {})
    candidates, evidence = [], []
    for candidate in state.ordered_candidates():
        recipe = (view.spec(candidate.id) or {}).get("recipe") or {}
        result = view.candidate_result(candidate.id) if candidate.status == "ok" else None
        candidates.append({
            "id": candidate.id, "round": candidate.round, "slot": candidate.slot, "status": candidate.status,
            "steps": candidate.steps, "parent": candidate.parent, "start": candidate.start,
            "enabled": recipe.get("enabled_oracles"), "held_out": recipe.get("held_out_oracles"),
            "utilities": recipe.get("oracle_utilities"),
            "qualified": None if result is None else {
                condition: info["qualified"] for condition, info in result["by_condition"].items()},
        })
        evidence.append(Evidence(candidate.id, candidate.ordinal, result is not None,
                                 None if result is None else result["delivery"]["qualification"]))
    artifact = view.delivery_artifact()
    audit = {}
    for path in sorted((root / "terminal_audit").glob("*.json")):
        payload = read_json(path) or {}
        audit[path.stem] = payload.get("primary") or payload
    frozen, synthesis = capability(view.config())
    total, spent = summary["total_updates"], summary["committed_updates"]
    decisions = [state.decisions[key] for key in state.decisions]
    return {
        "cell": root.name,
        "status": summary["status"],
        "spent": spent,
        "total": total,
        "share": spent / total if total else None,
        "rounds": summary["rounds"],
        "candidates": candidates,
        "decisions": {"recorded": len(decisions), "invalid": sum(1 for d in decisions if d.status == "invalid")},
        "stop": summary["stop"],
        "qualified": None if state.delivery is None else all(state.delivery.qualified.values()),
        "ranking": {} if artifact is None else ranking_values(artifact, spec),
        "declared_ranking": [(item["metric"], item.get("reduction", "min")) for item in spec.get("ranking") or []],
        "best_so_far": {} if success is None else leaderboard(success, evidence, view.conditions, top=1),
        "frozen": frozen,
        "coder": synthesis,
        "component_requests": len(state.components),
        "resumes": summary["resumes"],
        "tokens": summary["tokens"]["total"],
        "audit": audit,
    }


def _oracle_groups(candidates: list[dict]) -> list[tuple[dict, list[str]]]:
    """Candidates grouped by the oracle choice they trained with, in order of first appearance."""
    groups: list[tuple[dict, list[str]]] = []
    for item in candidates:
        key = {"enabled": item["enabled"], "held_out": item["held_out"], "utilities": item["utilities"]}
        if not key["enabled"]:
            continue
        for known, members in groups:
            if known == key:
                members.append(item["id"][-3:])
                break
        else:
            groups.append((key, [item["id"][-3:]]))
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="campaign roots or individual cells")
    parser.add_argument("--json", type=Path, default=None, help="write the full report here")
    arguments = parser.parse_args()

    cells = []
    for root in arguments.roots:
        found = [root] if (root / LOG_PATH).exists() else sorted(
            path for path in root.iterdir() if path.is_dir() and (path / LOG_PATH).exists())
        cells.extend(cell_report(path) for path in found)

    width = max((len(item["cell"]) for item in cells), default=4)
    print(f"{'cell':<{width}}  {'status':<22} {'spent/total':>16} {'share':>6}  {'qual':>5}  candidates")
    for item in cells:
        share = "" if item["share"] is None else f"{item['share'] * 100:5.1f}%"
        budget = f"{item['spent']}/{item['total']}"
        kinds = " ".join(f"{step['id'][-3:]}:{step['status']}+{step['steps']}" for step in item["candidates"])
        print(f"{item['cell']:<{width}}  {item['status']:<22} {budget:>16} {share:>6}  "
              f"{str(item['qualified']):>5}  {kinds}")
        rounds, stop = item["rounds"], item["stop"]
        print(f"{'':<{width}}  rounds {rounds['completed']}/{rounds['allocated']} done "
              f"(min {rounds['minimum']}, max {rounds['maximum']}); decisions {item['decisions']['recorded']} "
              f"({item['decisions']['invalid']} invalid); resumes {item['resumes']}; tokens {item['tokens']}"
              + ("" if stop is None else f"; stop {stop['outcome']} by {stop['source']}"))
        if item["ranking"]:
            values = ", ".join(f"{k}={'unmeasured' if v is None else format(v, '.4f')}"
                               for k, v in item["ranking"].items())
            print(f"{'':<{width}}  declared ranking of the delivered library: {values}")
        best = {condition: rows[0] for condition, rows in item["best_so_far"].items() if rows}
        if best:
            print(f"{'':<{width}}  best qualified so far: " + " | ".join(
                f"{condition} {row['candidate']} " + ",".join(
                    f"{metric}={'n/a' if value is None else format(value, '.4f')}"
                    for metric, value in row["values"].items()) for condition, row in best.items()))
        for key, members in _oracle_groups(item["candidates"]):
            print(f"{'':<{width}}  oracles: enabled={key['enabled']} held_out={key['held_out']}"
                  f"{'' if not key['utilities'] else ' utilities=' + json.dumps(key['utilities'])}"
                  f" candidates {','.join(members)}")
        if item["audit"]:
            print(f"{'':<{width}}  blind: " + json.dumps(item["audit"])[:200])
        if item["frozen"] or item["coder"] is False:
            print(f"{'':<{width}}  CAPABILITY frozen={item['frozen']} coder={item['coder']} "
                  f"component_requests={item['component_requests']}")
    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(cells, indent=2, default=str) + "\n")
        print(f"\nwrote {arguments.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
