#!/usr/bin/env python
"""Does a cell of the MAC-AMP port show, from its own record, the loop of MAC-AMP?

    macamp_trace_check.py CELL [--json]

Reads the cell's hash-chained event log (candidates, parents, steps, recipes, the accepted stop) and `macamp_trace.jsonl` and
`macamp_designs.json` written by `recital.control.macamp.MacAmpTransport`, and prints one line per check with PASS or FAIL. Exit
status 1 when a check fails. A cell with a failing check is a deviation of the port, reported as such.

Checks: three stages; per stage six sandbox candidates (two inner iterations of three) of SANDBOX_STEPS updates and one real
candidate of REAL_STEPS updates, all with the stage's starting candidate as parent (the initial policy in stage 1, the previous real
candidate after); no sandbox candidate is a parent; the real candidate uses the reward of the judge's pick of the last iteration;
no training setting other than `steps` differs from the initial recipe; the delivered candidate is a real one; every dialogue has at
most four designer turns; a critic prompt holds no number log and a designer prompt no meta-review; the judge's pick is 1 to 3;
the agent calls' tokens in the trace equal the decision tokens the engine metered.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from recital.control import macamp as mp  # noqa: E402
from recital.control.export import load_run, summary_of  # noqa: E402

HYPERPARAMETERS = ("lr", "kl", "group_size")


def check(cell: Path) -> list[tuple[str, bool, str]]:
    view = load_run(cell)
    state, blobs = view.state, view.blobs
    out: list[tuple[str, bool, str]] = []
    candidates = {c.round: c for c in state.ordered_candidates()}
    out.append(("21 candidates, one per round", sorted(candidates) == list(range(1, mp.ROUNDS + 1)), f"rounds {sorted(candidates)}"))
    sandbox_steps = {candidates[r].steps for r in candidates if mp.slot_of(r).kind == "sandbox"}
    real_steps = {candidates[r].steps for r in candidates if mp.slot_of(r).kind == "real"}
    out.append(("sandbox candidates share one step count, real candidates another, real longer",
                len(sandbox_steps) == 1 and len(real_steps) == 1 and min(real_steps) > min(sandbox_steps),
                f"sandbox {sorted(sandbox_steps)} real {sorted(real_steps)}"))
    bad = []
    for r, c in sorted(candidates.items()):
        slot = mp.slot_of(r)
        expected = None if slot.stage == 1 else candidates[mp.real_round(slot.stage - 1)].id
        if c.parent != expected:
            bad.append(f"round {r}: parent {c.parent}, expected {expected}")
    out.append(("every sandbox and real candidate starts from its stage's starting candidate", not bad, "; ".join(bad) or "ok"))
    sandbox_ids = {c.id for r, c in candidates.items() if mp.slot_of(r).kind == "sandbox"}
    children = [c.id for c in candidates.values() if c.parent in sandbox_ids]
    out.append(("no sandbox candidate is a parent", not children, f"children of sandbox candidates: {children}"))
    initial = blobs.get_json(candidates[1].recipe_sha)["recipe"]["hyperparameters"]
    drift = []
    for r, c in sorted(candidates.items()):
        hp = blobs.get_json(c.recipe_sha)["recipe"]["hyperparameters"]
        drift += [f"round {r} {k}" for k in HYPERPARAMETERS if hp.get(k) != initial.get(k)]
    out.append(("no training setting other than steps differs from the first candidate's", not drift, ", ".join(drift) or "ok"))
    designs_path, trace_path = cell / "macamp_designs.json", cell / "macamp_trace.jsonl"
    designs = json.loads(designs_path.read_text()) if designs_path.exists() else {}
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()] if trace_path.exists() else []
    picks = {r["stage"]: r["pick"] for r in trace if r["role"] == "judge_pick"}
    wrong = []
    for stage in range(1, mp.STAGES + 1):
        real = blobs.get_json(candidates[mp.real_round(stage)].recipe_sha)["recipe"]
        pick = picks.get(stage)
        design = designs.get(f"s{stage}i{mp.ITERATIONS}", {}).get(str(pick - 1)) if pick else None
        if design is None:
            wrong.append(f"stage {stage}: no judge pick or design on record")
            continue
        weights = json.loads(design["aggregator"]["params_json"]).get("weights")
        if sorted(real.get("enabled_oracles") or []) != sorted(design["enabled_oracles"]) or (
                weights is not None and real["aggregator"].get("weights") != weights):
            wrong.append(f"stage {stage}: the real candidate's reward is not the judged trial {pick}'s")
    out.append(("the real candidate uses the judge's pick of the last iteration", not wrong, "; ".join(wrong) or f"picks {picks}"))
    out.append(("every judge pick is 1 to 3", all(p in (1, 2, 3) for p in picks.values()) and len(picks) == mp.STAGES, str(picks)))
    accepted = state.stop.accepted
    delivered = set(accepted.assignment.values()) if accepted else set()
    reals = {candidates[mp.real_round(s)].id for s in range(1, mp.STAGES + 1)}
    out.append(("the delivered candidate is a real one", bool(delivered) and delivered <= reals, f"delivered {sorted(delivered)}"))
    turns = {}
    for r in trace:
        if r["role"] == "designer":
            turns[r["round"]] = max(turns.get(r["round"], 0), r["turn"])
    out.append(("at most four designer turns per reward", all(t <= mp.MAX_DIALOGUE for t in turns.values()) and bool(turns), str(turns)))
    out.append(("no critic prompt holds the number log", not [r for r in trace if r["role"] == "critic" and r["has_signals"]],
                f"critic calls {sum(r['role'] == 'critic' for r in trace)}"))
    out.append(("no designer prompt holds the meta-review", not [r for r in trace if r["role"] == "designer" and r["has_meta_review"]],
                f"designer calls {sum(r['role'] == 'designer' for r in trace)}"))
    traced = sum(r["usage"]["input"] + r["usage"]["output"] for r in trace if "usage" in r)
    metered = summary_of(view)["tokens"]["by_call"]["decision"]
    metered_total = metered["prompt"] + metered["completion"]
    out.append(("the agent calls' tokens in the trace equal the decision tokens the engine metered", traced == metered_total,
                f"trace {traced}, engine {metered_total}"))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cell", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    results = check(args.cell)
    if args.json:
        print(json.dumps([{"check": n, "pass": ok, "detail": d} for n, ok, d in results], indent=1))
    else:
        for name, ok, detail in results:
            print(f"{'PASS' if ok else 'FAIL'}  {name}  [{detail}]")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
