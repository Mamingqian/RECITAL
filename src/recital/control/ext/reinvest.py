"""Module M2, reinvest: the round's leader at the stage boundary gets the updates its siblings did not spend.

    control.modules.reinvest: {measure: "mingap/malinois", rung: 1000, first_round: 2}

In a round of `first_round` or later the Strategist plans `S_i` updates for each of its non-anchor candidates and the
engine trains only the first `rung`. When every sibling's ledger row is frozen, the module ranks them by the minimum over
conditions of the row's pool median of `measure`, takes the highest as the leader and sizes one continuation of it:
`sum_i (S_i - rung)` updates, capped so the leader's two stages total at most the declared `steps` maximum. The siblings
stay as stopped candidates with their rows. The continuation is proposed by the engine in the next round, as the anchor is
in the first (`continue_parent` of the leader's end checkpoint, no edits), and it is resolved before the Strategist's
candidates of that round, which are budget-checked after it; the freed updates the cap leaves over stay in the budget.
In the last round the contract allows no stage is cut, because no next round could hold the continuation.

This file holds the rule (`stage_value`, `rank_stage`, `continuation`), the module's events and the phases that write
them. `stage_steps` and `engine_proposals` are the module's side of two engine hooks the build lead adds: the first cuts a
stage to `rung` updates (the plan stays in the spec as `planned_steps`), the second proposes the continuation. The engine
does not call them yet, so with this module named a stage still trains its planned steps and no continuation is
allocated; the ranking and its events are written. A cut stage commits only the steps it trains, so the unspent plan stays
in the budget.
The rule is the one `scripts/analysis/probe_reinvest_replay.py` replays from stored checkpoints
(docs/reports/m2_probes_reinvest_replay_prereg_2026-11-04.md).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from .. import events as ev
from ..errors import ControlError
from ..projection import Block
from . import Extension, Phase, event, facts, folds

DEFAULTS = {"rung": 1000, "first_round": 2}
SETTINGS = {"measure", "rung", "first_round"}
NAME = "reinvest"
CONTINUE = "continue_parent"


# ----- the rule ----------------------------------------------------------------------------------------------------------

def stage_value(row: Mapping[str, Any], measure: str) -> float | None:
    """The minimum over conditions of the row's pool median of `measure`; None when the job failed or a condition has none."""
    pool = row.get("pool") if row.get("status") == "ok" else None
    if not pool:
        return None
    values = [(by_measure or {}).get(measure) for by_measure in pool.values()]
    if not values or any(v is None or not math.isfinite(v) for v in values):
        return None
    return float(min(values))


def rank_stage(values: Mapping[str, float | None], ordinal: Mapping[str, int]) -> list[str]:
    """Candidate ids from the highest value to the lowest; a candidate with no value is last, and a tie goes to the lower ordinal."""
    return sorted(values, key=lambda name: (values[name] is None, -(values[name] or 0.0), ordinal[name]))


@dataclass(frozen=True)
class Continuation:
    steps: int          # the leader's second stage, 0 for none
    capped: bool        # the cap, not the siblings' plan, set the size
    released: int       # freed updates that stay in the budget


def continuation(planned: Sequence[int], leader_steps: int, *, rung: int, steps_max: int, steps_min: int) -> Continuation:
    """The updates the siblings' plans leave unspent after `rung` each, less what the cap and the declared minimum take back."""
    freed = sum(max(steps - rung, 0) for steps in planned)
    steps = min(freed, max(steps_max - leader_steps, 0))
    if steps < steps_min:
        steps = 0
    return Continuation(steps=steps, capped=0 < steps < freed, released=freed - steps)


# ----- events and what they fold to --------------------------------------------------------------------------------------

@event
class StageRanked(ev._Event):
    """The siblings of `round` ranked at their stage boundary, and the continuation sized for the leader (`steps` 0: none)."""

    type: Literal["reinvest.stage_ranked"] = "reinvest.stage_ranked"
    round: int
    ranking: list[str]
    values: dict[str, float | None]
    planned: dict[str, int]
    leader: str | None
    steps: int
    capped: bool
    released: int


@event
class ContinuationAllocated(ev._Event):
    """The continuation of `source_round`'s leader took a slot of `round`."""

    type: Literal["reinvest.continuation_allocated"] = "reinvest.continuation_allocated"
    round: int
    source_round: int
    candidate: str
    parent: str
    steps: int


@folds(StageRanked)
def _fold_ranked(state, event):
    record = facts(state, NAME)
    record.setdefault("ranked", {}).setdefault(str(event.round), {
        "ranking": list(event.ranking), "leader": event.leader, "steps": event.steps, "capped": event.capped,
        "released": event.released, "planned": dict(event.planned)})
    if event.leader is not None and event.steps > 0:
        record.setdefault("pending", []).append({"source_round": event.round, "parent": event.leader, "steps": event.steps})


@folds(ContinuationAllocated)
def _fold_allocated(state, event):
    record = facts(state, NAME)
    record["pending"] = [p for p in record.get("pending", ()) if p["source_round"] != event.source_round]
    record.setdefault("continuations", {}).setdefault(event.candidate, {
        "round": event.round, "source_round": event.source_round, "parent": event.parent, "steps": event.steps})


# ----- the module --------------------------------------------------------------------------------------------------------

class Reinvest(Extension):
    name = NAME

    def __init__(self, config: Mapping[str, Any]):
        extra = sorted(set(config) - SETTINGS)
        if extra:
            raise ControlError(f"control.modules.reinvest has no setting {extra}; the settings are {sorted(SETTINGS)}")
        measure = config.get("measure")
        if not isinstance(measure, str) or not measure:
            raise ControlError("control.modules.reinvest needs `measure`, the pool measurement the leader is ranked by")
        self.measure = measure
        self.rung = config.get("rung", DEFAULTS["rung"])
        self.first_round = config.get("first_round", DEFAULTS["first_round"])
        if isinstance(self.rung, bool) or not isinstance(self.rung, int) or self.rung < 1:
            raise ControlError(f"control.modules.reinvest.rung must be a positive integer, not {self.rung!r}")
        if isinstance(self.first_round, bool) or not isinstance(self.first_round, int) or self.first_round < 2:
            raise ControlError(f"control.modules.reinvest.first_round must be an integer of at least 2 (round 1 holds the anchor), not {self.first_round!r}")

    # -- the phases

    def phases(self) -> Sequence[Phase]:
        return (Phase("reinvest_rank", "rows", self._rank), Phase("reinvest_record", "baseline", self._record))

    def _siblings(self, state, rnd) -> list[str]:
        """The round's non-anchor candidates that are not themselves a continuation."""
        continuations = facts(state, NAME).get("continuations", {})
        return [name for name in rnd.candidates
                if state.candidates[name].status != "forfeited" and not state.candidates[name].anchor and name not in continuations]

    def _rank(self, engine, state) -> bool:
        ranked = facts(state, NAME).get("ranked", {})
        for rnd in state.rounds:
            if rnd.status != "allocated" or rnd.index < self.first_round or str(rnd.index) in ranked:
                continue
            siblings = self._siblings(state, rnd)
            if len(siblings) < 2 or any(state.candidates[name].ledger_row_sha is None for name in siblings):
                continue
            planned = {name: self._planned(engine, state, name) for name in siblings}
            if all(planned[name] == state.candidates[name].steps for name in siblings):
                continue        # no stage was cut (the last round the contract allows), so nothing was left unspent
            rows = {name: engine.blobs.get_json(state.candidates[name].ledger_row_sha) for name in siblings}
            values = {name: stage_value(row, self.measure) for name, row in rows.items()}
            order = rank_stage(values, {name: state.candidates[name].ordinal for name in siblings})
            leader = order[0] if values[order[0]] is not None else None
            sized = continuation(list(planned.values()), 0 if leader is None else state.candidates[leader].steps, rung=self.rung,
                                 steps_max=engine.s.recipe_ctx.steps_max, steps_min=engine.s.recipe_ctx.steps_min)
            engine.emit(StageRanked(round=rnd.index, ranking=order, values=values, planned=planned, leader=leader,
                                    steps=0 if leader is None else sized.steps, capped=sized.capped,
                                    released=sized.released if leader is not None else sized.released + sized.steps))
            return True
        return False

    @staticmethod
    def _planned(engine, state, name: str) -> int:
        """The steps the Strategist planned: the spec records them when a stage was cut short, else the steps it trained."""
        spec = engine.blobs.get_json(state.candidates[name].recipe_sha)
        return int(spec.get("planned_steps", spec["train"]["additional_updates"]))

    def _record(self, engine, state) -> bool:
        """Name the candidate that took the continuation's slot, once its round is allocated."""
        record = facts(state, NAME)
        for pending in record.get("pending", ()):
            for rnd in state.rounds[pending["source_round"]:]:
                taken = next((name for name in rnd.candidates
                              if (candidate := state.candidates[name]).start == CONTINUE and candidate.parent == pending["parent"]
                              and candidate.steps == pending["steps"] and name not in record.get("continuations", {})), None)
                if rnd.status == "allocated" and taken is not None:
                    engine.emit(ContinuationAllocated(round=rnd.index, source_round=pending["source_round"], candidate=taken,
                                                      parent=pending["parent"], steps=pending["steps"]))
                    return True
        return False

    # -- the module's side of the engine hooks the build lead adds

    def stage_steps(self, engine, state, rnd, entry: Mapping[str, Any]) -> int | None:
        """The steps a candidate's first stage trains, or None for its plan: `rung` from `first_round` on, unless this round is the
        last the contract allows (no next round would hold the continuation)."""
        plan = int(entry["resolved"]["steps"])
        allocated = sum(1 for r in state.rounds if r.status == "allocated")
        last = allocated + 1 >= engine.s.control.proposals.rounds_max
        return self.rung if rnd.index >= self.first_round and not last and plan > self.rung else None

    def engine_proposals(self, state, rnd) -> list[dict[str, Any]]:
        """The proposals the engine submits itself in `rnd`: the oldest pending continuation, as the anchor is proposed in round 1."""
        pending = facts(state, NAME).get("pending", ())
        if not pending or rnd.index <= pending[0]["source_round"]:
            return []
        first = pending[0]
        return [{"parent": first["parent"], "start": CONTINUE,
                 "hypothesis": f"continue {first['parent']}, round {first['source_round']}'s leader at {self.rung} updates, "
                               f"with the {first['steps']} updates its siblings did not spend",
                 "steps": first["steps"], "lr": None, "kl": None, "group_size": None, "enabled_oracles": None,
                 "aggregator": None, "oracle_utilities": None, "penalties": None, "gates": None}]

    # -- what the Strategist reads

    def round_blocks(self, state, store, arm, rnd) -> Sequence[Block]:
        ranked = facts(state, NAME).get("ranked", {}).get(str(rnd.index))
        if ranked is None:
            return ()
        if ranked["leader"] is None or ranked["steps"] == 0:
            text = (f"Round {rnd.index}: the siblings were ranked at {self.rung} updates, {ranked['released']} planned updates "
                    "were returned to the budget and no candidate was continued.")
        else:
            text = (f"Round {rnd.index}: the siblings trained {self.rung} updates each and {ranked['leader']} ranked first; "
                    f"the engine continues {ranked['leader']} in the next round for {ranked['steps']} more updates, which are "
                    f"earmarked, and {ranked['released']} planned updates were returned to the budget.")
        return (Block(text=text, system_text=text),)


def build(config: Mapping[str, Any]) -> Reinvest:
    return Reinvest(config)
