"""Module M4, cycles: the search alternates between a training cycle and a reward cycle.

`control.modules.cycles` declares a phase table. Each phase opens one family of levers (`recipes.LEVER_FIELDS`: the
training settings or the reward) for at most `rounds` rounds and fixes the other; after the last phase the table starts
over. The restriction sits where the engine enforces any closed lever (`control.levers`): the round's response schema
omits the fields of the closed family and `recipes.resolve` refuses an edit of one with `lever_frozen`. Inside an open
family any number of its levers may move in one candidate.

A cycle ends when its rounds are used, or earlier when it stalls: none of the last `k` candidates trained in the cycle is
qualified in every condition and better than the best earlier candidate by more than `tolerance` on the last declared
ranking criterion (`auto` is the ranking's own rule: more than the two standard errors combined). The next cycle starts
from the best qualified candidate so far: the first round of a cycle accepts only that candidate as parent.

A table of one phase that opens every lever alternates nothing: every round is then the first round of its cycle, so the
module only forces the best qualified candidate so far as the parent, and the prompt names the parent rule and no cycle.
Together with the alternating tables it separates the parent rule from the restriction to one family.

Each handoff is a `cycles.phase_entered` event written between rounds, from the frozen rows alone, so a restarted
controller and a replay read the same table of phases.

    control.modules.cycles:
      phases:
        - {name: training, levers: [training], rounds: 2}
        - {name: reward, levers: [reward], rounds: 2}
      stall: {k: 3, tolerance: auto}      # optional; without it a cycle lasts its rounds
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from recital.success import SuccessSpec, tied_with_best

from .. import events as ev
from ..errors import ControlError
from ..events import Problem
from ..projection import Block, evidence_of
from ..recipes import LEVER_FIELDS, LEVERS
from ..selection import Evidence, best_candidate, condition_errors, condition_status, condition_values
from . import Extension, Phase, event, facts, folds

SETTINGS = ("phases", "stall")
PHASE_KEYS = ("name", "levers", "rounds")
STALL_KEYS = ("k", "tolerance")
SWITCHES = ("rounds", "stall")


@dataclass(frozen=True)
class CyclePhase:
    name: str
    levers: tuple[str, ...]
    rounds: int

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(name for lever in self.levers for name in LEVER_FIELDS[lever])

    @property
    def fixed(self) -> tuple[str, ...]:
        return tuple(name for lever in LEVERS if lever not in self.levers for name in LEVER_FIELDS[lever])


@dataclass(frozen=True)
class Stall:
    k: int
    tolerance: float | Literal["auto"]


def parse_phases(raw: Any) -> tuple[CyclePhase, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ControlError("control.modules.cycles.phases is a nonempty list of {name, levers, rounds}")
    phases = []
    for index, entry in enumerate(raw):
        where = f"control.modules.cycles.phases[{index}]"
        if not isinstance(entry, Mapping) or set(entry) != set(PHASE_KEYS):
            raise ControlError(f"{where} has the keys {sorted(entry) if isinstance(entry, Mapping) else entry!r}; "
                               f"it takes exactly {list(PHASE_KEYS)}")
        name, levers, rounds = entry["name"], entry["levers"], entry["rounds"]
        if not isinstance(name, str) or not name.isidentifier():
            raise ControlError(f"{where}.name must be an identifier, got {name!r}")
        if (not isinstance(levers, Sequence) or isinstance(levers, str) or not levers
                or len(set(levers)) != len(levers) or not set(levers) <= set(LEVERS)):
            raise ControlError(f"{where}.levers names each of {list(LEVERS)} at most once, and at least one; got {levers!r}")
        if type(rounds) is not int or rounds < 1:
            raise ControlError(f"{where}.rounds is an integer of at least 1, got {rounds!r}")
        phases.append(CyclePhase(name, tuple(lever for lever in LEVERS if lever in levers), rounds))
    return tuple(phases)


def parse_stall(raw: Any) -> Stall | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or set(raw) != set(STALL_KEYS):
        raise ControlError(f"control.modules.cycles.stall takes exactly {list(STALL_KEYS)}, got {raw!r}")
    k, tolerance = raw["k"], raw["tolerance"]
    if type(k) is not int or k < 1:
        raise ControlError(f"control.modules.cycles.stall.k is an integer of at least 1, got {k!r}")
    if tolerance != "auto" and (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or tolerance < 0):
        raise ControlError(f"control.modules.cycles.stall.tolerance is 'auto' or a number of at least 0, got {tolerance!r}")
    return Stall(k, tolerance if tolerance == "auto" else float(tolerance))


# ----- the stall rule ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Standing:
    """A candidate qualified in every condition, at the value of its weakest condition on the endpoint criterion."""

    id: str
    ordinal: int
    value: float
    error: float | None


def standing(success: SuccessSpec, item: Evidence, conditions: Sequence[str]) -> Standing | None:
    """None unless the candidate is qualified in every condition and carries the endpoint value in each."""
    if not item.ok or not success.ranking or not all(condition_status(item.qualification, c).qualified for c in conditions):
        return None
    index = len(success.ranking) - 1
    sign = 1.0 if success.ranking[index].direction == "maximize" else -1.0
    per_condition = []
    for condition in conditions:
        value = condition_values(success, item.qualification, condition)[index]
        if value is None:
            return None
        per_condition.append((value, condition_errors(success, item.qualification, condition)[index]))
    value, error = min(per_condition, key=lambda pair: sign * pair[0])
    return Standing(item.id, item.ordinal, value, error)


def improves(success: SuccessSpec, tolerance: float | str, found: Standing, best: Standing | None) -> bool:
    """Whether `found` is better than `best` on the endpoint criterion by more than the tolerance."""
    if best is None:
        return True
    criterion = success.ranking[-1]
    if (found.value > best.value) != (criterion.direction == "maximize") or found.value == best.value:
        return False
    try:
        return not tied_with_best(criterion.model_copy(update={"tolerance": tolerance}), found.value, found.error,
                                  best.value, best.error)
    except ValueError as exc:
        raise ControlError(f"control.modules.cycles.stall.tolerance: {exc}") from exc


def best_standing(success: SuccessSpec, standings: Sequence[Standing]) -> Standing | None:
    if not standings:
        return None
    sign = 1.0 if success.ranking[-1].direction == "maximize" else -1.0
    return max(standings, key=lambda found: (sign * found.value, -found.ordinal))


def stalled(success: SuccessSpec, stall: Stall, conditions: Sequence[str], evidence: Sequence[Evidence],
            trained: Sequence[str]) -> dict[str, Any] | None:
    """The evidence of a stall when none of the last `stall.k` of `trained` improves on everything before them.

    `trained` holds the ids of the cycle's candidates in the order they were allocated. None while the cycle has trained
    fewer than `k`, or when one of the last `k` improves.
    """
    if len(trained) < stall.k:
        return None
    window = set(trained[-stall.k:])
    by_id = {item.id: item for item in evidence}
    first = min(by_id[name].ordinal for name in window)
    best = best_standing(success, [found for item in evidence if item.ordinal < first
                                   and (found := standing(success, item, conditions)) is not None])
    found = [s for name in sorted(window) if (s := standing(success, by_id[name], conditions)) is not None]
    if any(improves(success, stall.tolerance, s, best) for s in found):
        return None
    return {"window": sorted(window), "best_before": None if best is None else best.value,
            "best_in_window": None if not found else best_standing(success, found).value,
            "tolerance": stall.tolerance}


# ----- events and the table of phases they fold to -------------------------------------------------------------

@event
class PhaseEntered(ev._Event):
    """The phase of `round`, decided between rounds. `occurrence` counts the cycles of the search from 0."""

    type: Literal["cycles.phase_entered"] = "cycles.phase_entered"
    round: int
    index: int
    name: str
    occurrence: int
    reason: Literal["start", "continue", "rounds", "stall"]
    incumbent: str | None
    detail: dict[str, Any]


@folds(PhaseEntered)
def _fold_entered(state, event):
    entered = facts(state, "cycles").setdefault("entered", {})
    entered.setdefault(str(event.round), {"index": event.index, "name": event.name, "occurrence": event.occurrence,
                                          "reason": event.reason, "incumbent": event.incumbent})


def entered_of(state) -> Mapping[str, Mapping[str, Any]]:
    """The phase of each round that has one, by round index as text; reads without creating the module's facts."""
    return (state.ext.get("cycles") or {}).get("entered", {})


def rounds_used(state, occurrence: int) -> int:
    """Rounds of this occurrence that allocated candidates."""
    return sum(1 for index, entry in entered_of(state).items()
               if entry["occurrence"] == occurrence and state.rounds[int(index) - 1].status == "allocated")


class Cycles(Extension):
    """Module M4: the phase scheduler."""

    name = "cycles"

    def __init__(self, config: Mapping[str, Any]):
        unknown = sorted(set(config) - set(SETTINGS))
        if unknown:
            raise ControlError(f"control.modules.cycles has the keys {unknown}; it reads {list(SETTINGS)}")
        if "phases" not in config:
            raise ControlError("control.modules.cycles needs `phases`: the declared table of cycles")
        self.phases_table = parse_phases(config["phases"])
        self.stall = parse_stall(config.get("stall"))

    @property
    def parent_only(self) -> bool:
        """One phase that opens every lever: nothing alternates, and the module only forces the incumbent as the parent."""
        return len(self.phases_table) == 1 and set(self.phases_table[0].levers) == set(LEVERS)

    # ----- what the Strategist is told ---------------------------------------------------------------------------

    def instructions(self) -> str:
        if self.parent_only:
            return ("Every round after the first starts from the best qualified candidate so far: name it as the parent "
                    "of every candidate of that round.")
        table = "; ".join(f"{phase.name}: {', '.join(phase.fields)} for at most {phase.rounds} "
                          f"round{'s' if phase.rounds > 1 else ''}" for phase in self.phases_table)
        stall = ("" if self.stall is None else
                 f" A cycle ends early when none of the last {self.stall.k} candidates trained in it is qualified in "
                 "every condition and better than the best earlier candidate by more than the ranking's tolerance.")
        return ("This search alternates between cycles, and each cycle opens one family of levers: "
                f"{table}; then the table starts over. The round's header names the open cycle. The fields of a closed "
                "family are not offered, and a candidate leaves that family as it is. Inside the open family several "
                "levers may move in one candidate." + stall + " The first round of a cycle starts from the best "
                "qualified candidate so far: name it as the parent of every candidate of that round.")

    def candidate_fields(self, engine) -> Mapping[str, Mapping[str, Any]]:
        closed = sorted({lever for phase in self.phases_table for lever in phase.levers} - set(engine.s.control.levers))
        if closed:
            raise ControlError(f"cycles opens the {closed} lever, which control.levers closes")
        return {}

    def tail_lines(self, state, store, arm) -> Sequence[str]:
        index = state.open_round.index if state.open_round else state.next_round
        entry = entered_of(state).get(str(index))
        if entry is None:
            return ()
        if self.parent_only:
            if entry["incumbent"] is not None:
                return (f"The parent of every candidate of this round is {entry['incumbent']}, the best qualified "
                        "candidate so far.",)
            return ("No candidate is yet qualified in every condition, so the parent is free.",) if index > 1 else ()
        phase = self.phases_table[entry["index"]]
        position = rounds_used(state, entry["occurrence"]) + 1
        closed = (f" Closed: {', '.join(phase.fixed)}; leave them as the parent has them"
                  f"{' (steps take the declared default)' if 'steps' in phase.fixed else ''}." if phase.fixed else "")
        lines = [f"## Cycle\nRound {index} is in the {phase.name} cycle, round {position} of at most {phase.rounds}. "
                 f"Open: {', '.join(phase.fields)}.{closed}"]
        if entry["incumbent"] is not None:
            lines.append(f"This round opens the cycle: the parent of every candidate is {entry['incumbent']}, the best "
                         "qualified candidate so far.")
        elif position == 1 and entry["reason"] != "start":
            lines.append("This round opens the cycle, and no candidate is yet qualified in every condition, so the "
                         "parent is free.")
        return lines

    def round_blocks(self, state, store, arm, rnd) -> Sequence[Block]:
        entry = entered_of(state).get(str(rnd.index + 1))
        if self.parent_only or entry is None or entry["reason"] not in SWITCHES:
            return ()
        before = self.phases_table[entered_of(state)[str(rnd.index)]["index"]].name
        start = (f"from {entry['incumbent']}" if entry["incumbent"] is not None
                 else "with no candidate yet qualified in every condition")
        text = (f"[cycle] The {before} cycle ended after round {rnd.index} ({entry['reason']}); round {rnd.index + 1} "
                f"opens the {entry['name']} cycle {start}.")
        return (Block(text, text),)

    # ----- what the engine enforces ----------------------------------------------------------------------------

    def open_levers(self, state, rnd) -> Sequence[str] | None:
        entry = entered_of(state).get(str(rnd.index))
        return None if entry is None else self.phases_table[entry["index"]].levers

    def validate_candidate(self, engine, state, rnd, index, proposal, resolved) -> Sequence[Problem]:
        entry = entered_of(state).get(str(rnd.index))
        if entry is None or entry["incumbent"] is None or proposal["parent"] == entry["incumbent"]:
            return ()
        where = "every round starts" if self.parent_only else f"the first round of the {entry['name']} cycle starts"
        return (Problem(code="phase_parent", path="parent",
                        detail=f"{where} from the best qualified candidate so far, {entry['incumbent']}; "
                               f"got {proposal['parent']!r}"),)

    # ----- the handoff -------------------------------------------------------------------------------------------

    def phases(self) -> Sequence[Phase]:
        return (Phase("cycles_handoff", "critic", self._hand_off),)

    def _hand_off(self, engine, state) -> bool:
        """Decide the phase of the next round once the last round is complete and its rows are frozen."""
        if (state.run is None or state.baseline.status != "ingested" or state.stop.accepted is not None
                or state.open_round is not None or state.uningested or state.unscored or state.unfrozen):
            return False
        target = state.next_round
        entered = entered_of(state)
        if str(target) in entered:
            return False
        if target == 1:
            engine.emit(PhaseEntered(round=1, index=0, name=self.phases_table[0].name, occurrence=0, reason="start",
                                     incumbent=None, detail={}))
            return True
        previous = entered.get(str(target - 1))
        if previous is None:
            raise ControlError(f"cycles has no phase for round {target - 1}; the module must be declared from the start")
        evidence = evidence_of(state, engine.blobs)
        conditions = list(engine.s.task.conditions)
        occurrence, used = previous["occurrence"], rounds_used(state, previous["occurrence"])
        phase = self.phases_table[previous["index"]]
        reason, detail = "continue", {"rounds_used": used}
        if used >= phase.rounds:
            reason = "rounds"
        elif self.stall is not None:
            trained = [c.id for c in state.ordered_candidates()
                       if entered.get(str(c.round), {}).get("occurrence") == occurrence and not c.anchor
                       and c.status != "forfeited"]
            found = stalled(engine.s.success, self.stall, conditions, evidence, trained)
            if found is not None:
                reason, detail = "stall", {**detail, **found}
        if reason == "continue":
            incumbent = previous["incumbent"] if used == 0 else None
            index = previous["index"]
        else:
            incumbent = best_candidate(engine.s.success, evidence, conditions)
            index, occurrence = (previous["index"] + 1) % len(self.phases_table), occurrence + 1
        engine.emit(PhaseEntered(round=target, index=index, name=self.phases_table[index].name,
                                 occurrence=occurrence, reason=reason, incumbent=incumbent, detail=detail))
        return True


def build(config: Mapping[str, Any]) -> Cycles:
    return Cycles(config)
