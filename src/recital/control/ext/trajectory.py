"""Module B1, trajectory: a digest of how the latest round's candidates trained, shown beside the ledger.

    control.modules.trajectory: {view: reward}                  # the reward components over the run, for the round's best candidate
    control.modules.trajectory: {view: training, rounds: 2}     # the training log, for every candidate of the last two rounds

The feedback channel of two published LLM agents, each given to the Strategist as ledger text and nothing else. `reward`
is Eureka's reward reflection (Ma et al. 2023, section 3.3): the mean of each reward component over the run, for the one
candidate of the latest round that the declared ranking puts first among those qualified in every condition (its own
prompt keeps one reward and its reflection). `training` is AgentHPO's trajectory log (Liu et al. 2024, section 3.3): the
training curves of the trials, here five window means of the logged reward, loss, KL, entropy and gradient norm of every
candidate of the last `rounds` rounds, where AgentHPO shows every earlier trial and an LLM-written analysis of each.

Which candidates are shown is decided between rounds and recorded as a `trajectory.shown` event, so the log reproduces
the prompt. The digest is computed from the candidate's `training_statistics.json` when its ledger row is frozen and stored
under `row["ext"]["trajectory"]`. Both views state facts about the run; what they imply for the next edit is left to the
reader, as `reflect.py` does.
"""
from __future__ import annotations

import json
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from .. import events as ev
from ..errors import ControlError
from ..events import Problem
from ..guard import count_tokens
from ..projection import Block, evidence_of
from ..selection import best_candidate
from . import Extension, Phase, event, facts, folds, reflect

VIEWS = ("reward", "training")
SETTINGS = {"view": "reward", "rounds": 2, "restart": True}
TOKEN_CAP = 250
WINDOWS = reflect.WINDOWS
TRAINING_SERIES = (("reward", "reward mean"), ("reward_std", "within-batch spread"), ("loss", "loss"),
                   ("grad_norm", "gradient norm"), ("kl", "kl"), ("entropy", "entropy"),
                   ("clip_ratio/region_mean", "clip ratio"), ("frac_reward_zero_std", "groups without spread"))

REWARD_INSTRUCTIONS = (
    "The ledger carries a reflection on the training of the best candidate of the latest round: the mean of every reward "
    "component, the total reward and its within-batch spread over five windows of the run, first to last. Read it "
    "component by component before the next edit. A component whose mean does not move over the run is not being "
    "optimized as written. A component whose magnitude is far below the others does not change the gradient. A run with "
    "no within-batch reward spread carries no learning signal.")
RESTART_INSTRUCTIONS = ("Every candidate trains from the initial policy (start from_initial); the parent names the recipe "
                        "you edit.")
TRAINING_INSTRUCTIONS = (
    "The ledger carries the training log of every candidate of the last rounds: reward, loss, gradient norm, KL and "
    "entropy as five window means over the run, first to last. Read the logs of the completed trials before the next "
    "edit, and choose settings that differ from those already tried.")


def parse(config: Mapping[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(config) - set(SETTINGS))
    if unknown:
        raise ControlError(f"control.modules.trajectory has the keys {unknown}; it reads {sorted(SETTINGS)}")
    settings = {**SETTINGS, **config}
    if settings["view"] not in VIEWS:
        raise ControlError(f"control.modules.trajectory.view is one of {list(VIEWS)}, got {settings['view']!r}")
    if type(settings["rounds"]) is not int or settings["rounds"] < 1:
        raise ControlError("control.modules.trajectory.rounds is a positive integer")
    if type(settings["restart"]) is not bool:
        raise ControlError("control.modules.trajectory.restart is true or false")
    return settings


# ----- the digest of one run -----------------------------------------------------------------------------------------

def _means(entries: Sequence[Mapping[str, Any]], key: str, windows: int) -> list[float | None]:
    out: list[float | None] = []
    for start, stop in reflect._windows(len(entries), windows):
        values = [float(e[key]) for e in entries[start:stop] if isinstance(e.get(key), (int, float))]
        out.append(float(np.mean(values)) if values else None)
    return out


def training_digest(statistics: Mapping[str, Any], *, windows: int = WINDOWS, token_cap: int = TOKEN_CAP) -> str | None:
    """Window means of the trainer's own log, one line per pair of series; None when the run logged nothing."""
    logged = [entry for entry in statistics.get("training_log") or [] if "reward" in entry and "step" in entry]
    if len(logged) < 2:
        return None
    steps = [int(entry["step"]) for entry in logged]
    lines = [f"training log ({len(logged)} logs, steps {steps[0]} to {steps[-1]}, {windows} windows, first to last):"]
    series = [(label, _means(logged, key, windows)) for key, label in TRAINING_SERIES]
    series = [(label, values) for label, values in series if any(v is not None for v in values)]
    for index in range(0, len(series), 2):
        lines.append(" " + "  ".join(f"{label} {reflect._list(values)}" for label, values in series[index:index + 2]))
    text = "\n".join(lines)
    return text if count_tokens(text) <= token_cap else "\n".join(lines[:1] + lines[1:3])


def reward_digest(statistics: Mapping[str, Any]) -> str | None:
    if not statistics.get("reward_execution_history"):
        return None
    summary = reflect.summarize(statistics, statistics.get("training_log") or [])
    return reflect.render(summary, reflect.flags(summary), token_cap=TOKEN_CAP, count_tokens=count_tokens)


def anonymous(text: str, names: Sequence[str]) -> str:
    """The digest with each reward component's name replaced by its position, for the tripwire's reading."""
    label = {name: f"component{index + 1}" for index, name in enumerate(sorted(set(names)))}
    for name in sorted(label, key=len, reverse=True):
        text = text.replace(name, label[name])
    return text


# ----- the event ------------------------------------------------------------------------------------------------------

@event
class Shown(ev._Event):
    """The candidates whose digest the ledger shows from round `round` on."""

    type: Literal["trajectory.shown"] = "trajectory.shown"
    round: int
    view: str
    candidates: list[str]


@folds(Shown)
def _fold_shown(state, event):
    facts(state, "trajectory").setdefault("shown", {}).setdefault(str(event.round), list(event.candidates))


def shown_of(state) -> list[str]:
    """The candidates of the latest `trajectory.shown` event, in the order the event lists them."""
    entries = (state.ext.get("trajectory") or {}).get("shown") or {}
    return entries[str(max(map(int, entries)))] if entries else []


# ----- the module -----------------------------------------------------------------------------------------------------

class Trajectory(Extension):
    name = "trajectory"

    def __init__(self, config: Mapping[str, Any]):
        self.settings = parse(config)
        self.view = self.settings["view"]

    def instructions(self) -> str:
        text = REWARD_INSTRUCTIONS if self.view == "reward" else TRAINING_INSTRUCTIONS
        return text + (" " + RESTART_INSTRUCTIONS if self.settings["restart"] else "")

    def validate_candidate(self, engine, state, rnd, index, proposal, resolved) -> Sequence[Problem]:
        if not self.settings["restart"] or proposal["start"] == "from_initial":
            return ()
        return (Problem(code="restart_only", path="start",
                        detail=f"every candidate of this search trains from the initial policy (start from_initial); "
                               f"got {proposal['start']!r}"),)

    def row_fields(self, engine, state, candidate, spec, result) -> Mapping[str, Any] | None:
        if result is None:
            return None
        path = engine.candidate_dir(candidate) / "training_statistics.json"
        if not path.exists():
            return None
        statistics = json.loads(path.read_text())
        text = reward_digest(statistics) if self.view == "reward" else training_digest(statistics)
        if text is None:
            return None
        names = sorted({*(statistics.get("enabled_oracles") or []),
                        *(p["name"] for p in ((statistics.get("reward_execution") or {}).get("penalties") or {}).values())})
        return {"digest": text, "names": names}

    def ledger_blocks(self, state, store, arm) -> Sequence[Block]:
        parts, system = [], []
        for name in shown_of(state):
            candidate = state.candidates[name]
            row = store.get_json(candidate.ledger_row_sha) if candidate.ledger_row_sha else {}
            found = (row.get("ext") or {}).get(self.name)
            if found is None:
                continue
            parts.append(f"[{name}] {found['digest']}")
            system.append(f"[{name}] {anonymous(found['digest'], found['names'])}")
        if not parts:
            return ()
        head = "Reflection on the best candidate of the latest round:" if self.view == "reward" else \
            "Training logs of the latest rounds:"
        return (Block("\n".join([head, *parts]), "\n".join([head, *system])),)

    def phases(self) -> Sequence[Phase]:
        return (Phase("trajectory_shown", "critic", self._choose),)

    def _choose(self, engine, state) -> bool:
        """Once the latest round is complete and its rows are frozen, record whose digest the next prompt shows."""
        if (state.run is None or state.baseline.status != "ingested" or state.stop.accepted is not None
                or state.open_round is not None or state.uningested or state.unscored or state.unfrozen):
            return False
        target = state.next_round
        if target == 1 or str(target) in ((state.ext.get("trajectory") or {}).get("shown") or {}):
            return False
        evidence = evidence_of(state, engine.blobs)
        conditions = list(engine.s.task.conditions)
        if self.view == "reward":
            latest = {c for c in state.rounds[target - 2].candidates if state.candidates[c].status == "ok"}
            pick = best_candidate(engine.s.success, [e for e in evidence if e.id in latest], conditions)
            chosen = [] if pick is None else [pick]
        else:
            rounds = state.rounds[max(0, target - 1 - self.settings["rounds"]):target - 1]
            chosen = [c for r in rounds for c in r.candidates if state.candidates[c].status == "ok"]
        engine.emit(Shown(round=target, view=self.view, candidates=chosen))
        return True


def build(config: Mapping[str, Any]) -> Trajectory:
    return Trajectory(config)
