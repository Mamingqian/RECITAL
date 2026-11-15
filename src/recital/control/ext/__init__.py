"""Optional modules of the control engine; with none declared the engine is the core.

A module is the file `recital/control/ext/<name>.py` with a function `build(config) -> Extension`. It is switched on
by naming it in the contract, `control.modules: {<name>: <its own config>}`, and `launch.prepare` builds every named
module (sorted by name). The engine calls an `Extension` at the points below and nowhere else; each hook has an empty
default, so a module overrides only what it uses, and with no module (or one whose hooks return nothing) every prompt,
schema, row, state and job the engine writes is what it wrote before modules existed (gate G0:
`tests/test_control/test_modules_off_identity.py`).

Hooks, and where the engine calls them

  candidate_fields(engine)            `Engine._round_schema`: JSON-schema property fragments added to every candidate
                                      object of the round response (the strict schema lists each property as required);
                                      a name that a core field already has is refused.
  instructions()                      `launch.prepare`: text appended to the Strategist's instructions.
  tail_lines(state, store, arm)       `projection.render`: lines added to the volatile tail before its instruction.
  round_blocks(state, store, arm, rnd)  `projection.ledger_blocks`: blocks placed after a round's rows.
  ledger_blocks(state, store, arm)    `projection.ledger_blocks`: blocks placed after the last round.
  row_fields(engine, state, candidate, spec, result)
                                      `Engine._rows`: a JSON object stored in the frozen ledger row under
                                      `row["ext"][<name>]`, or None. The row an agent sees is built field by field from
                                      the core fields, so what a module shows of it goes through a block.
  prepare_validation(engine, state, decision_id)
                                      `Engine._decide`, before a recorded decision is validated: returns True when it
                                      wrote events (the engine then steps again), False when it has nothing to do. It
                                      must be idempotent over the log, because it is asked again after it returned True.
  validate_candidate(engine, state, rnd, index, proposal, resolved)
                                      `Engine._validate_propose`, once per proposed candidate after its recipe resolved
                                      (`resolved` is None when it did not): typed `Problem`s, whose paths are relative to
                                      the candidate. A problem refuses the answer through the existing retry path.
  open_levers(state, rnd)             `Engine._round_schema` and `Engine._validate_propose`: the levers (`reward`,
                                      `training`) a module leaves open in round `rnd`, or None for no opinion. The
                                      engine intersects every module's answer with `control.levers`, so a module can
                                      only narrow what the contract opens; the response schema of that round omits the
                                      fields of the others and `recipes.resolve` refuses an edit of them
                                      (`lever_frozen`). The answer must depend only on `state`, because the round's
                                      decision is validated against it again after it was recorded.
  holds(engine, state)                `Engine._decide`, after the stop handling and before a recorded decision is validated,
                                      an action is taken or the Strategist is asked: a reason (text) while the engine must
                                      wait for a person, None otherwise. The engine does nothing in `_decide` while any
                                      module holds, and `step` reports WAITING. The answer must depend only on `state`,
                                      so a restarted controller holds, and releases, where it did.
  phases()                            `Engine.step`: `Phase(name, after, run)` entries run right after the core phase
                                      called `after` (one of `engine.PHASES`); `run(engine, state)` returns True when it
                                      made progress.
  stage_steps(engine, state, rnd, entry)
                                      `Engine._allocate`, for each candidate that is neither the anchor nor a module's
                                      own proposal (`entry` is its validated entry; `entry["resolved"]["steps"]` is the
                                      plan): the updates it trains in this stage, fewer than the plan, or None to train
                                      the plan. The candidate's `steps` is that number (so `committed_updates` counts
                                      what trains), its spec blob has `train.additional_updates` and `checkpoint_steps`
                                      for it and `planned_steps` for the plan, and its learning-rate schedule keeps the
                                      horizon of the plan, so a `continue_parent` candidate can resume it. The answer is
                                      read when the round is allocated and is not asked again.
  engine_proposals(state, rnd)        `Engine._validate_propose`, `_round_schema` and `ask_prompt`: proposals (the shape
                                      of a candidate of the response, every edit field None for none) the engine submits
                                      itself in round `rnd`, as it does the anchor in round 1. They are resolved with
                                      `recipes.resolve` before the Strategist's candidates, take the first slots after
                                      the anchor and are flagged `"source": "engine"` in the normalized answer; the
                                      Strategist is asked for the slots that remain. A proposal that does not resolve
                                      raises `ControlError` (a module bug, as for the anchor); the other modules'
                                      `validate_candidate` and `open_levers` do not apply to them. The answer must
                                      depend only on `state`, because the engine asks it for the schema, the prompt and
                                      the validation of the round.

Everything a module adds to a prompt is a `projection.Block` or a line of the tail or the instructions, so
`guard.tripwire` reads it with the rest. A module's events are named `<name>.<event>`, are declared in the module with
`ext.event` and folded with `ext.folds` into `state.ext[<name>]` (`ext.facts`), and register when
`recital.control.ext.<name>` is imported, which reading a log that holds one does.
"""
from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from recital.control import events as ev
from recital.control import state as st
from recital.control.errors import ControlError

NAME = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class Phase:
    name: str
    after: str                                   # a core phase: one of `engine.PHASES`
    run: Callable[[Any, st.RunState], bool]      # (engine, state) -> whether it made progress


class Extension:
    """The interface of a module; every hook does nothing until a module overrides it."""

    name: str = ""

    def candidate_fields(self, engine) -> Mapping[str, Mapping[str, Any]]:
        return {}

    def instructions(self) -> str:
        return ""

    def tail_lines(self, state, store, arm) -> Sequence[str]:
        return ()

    def round_blocks(self, state, store, arm, rnd) -> Sequence[Any]:
        return ()

    def ledger_blocks(self, state, store, arm) -> Sequence[Any]:
        return ()

    def row_fields(self, engine, state, candidate: str, spec: Mapping[str, Any],
                   result: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        return None

    def prepare_validation(self, engine, state, decision_id: str) -> bool:
        return False

    def validate_candidate(self, engine, state, rnd, index: int, proposal: Mapping[str, Any],
                           resolved) -> Sequence[ev.Problem]:
        return ()

    def open_levers(self, state, rnd) -> Sequence[str] | None:
        return None

    def holds(self, engine, state) -> str | None:
        return None

    def phases(self) -> Sequence[Phase]:
        return ()

    def stage_steps(self, engine, state, rnd, entry: Mapping[str, Any]) -> int | None:
        return None

    def engine_proposals(self, state, rnd) -> Sequence[Mapping[str, Any]]:
        return ()


def load(name: str, config: Mapping[str, Any]) -> Extension:
    if not NAME.match(name):
        raise ControlError(f"control.modules names {name!r}, which is not a module name (lowercase letters, digits, _)")
    path = f"recital.control.ext.{name}"
    try:
        module = importlib.import_module(path)
    except ModuleNotFoundError as exc:
        if exc.name != path:
            raise
        raise ControlError(f"control.modules names {name!r} and there is no module {path}") from exc
    build_module = getattr(module, "build", None)
    if build_module is None:
        raise ControlError(f"{path} has no build(config) function")
    extension = build_module(dict(config))
    if extension.name != name:
        raise ControlError(f"{path} built an extension named {extension.name!r}, not {name!r}")
    return extension


def build(modules: Mapping[str, Mapping[str, Any]]) -> tuple[Extension, ...]:
    """The extensions `control.modules` names, in name order."""
    return tuple(load(name, modules[name]) for name in sorted(modules))


def event(cls):
    """Declare a module's event: its type is `<module>.<event>` and its class lives under that module."""
    kind = cls.model_fields["type"].default
    module = kind.split(".", 1)[0]
    if "." not in kind or not NAME.match(module) or not cls.__module__.startswith(f"recital.control.ext.{module}"):
        raise ControlError(f"{cls.__module__}.{cls.__name__} has type {kind!r}; a module's events are named "
                           "<module>.<event> and declared in recital.control.ext.<module>")
    return ev.register(cls)


def folds(cls):
    """Register a function `(state, event)` that folds a module's event into `state.ext`."""
    return st.handles(cls)


def facts(state: st.RunState, name: str) -> dict[str, Any]:
    """The mutable record of module `name` inside `state.ext`, created on first use."""
    return state.ext.setdefault(name, {})
