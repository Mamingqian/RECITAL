"""Search arms that propose candidates without a language model, run through the control layer's own engine.

An arm replaces the Strategist and nothing else. `ArmTransport` stands where `ResponsesTransport` stands: the engine
asks it for a proposal, a Critic review or a stop, and every proposal passes `recipes.resolve`, the same validator an
LLM answer passes, with the round's siblings, the run's history and the unallocated update budget. The cell keeps the
contract's candidates per round, rounds, selection plan, library-health floor, ranking, delivery and terminal audit.

A new arm (another proposer, a model-based search, a module under test) is a subclass of `Arm` with one method,
`candidate(ctx)`, that returns one proposal in the Strategist's candidate schema; `ctx` carries the finished candidates'
frozen ledger rows, the declared ranking, the bounds, the primitive catalog and a seeded random generator. The arm
may draw as often as it likes: `ArmTransport` keeps the first draws the validators accept and counts the refused ones
in the answer's rationale.

The arm spends trained updates the way the Strategist does, out of `total_updates`: a round's steps are refused when
they exceed what is unallocated, and the engine offers only a stop once the rounds or the update budget are used up.
The stop an arm gives is the declared ranking's best qualified candidate per condition (`selection.best_assignment`),
the pick the engine takes itself when a Strategist's stops are refused; the Critic's answer is empty.
"""
from __future__ import annotations

import importlib
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from . import recipes as rc
from .errors import ControlError
from .projection import evidence_of
from .selection import best_assignment, condition_status, condition_values
from .transport import Turn, Usage

MAX_DRAWS = 200  # per open slot, before the arm falls back to restarting the declared recipe


@dataclass(frozen=True)
class Finished:
    """A completed candidate as an arm sees it: its recipe, its frozen ledger row and how it ranks."""

    id: str
    round: int
    parent: str | None
    start: str
    steps: int
    recipe: Mapping[str, Any]
    row: Mapping[str, Any]
    qualified: Mapping[str, bool]
    values: Mapping[str, tuple[float | None, ...]]  # per condition, one value per declared ranking criterion

    @property
    def all_qualified(self) -> bool:
        return all(self.qualified.values())

    def score(self, direction: str) -> float | None:
        """The first ranking criterion averaged over conditions, signed so that larger is better."""
        found = [value[0] for value in self.values.values() if value and value[0] is not None]
        if not found or len(found) < len(self.values):
            return None
        mean = sum(found) / len(found)
        return mean if direction == "maximize" else -mean


@dataclass
class ArmContext:
    round: int
    slots: int
    finished: list[Finished]
    remaining_updates: int
    steps: tuple[int, int, int]  # declared minimum, default, maximum
    bounds: Mapping[str, Any]
    names: Mapping[str, Sequence[str]]  # reward_oracles, aggregators, utilities, penalties, gates
    conditions: Sequence[str]
    direction: str
    rng: random.Random
    initial_recipe: Mapping[str, Any]
    allocated: int = 0  # candidates the agent proposed in earlier rounds (the anchor excluded)
    slot: int = 0       # index of the open slot being filled in this answer
    levers: tuple[str, ...] = ("reward", "training")  # the levers of a recipe the search may edit (`control.levers`)

    def ranked(self) -> list[Finished]:
        """Finished candidates best first by the first ranking criterion; those qualified in every condition lead."""
        scored = [(item, item.score(self.direction)) for item in self.finished]
        scored = [(item, value) for item, value in scored if value is not None]
        scored.sort(key=lambda pair: (not pair[0].all_qualified, -pair[1], pair[0].id))
        return [item for item, _ in scored]

    def recipe_of(self, parent: str) -> Mapping[str, Any]:
        if parent == "initial":
            return self.initial_recipe
        return next(item.recipe for item in self.finished if item.id == parent)

    def bound(self, key: str) -> tuple[float, float] | None:
        found = self.bounds.get(key) or {}
        if "minimum" in found and "maximum" in found:
            return float(found["minimum"]), float(found["maximum"])
        return None


class Arm:
    """One proposer. `candidate` returns one proposal in the Strategist's candidate schema."""

    name = "arm"

    def __init__(self, seed: int):
        self.seed = seed

    def candidate(self, ctx: ArmContext) -> dict[str, Any]:
        raise NotImplementedError

    def fallback(self, ctx: ArmContext) -> dict[str, Any]:
        """What a slot gets when `MAX_DRAWS` draws were all refused: the declared recipe restarted, with fewer steps."""
        steps = max(min(ctx.steps[0], ctx.remaining_updates), 1) if "training" in ctx.levers else None
        return proposal(parent="initial", start="from_initial", hypothesis=f"{self.name}: restart of the declared recipe",
                        steps=steps)


def proposal(*, parent: str, start: str, hypothesis: str, steps: int | None = None, **edits: Any) -> dict[str, Any]:
    """A candidate in the Strategist's schema; every field not named inherits (null)."""
    base = {"parent": parent, "start": start, "hypothesis": hypothesis, "steps": steps, "lr": None, "kl": None,
            "group_size": None, "enabled_oracles": None, "aggregator": None, "oracle_utilities": None,
            "penalties": None, "gates": None}
    unknown = set(edits) - set(base)
    if unknown:
        raise ControlError(f"unknown candidate fields {sorted(unknown)}")
    return {**base, **edits}


def call(name: str, params: Mapping[str, Any] | None = None) -> dict[str, str]:
    return {"name": name, "params_json": json.dumps(dict(params or {}), sort_keys=True)}


def as_calls(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """A recipe's `penalties` or `gates` (`{"name": ..., **params}`) as the proposal's name and params_json pairs."""
    return [call(entry["name"], {key: value for key, value in entry.items() if key != "name"}) for entry in entries]


def grid(low: float, high: float, n: int, *, log: bool) -> list[float]:
    if n < 2 or low >= high:
        return [low]
    if log:
        inner = [math.exp(math.log(low) + (math.log(high) - math.log(low)) * i / (n - 1)) for i in range(1, n - 1)]
    else:
        inner = [low + (high - low) * i / (n - 1) for i in range(1, n - 1)]
    return [low, *inner, high]  # the declared bounds exactly: a rounded endpoint outside them is refused


def steps_menu(ctx: ArmContext) -> list[int]:
    low, default, high = ctx.steps
    return sorted({value for value in (low, (low + default) // 2, default, (default + high) // 2, high)
                   if value <= ctx.remaining_updates})


class RandomArm(Arm):
    """Parent-aware random edits from the Strategist's own menu.

    The parent is drawn from the ledger the way a Strategist reads it: the three best finished candidates by the
    declared ranking (those qualified in every condition first), the declared recipe with probability `RESTART`. The
    candidate starts from that parent's weights, restarts from the initial policy, or continues the parent's schedule,
    and each lever of the menu (updates, learning rate, KL coefficient, one gate, one penalty, one reward source) is
    edited with probability `EDIT` relative to the parent's recipe, at least one lever per candidate. A draw the
    validators refuse is discarded and drawn again.
    """

    name = "random"
    RESTART = 0.10
    EDIT = 0.35
    TOP = 3
    STARTS = (("from_parent", 0.5), ("from_initial", 0.25), ("continue_parent", 0.25))

    def _parent(self, ctx: ArmContext) -> str:
        ranked = ctx.ranked()
        if not ranked or ctx.rng.random() < self.RESTART:
            return "initial"
        return ctx.rng.choice(ranked[:self.TOP]).id

    def _toggle(self, ctx: ArmContext, current: Sequence[Mapping[str, Any]], pool: Sequence[str]) -> list[dict[str, str]]:
        names = list(pool)
        if not names:
            return as_calls(current)
        chosen = ctx.rng.choice(names)
        present = [entry for entry in current if entry["name"] == chosen]
        if present:
            return as_calls([entry for entry in current if entry["name"] != chosen])
        return as_calls([*current, {"name": chosen}])

    def candidate(self, ctx: ArmContext) -> dict[str, Any]:
        rng = ctx.rng
        parent = self._parent(ctx)
        if parent == "initial":
            start = "from_initial"
        else:
            start = rng.choices([name for name, _ in self.STARTS], [weight for _, weight in self.STARTS])[0]
        base = ctx.recipe_of(parent)
        edits: dict[str, Any] = {}
        levers = [lever for lever in ("steps", "lr", "kl") if "training" in ctx.levers]
        levers += [lever for lever in ("gate", "penalty", "reward") if "reward" in ctx.levers]
        chosen = [lever for lever in levers if rng.random() < self.EDIT] or [rng.choice(levers)]
        steps = None
        for lever in chosen:
            if lever == "steps":
                steps = rng.choice(steps_menu(ctx) or [ctx.steps[0]])
            elif lever in ("lr", "kl"):
                bound = ctx.bound(lever)
                if bound is not None:
                    edits[lever] = rng.choice(grid(*bound, 5, log=True))
            elif lever == "gate":
                edits["gates"] = self._toggle(ctx, base.get("gates") or [], ctx.names.get("gates", ()))
            elif lever == "penalty":
                edits["penalties"] = self._toggle(ctx, base.get("penalties") or [], ctx.names.get("penalties", ()))
            elif lever == "reward":
                pool = list(ctx.names.get("reward_oracles", ()))
                enabled = list(base.get("enabled_oracles") or [])
                if pool:
                    flip = rng.choice(pool)
                    edits["enabled_oracles"] = ([name for name in enabled if name != flip] if flip in enabled
                                                else [*enabled, flip])
        if start == "continue_parent":
            edits.pop("lr", None)  # continue_parent keeps the parent's lr (recipes._training)
        return proposal(parent=parent, start=start, steps=steps, hypothesis=f"{self.name}: edit {', '.join(chosen)}",
                        **edits)


GATE_FOR = {"dinucleotide_entropy": ("dinucleotide_entropy_gate", lambda low, high: {"min_entropy": low}),
            "gc": ("gc_bounds_gate", lambda low, high: {"min_gc": low, "max_gc": high})}


def gate_for(measurement: str) -> tuple[str, Callable[[float, float], dict[str, float]]] | None:
    lowered = measurement.lower()
    for key, entry in GATE_FOR.items():
        if lowered == key or lowered.startswith(key + "_") or lowered.endswith("_" + key):
            return entry
    return None


class RuleArm(Arm):
    """A fixed rule on the incumbent: gate what the composition line flags, then sweep the KL coefficient.

    Round 1 has no finished candidate: its open slots restart the declared recipe at the first values of the KL grid.
    Later rounds take the incumbent (the best finished candidate by the declared ranking, those
    qualified in every condition first), add a gate for every source-free measurement whose pool median lies outside the
    initial pool's 5th-95th percentile band in any condition (the gate primitive of that measurement, its bound set at the
    band's edge) and give each open slot the next KL value of the declared bounds' three-point geometric grid (low, middle,
    high, in order across all slots of the cell), steps at the declared default, started from the incumbent's weights. A measurement without a gate primitive in the task's catalog
    is left alone: a task with no gate for what its composition line flags gets the KL sweep only.
    """

    name = "rule"

    def _kl_grid(self, ctx: ArmContext) -> list[float | None]:
        bound = ctx.bound("kl")
        return grid(*bound, 3, log=True) if bound is not None else [None]

    def _gates(self, ctx: ArmContext, incumbent: Finished) -> list[dict[str, str]] | None:
        catalog = set(ctx.names.get("gates", ()))
        present = {entry["name"]: dict(entry) for entry in incumbent.recipe.get("gates") or []}
        flagged: dict[str, Any] = {}
        for found in (incumbent.row.get("composition") or {}).values():
            for measurement, item in found.items():
                match = gate_for(measurement)
                if match is not None and match[0] in catalog:
                    low, high = item["band"]
                    flagged[match[0]] = match[1](low, high)
        if not flagged:
            return None
        merged = {**{name: {key: value for key, value in entry.items() if key != "name"}
                     for name, entry in present.items()}, **flagged}
        if merged == {name: {key: value for key, value in entry.items() if key != "name"}
                      for name, entry in present.items()}:
            return None
        return [call(name, params) for name, params in sorted(merged.items())]

    def candidate(self, ctx: ArmContext) -> dict[str, Any]:
        grid_values = self._kl_grid(ctx)
        training = "training" in ctx.levers
        kl = grid_values[(ctx.allocated + ctx.slot) % len(grid_values)] if training else None
        steps = ctx.steps[1] if training else None
        ranked = ctx.ranked()
        if not ranked:
            return proposal(parent="initial", start="from_initial", steps=steps, kl=kl,
                            hypothesis=f"{self.name}: declared recipe at KL {kl}")
        incumbent = ranked[0]
        gates = self._gates(ctx, incumbent) if "reward" in ctx.levers else None
        return proposal(parent=incumbent.id, start="from_parent", steps=steps, kl=kl, gates=gates,
                        hypothesis=f"{self.name}: incumbent {incumbent.id} at KL {kl}"
                                   + (", gates added for flagged composition" if gates else ""))


ARMS: dict[str, type[Arm]] = {"random": RandomArm, "rule": RuleArm}


def arm_class(spec: str) -> type[Arm]:
    """A built-in arm by name, or `package.module:Class` for an arm under test."""
    if spec in ARMS:
        return ARMS[spec]
    if ":" not in spec:
        raise ControlError(f"unknown arm {spec!r}; built in: {sorted(ARMS)}, or give module:Class")
    module, _, name = spec.partition(":")
    found = getattr(importlib.import_module(module), name)
    if not (isinstance(found, type) and issubclass(found, Arm)):
        raise ControlError(f"{spec} is not a subclass of Arm")
    return found


def arm_config(config: Mapping[str, Any], arm_name: str, *, total_updates: int | None = None) -> dict[str, Any]:
    """`config` (a cell config of a control campaign) with the Strategist replaced by an arm: no Coder, no model.

    The contract's candidates per round, rounds, budget, selection plan, health floor and scheduler are untouched.
    `total_updates` equalizes the arm's trained updates with a comparison search's.
    """
    import copy
    out = copy.deepcopy(dict(config))
    out["control"]["synthesis"] = {**out["control"]["synthesis"], "enabled": False, "roles": [], "max_requests": 0}
    out["llm"] = {**out["llm"], "model": f"arm-{arm_name}", "base_url": None, "api_key_file": None,
                  "max_total_tokens": None}
    if total_updates is not None:
        out["total_updates"] = int(total_updates)
    return out


def _names(schema: Mapping[str, Any]) -> dict[str, list[str]]:
    """The catalog of one round, read from that round's own response schema, so an arm offers what an agent is offered."""
    item = schema["properties"]["candidates"]["items"]["properties"]

    def enum(field: Mapping[str, Any], path: Sequence[str]) -> list[str]:
        node: Any = field
        for key in path:
            node = node[key]
        return list(node.get("enum", ()))

    return {
        "reward_oracles": enum(item["enabled_oracles"], ("items",)),
        "aggregators": enum(item["aggregator"], ("properties", "name")),
        "utilities": enum(item["oracle_utilities"], ("items", "properties", "name")),
        "penalties": enum(item["penalties"], ("items", "properties", "name")),
        "gates": enum(item["gates"], ("items", "properties", "name")),
    }


@dataclass
class ArmTransport:
    """The engine's model transport, answered by an arm. Bind it to the engine after `launch.prepare`."""

    arm: Arm
    model: str = ""
    engine: Any = None
    refused_draws: int = 0
    requests: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.model = self.model or f"arm-{self.arm.name}"

    def bind(self, engine: Any) -> "ArmTransport":
        self.engine = engine
        return self

    def respond(self, input_items, *, instructions=None, schema=None, schema_name="response", tools=()):
        if self.engine is None:
            raise ControlError("the arm transport is not bound to an engine")
        self.requests.append(schema_name)
        if schema_name == "review":
            parsed: dict[str, Any] = {"concerns": [], "requested_evidence": [], "rationale": f"no critic: {self.model}"}
        elif "propose" in schema["properties"]["kind"]["enum"]:
            parsed = self._propose(schema)
        else:
            parsed = self._stop()
        return Turn(text=json.dumps(parsed), parsed=parsed, tool_calls=(), items=(), usage=Usage(0, 0, 0, 0),
                    model=self.model, latency_s=0.0)

    # ----- the answers ----------------------------------------------------------------------------

    def _stop(self) -> dict[str, Any]:
        engine = self.engine
        state = engine.state()
        conditions = list(engine.s.task.conditions)
        picks = best_assignment(engine.s.success, evidence_of(state, engine.blobs), conditions)
        base = {"kind": "stop", "rationale": f"{self.model}: rounds or updates used up", "candidates": [],
                "component": None}
        if picks is None:
            return {**base, "stop": {"outcome": "no_qualified_candidate", "assignment": [],
                                     "summary": "no candidate qualifies in every condition"}}
        return {**base, "stop": {"outcome": "deliver", "summary": "declared ranking's best qualified candidate",
                                 "assignment": [{"condition": c, "candidate": picks[c]} for c in conditions]}}

    def _context(self, schema: Mapping[str, Any], state, rnd) -> tuple[ArmContext, dict, dict, list]:
        engine = self.engine
        settings = engine.s
        parents = engine._parents(state)
        history: dict[str, str] = {}
        for item in state.ordered_candidates():
            if item.status != "forfeited" and item.fingerprint:
                history.setdefault(item.fingerprint, item.id)
        siblings = []
        if rnd.index == 1 and settings.control.proposals.anchor_candidate:
            anchor, problems = rc.resolve(rc.anchor_proposal(), settings.recipe_ctx, parents,
                                          committed_updates=state.committed_updates, history=history, anchor=True)
            if problems:
                raise ControlError("the declared initial recipe does not resolve: "
                                   + "; ".join(f"{p.code} {p.detail}" for p in problems))
            siblings.append(anchor)
        siblings += [rc.Resolved(**entry["resolved"]) for entry in engine._kept(state, rnd)]
        finished = []
        for item in state.ordered_candidates():
            if item.status != "ok" or item.ledger_row_sha is None:
                continue
            row = engine.blobs.get_json(item.ledger_row_sha)
            spec = engine.blobs.get_json(item.recipe_sha)
            qualification = row.get("qualification")
            finished.append(Finished(
                id=item.id, round=item.round, parent=item.parent, start=item.start or "", steps=item.steps,
                recipe=spec["recipe"], row=row,
                qualified={c: condition_status(qualification, c).qualified for c in settings.task.conditions},
                values={c: condition_values(settings.success, qualification, c) for c in settings.task.conditions}))
        ctx = settings.recipe_ctx
        # Seeded by the arm, the round and the attempt, so a controller restarted mid-cell draws what it would have drawn.
        rng = random.Random(f"{self.arm.name}:{self.arm.seed}:{rnd.index}:{rnd.attempts}")
        arm_ctx = ArmContext(
            round=rnd.index, slots=schema["properties"]["candidates"]["maxItems"], finished=finished,
            remaining_updates=state.remaining_updates - sum(item.steps for item in siblings),
            steps=(ctx.steps_min, ctx.steps_default, ctx.steps_max), bounds=ctx.hyperparameter_bounds,
            names=_names(schema), conditions=list(settings.task.conditions),
            direction=settings.success.ranking[0].direction if settings.success.ranking else "maximize",
            rng=rng, initial_recipe=ctx.initial_recipe, levers=ctx.levers,
            allocated=sum(1 for item in state.ordered_candidates()
                          if item.round < rnd.index and not item.anchor and item.status != "forfeited"))
        return arm_ctx, parents, history, siblings

    def _propose(self, schema: Mapping[str, Any]) -> dict[str, Any]:
        engine = self.engine
        state = engine.state()
        rnd = state.open_round
        arm_ctx, parents, history, siblings = self._context(schema, state, rnd)
        accepted: list[dict[str, Any]] = []
        resolved = list(siblings)
        refused = 0
        for slot in range(arm_ctx.slots):
            arm_ctx.slot = slot
            for draw in range(MAX_DRAWS + 1):
                candidate = self.arm.candidate(arm_ctx) if draw < MAX_DRAWS else self.arm.fallback(arm_ctx)
                spent = state.committed_updates + sum(item.steps for item in resolved)
                item, problems = rc.resolve(candidate, engine.s.recipe_ctx, parents, committed_updates=state.committed_updates,
                                            siblings=resolved, history=history)
                if item is not None:
                    accepted.append(candidate)
                    resolved.append(item)
                    arm_ctx.remaining_updates = engine.s.recipe_ctx.total_updates - (spent + item.steps)
                    break
                refused += 1
        self.refused_draws += refused
        return {"kind": "propose", "rationale": f"{self.model}: {len(accepted)} candidates, {refused} draws refused",
                "candidates": accepted, "stop": None, "component": None}
