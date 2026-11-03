"""A drill of the Coder step on a real run: scripted Strategist answers, a real Coder, real training.

The Strategist and Critic are scripted so the drill is deterministic and cheap; the Coder, the sandbox, the
composition screen, the workers and the GPU are real. It asks for a per-oracle utility and a cross-oracle
aggregator, then proposes two candidates that use them, which is the only way to learn whether the model writes
code the sandbox admits and a recipe the trainer runs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .errors import ControlError
from .launch import prepare
from .scheduler import Scheduler
from .transport import ResponsesTransport, Turn, Usage

DEFAULT_ORACLES = ("malinois", "grammar_prior")


def utility_request(oracle: str) -> dict[str, Any]:
    return {
        "role": "per_oracle_utility", "name": "soft_clip", "oracle": oracle,
        "description": ("Per-row utility of the oracle's raw score `raw` (1-D, one finite number per row): "
                        "tanh(raw / scale) * scale with parameter scale=3.0, so an extreme raw value cannot "
                        "dominate the reward."),
        "biological_motivation": "Bound the pull of a few extreme predicted values on the policy.",
        "reference_candidate": "initial", "reference_bindings": []}


AGGREGATOR_REQUEST = {
    "role": "cross_oracle_aggregation", "name": "worst_utility", "oracle": None,
    "description": ("Combine the per-oracle arrays into one reward per sequence as the element-wise minimum "
                    "across oracles, so a candidate is only as good as its least favourable oracle."),
    "biological_motivation": "Require agreement between the oracles before rewarding.",
    "reference_candidate": "initial", "reference_bindings": []}

PENALTY_REQUEST = {
    "role": "per_sequence_penalty", "name": "run_penalty", "oracle": None,
    "description": ("Per-sequence penalty, non-negative and finite: the fraction of positions that lie inside a "
                    "run of 4 or more identical consecutive characters (0 when there is none)."),
    "biological_motivation": "Discourage low-complexity repeats.",
    "reference_candidate": "initial", "reference_bindings": []}

GATE_REQUEST = {
    "role": "hard_biological_gate", "name": "run_length_gate", "oracle": None,
    "description": ("Hard gate returning one boolean per sequence: True when the longest run of identical "
                    "consecutive characters is at most 8."),
    "biological_motivation": "Exclude degenerate homopolymer sequences outright.",
    "reference_candidate": "initial", "reference_bindings": []}


def _empty(**fields: Any) -> dict[str, Any]:
    return {"rationale": "drill", "candidates": [], "stop": None, "component": None, **fields}


def _candidate(steps: int, **fields: Any) -> dict[str, Any]:
    return {"parent": "initial", "start": "from_initial", "hypothesis": "drill", "steps": steps, "lr": None, "kl": None,
            "group_size": None, "enabled_oracles": None, "aggregator": None, "oracle_utilities": None,
            "penalties": None, "gates": None, **fields}


def script(oracles: tuple[str, ...] = DEFAULT_ORACLES, steps: int = 100,
           roles: tuple[str, ...] = ("per_oracle_utility", "cross_oracle_aggregation")) -> list[Any]:
    """Answers in the order the engine asks: one component request per role, one proposal, a critic review, a stop.

    `oracles` are enabled oracles of the task; `roles` are the roles its contract lets the Coder write. The first
    candidate uses the utility (when offered) or the penalty; the second uses the aggregator and the gate over
    every oracle. `steps` is the task's minimum.
    """
    requests = {"per_oracle_utility": utility_request(oracles[0]), "cross_oracle_aggregation": AGGREGATOR_REQUEST,
                "per_sequence_penalty": PENALTY_REQUEST, "hard_biological_gate": GATE_REQUEST}
    utilities = [{"oracle": oracle, "name": "soft_clip", "params_json": json.dumps({"scale": 3.0})}
                 for oracle in oracles]
    first: dict[str, Any] = {}
    if "per_oracle_utility" in roles:
        first["oracle_utilities"] = utilities[:1]
    if "per_sequence_penalty" in roles:
        first["penalties"] = [{"name": "run_penalty", "params_json": "{}"}]
    second: dict[str, Any] = {"enabled_oracles": list(oracles)}
    if "cross_oracle_aggregation" in roles:
        second["aggregator"] = {"name": "worst_utility", "params_json": "{}"}
    if "per_oracle_utility" in roles:
        second["oracle_utilities"] = utilities
    if "hard_biological_gate" in roles:
        second["gates"] = [{"name": "run_length_gate", "params_json": "{}"}]
    return [
        *[_empty(kind="develop_component", component=requests[role]) for role in roles if role in requests],
        _empty(kind="propose", candidates=[_candidate(steps, hypothesis="first drill candidate", **first),
                                           _candidate(steps, hypothesis="second drill candidate", **second)]),
        {"concerns": [], "requested_evidence": [], "rationale": "drill"},
        _empty(kind="stop", stop={"outcome": "deliver", "summary": "drill", "assignment": []}),
    ]


@dataclass
class ScriptedStrategist:
    """Stands in for the Strategist and Critic transport; the stop assignment is filled from the candidates.

    A request is answered by the first unused answer of a kind its schema allows, so the answer follows from what the
    engine asks and not from how many requests came before: a controller restarted on a half-finished cell is answered
    correctly by a fresh script. The Critic's schema has no `kind` and its answer has none. When every answer of an
    allowed kind has been given, the last one given is given again, as a Strategist that insists on its answer: a
    refused stop is refused again until the engine's own protocol ends the cell. `answers` holds those never given.
    """

    answers: list[Any]
    conditions: list[str]
    model: str = "scripted"
    requests: list[dict[str, Any]] = field(default_factory=list)
    given: list[Any] = field(default_factory=list)

    def _answer(self, schema: Mapping[str, Any]) -> Any:
        kinds = schema["properties"].get("kind", {}).get("enum", [None])
        unused = next((i for i, answer in enumerate(self.answers) if answer.get("kind") in kinds), None)
        if unused is not None:
            self.given.append(self.answers.pop(unused))
        for answer in reversed(self.given):
            if answer.get("kind") in kinds:
                return answer
        raise ControlError(f"the script holds no answer of kind {kinds}")

    def respond(self, input_items, *, instructions=None, schema=None, schema_name="response", tools=()):
        self.requests.append({"schema_name": schema_name})
        step = self._answer(schema)
        if isinstance(step, dict) and step.get("kind") == "stop":
            step = {**step, "stop": {**step["stop"], "assignment": [
                {"condition": condition, "candidate": "cand_000001"} for condition in self.conditions]}}
        return Turn(text=json.dumps(step), parsed=step, tool_calls=(), items=(), usage=Usage(0, 0, 0, 0),
                    model=self.model, latency_s=0.0)


def run_coder_drill(config: Mapping[str, Any], *, run_dir: Path, repo_root: Path, scheduler: Scheduler,
                    coder_transport: ResponsesTransport, device: str = "cuda",
                    poll_seconds: float = 30.0,
                    build_runner: Callable[..., Any] | None = None,
                    oracles: tuple[str, ...] = DEFAULT_ORACLES) -> dict[str, Any]:
    """Run the scripted cell to its end and report what the Coder produced."""
    from .coder import build_runner as default_runner

    runner_factory = build_runner or default_runner
    conditions = list(config["control"]["selection"]["plan"]["count"])
    control = config["control"]
    # One propose round is scripted, so the stop must be allowed after it.
    config = {**config, "control": {**control, "proposals": {**control["proposals"], "rounds_min": 1}}}
    strategist = ScriptedStrategist(script(oracles, control["budget"]["steps"]["min"],
                                           tuple(control["synthesis"]["roles"])), conditions)
    prepared = prepare(
        config, run_dir=run_dir, repo_root=repo_root, scheduler=scheduler, transport=strategist, device=device,
        components=lambda parts: runner_factory(parts["settings"].control, backend=parts["backend"],
                                                synthesis=parts["synthesis"], transport=coder_transport))
    try:
        state = prepared.engine.run(poll_seconds=poll_seconds)
        components = {request_id: {"name": item.name, "role": item.role, "status": item.status,
                                   "attempts": [(a.attempt, a.outcome, a.failure_stage) for a in item.attempts],
                                   "reason": item.reason_code}
                      for request_id, item in state.components.items()}
        return {"status": state.finished_status, "components": components,
                "candidates": {c.id: c.status for c in state.ordered_candidates()},
                "scripted_answers_left": len(strategist.answers)}
    finally:
        prepared.close()
