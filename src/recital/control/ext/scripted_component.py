"""Arm B of the Coder true test: one component request, written by the declaration and answered by the real Coder.

    control.modules.scripted_component: {round: 1, request: {role, name, oracle, description, biological_motivation,
                                                              reference_candidate, reference_bindings}}

In the declared round, before the Strategist is asked, the module writes a `scripted_component.injected` event (whose
`source` field is `scripted_component`) and a `decision_recorded` event of kind develop_component whose response is the
request. The engine validates and acts on that decision as on any recorded decision (`decision_validated`,
`component_requested`) and runs the Coder on the request exactly as for an agent's request (the same runner, sandbox, screen
and attempts). Every count of agent decisions reads `scripted_component.injected` to leave this decision out.
The Strategist's first propose decision of the round is asked after the request ends, so it sees the result in the ledger and
the component's name among the penalties it may name.

What the log says about the scripted decision: model and resolved model are the cell's configured model, usage is zero,
`finish_reason` is `scripted_by_module` and the prompt blob is the request itself. No model produced it. The configured model
is written so that the cell keeps one resolved model id, which the contrast scripts require.

The module fires once: when the round is open, no decision of the round exists, no component exists and the baseline is
ingested. It is idempotent over the log, so a restarted controller does not repeat it.
"""
from __future__ import annotations

import hashlib
from typing import Any, Literal, Mapping

from .. import events as ev
from ..errors import ControlError
from ..jsonio import canonical_json
from . import Extension, Phase, event, facts, folds

FIELDS = ("role", "name", "oracle", "description", "biological_motivation", "reference_candidate", "reference_bindings")
FINISH = "scripted_by_module"


SOURCE = "scripted_component"


@event
class Injected(ev._Event):
    """The decision `decision_id` was written by the declaration of this module and not by a model."""

    type: Literal["scripted_component.injected"] = "scripted_component.injected"
    source: Literal["scripted_component"]
    round: int
    decision_id: str
    name: str


@folds(Injected)
def _injected(state, e: Injected) -> None:
    facts(state, SOURCE)[e.decision_id] = {"source": e.source, "round": e.round, "name": e.name}


class ScriptedComponent(Extension):
    name = "scripted_component"

    def __init__(self, round_index: int, request: Mapping[str, Any]):
        self.round, self.request = round_index, dict(request)

    def phases(self):
        return (Phase("scripted_component", "critic", self._request),)

    def _request(self, engine, state) -> bool:
        rnd = state.open_round
        if (state.baseline.status != "ingested" or state.components or rnd is None or rnd.index != self.round
                or rnd.decision_ids or rnd.b_effective == 0):
            return False
        parsed = {"kind": "develop_component", "component": self.request}
        ok, _, problems, normalized = engine._validate_component(state, rnd, parsed)
        if not ok:
            raise ControlError(f"the scripted request is refused by the engine: {[p.code for p in problems]}")
        blobs, model = engine.blobs, engine.s.control.llm.model
        response_sha = blobs.put_json(parsed)
        decision_id = state.next_decision_id
        engine.emit(Injected(source=SOURCE, round=rnd.index, decision_id=decision_id, name=self.request["name"]))
        engine.emit(ev.DecisionRecorded(
            decision_id=decision_id, round=rnd.index, kind="develop_component", attempt=1,
            state_hash=hashlib.sha256(f"{self.name}:{rnd.index}".encode()).hexdigest(),
            prompt_hash=hashlib.sha256(canonical_json(self.request)).hexdigest(), prompt_manifest=[response_sha],
            schema_sha=response_sha, response_sha=response_sha, tool_transcript_sha=None,
            usage=ev.Usage(prompt=0, completion=0, cached=0, reasoning=0), model=model, resolved_model=model,
            latency_s=0.0, finish_reason=FINISH))
        return True


def build(config: Mapping[str, Any]) -> ScriptedComponent:
    unknown = sorted(set(config) - {"round", "request"})
    request = config.get("request")
    round_index = config.get("round")
    if (unknown or type(round_index) is not int or round_index < 1 or not isinstance(request, dict)
            or sorted(request) != sorted(FIELDS)):
        raise ControlError(f"control.modules.scripted_component takes `round` (a positive integer) and `request` with exactly "
                           f"the fields {list(FIELDS)}")
    return ScriptedComponent(round_index, request)
