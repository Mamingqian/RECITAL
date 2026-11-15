"""The Coder with K drafts, framework-written property checks, a bounded repair, and a choice by agreement.

`DraftingRunner` replaces `SynthesisRunner` for a request. It asks the Coder for `drafts` independent drafts of the
primitive. A draft that the sandbox admits and the composition screen accepts is run against the property battery of
`coder_checks` (the model's own unit test is not the only test). A draft that fails gets the concrete failing case back and
at most `repairs` rewrites. Among the drafts that pass, `coder_checks.choose` takes the one whose outputs agree most with
the others', which needs no model call and no GPU. Only the chosen draft is published; every other staged draft is aborted.

The attempts of a request are written to the log after the choice, one `component_attempt` per model call in the order
they were made: `published` for the chosen draft, `failed` with the failing stage for the others, and
`failed` with stage `not_selected` for a draft that passed but was not chosen. The chosen draft's component file and the
result row carry the checks of every draft and each draft's agreement. A controller that stops before the choice repeats
the calls of that request.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from recital.capabilities import validate_recipe_capabilities
from recital.dna_session import json_safe
from recital.spec.composer import validate_spec

from . import events as ev
from .coder import (
    Authored, CoderOutput, Rejected, SynthesisRunner, parse_output,
)
from .coder_checks import (
    AGGREGATION, GATE, MAXIMIZE, PENALTY, UTILITY, Check, battery, choose, feedback, passed, probe_output, probe_sequences,
)
from .errors import ControlError
from .guard import tripwire
from .jobs import write_json
from .projection import ArmView
from .state import ComponentState, RunState

DETAIL_LIMIT = 4000


@dataclass
class Draft:
    """One model call and what became of it."""

    number: int
    authored: Authored
    stage: str | None = None          # the stage that rejected it, or None
    error: str = ""
    prepared: Any = None              # a staged extension that passed everything, not yet published
    checks: list[Check] = field(default_factory=list)
    output: np.ndarray | None = None  # its output on the probe inputs
    diagnostics: dict[str, Any] = field(default_factory=dict)
    origin: int | None = None         # the draft this one repairs
    parameters: dict[str, Any] = field(default_factory=dict)
    inspection: Any = None

    @property
    def ok(self) -> bool:
        return self.stage is None


class DraftingRunner(SynthesisRunner):
    def __init__(self, *, backend, synthesis, coder, drafts: int = 3, repairs: int = 1, inspect=None, batch_level=lambda request: False):
        """`inspect(draft, request, call)`, when given, is called for every draft the sandbox and the screen admit, while its
        staged extension is open; what it returns is kept as `draft.inspection` (an offline study of the drafts uses it)."""
        super().__init__(backend=backend, synthesis=synthesis, coder=coder)
        if drafts < 1 or repairs < 0:
            raise ValueError("drafts must be at least 1 and repairs at least 0")
        self.drafts, self.repairs, self.inspect, self.batch_level = drafts, repairs, inspect, batch_level
        self.last_drafts: list[Draft] = []

    # ----- the engine's entry point -----------------------------------------------------------------

    def run(self, engine, state: RunState, request_id: str) -> None:
        component = state.components[request_id]
        request = engine.blobs.get_json(component.request_sha)
        directory = Path(engine.s.run_dir) / "components" / request_id
        if any(attempt.outcome == "published" for attempt in component.attempts):
            import json
            self._published(engine, component, request, json.loads((directory / "component.json").read_text()))
            return
        recipe, problem = self._reference(engine, state, request)
        if not component.attempts and problem is None:
            problem = self._request_problem(engine, request, recipe)
        if problem is not None:
            self._rejected(engine, component, request, *problem)
            return
        self._require_cached_scores(recipe)
        if component.attempts:
            # The attempts of a request are recorded together with its choice; a log with attempts and no result holds
            # only failures, so the request is rejected on the last of them.
            self._rejected(engine, component, request, "attempts_exhausted", "every draft was rejected")
            return
        drafts = self.last_drafts = self._collect(engine, component, request, recipe, directory)
        passing = [draft for draft in drafts if draft.ok]
        if not passing:
            self._record(engine, component, drafts, chosen=None, means={})
            last = drafts[-1]
            self._rejected(engine, component, request, "attempts_exhausted", f"{last.stage}: {last.error}")
            return
        outputs = {draft.number: draft.output for draft in passing}
        number, means = choose(request["role"], outputs)
        chosen = next(draft for draft in drafts if draft.number == number)
        try:
            chosen.prepared.stage.publish()
        except (ValueError, KeyError, TypeError) as exc:
            chosen.stage, chosen.error = "publish", f"{type(exc).__name__}: {exc}"
            self._record(engine, component, drafts, chosen=None, means=means)
            self._rejected(engine, component, request, "attempts_exhausted", f"publish: {exc}")
            return
        for draft in drafts:
            if draft is not chosen and draft.prepared is not None:
                draft.prepared.abort()
        declaration = chosen.prepared.stage.declaration()
        payload = {"request": dict(request), "role": request["role"], "name": request["name"],
                   "reference_candidate": request.get("reference_candidate"), "attempt": chosen.number,
                   "parameters": json_safe(chosen.parameters), "parameter_schema": declaration["parameters"],
                   "declaration": declaration, "composition_screen": json_safe(chosen.diagnostics),
                   "selection": self._selection(drafts, chosen, means)}
        write_json(directory / "component.json", payload)
        self._record(engine, component, drafts, chosen=chosen, means=means)
        self._published(engine, component, request, payload)

    # ----- drafts -----------------------------------------------------------------------------------

    def _collect(self, engine, component: ComponentState, request, recipe, directory: Path) -> list[Draft]:
        drafts: list[Draft] = []
        for _ in range(self.drafts):
            draft = self._draft(engine, request, recipe, len(drafts) + 1, [], directory)
            drafts.append(draft)
            origin = draft
            for _ in range(self.repairs):
                if origin.ok or origin.stage in ("coder_identity", "coder_output"):
                    break
                failure = {"stage": origin.stage, "error": origin.error, "implementation": origin.authored.raw}
                repaired = self._draft(engine, request, recipe, len(drafts) + 1, [failure], directory)
                repaired.origin = origin.number
                drafts.append(repaired)
                origin = repaired
        return drafts

    def _draft(self, engine, request, recipe, number: int, failures: list[dict[str, Any]], directory: Path) -> Draft:
        arm = ArmView.from_task(engine.s.task)
        context = self._context(engine, request, recipe)
        folder = directory / f"attempt_{number}"
        write_json(folder / "coder_request.json", {"request": dict(request), "context": context,
                                                   "previous_failures": [{"stage": f["stage"], "error": f["error"]} for f in failures]})
        try:
            authored = self.coder.author(request, context, failures, arm=arm, limit=engine.s.control.llm.max_prompt_tokens,
                                         blobs=engine.blobs)
        except ControlError as exc:
            spent = getattr(exc, "usage", None)
            if spent is not None:
                engine.meter.add(spent)
            raise
        engine.meter.add(authored.usage)
        write_json(folder / "coder_output.json", authored.raw)
        draft = Draft(number=number, authored=authored)
        prepared = None
        try:
            output = parse_output(authored.raw, request)
            draft.parameters = dict(output.parameters)
            self._scan(engine, output)
            prepared = self._stage(request, output, recipe)
            problems = validate_spec(prepared.spec, primitive_resolver=prepared.stage.resolve)
            if problems:
                raise Rejected("spec", "; ".join(problems))
            problems = validate_recipe_capabilities(prepared.spec, prepared.capabilities, current_spec=recipe)
            if problems:
                raise Rejected("capabilities", "; ".join(problems))
            draft.diagnostics = self._screen(prepared)
            draft.checks = self._properties(engine, request, prepared, output)
            try:
                draft.output = self._output(engine, request, prepared, output)
            except Exception:  # a draft whose probe output cannot be read has none; the checks already say why
                draft.output = None
            if self.inspect is not None:
                draft.inspection = self.inspect(draft, request, self._call(request, prepared, output))
            if not passed(draft.checks):
                raise Rejected("properties", feedback(draft.checks))
            draft.prepared = prepared
        except Rejected as rejected:
            if prepared is not None:
                prepared.abort()
            draft.stage, draft.error = rejected.stage, arm.scrub(rejected.error)[:DETAIL_LIMIT]
            write_json(folder / "failure.json", {"stage": draft.stage, "error": draft.error, "implementation": authored.raw})
        except BaseException:
            if prepared is not None:
                prepared.abort()
            raise
        return draft

    @staticmethod
    def _scan(engine, output: CoderOutput) -> None:
        """The authored source must not name a source the arm cannot see."""
        arm = ArmView.from_task(engine.s.task)
        try:
            tripwire(system_texts=[output.code, output.unit_test], hidden_names=arm.hidden_names)
        except Exception as exc:  # a BlindingViolation on code the model wrote is a rejected draft, not a halted cell
            raise Rejected("leakage", f"the source names a source this search cannot see: {exc}") from exc

    # ----- properties -------------------------------------------------------------------------------

    def _call(self, request: Mapping[str, Any], prepared, output: CoderOutput):
        constructor = prepared.stage.constructor
        proxy = constructor(**output.parameters)
        return proxy

    def _inputs(self, engine, request: Mapping[str, Any]) -> dict[str, Any]:
        role = request["role"]
        if role in (GATE, PENALTY):
            return {"sequences": probe_sequences(self.backend.training_sequences),
                    "batch_level": bool(self.batch_level(request))}
        if role == AGGREGATION:
            return {"oracles": sorted(self.synthesis.catalog.allowed_oracles)[:3]}
        return {"target": MAXIMIZE}

    def _properties(self, engine, request, prepared, output: CoderOutput) -> list[Check]:
        try:
            proxy = self._call(request, prepared, output)
        except Exception as exc:
            return [Check("constructs", False, True, f"{type(exc).__name__}: {str(exc)[:300]}")]
        return battery(request["role"], proxy, **self._inputs(engine, request))

    def _output(self, engine, request, prepared, output: CoderOutput) -> np.ndarray:
        inputs = {key: value for key, value in self._inputs(engine, request).items() if key != "batch_level"}
        return probe_output(request["role"], self._call(request, prepared, output), **inputs)

    # ----- records ----------------------------------------------------------------------------------

    @staticmethod
    def _selection(drafts: list[Draft], chosen: Draft, means: Mapping[int, float]) -> dict[str, Any]:
        summary = []
        for draft in drafts:
            summary.append({
                "draft": draft.number, "repairs": draft.origin, "stage": draft.stage, "passed": draft.ok,
                "agreement": means.get(draft.number),
                "checks": {check.name: check.passed for check in draft.checks},
                "informational_failures": [check.name for check in draft.checks if not check.blocking and not check.passed]})
        return {"chosen": chosen.number, "drafts": summary, "model_calls": len(drafts),
                "property_pass_rate": sum(draft.ok for draft in drafts) / len(drafts)}

    def _record(self, engine, component: ComponentState, drafts: list[Draft], *, chosen: Draft | None,
                means: Mapping[int, float]) -> None:
        for draft in drafts:
            if draft is chosen:
                outcome, stage, detail = "published", None, None
            else:
                stage = draft.stage or "not_selected"
                detail = engine.blobs.put_json({
                    "stage": stage, "error": draft.error, "implementation": draft.authored.raw,
                    "agreement": means.get(draft.number),
                    "checks": [(check.name, check.passed, check.blocking, check.detail) for check in draft.checks]})
                outcome = "failed"
            engine.emit(ev.ComponentAttempt(request_id=component.request_id, attempt=draft.number, outcome=outcome,
                                            failure_stage=stage, detail_sha=detail, **draft.authored.call))
