"""The Coder step: the authoring call, and the lifecycle that publishes one requested component.

`Coder` makes one strict structured call that returns the source of one primitive. `SynthesisRunner` is the
engine's `ComponentRunner`: it validates a request deterministically (a mistake there is the Strategist's and
costs no attempt and no model call), then makes up to `attempts_per_request` attempts, each of which authors a
primitive, stages it in the sandbox, checks the reference recipe with it inserted, runs the composition screen
and publishes it. A component is published against its reference recipe and is inserted into no spec; later
candidates name it.
"""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema

from recital.capabilities import primitive_request_problems, validate_recipe_capabilities
from recital.dna_session import json_safe
from recital.spec.composer import validate_spec
from recital.synthesis.models import IsolationError, RuntimeLimits

from . import events as ev
from .contract import ControlContract
from .errors import ControlError
from .guard import blind_projection, check_prompt_budget, tripwire
from .jobs import write_json
from .jsonio import canonical_json, sha256_hex
from .projection import ArmView
from .schemas import coder_output_schema
from .state import ComponentState, RunState
from .store import BlobStore
from .transport import ResponsesTransport, Usage

ERROR_LIMIT = 300
DETAIL_LIMIT = 4000
INITIAL = "initial"

# What the sandbox calls, per role. These are the roles `RecipeSynthesis` can publish.
ROLE_SIGNATURES = {
    "per_oracle_utility": "__call__(self, raw: np.ndarray, *, target: Mapping, conditioning=None) -> np.ndarray",
    "cross_oracle_aggregation": "__call__(self, utilities: dict[str, np.ndarray]) -> np.ndarray",
    "per_sequence_penalty": "__call__(self, sequences: list[str]) -> np.ndarray",
    "hard_biological_gate": "__call__(self, sequences: list[str]) -> np.ndarray  # dtype bool",
}

ROLE_NOTES = {
    "per_oracle_utility": (
        "A per_oracle_utility receives `raw`, one finite number per row for one oracle (see the reward input "
        "below); `target`, a read-only mapping that declares that oracle's immutable raw target (kind, "
        "coordinate, units, scale, and reference, threshold or lower/upper); and `conditioning`, the condition "
        "label of each row, or None. It returns finite signed utilities, shape (n,). A conditional target has "
        "common kind/coordinate/units/scale and a by_condition mapping: select the bounds for EACH row using its "
        "label. A null lower or upper bound means a one-sided interval, not an invalid target. Handle empty "
        "batches and mixed conditions. Admission calls it with raw=[-1.0, 0.0, 1.0], the unconditional maximize "
        "target {kind: maximize, coordinate: raw, units: score, scale: 1.0, reference: 0.0} and conditioning "
        "None, then runs your unit test."),
    "cross_oracle_aggregation": (
        "A cross_oracle_aggregation receives {oracle name: utility array of shape (n,)} for the enabled "
        "oracles, utilities only and never raw predictions, and returns one finite value per row. Admission "
        "calls it with every approved oracle name, each mapped to [-1.0, 0.0, 1.0], so accept any non-empty "
        "set of names."),
    "per_sequence_penalty": (
        "A per_sequence_penalty returns finite nonnegative values, shape (n,); the reward subtracts them. "
        "It may declare the keyword conditioning=None to receive the condition label of each row. Admission "
        "calls it on three sequences drawn from the declared alphabet and length range."),
    "hard_biological_gate": (
        "A hard_biological_gate returns a boolean array, shape (n,), True where a sequence violates the gate. "
        "It may declare the keyword conditioning=None to receive the condition label of each row. Admission "
        "calls it on three sequences drawn from the declared alphabet and length range."),
}

REWARD_INPUT_NOTES = {
    "cell_gap": (
        "With reward_input `cell_gap`, each enabled oracle's per-cell output is reduced to one number per "
        "sequence before any primitive runs: GAP = (the value in the column of the sequence's target condition) "
        "minus (the largest of the other columns), in the oracle's raw head coordinates. Larger is better and 0 "
        "means the target cell does not beat the strongest off-target cell. A per_oracle_utility therefore "
        "receives that 1-D GAP array as `raw`, one finite value per sequence; its `target` is the "
        "framework-owned objective {kind: maximize, reference: 0.0, scale: 1.0, coordinate: raw, units: raw "
        "target-minus-max-offtarget gap}; its `conditioning` lists the condition label of each row. It returns "
        "a 1-D array of utilities, one per sequence. The utilities of the enabled oracles are averaged unless "
        "the recipe names a cross_oracle_aggregation, which receives {oracle name: utility array}. Penalties "
        "and gates see sequences only."),
}
NO_REWARD_INPUT_NOTE = (
    "Without a reward_input, per-oracle target utilities precede aggregation when target_objectives is "
    "present, acting on the oracle's scalar output; otherwise the declared transforms process raw "
    "predictions before aggregation.")


def system_prompt(roles: Sequence[str], limits: RuntimeLimits) -> str:
    """The Coder's instructions for the roles the contract allows, under the sandbox's limits."""
    unknown = [role for role in roles if role not in ROLE_SIGNATURES]
    if unknown:
        raise ControlError(f"no component of role {unknown} can be synthesized; roles: {sorted(ROLE_SIGNATURES)}")
    signatures = "\n".join(f"  - {role}: {ROLE_SIGNATURES[role]}" for role in roles)
    notes = "\n\n".join(ROLE_NOTES[role] for role in roles)
    reward_roles = {"per_oracle_utility", "cross_oracle_aggregation"} & set(roles)
    reward_input = ("\n\nReward input. runtime_context.reward_input_semantics states which of these applies to "
                    "your request. " + " ".join([*REWARD_INPUT_NOTES.values(), NO_REWARD_INPUT_NOTE])
                    ) if reward_roles else ""
    return (
        "You are the Coder of a sequence-design search. You write the Python implementation of one reward "
        "primitive that the Strategist requested, and a unit test for it.\n\n"
        "Each primitive is a class decorated with @register_primitive(role, name). The class must follow the "
        "signature of its role:\n\n" + signatures + "\n\n" + notes + reward_input + "\n\n"
        "Use the supplied runtime_context as the authoritative input format; do not guess units, directions "
        "or calibration. Code executes in an isolated Python/NumPy worker, not the training process: it has no "
        "network, cannot create processes and cannot write files. Use only numpy, dataclasses, math and "
        "recital.spec.registry (which supplies register_primitive). A call is cut off after "
        f"{limits.cpu_seconds} CPU seconds ({limits.wall_seconds:g} s wall) and "
        f"{limits.memory_bytes / 2 ** 30:g} GiB of memory, and admission after {limits.admission_seconds:g} "
        "seconds, so vectorize with numpy. Every call builds a fresh instance in a fresh process: keep no "
        "state between calls and derive everything from the batch you are given.\n\n"
        "Fixed authorized reference objects are available in the read-only `references` mapping, keyed by "
        "approved id. `references` is an injected worker variable: use it directly (for example "
        "`references[\"training_reference\"]`) and do NOT import it. A reference grant describes its actual "
        "schema: an array of DNA strings holds sequences, not oracle predictions. Do not invent a reference, "
        "read data from a path, or pretend that missing scores exist.\n\n"
        "Give constructor parameters defaults so admission can construct the class. Put the defaults in "
        "parameters_json and a closed, self-contained JSON Schema object for them in parameter_schema_json "
        "(additionalProperties false); keep the defaults valid under it. Do not put `name` in either: the "
        "framework supplies it. Return the exact requested name and role; a request does not authorize "
        "another role or a new oracle input.\n\n"
        "Return:\n"
        "  - primitive_name: the requested name,\n"
        "  - role: the requested role,\n"
        "  - code: a complete Python module string defining the class with @register_primitive,\n"
        "  - unit_test: a complete pytest-style module string defining at least one zero-argument test_* "
        "function that constructs the primitive and asserts expected behaviour,\n"
        "  - parameters_json: JSON object text of the constructor defaults ('{}' for none),\n"
        "  - parameter_schema_json: JSON object text of the parameter schema ('{}' for none).\n\n"
        "The unit test runs in a namespace where your class (exactly as named in `code`), `np`, `references` "
        "and `get_primitive(role, name)` are already available: refer to the class by the name you defined, "
        "and do not re-import or re-define it. Return only the JSON object allowed by the supplied schema.")


@dataclass(frozen=True)
class CoderOutput:
    primitive_name: str
    role: str
    code: str
    unit_test: str
    parameters: dict[str, Any]
    parameter_schema: dict[str, Any] | None


class Rejected(Exception):
    """A defect of one attempt: the stage that found it and the text the author is shown."""

    def __init__(self, stage: str, error: str):
        super().__init__(error)
        self.stage, self.error = stage, error


def _object_text(text: str, field: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise Rejected("coder_output", f"{field} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise Rejected("coder_output", f"{field} must be a JSON object")
    return value


def parse_output(raw: Mapping[str, Any], request: Mapping[str, Any]) -> CoderOutput:
    """The model's answer as a primitive, refusing a role or name other than the request's."""
    if raw["role"] != request["role"] or raw["primitive_name"] != request["name"]:
        raise Rejected("coder_identity",
                       f"you returned {raw['role']}/{raw['primitive_name']}, the request is "
                       f"{request['role']}/{request['name']}; keep the requested role and name")
    # `name` is catalog-owned identity; a model may echo it among the constructor defaults.
    parameters = _object_text(raw["parameters_json"], "parameters_json")
    parameters.pop("name", None)
    schema = _object_text(raw["parameter_schema_json"], "parameter_schema_json")
    return CoderOutput(primitive_name=raw["primitive_name"], role=raw["role"], code=raw["code"],
                       unit_test=raw["unit_test"], parameters=parameters, parameter_schema=schema or None)


def _implementation_text(implementation: Mapping[str, Any] | None) -> str:
    if not implementation:
        return "(none)"
    return (f"{implementation.get('code')}\n  unit test you submitted:\n{implementation.get('unit_test')}\n"
            f"  parameters_json: {implementation.get('parameters_json')}\n"
            f"  parameter_schema_json: {implementation.get('parameter_schema_json')}")


def _compose(request: Mapping[str, Any], context: Mapping[str, Any],
             previous_failures: Sequence[Mapping[str, Any]], *, written: bool) -> str:
    """The user message; with `written` false, the strings an agent or the sandbox wrote are left blank."""
    def mine(value: Any) -> str:
        return str(value) if written else ""

    text = (
        "Primitive request:\n"
        f"  name: {request['name']}\n"
        f"  role: {request['role']}\n"
        f"  description: {mine(request.get('description'))}\n"
        f"  biological_motivation: {mine(request.get('biological_motivation'))}\n\n"
        f"  oracle: {request.get('oracle')}\n"
        f"  approved_reference_ids: {list(request.get('reference_bindings') or [])}\n"
        f"  runtime_context: {json.dumps(json_safe(context), sort_keys=True)}\n"
        "Return the implementation, unit test, parameter defaults and parameter schema.")
    for ordinal, failure in enumerate(previous_failures, start=1):
        text += (f"\n\nAttempt {ordinal} of this same request was REJECTED.\n"
                 f"  rejection: {mine(failure['error'])}\n"
                 f"  implementation you submitted:\n{mine(_implementation_text(failure.get('implementation')))}\n"
                 "Fix the cause and resubmit. Keep the requested role and name.")
    return text


def render_request(request: Mapping[str, Any], context: Mapping[str, Any],
                   previous_failures: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """The user message of one authoring call, and the same message without the words others wrote.

    Each earlier rejection of this request is shown with the implementation that was rejected, so the author
    learns what it got wrong. The second text is what the blind-word scan reads: an agent or the sandbox can
    only write a word it was shown, and a generic phrase of theirs must not halt a cell.
    """
    return (_compose(request, context, previous_failures, written=True),
            _compose(request, context, previous_failures, written=False))


@dataclass(frozen=True)
class Authored:
    """One authoring call: the model's answer, the `LlmCall` fields it leaves in the log, and what it cost."""

    raw: dict[str, Any]
    call: dict[str, Any]
    usage: Usage


class Coder:
    """One strict structured request per primitive, over the Responses transport."""

    def __init__(self, transport: ResponsesTransport, roles: Sequence[str], limits: RuntimeLimits):
        self.transport = transport
        self.roles = tuple(roles)
        self.system = system_prompt(self.roles, limits)
        self.schema = coder_output_schema(self.roles)

    def author(self, request: Mapping[str, Any], context: Mapping[str, Any],
               previous_failures: Sequence[Mapping[str, Any]], *, arm: ArmView, limit: int,
               blobs: BlobStore) -> Authored:
        """Render, check the prompt for blinding and size, and make the call."""
        user, system_part = render_request(request, context, previous_failures)
        tripwire(system_texts=[self.system, system_part], hidden_names=arm.hidden_names)
        check_prompt_budget([self.system, user], limit=limit, schema=self.schema)
        items = [{"role": "user", "content": user}]
        turn = self.transport.respond(items, instructions=self.system, schema=self.schema,
                                      schema_name="component")
        if turn.parsed is None:
            raise ControlError("the authoring turn ended without a structured answer")
        usage = turn.usage
        call = dict(
            prompt_hash=sha256_hex(canonical_json({"instructions": self.system, "input": items})),
            prompt_manifest=[blobs.put(self.system.encode("utf-8")), blobs.put(user.encode("utf-8"))],
            schema_sha=blobs.put_json(self.schema), response_sha=blobs.put_json(turn.parsed),
            tool_transcript_sha=None,
            usage=ev.Usage(prompt=usage.input, completion=usage.output, cached=usage.cached,
                           reasoning=usage.reasoning),
            model=self.transport.model, resolved_model=turn.model, latency_s=turn.latency_s)
        return Authored(raw=dict(turn.parsed), call=call, usage=usage)


def reward_input_note(recipe: Mapping[str, Any]) -> str:
    reference = recipe.get("reward_input")
    if not reference:
        return NO_REWARD_INPUT_NOTE
    if reference["name"] not in REWARD_INPUT_NOTES:
        raise ControlError(f"no Coder description of the reward input {reference['name']!r}")
    return REWARD_INPUT_NOTES[reference["name"]]


def signature_of(name: str, parameters: Mapping[str, Any]) -> str:
    return f"{name}({', '.join(f'{key}={json.dumps(value, sort_keys=True)}' for key, value in parameters.items())})"


class SynthesisRunner:
    """The engine's `ComponentRunner` for a run whose contract enables synthesis."""

    def __init__(self, *, backend, synthesis, coder: Coder):
        self.backend, self.synthesis, self.coder = backend, synthesis, coder

    # ----- the engine's entry point ---------------------------------------------------------------

    def run(self, engine, state: RunState, request_id: str) -> None:
        """Emit the request's attempt events and its result; a repeated call continues from the log."""
        component = state.components[request_id]
        request = engine.blobs.get_json(component.request_sha)
        directory = Path(engine.s.run_dir) / "components" / request_id
        if any(attempt.outcome == "published" for attempt in component.attempts):
            # The attempt that published was recorded and its result was not.
            payload = json.loads((directory / "component.json").read_text())
            self._published(engine, component, request, payload)
            return
        recipe, problem = self._reference(engine, state, request)
        if not component.attempts and problem is None:
            problem = self._request_problem(engine, request, recipe)
        if problem is not None:
            self._rejected(engine, component, request, *problem)
            return
        self._require_cached_scores(recipe)
        allowed = engine.s.control.synthesis.attempts_per_request
        failures = self._recorded_failures(engine, request_id)
        for number in range(len(component.attempts) + 1, allowed + 1):
            failure = self._attempt(engine, component, request, recipe, number, failures, directory)
            if failure is None:
                return
            failures.append(failure)
        self._rejected(engine, component, request, "attempts_exhausted", failures[-1]["error"])

    # ----- request validation ---------------------------------------------------------------------

    def _reference(self, engine, state: RunState, request: Mapping[str, Any]):
        """The recipe the component is tested in, or a typed refusal."""
        reference = request.get("reference_candidate")
        if reference == INITIAL:
            return deepcopy(dict(engine.s.recipe_ctx.initial_recipe)), None
        candidate = state.candidates.get(reference)
        if candidate is None:
            return None, ("unknown_reference_candidate", f"{reference!r} is not a candidate of this run")
        if candidate.status != "ok":
            return None, ("reference_candidate_not_completed",
                          f"{reference} is {candidate.status}; name a completed candidate or 'initial'")
        return deepcopy(engine.blobs.get_json(candidate.recipe_sha)["recipe"]), None

    def _request_problem(self, engine, request: Mapping[str, Any],
                         recipe: Mapping[str, Any]) -> tuple[str, str] | None:
        roles = engine.s.control.synthesis.roles
        if request["role"] not in roles:
            return "role_not_allowed", f"{request['role']} is not among the contract's synthesis roles {list(roles)}"
        problems = primitive_request_problems(dict(request), self.synthesis.capabilities, dict(recipe))
        if problems:
            return "request_invalid", "; ".join(problems)
        return None

    # ----- one attempt ----------------------------------------------------------------------------

    def _context(self, engine, request: Mapping[str, Any], recipe: Mapping[str, Any]) -> dict[str, Any]:
        """What the Coder is told: only what the arm may see, and the real input format of the reward."""
        task = engine.s.task
        visible = set(task.visible_sources)
        bound = self.synthesis.coder_request(dict(request), dict(recipe))["runtime_context"]
        context = {
            "task": task.public_description()["task"],
            "conditions": list(task.conditions),
            "representation": deepcopy(self.backend.representation),
            "recipe": {key: value for key, value in recipe.items() if key != "held_out_oracles"},
            "reward_input_semantics": reward_input_note(recipe),
            "runtime_parameters": deepcopy(self.synthesis.capabilities.runtime_parameters),
            "measurement_definitions": {
                name: deepcopy(definition) for name, definition in engine.s.definitions.items()
                if set(definition.get("sources", ())) <= visible},
            **bound,
        }
        # Source notes about sentinel measurements this arm does not make are left out of the context, as they are from
        # the Strategist header.
        return blind_projection(context) if ArmView.from_task(task).blind else context

    def _attempt(self, engine, component: ComponentState, request: Mapping[str, Any], recipe: dict[str, Any],
                 number: int, failures: list[dict[str, Any]], directory: Path) -> dict[str, Any] | None:
        """One authoring attempt. Returns None when it published, else the failure to show the next one."""
        folder = directory / f"attempt_{number}"
        arm = ArmView.from_task(engine.s.task)
        context = self._context(engine, request, recipe)
        write_json(folder / "coder_request.json", {
            "request": dict(request), "context": context,
            "previous_failures": [{"stage": item["stage"], "error": item["error"]} for item in failures]})
        try:
            authored = self.coder.author(
                request, context, failures, arm=arm, limit=engine.s.control.llm.max_prompt_tokens,
                blobs=engine.blobs)
        except ControlError as exc:
            spent = getattr(exc, "usage", None)
            if spent is not None:
                engine.meter.add(spent)  # tokens a failed call still cost count against the run's cap
            raise
        write_json(folder / "coder_output.json", authored.raw)
        try:
            payload = self._publish(engine, request, recipe, authored.raw, number)
        except Rejected as rejected:
            error = arm.scrub(rejected.error)
            detail = {"stage": rejected.stage, "error": error[:DETAIL_LIMIT], "implementation": authored.raw}
            write_json(folder / "failure.json", detail)
            engine.emit(ev.ComponentAttempt(
                request_id=component.request_id, attempt=number, outcome="failed", failure_stage=rejected.stage,
                detail_sha=engine.blobs.put_json(detail), **authored.call))
            engine.meter.add(authored.usage)
            return detail
        write_json(directory / "component.json", payload)
        engine.emit(ev.ComponentAttempt(
            request_id=component.request_id, attempt=number, outcome="published", failure_stage=None,
            detail_sha=None, **authored.call))
        engine.meter.add(authored.usage)
        self._published(engine, component, request, payload)
        return None

    def _publish(self, engine, request: Mapping[str, Any], recipe: dict[str, Any], raw: Mapping[str, Any],
                 number: int) -> dict[str, Any]:
        """Stage the authored primitive and publish it if the reference recipe with it inserted passes."""
        output = parse_output(raw, request)
        prepared = self._stage(request, output, recipe)
        try:
            problems = validate_spec(prepared.spec, primitive_resolver=prepared.stage.resolve)
            if problems:
                raise Rejected("spec", "; ".join(problems))
            problems = validate_recipe_capabilities(prepared.spec, prepared.capabilities, current_spec=recipe)
            if problems:
                raise Rejected("capabilities", "; ".join(problems))
            diagnostics = self._screen(prepared)
            try:
                prepared.stage.publish()
            except (ValueError, KeyError, TypeError) as exc:
                raise Rejected("publish", f"{type(exc).__name__}: {exc}") from exc
        except BaseException:
            prepared.abort()
            raise
        declaration = prepared.stage.declaration()
        return {"request": dict(request), "role": request["role"], "name": request["name"],
                "reference_candidate": request.get("reference_candidate"), "attempt": number,
                "parameters": json_safe(output.parameters), "parameter_schema": declaration["parameters"],
                "declaration": declaration, "composition_screen": json_safe(diagnostics)}

    def _stage(self, request: Mapping[str, Any], output: CoderOutput, recipe: dict[str, Any]):
        try:
            return self.synthesis.stage(dict(request), output, recipe)
        except IsolationError:
            raise
        except (ValueError, KeyError, TypeError, jsonschema.exceptions.SchemaError) as exc:
            raise Rejected("stage", f"{type(exc).__name__}: {exc}") from exc

    def _require_cached_scores(self, recipe: Mapping[str, Any]) -> None:
        """The screen reads the baseline job's cached composition-reference scores; this process scores nothing."""
        scores = Path(self.backend.output) / "composition_reference" / "scores"
        missing = [name for name in recipe["enabled_oracles"] if not (scores / f"{name}.json").is_file()]
        if missing:
            raise ControlError(f"the baseline job cached no composition-reference scores for {missing}")

    def _screen(self, prepared) -> dict[str, Any]:
        """Execute the recipe with the component inserted on the cached composition reference."""
        screen = self.backend._proposal_screen(prepared.spec)
        screen.primitive_resolver = prepared.stage.resolve
        problems = screen(prepared.spec)
        if problems or screen.last_diagnostics.get("status") != "passed":
            raise Rejected("screen", "composed reward failed execution: " + "; ".join(problems))
        return screen.last_diagnostics

    # ----- results --------------------------------------------------------------------------------

    @staticmethod
    def _recorded_failures(engine, request_id: str) -> list[dict[str, Any]]:
        return [engine.blobs.get_json(event.data["detail_sha"]) for event in engine.log.events()
                if event.type == "component_attempt" and event.data["request_id"] == request_id
                and event.data["outcome"] == "failed"]

    @staticmethod
    def _row(component: ComponentState, request: Mapping[str, Any], **fields: Any) -> dict[str, Any]:
        return {"request_id": component.request_id, "round": component.round, "role": component.role,
                "name": component.name, "description": request.get("description", ""),
                "reference_candidate": request.get("reference_candidate"), **fields}

    def _rejected(self, engine, component: ComponentState, request: Mapping[str, Any], code: str,
                  error: str) -> None:
        text = ArmView.from_task(engine.s.task).scrub(error)[:ERROR_LIMIT]
        row = self._row(component, request, status="rejected", reason_code=code, signature=None, error=text)
        engine.emit(ev.ComponentResult(request_id=component.request_id, status="rejected", reason_code=code,
                                       row_sha=engine.blobs.put_json(row), component_sha=None))

    def _published(self, engine, component: ComponentState, request: Mapping[str, Any],
                   payload: Mapping[str, Any]) -> None:
        row = self._row(component, request, status="published", reason_code=None, error=None,
                        signature=signature_of(payload["name"], payload["parameters"]),
                        parameter_schema=payload["parameter_schema"])
        engine.emit(ev.ComponentResult(
            request_id=component.request_id, status="published", reason_code=None,
            row_sha=engine.blobs.put_json(row), component_sha=engine.blobs.put_json(payload)))


def build_runner(control: ControlContract, *, backend, synthesis,
                 transport: ResponsesTransport) -> SynthesisRunner | None:
    """The run's component runner, or None when its contract disables synthesis; no Coder is built then."""
    if not control.synthesis.enabled:
        return None
    coder = Coder(transport, control.synthesis.roles, synthesis.catalog.limits)
    return SynthesisRunner(backend=backend, synthesis=synthesis, coder=coder)
