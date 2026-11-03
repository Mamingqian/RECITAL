"""RunState: everything the engine knows about a run, folded from its events.

`fold` and `apply` are pure. They read no file, clock or random source, and a
state is a function of the event sequence alone. An event repeating the key of an
earlier one (same candidate and attempt, same round, same decision id, ...) is
ignored, first one wins. An event that refers to something that does not exist, or
that contradicts an earlier event, raises LedgerCorruption.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field

from recital.control import events as ev
from recital.control.errors import ControlError, LedgerCorruption
from recital.control.events import BASELINE
from recital.control.jsonio import canonical_json, sha256_hex

CandidateStatus = Literal[
    "allocated", "submitted", "running", "ok", "failed", "timeout", "cancelled", "forfeited"
]
COMPLETED = frozenset({"ok", "failed", "timeout", "cancelled"})
IN_FLIGHT = frozenset({"submitted", "running"})


def ordinal_of(round: int, slot: int, B: int) -> int:
    """Candidate ordinal of `slot` (1-based) in `round` (1-based) with B slots per round."""
    if B < 1 or round < 1 or not 1 <= slot <= B:
        raise ControlError(f"no ordinal for round {round}, slot {slot}, B {B}")
    return (round - 1) * B + slot


def candidate_id(ordinal: int) -> str:
    if type(ordinal) is not int or ordinal < 1:
        raise ControlError(f"candidate ordinals start at 1, got {ordinal!r}")
    return f"cand_{ordinal:06d}"


class _State(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Job(_State):
    job_attempt: int = 0
    job_token: str | None = None
    slurm_id: str | None = None
    infra_failures: list[str] = Field(default_factory=list)


class BaselineState(_Job):
    status: Literal["pending", "submitted", "running", "ingested"] = "pending"
    pool_sha: str | None = None
    policy_sample_sha: str | None = None
    n_rows: int = 0


class CandidateState(_Job):
    id: str
    ordinal: int
    round: int
    slot: int
    parent: str | None = None
    start: str | None = None
    recipe_sha: str | None = None
    fingerprint: str | None = None
    steps: int = 0
    repeat_of: str | None = None
    anchor: bool = False
    status: CandidateStatus = "allocated"
    completed: ev.CandidateSummary | None = None
    completed_attempt: int | None = None
    selfval_sha: str | None = None
    ledger_row_sha: str | None = None


class RoundState(_State):
    index: int
    b_effective: int
    status: Literal["open", "allocated", "skipped", "stopped"] = "open"
    attempts: int = 0
    decision_ids: list[str] = Field(default_factory=list)
    candidates: list[str] = Field(default_factory=list)
    skip_reason: str | None = None
    critic_row_sha: str | None = None
    critic_unavailable: bool = False


class DecisionState(_State):
    id: str
    round: int
    kind: str
    attempt: int
    state_hash: str
    response_sha: str
    status: Literal["recorded", "valid", "invalid"] = "recorded"
    n_valid: int = 0
    problems: list[ev.Problem] = Field(default_factory=list)
    normalized_sha: str | None = None
    consumed: bool = False


class ComponentAttemptRecord(_State):
    attempt: int
    outcome: str
    failure_stage: str | None = None


class ComponentState(_State):
    request_id: str
    round: int
    decision_id: str
    role: str
    name: str
    reference_candidate: str | None = None
    request_sha: str
    status: Literal["requested", "published", "rejected"] = "requested"
    attempts: list[ComponentAttemptRecord] = Field(default_factory=list)
    reason_code: str | None = None
    row_sha: str | None = None
    component_sha: str | None = None


class StopState(_State):
    pending: ev.StopRequested | None = None
    accepted: ev.StopAccepted | None = None
    rejections: list[ev.StopRejected] = Field(default_factory=list)
    consecutive_rejections: int = 0


def _no_usage() -> ev.Usage:
    return ev.Usage(prompt=0, completion=0, cached=0, reasoning=0)


class RunState(_State):
    """Treat as immutable: `apply` and `fold` build new instances."""

    n_events: int = 0
    run: ev.RunStarted | None = None
    baseline: BaselineState = Field(default_factory=BaselineState)
    rounds: list[RoundState] = Field(default_factory=list)
    candidates: dict[str, CandidateState] = Field(default_factory=dict)
    decisions: dict[str, DecisionState] = Field(default_factory=dict)
    components: dict[str, ComponentState] = Field(default_factory=dict)
    stop: StopState = Field(default_factory=StopState)
    delivery: ev.DeliveryAssembled | None = None
    library: ev.LibraryFrozen | None = None
    usage: ev.Usage = Field(default_factory=_no_usage)
    resolved_models: list[str] = Field(default_factory=list)
    consecutive_skips: int = 0
    resumes: int = 0
    terminal: bool = False
    finished_status: str | None = None

    @property
    def header(self) -> ev.RunStarted:
        if self.run is None:
            raise ControlError("the run has not started")
        return self.run

    @property
    def B(self) -> int:
        return self.header.candidates_per_round

    def ordered_candidates(self) -> list[CandidateState]:
        return sorted(self.candidates.values(), key=lambda c: c.ordinal)

    @property
    def unsubmitted(self) -> list[str]:
        """Allocated candidates with no live job: submit, or resubmit after a failed attempt."""
        return [c.id for c in self.ordered_candidates() if c.status == "allocated"]

    @property
    def inflight(self) -> list[str]:
        """Candidates whose job was submitted and whose completion is not yet ingested."""
        return [c.id for c in self.ordered_candidates() if c.status in IN_FLIGHT]

    @property
    def uningested(self) -> list[str]:
        """Candidates still owed a `candidate_completed`: unsubmitted plus in flight."""
        return [c.id for c in self.ordered_candidates()
                if c.status == "allocated" or c.status in IN_FLIGHT]

    @property
    def unscored(self) -> list[str]:
        """Candidates that finished ok and have no `selfval_computed` yet."""
        return [c.id for c in self.ordered_candidates()
                if c.status == "ok" and c.selfval_sha is None]

    @property
    def unfrozen(self) -> list[str]:
        """Completed candidates whose ledger row is not yet frozen."""
        return [c.id for c in self.ordered_candidates()
                if c.status in COMPLETED and c.ledger_row_sha is None]

    @property
    def committed_updates(self) -> int:
        """Steps of every allocated, non-forfeited candidate, in-flight siblings included."""
        return sum(c.steps for c in self.candidates.values() if c.status != "forfeited")

    @property
    def remaining_updates(self) -> int:
        return self.header.budgets.total_updates - self.committed_updates

    @property
    def current_round(self) -> int:
        return len(self.rounds)

    @property
    def next_round(self) -> int:
        return len(self.rounds) + 1

    @property
    def open_round(self) -> RoundState | None:
        return self.rounds[-1] if self.rounds and self.rounds[-1].status == "open" else None

    @property
    def rounds_done(self) -> int:
        """Rounds that allocated at least one candidate and whose candidates all completed."""
        done = 0
        for rnd in self.rounds:
            members = [self.candidates[i] for i in rnd.candidates
                       if self.candidates[i].status != "forfeited"]
            if rnd.status == "allocated" and members and all(c.status in COMPLETED for c in members):
                done += 1
        return done

    @property
    def decisions_awaiting_validation(self) -> list[str]:
        return [d.id for d in self.decisions.values() if d.status == "recorded"]

    @property
    def decisions_awaiting_action(self) -> list[str]:
        """Valid decisions of the current round that nothing was allocated or requested from."""
        return [d.id for d in self.decisions.values()
                if d.status == "valid" and not d.consumed and d.round == self.current_round]

    @property
    def components_in_flight(self) -> list[str]:
        return [c.request_id for c in self.components.values() if c.status == "requested"]

    @property
    def next_decision_id(self) -> str:
        return f"dec_{len(self.decisions) + 1:06d}"

    @property
    def next_request_id(self) -> str:
        return f"req_{len(self.components) + 1:04d}"


Handler = Callable[[RunState, Any], None]
_HANDLERS: dict[str, Handler] = {}


def _handles(cls: type[ev._Event]) -> Callable[[Handler], Handler]:
    def register(handler: Handler) -> Handler:
        _HANDLERS[cls.model_fields["type"].default] = handler
        return handler

    return register


def _bad(event: ev._Event, why: str) -> LedgerCorruption:
    return LedgerCorruption(f"{event.type}: {why}")


def _round(state: RunState, index: int, event: ev._Event) -> RoundState:
    if not 1 <= index <= len(state.rounds):
        raise _bad(event, f"round {index} was never opened")
    return state.rounds[index - 1]


def _candidate(state: RunState, candidate: str, event: ev._Event) -> CandidateState:
    if candidate not in state.candidates:
        raise _bad(event, f"unknown candidate {candidate!r}")
    return state.candidates[candidate]


def _job(state: RunState, target: str, event: ev._Event) -> _Job:
    return state.baseline if target == BASELINE else _candidate(state, target, event)


def _decision(state: RunState, decision_id: str, event: ev._Event) -> DecisionState:
    if decision_id not in state.decisions:
        raise _bad(event, f"unknown decision {decision_id!r}")
    return state.decisions[decision_id]


def _spend(state: RunState, call: ev.LlmCall) -> None:
    used = call.usage
    state.usage = ev.Usage(
        prompt=state.usage.prompt + used.prompt,
        completion=state.usage.completion + used.completion,
        cached=state.usage.cached + used.cached,
        reasoning=state.usage.reasoning + used.reasoning,
    )
    if call.resolved_model not in state.resolved_models:
        state.resolved_models = sorted([*state.resolved_models, call.resolved_model])


@_handles(ev.RunStarted)
def _run_started(state: RunState, e: ev.RunStarted) -> None:
    if state.run is not None:
        raise _bad(e, "the run already started")
    state.run = e


@_handles(ev.JobSubmitIntent)
def _job_submit_intent(state: RunState, e: ev.JobSubmitIntent) -> None:
    job = _job(state, e.target, e)
    if e.attempt <= job.job_attempt:
        return
    if e.attempt != job.job_attempt + 1:
        raise _bad(e, f"{e.target} attempt {e.attempt} skips {job.job_attempt + 1}")
    waiting = state.baseline.status == "pending" if e.target == BASELINE \
        else state.candidates[e.target].status == "allocated"
    if not waiting:
        raise _bad(e, f"{e.target} has a live job or is finished")
    job.job_attempt, job.job_token, job.slurm_id = e.attempt, e.token, None


@_handles(ev.JobSubmitted)
def _job_submitted(state: RunState, e: ev.JobSubmitted) -> None:
    job = _job(state, e.target, e)
    if e.attempt > job.job_attempt:
        raise _bad(e, f"{e.target} attempt {e.attempt} has no submit intent")
    if e.attempt < job.job_attempt or job.job_token is None or job.slurm_id is not None:
        return
    if job.job_token != e.token:
        raise _bad(e, f"{e.target} attempt {e.attempt} was submitted under another token")
    job.slurm_id = e.slurm_id
    if e.target == BASELINE:
        state.baseline.status = "submitted"
    else:
        state.candidates[e.target].status = "submitted"


@_handles(ev.JobStarted)
def _job_started(state: RunState, e: ev.JobStarted) -> None:
    job = _job(state, e.target, e)
    if e.attempt != job.job_attempt:
        return
    if e.target == BASELINE and state.baseline.status == "submitted":
        state.baseline.status = "running"
    elif e.target != BASELINE and state.candidates[e.target].status == "submitted":
        state.candidates[e.target].status = "running"


@_handles(ev.JobAttemptFailed)
def _job_attempt_failed(state: RunState, e: ev.JobAttemptFailed) -> None:
    job = _job(state, e.target, e)
    if e.attempt != job.job_attempt or job.job_token is None:
        return
    if e.target == BASELINE:
        if state.baseline.status == "ingested":
            return
        state.baseline.status = "pending"
    else:
        candidate = state.candidates[e.target]
        if candidate.status in COMPLETED or candidate.status == "forfeited":
            return
        candidate.status = "allocated"
    job.infra_failures.append(e.cause)
    job.job_token, job.slurm_id = None, None


@_handles(ev.BaselineIngested)
def _baseline_ingested(state: RunState, e: ev.BaselineIngested) -> None:
    if state.baseline.status == "ingested":
        return
    state.baseline.status = "ingested"
    state.baseline.pool_sha = e.pool_sha
    state.baseline.policy_sample_sha = e.policy_sample_sha
    state.baseline.n_rows = e.n_rows


@_handles(ev.RoundOpened)
def _round_opened(state: RunState, e: ev.RoundOpened) -> None:
    if e.round <= len(state.rounds):
        return
    if e.round != len(state.rounds) + 1:
        raise _bad(e, f"round {e.round} skips round {len(state.rounds) + 1}")
    if e.b_effective > state.B:
        raise _bad(e, f"b_effective {e.b_effective} exceeds B {state.B}")
    state.rounds.append(RoundState(index=e.round, b_effective=e.b_effective))


@_handles(ev.DecisionRecorded)
def _decision_recorded(state: RunState, e: ev.DecisionRecorded) -> None:
    if e.decision_id in state.decisions:
        return
    rnd = _round(state, e.round, e)
    state.decisions[e.decision_id] = DecisionState(
        id=e.decision_id, round=e.round, kind=e.kind, attempt=e.attempt,
        state_hash=e.state_hash, response_sha=e.response_sha)
    rnd.decision_ids.append(e.decision_id)
    if e.kind != "develop_component":
        rnd.attempts += 1
    _spend(state, e)


@_handles(ev.DecisionValidated)
def _decision_validated(state: RunState, e: ev.DecisionValidated) -> None:
    decision = _decision(state, e.decision_id, e)
    if decision.status != "recorded":
        return
    decision.status = "valid" if e.ok else "invalid"
    decision.n_valid = e.n_valid
    decision.problems = list(e.problems)
    decision.normalized_sha = e.normalized_sha


@_handles(ev.CandidatesAllocated)
def _candidates_allocated(state: RunState, e: ev.CandidatesAllocated) -> None:
    rnd = _round(state, e.round, e)
    if rnd.status == "allocated":
        return
    if rnd.status != "open":
        raise _bad(e, f"round {e.round} is {rnd.status}")
    slots = [c.slot for c in e.candidates] + list(e.forfeited_slots)
    if sorted(slots) != list(range(1, rnd.b_effective + 1)):
        raise _bad(e, f"slots {sorted(slots)} do not cover 1..{rnd.b_effective} once each")
    entries = []
    for c in e.candidates:
        ordinal = ordinal_of(e.round, c.slot, state.B)
        if c.ordinal != ordinal or c.id != candidate_id(ordinal):
            raise _bad(e, f"{c.id} is not the candidate of round {e.round} slot {c.slot}")
        for reference in (c.parent, c.repeat_of):
            if reference is not None and _candidate(state, reference, e).status == "forfeited":
                raise _bad(e, f"{c.id} refers to forfeited {reference}")
        entries.append(CandidateState(
            id=c.id, ordinal=ordinal, round=e.round, slot=c.slot, parent=c.parent, start=c.start,
            recipe_sha=c.recipe_sha, fingerprint=c.fingerprint, steps=c.steps,
            repeat_of=c.repeat_of, anchor=c.anchor))
    for slot in e.forfeited_slots:
        ordinal = ordinal_of(e.round, slot, state.B)
        entries.append(CandidateState(
            id=candidate_id(ordinal), ordinal=ordinal, round=e.round, slot=slot,
            status="forfeited"))
    entries.sort(key=lambda c: c.ordinal)
    if any(c.id in state.candidates for c in entries):
        raise _bad(e, f"round {e.round} reuses a candidate id")
    if e.decision_id is not None:
        _decision(state, e.decision_id, e).consumed = True
    for c in entries:
        state.candidates[c.id] = c
    rnd.status = "allocated"
    rnd.candidates = [c.id for c in entries]
    state.consecutive_skips = 0
    state.stop.consecutive_rejections = 0


@_handles(ev.CandidateCompleted)
def _candidate_completed(state: RunState, e: ev.CandidateCompleted) -> None:
    candidate = _candidate(state, e.candidate, e)
    if candidate.status in COMPLETED:
        return
    if candidate.status == "forfeited":
        raise _bad(e, f"{e.candidate} was forfeited")
    candidate.status = e.status
    candidate.completed = e.summary
    candidate.completed_attempt = e.attempt


@_handles(ev.SelfvalComputed)
def _selfval_computed(state: RunState, e: ev.SelfvalComputed) -> None:
    candidate = _candidate(state, e.candidate, e)
    if candidate.selfval_sha is None:
        candidate.selfval_sha = e.selfval_sha


@_handles(ev.LedgerRowFrozen)
def _ledger_row_frozen(state: RunState, e: ev.LedgerRowFrozen) -> None:
    candidate = _candidate(state, e.candidate, e)
    if candidate.ledger_row_sha is None:
        candidate.ledger_row_sha = e.row_sha


@_handles(ev.CriticAdvisory)
def _critic_advisory(state: RunState, e: ev.CriticAdvisory) -> None:
    rnd = _round(state, e.round, e)
    if rnd.critic_row_sha is not None or rnd.critic_unavailable:
        return
    rnd.critic_row_sha = e.row_sha
    _spend(state, e)


@_handles(ev.CriticUnavailable)
def _critic_unavailable(state: RunState, e: ev.CriticUnavailable) -> None:
    rnd = _round(state, e.round, e)
    if rnd.critic_row_sha is None:
        rnd.critic_unavailable = True


@_handles(ev.ComponentRequested)
def _component_requested(state: RunState, e: ev.ComponentRequested) -> None:
    if e.request_id in state.components:
        return
    _round(state, e.round, e)
    _decision(state, e.decision_id, e).consumed = True
    state.components[e.request_id] = ComponentState(
        request_id=e.request_id, round=e.round, decision_id=e.decision_id, role=e.role,
        name=e.name, reference_candidate=e.reference_candidate, request_sha=e.request_sha)


@_handles(ev.ComponentAttempt)
def _component_attempt(state: RunState, e: ev.ComponentAttempt) -> None:
    if e.request_id not in state.components:
        raise _bad(e, f"unknown component request {e.request_id!r}")
    component = state.components[e.request_id]
    if component.status != "requested" or any(a.attempt == e.attempt for a in component.attempts):
        return
    component.attempts.append(ComponentAttemptRecord(
        attempt=e.attempt, outcome=e.outcome, failure_stage=e.failure_stage))
    _spend(state, e)


@_handles(ev.ComponentResult)
def _component_result(state: RunState, e: ev.ComponentResult) -> None:
    if e.request_id not in state.components:
        raise _bad(e, f"unknown component request {e.request_id!r}")
    component = state.components[e.request_id]
    if component.status != "requested":
        return
    component.status = e.status
    component.reason_code = e.reason_code
    component.row_sha = e.row_sha
    component.component_sha = e.component_sha


@_handles(ev.StopRequested)
def _stop_requested(state: RunState, e: ev.StopRequested) -> None:
    _round(state, e.round, e)
    if e.decision_id is not None:
        decision = _decision(state, e.decision_id, e)
        if decision.consumed:
            return
        decision.consumed = True
    state.stop.pending = e


@_handles(ev.StopAccepted)
def _stop_accepted(state: RunState, e: ev.StopAccepted) -> None:
    rnd = _round(state, e.round, e)
    if state.stop.accepted is not None:
        return
    if rnd.status != "open":
        raise _bad(e, f"round {e.round} is {rnd.status}")
    state.stop.accepted = e
    state.stop.pending = None
    rnd.status = "stopped"


@_handles(ev.StopRejected)
def _stop_rejected(state: RunState, e: ev.StopRejected) -> None:
    _round(state, e.round, e)
    if e.decision_id is not None and any(
            r.decision_id == e.decision_id for r in state.stop.rejections):
        return
    state.stop.rejections.append(e)
    state.stop.pending = None
    state.stop.consecutive_rejections += 1


@_handles(ev.DeliveryAssembled)
def _delivery_assembled(state: RunState, e: ev.DeliveryAssembled) -> None:
    if state.delivery is not None:
        return
    if state.stop.accepted is None:
        raise _bad(e, "no accepted stop")
    state.delivery = e


@_handles(ev.LibraryFrozen)
def _library_frozen(state: RunState, e: ev.LibraryFrozen) -> None:
    if state.library is not None:
        return
    if state.stop.accepted is None:
        raise _bad(e, "no accepted stop")
    state.library = e


@_handles(ev.RunFinished)
def _run_finished(state: RunState, e: ev.RunFinished) -> None:
    if state.terminal:
        return
    if e.status != "needs_attention" and (state.library is None or state.library.status != e.status):
        raise _bad(e, f"{e.status} without a library frozen as {e.status}")
    state.terminal = True
    state.finished_status = e.status


@_handles(ev.RoundSkipped)
def _round_skipped(state: RunState, e: ev.RoundSkipped) -> None:
    rnd = _round(state, e.round, e)
    if rnd.status == "skipped":
        return
    if rnd.status != "open":
        raise _bad(e, f"round {e.round} is {rnd.status}")
    rnd.status = "skipped"
    rnd.skip_reason = e.reason_code
    state.consecutive_skips += 1


@_handles(ev.Resumed)
def _resumed(state: RunState, e: ev.Resumed) -> None:
    state.resumes += 1
    if state.terminal:
        state.terminal, state.finished_status = False, None


def _step(state: RunState, event: ev._Event) -> None:
    handler = _HANDLERS.get(event.type)
    if handler is None:
        raise LedgerCorruption(f"no handler for event type {event.type!r}")
    if state.run is None and event.type != "run_started":
        raise _bad(event, "the run has not started")
    if state.terminal and not _allowed_after_finish(state, event):
        raise _bad(event, f"the run already finished as {state.finished_status}")
    state.n_events += 1
    handler(state, event)


def _allowed_after_finish(state: RunState, event: ev._Event) -> bool:
    if event.type == "resumed":
        return state.finished_status == "needs_attention"
    return event.type == "run_finished" and event.status == state.finished_status


def apply(state: RunState, event: ev._Event) -> RunState:
    """The state after `event`; `state` itself is left untouched."""
    updated = state.model_copy(deep=True)
    _step(updated, event)
    return updated


def fold(events: Iterable[ev._Event]) -> RunState:
    """The state after every event, equal to applying them one by one from RunState()."""
    state = RunState()
    for event in events:
        _step(state, event)
    return state


def state_hash(state: RunState) -> str:
    return sha256_hex(canonical_json(state.model_dump(mode="json")))
