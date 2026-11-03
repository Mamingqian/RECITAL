"""Typed payloads of every event the control layer writes.

Payloads carry digests (blob shas), counts and identifiers, never prompts or
tables. The store keeps `{seq, ts, type, data}`; `to_record` / `from_record` map a
model to and from that `(type, data)` pair, where `data` omits `type`.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter, ValidationError

from recital.control.errors import LedgerCorruption

SCHEMA_VERSION = 1

BASELINE = "baseline"
Sha = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Count = Annotated[int, Field(ge=0)]
Positive = Annotated[int, Field(ge=1)]

Role = Literal[
    "per_oracle_normalization", "per_oracle_utility", "cross_oracle_aggregation",
    "per_sequence_shaping", "per_sequence_penalty", "hard_biological_gate",
]
StartMode = Literal["from_initial", "from_parent", "continue_parent"]
DecisionKind = Literal["propose", "stop", "develop_component"]
JobKind = Literal["baseline_pool", "train_candidate"]
CompletionStatus = Literal["ok", "failed", "timeout", "cancelled"]
StopSource = Literal["agent", "contract_default"]
StopOutcome = Literal["delivered", "no_qualified_candidate"]
FinalStatus = Literal["frozen", "completed_no_qualified_candidate"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Usage(_Model):
    prompt: Count
    completion: Count
    cached: Count
    reasoning: Count


class Problem(_Model):
    """A typed validation finding; the only free text is `detail`, capped."""

    code: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    path: Annotated[str, StringConstraints(max_length=200)] = ""
    detail: Annotated[str, StringConstraints(max_length=300)] = ""


class LlmCall(_Model):
    """What one model call left behind: digests of what went in and came out."""

    prompt_hash: Sha
    prompt_manifest: list[Sha]
    schema_sha: Sha
    response_sha: Sha
    tool_transcript_sha: Sha | None
    usage: Usage
    model: str
    resolved_model: str
    latency_s: Annotated[float, Field(ge=0)]
    finish_reason: str | None = None


class Budgets(_Model):
    candidate_budget: Positive
    rounds_min: Positive
    rounds_max: Positive
    total_updates: Positive
    max_total_tokens: Positive | None


class AllocatedCandidate(_Model):
    id: str
    ordinal: Positive
    slot: Positive
    parent: str | None
    start: StartMode
    recipe_sha: Sha
    fingerprint: Sha
    steps: Positive
    repeat_of: str | None
    anchor: bool


class CandidateSummary(_Model):
    steps_done: Count | None = None
    wall_s: Annotated[float, Field(ge=0)] | None = None
    reward_first: float | None = None
    reward_last: float | None = None
    reward_std: float | None = None
    frac_zero_std: float | None = None
    kl_ref: float | None = None
    pool_sha: Sha | None = None
    evaluation_sha: Sha | None = None
    failure_type: str | None = None
    failure_sha: Sha | None = None


class _Event(_Model):
    type: str


EVENT_TYPES: dict[str, type[_Event]] = {}


def _register(cls: type[_Event]) -> type[_Event]:
    EVENT_TYPES[cls.model_fields["type"].default] = cls
    return cls


@_register
class RunStarted(_Event):
    """Provenance only: hashes and options are recorded and never compared."""

    type: Literal["run_started"] = "run_started"
    schema_version: Positive
    task: str
    arm: str
    seed: int
    conditions: list[str]
    candidates_per_round: Positive
    anchor_candidate: bool
    budgets: Budgets
    contract_hash: Sha
    config_hash: Sha
    source_commit: str | None
    oracle_versions_hash: Sha
    model: str
    llm: dict[str, Any]


@_register
class JobSubmitIntent(_Event):
    type: Literal["job_submit_intent"] = "job_submit_intent"
    target: str
    kind: JobKind
    attempt: Positive
    token: str
    spec_sha: Sha


@_register
class JobSubmitted(_Event):
    type: Literal["job_submitted"] = "job_submitted"
    target: str
    attempt: Positive
    token: str
    slurm_id: str


@_register
class JobStarted(_Event):
    type: Literal["job_started"] = "job_started"
    target: str
    attempt: Positive


@_register
class JobAttemptFailed(_Event):
    """The attempt died for a cluster reason; the target may be submitted again."""

    type: Literal["job_attempt_failed"] = "job_attempt_failed"
    target: str
    attempt: Positive
    cause: str
    slurm_id: str | None


@_register
class BaselineIngested(_Event):
    type: Literal["baseline_ingested"] = "baseline_ingested"
    attempt: Positive
    pool_sha: Sha
    policy_sample_sha: Sha | None
    n_rows: Count


@_register
class RoundOpened(_Event):
    type: Literal["round_opened"] = "round_opened"
    round: Positive
    b_effective: Count


@_register
class DecisionRecorded(_Event, LlmCall):
    """Written before validation, so a paid response is never lost."""

    type: Literal["decision_recorded"] = "decision_recorded"
    decision_id: str
    round: Positive
    kind: DecisionKind
    attempt: Positive
    state_hash: Sha


@_register
class DecisionValidated(_Event):
    type: Literal["decision_validated"] = "decision_validated"
    decision_id: str
    ok: bool
    n_valid: Count
    problems: list[Problem]
    normalized_sha: Sha | None


@_register
class CandidatesAllocated(_Event):
    """Every slot of the round appears once, as a candidate or as forfeited."""

    type: Literal["candidates_allocated"] = "candidates_allocated"
    round: Positive
    decision_id: str | None
    candidates: list[AllocatedCandidate]
    forfeited_slots: list[Positive]


@_register
class CandidateCompleted(_Event):
    type: Literal["candidate_completed"] = "candidate_completed"
    candidate: str
    attempt: Positive
    status: CompletionStatus
    summary: CandidateSummary


@_register
class SelfvalComputed(_Event):
    type: Literal["selfval_computed"] = "selfval_computed"
    candidate: str
    selfval_sha: Sha


@_register
class LedgerRowFrozen(_Event):
    type: Literal["ledger_row_frozen"] = "ledger_row_frozen"
    candidate: str
    row_sha: Sha
    tokens: Count


@_register
class CriticAdvisory(_Event, LlmCall):
    type: Literal["critic_advisory"] = "critic_advisory"
    round: Positive
    row_sha: Sha
    n_concerns: Count


@_register
class CriticUnavailable(_Event):
    type: Literal["critic_unavailable"] = "critic_unavailable"
    round: Positive
    attempts: Positive
    cause: str


@_register
class ComponentRequested(_Event):
    type: Literal["component_requested"] = "component_requested"
    request_id: str
    round: Positive
    decision_id: str
    role: Role
    name: str
    reference_candidate: str | None
    request_sha: Sha


@_register
class ComponentAttempt(_Event, LlmCall):
    type: Literal["component_attempt"] = "component_attempt"
    request_id: str
    attempt: Positive
    outcome: Literal["published", "failed"]
    failure_stage: str | None
    detail_sha: Sha | None


@_register
class ComponentResult(_Event):
    type: Literal["component_result"] = "component_result"
    request_id: str
    status: Literal["published", "rejected"]
    reason_code: str | None
    row_sha: Sha
    component_sha: Sha | None


@_register
class StopRequested(_Event):
    type: Literal["stop_requested"] = "stop_requested"
    round: Positive
    decision_id: str | None
    source: StopSource
    outcome: StopOutcome
    assignment: dict[str, str]
    summary_sha: Sha | None


@_register
class StopAccepted(_Event):
    type: Literal["stop_accepted"] = "stop_accepted"
    round: Positive
    source: StopSource
    outcome: StopOutcome
    assignment: dict[str, str]


@_register
class StopRejected(_Event):
    type: Literal["stop_rejected"] = "stop_rejected"
    round: Positive
    decision_id: str | None
    source: StopSource
    reasons: list[Problem]


@_register
class DeliveryAssembled(_Event):
    type: Literal["delivery_assembled"] = "delivery_assembled"
    assignment: dict[str, str]
    pool_sha: Sha
    plan_sha: Sha
    qualified: dict[str, bool]
    counts: dict[str, Count]


@_register
class LibraryFrozen(_Event):
    type: Literal["library_frozen"] = "library_frozen"
    status: FinalStatus
    library_sha: Sha | None
    reason_code: str | None


@_register
class RunFinished(_Event):
    """`needs_attention` ends the run until a `resumed` event; the others are final."""

    type: Literal["run_finished"] = "run_finished"
    status: Literal["frozen", "completed_no_qualified_candidate", "needs_attention"]
    reason_code: str | None


@_register
class RoundSkipped(_Event):
    type: Literal["round_skipped"] = "round_skipped"
    round: Positive
    attempts: Positive
    reason_code: str


@_register
class Resumed(_Event):
    """Hashes are recorded for provenance; a difference from the start is not an error."""

    type: Literal["resumed"] = "resumed"
    torn_bytes: Count
    source_commit: str | None
    contract_hash: Sha
    config_hash: Sha
    oracle_versions_hash: Sha


EventModel = Annotated[Union[tuple(EVENT_TYPES.values())], Field(discriminator="type")]
_ADAPTER = TypeAdapter(EventModel)


def parse_event(raw: dict[str, Any]) -> _Event:
    """Build the model for a flat `{type, **payload}` mapping."""
    try:
        return _ADAPTER.validate_python(raw)
    except ValidationError as exc:
        raise LedgerCorruption(f"event {raw.get('type')!r} fails its schema: {exc}") from exc


def to_record(event: _Event) -> tuple[str, dict[str, Any]]:
    """The `(type, data)` pair `EventLog.append` stores for `event`."""
    return event.type, event.model_dump(mode="json", exclude={"type"})


def from_record(record) -> _Event:
    """Parse a stored record, anything with `.type` and `.data`."""
    return parse_event({"type": record.type, **record.data})
