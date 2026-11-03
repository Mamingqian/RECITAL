"""The control loop of one cell: a state machine over the event log.

`Engine.step()` folds the log, advances the first phase that has work and returns; calling it again
continues from whatever the log says, so a crashed process resumes by constructing a new engine on the same
files. Every outward action is written as an intent event before it is taken and every completion as an
event after, and all state a decision needs comes from events and the blobs they name.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from recital.success import SuccessSpec
from recital.synthesis.models import IsolationError

from . import events as ev
from . import recipes as rc
from .agents import AgentTurn, run_critic, run_strategist
from .contract import ControlContract
from .errors import (BlindingViolation, ControlError, LedgerCorruption, PromptBudgetExceeded,
                     RunTokenBudgetExceeded, SchedulerUnreachable)
from .events import from_record, to_record
from .guard import count_tokens
from .health import UNIQUE_KMER_MEASUREMENT, HealthRule, baseline_reference, diversity_floor, plan_with_floor
from .jobs import (BASELINE_ACTION_ID, BASELINE_RESULT_NAME, CANDIDATE_RESULT_NAME, BaselineResult, Candidate,
                   CandidateResult, CheckpointRecord, Evaluation, JobSpec, Resources, TrainParameters,
                   file_sha256, read_failure, read_result, read_spec, write_json, write_spec)
from .jsonio import canonical_json, sha256_hex
from .ledger import build_row, composition_bands, medians, render_row, visible_row
from .projection import ArmView, Retriever, evidence_of, render, render_with_tail
from .prompts import CRITIC_SYSTEM
from .schemas import critic_schema, retrieval_tools, round_response_schema
from .scheduler import JobState, Scheduler
from .selection import best_assignment, validate_assignment, validate_no_qualified
from .selfval import scalar_name
from .state import RunState, candidate_id, fold, ordinal_of, state_hash
from .store import BlobStore, EventLog
from .transport import (ResponseIncomplete, ResponseRefused, ResponsesTransport, ResponseUnparsable,
                        TransportUnavailable, Usage as TransportUsage, UsageMeter)

BASELINE = "baseline"
TRAIN_SECONDS_PER_STEP = 2.6
FIXED_JOB_SECONDS = 2400
TIMEOUT_GROWTH = 1.5
# Failures a later attempt may not repeat: the model endpoint or the shared filesystem blipped. A run that meets this
# many in a row without making progress stops for a person instead of waiting forever.
TRANSIENT = (OSError, TransportUnavailable, ResponseIncomplete, ResponseUnparsable, ResponseRefused)
TRANSIENT_LIMIT = 6


class StepResult(str, Enum):
    PROGRESS = "progress"
    WAITING = "waiting"
    DONE = "done"


class ComponentRunner(Protocol):
    def run(self, engine: "Engine", state: RunState, request_id: str) -> None:
        """Emit the request's attempt and result events."""


@dataclass
class EngineSettings:
    run_dir: Path
    config_path: Path
    repo_root: Path
    device: str
    task: Any                              # actions.ExperimentContract
    control: ControlContract
    recipe_ctx: rc.RecipeContext
    success: SuccessSpec
    definitions: Mapping[str, Mapping[str, Any]]
    generation_plan: dict[str, Any]
    header_text: str
    instructions: str
    run_header: ev.RunStarted
    # Names admitted per role, or a callable that reads them from the live catalog each round.
    primitives: Mapping[str, Sequence[str]] | Callable[[], Mapping[str, Sequence[str]]]
    approved_references: Sequence[str] = ()


def _llm_fields(turn: AgentTurn, rendered, schema_sha: str, response_sha: str, transcript_sha: str | None):
    usage = turn.usage
    return dict(
        prompt_hash=rendered.prompt_hash, prompt_manifest=list(rendered.manifest), schema_sha=schema_sha,
        response_sha=response_sha, tool_transcript_sha=transcript_sha,
        usage=ev.Usage(prompt=usage.input, completion=usage.output, cached=usage.cached, reasoning=usage.reasoning),
        model=turn.model, resolved_model=turn.resolved_model, latency_s=turn.latency_s)


def time_limit(steps: int, factor: float, timeouts: int) -> str:
    seconds = int((steps * TRAIN_SECONDS_PER_STEP * factor + FIXED_JOB_SECONDS) * TIMEOUT_GROWTH ** timeouts) + 1
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"


class Engine:
    def __init__(self, settings: EngineSettings, *, log: EventLog, blobs: BlobStore, scheduler: Scheduler,
                 transport: ResponsesTransport, assemble: Callable[..., dict[str, Any]],
                 components: ComponentRunner | None = None, sleep: Callable[[float], None] = time.sleep,
                 may_submit: Callable[[], bool] = lambda: True):
        self.s, self.log, self.blobs = settings, log, blobs
        self.may_submit = may_submit
        self.scheduler, self.transport, self.assemble, self.components = scheduler, transport, assemble, components
        self.sleep = sleep
        control = settings.control
        if control.synthesis.enabled and components is None:
            raise ControlError("synthesis is enabled but no component runner is mounted")
        self.arm = ArmView.from_task(settings.task)
        self.visible = sorted(settings.task.visible_sources)
        self.headline = [scalar_name(control.selfval.scalar, source) for source in self.visible]
        self.meter = UsageMeter(control.llm.max_total_tokens)
        state = self.state()
        if state.run is not None:
            used = state.usage
            self.meter.total = TransportUsage(used.prompt, used.cached, used.completion, used.reasoning)
        self._results: dict[str, CandidateResult] = {}
        self._transient = 0
        self._baseline_cache: BaselineResult | None = None
        self._delivery_rule_cache: tuple[dict[str, Any], HealthRule] | None = None

    # ----- plumbing -------------------------------------------------------------------------------

    def state(self) -> RunState:
        return fold(from_record(record) for record in self.log.events())

    def emit(self, model: ev._Event) -> None:
        self.log.append(*to_record(model))

    def candidate_dir(self, candidate: str) -> Path:
        return self.s.run_dir / "candidates" / candidate

    def _job_dir(self, target: str, attempt: int) -> Path:
        return self.s.run_dir / "jobs" / f"{target}_a{attempt}"

    def _output_dir(self, target: str) -> Path:
        return self.s.run_dir / ("baseline" if target == BASELINE else f"candidates/{target}")

    @property
    def run_tag(self) -> str:
        """Names this run's jobs apart from any other run's, including an earlier run of the same config in a
        recreated directory: the hash of the log's first event, which carries the time the run began."""
        first = next(iter(self.log.events()), None)
        if first is None:
            raise ControlError("a job token needs a started run")
        return first.hash[:10]

    def _token(self, target: str, attempt: int) -> str:
        return f"recital.{self.run_tag}.{target.replace('_', '-')}.a{attempt}"

    def run(self, poll_seconds: float | None = None) -> RunState:
        """Step until the run is terminal, sleeping between polls while jobs run."""
        wait = poll_seconds if poll_seconds is not None else self.s.control.scheduler.poll_seconds
        while True:
            result = self.step()
            if result is StepResult.DONE:
                return self.state()
            if result is StepResult.WAITING:
                self.sleep(wait)

    def step(self) -> StepResult:
        state = self.state()
        if state.terminal:
            return StepResult.DONE
        if state.run is None:
            self.emit(self.s.run_header)
            return StepResult.PROGRESS
        try:
            for phase in (self._baseline, self._jobs, self._selfval, self._rows, self._finish, self._submit,
                          self._critic, self._decide):
                if phase(state):
                    self._transient = 0
                    return StepResult.PROGRESS
        except LedgerCorruption:
            raise
        except (ControlError, IsolationError, OSError) as exc:
            if isinstance(exc, TRANSIENT) and self._transient < TRANSIENT_LIMIT:
                self._transient += 1
                return StepResult.WAITING
            self._halt(exc)
            return StepResult.DONE
        self._transient = 0
        return StepResult.DONE if self.state().terminal else StepResult.WAITING

    def _halt(self, exc: Exception) -> None:
        write_json(self.s.run_dir / "needs_attention.json", {"type": type(exc).__name__, "message": str(exc)})
        self.emit(ev.RunFinished(status="needs_attention", reason_code=type(exc).__name__[:64]))

    # ----- jobs -----------------------------------------------------------------------------------

    def _resources(self, steps: int | None, timeouts: int) -> Resources:
        scheduler = self.s.control.scheduler
        limit = time_limit(steps or 0, scheduler.time_factor, timeouts)  # the baseline job is a job with no updates
        return Resources(partition=scheduler.partition, gres=scheduler.gres, cpus=scheduler.cpus,
                         mem=scheduler.mem, time_limit=limit)

    def _baseline_rows(self) -> tuple[dict[str, list[str]], dict[str, list[float]] | None]:
        """Each condition's legal baseline sequences, and their unique k-mer counts when the task emits them."""
        directory = self.s.run_dir / "baseline"
        pool = json.loads((directory / "pool.json").read_text())
        measured = json.loads((directory / "public_measurements.json").read_text())["measurements"]
        if "legal" not in measured:
            raise ControlError(f"{directory} holds no `legal` measurement, which the library-health reference needs")
        legal = [bool(value) for value in measured["legal"]["values"]]
        kmers = measured.get(UNIQUE_KMER_MEASUREMENT, {}).get("values")
        sequences: dict[str, list[str]] = {condition: [] for condition in self.s.task.conditions}
        counts: dict[str, list[float]] = {condition: [] for condition in self.s.task.conditions}
        for index, (sequence, condition) in enumerate(zip(pool["sequences"], pool["conditions"])):
            if legal[index] and condition in sequences:
                sequences[condition].append(sequence)
                if kmers is not None:
                    counts[condition].append(float(kmers[index]))
        return sequences, None if kmers is None else counts

    def _delivery_rule(self) -> tuple[dict[str, Any], HealthRule]:
        """The selection plan every candidate and the delivery run under, and the library-health rule beside it.

        The contract declares two fractions and the baseline pool says what they are fractions of. The plan's
        minimum distance is `min_distance_fraction` of the smallest condition's median nearest-neighbour distance,
        because a plan carries one number and the smallest keeps every condition's floor at or below its own.
        """
        if self._delivery_rule_cache is None:
            declared = self.s.control
            try:
                reference = baseline_reference(*self._baseline_rows())
            except ValueError as exc:
                raise ControlError(f"the baseline pool cannot ground the library-health rule: {exc}") from exc
            floor = diversity_floor(reference, declared.health.min_distance_fraction)
            self._delivery_rule_cache = (
                plan_with_floor(declared.selection.plan, floor),
                HealthRule(min_pairwise_ratio=declared.health.min_pairwise_ratio, reference=reference))
        return self._delivery_rule_cache

    def _evaluation(self, target: str) -> Evaluation:
        """The baseline pool is selected from by no rule: the rule is derived from it."""
        plan, health = (None, None) if target == BASELINE else self._delivery_rule()
        return Evaluation(generation_plan=dict(self.s.generation_plan), scorers=list(self.visible),
                          selection_plan=plan, health=health)

    def _initial_record(self) -> CheckpointRecord:
        return CheckpointRecord(**rc.initial_parent_record(self.s.recipe_ctx), measurement_refs=[])

    def _records(self, state: RunState) -> dict[str, CheckpointRecord]:
        records = {"initial": self._initial_record()}
        for candidate in state.ordered_candidates():
            if candidate.status == "ok":
                result = self._result(candidate.id)
                for record in result.checkpoints:
                    records[record.id] = record
        return records

    def _result(self, candidate: str) -> CandidateResult:
        if candidate not in self._results:
            path = self.candidate_dir(candidate) / CANDIDATE_RESULT_NAME
            self._results[candidate] = CandidateResult.model_validate(json.loads(path.read_text()))
        return self._results[candidate]

    def _baseline_result(self) -> BaselineResult:
        if self._baseline_cache is None:
            path = self.s.run_dir / "baseline" / BASELINE_RESULT_NAME
            self._baseline_cache = BaselineResult.model_validate(json.loads(path.read_text()))
        return self._baseline_cache

    def _spec(self, state: RunState, target: str, attempt: int, token: str) -> JobSpec:
        timeouts = self._timeouts(state, target)
        common = dict(run_dir=str(self.s.run_dir), config_path=str(self.s.config_path),
                      repo_root=str(self.s.repo_root), device=self.s.device, attempt=attempt, submit_token=token,
                      evaluation=self._evaluation(target))
        if target == BASELINE:
            return JobSpec(kind="baseline_pool", output_dir=str(self._output_dir(BASELINE)), train=None,
                           candidate=Candidate(id=BASELINE_ACTION_ID, ordinal=0, round=0, slot=0,
                                               action_id=BASELINE_ACTION_ID, seed=self.s.task.seed),
                           parent_checkpoints={"initial": self._initial_record()},
                           resources=self._resources(None, timeouts), **common)
        candidate = state.candidates[target]
        spec = self.blobs.get_json(candidate.recipe_sha)
        train = TrainParameters(**spec["train"])
        records = self._records(state)
        needed = {"initial", train.start_checkpoint, train.kl_reference}
        return JobSpec(kind="train_candidate", output_dir=str(self._output_dir(target)), train=train,
                       candidate=Candidate(id=target, ordinal=candidate.ordinal, round=candidate.round,
                                           slot=candidate.slot, action_id=target, seed=self.s.task.seed),
                       parent_checkpoints={name: records[name] for name in sorted(needed)},
                       resources=self._resources(train.additional_updates, timeouts), **common)

    @staticmethod
    def _timeouts(state: RunState, target: str) -> int:
        job = state.baseline if target == BASELINE else state.candidates[target]
        return sum(1 for cause in job.infra_failures if cause == "timeout")

    def _start_job(self, state: RunState, target: str) -> bool:
        """Write the spec, record the intent, then submit; recovers a job whose submission was never recorded."""
        job = state.baseline if target == BASELINE else state.candidates[target]
        if job.job_token is None:
            attempt = job.job_attempt + 1
            token = self._token(target, attempt)
            spec = self._spec(state, target, attempt, token)
            path = self._job_dir(target, attempt) / "spec.json"
            write_spec(path, spec)
            self.emit(ev.JobSubmitIntent(target=target, kind=spec.kind, attempt=attempt, token=token,
                                         spec_sha=spec.digest()))
            return self._submit_recorded(target, attempt, token, path, spec.resources)
        attempt, token = job.job_attempt, job.job_token
        path = self._job_dir(target, attempt) / "spec.json"
        return self._submit_recorded(target, attempt, token, path, read_spec(path).resources)

    def _submit_recorded(self, target: str, attempt: int, token: str, path: Path, resources: Resources) -> bool:
        handle = self.scheduler.find_by_token(token)
        if handle is None:
            handle = self.scheduler.submit(path, resources, token)
        self.emit(ev.JobSubmitted(target=target, attempt=attempt, token=token, slurm_id=handle.job_id))
        return True

    def _baseline(self, state: RunState) -> bool:
        job = state.baseline
        if job.status == "ingested":
            return False
        if job.status == "pending":
            if not self.may_submit():
                return False
            try:
                return self._start_job(state, BASELINE)
            except SchedulerUnreachable:
                return False
        return self._poll(state, BASELINE, job)

    def _jobs(self, state: RunState) -> bool:
        progressed = False
        for candidate_name in state.inflight:
            progressed |= self._poll(state, candidate_name, state.candidates[candidate_name])
        return progressed

    def _submit(self, state: RunState) -> bool:
        if state.baseline.status != "ingested":
            return False
        cap = self.s.control.scheduler.max_concurrent_jobs - len(state.inflight)
        for name in state.unsubmitted:
            if cap <= 0 or not self.may_submit():
                return False
            try:
                self._start_job(state, name)
            except SchedulerUnreachable:
                return False
            return True
        return False

    def _poll(self, state: RunState, target: str, job) -> bool:
        attempt, token = job.job_attempt, job.job_token
        try:
            handle = self.scheduler.find_by_token(token)
            status = self.scheduler.poll(handle) if handle is not None else None
        except SchedulerUnreachable:
            return False
        output = self._output_dir(target)
        result = read_result(output)
        if result is not None and result.attempt == attempt and result.submit_token == token:
            if result.status == "ok":
                return self._ingest(state, target, attempt)
            failure = read_failure(output) or result.failure
            return self._failed(state, target, job, attempt, failure.type, failure.message, failure.retryable)
        if status is None:
            return self._failed(state, target, job, attempt, "job_vanished", "the scheduler holds no such job", True)
        if status.state is JobState.RUNNING:
            if job.status == "submitted":
                self.emit(ev.JobStarted(target=target, attempt=attempt))
                return True
            return False
        if status.state is JobState.COMPLETED:
            return self._failed(state, target, job, attempt, "completed_without_result",
                                "the job exited cleanly without a result marker", True)
        if status.state in (JobState.FAILED, JobState.CANCELLED):
            # An out-of-memory kill repeats under the same request, so it is the recipe's failure, not the cluster's.
            return self._failed(state, target, job, attempt, status.reason or status.state.value,
                                f"job ended {status.state.value} (exit {status.exit_code})",
                                status.reason != "OUT_OF_MEMORY")
        if status.state is JobState.TIMEOUT:
            return self._failed(state, target, job, attempt, "timeout", "the job hit its time limit", True)
        return False

    def _failed(self, state: RunState, target: str, job, attempt: int, cause: str, message: str,
                retryable: bool) -> bool:
        if retryable and len(job.infra_failures) < self.s.control.scheduler.max_infra_retries:
            self.emit(ev.JobAttemptFailed(target=target, attempt=attempt, cause=cause[:120], slurm_id=job.slurm_id))
            return True
        if target == BASELINE:
            # Return the baseline to pending first, so a resume submits it again instead of polling the dead job.
            self.emit(ev.JobAttemptFailed(target=target, attempt=attempt, cause=cause[:120], slurm_id=job.slurm_id))
            raise ControlError(f"the baseline job failed: {cause}: {message}")
        detail = {"type": cause, "message": message}
        failure_sha = self.blobs.put_json(detail)
        status = "timeout" if cause == "timeout" else "failed"
        self.emit(ev.CandidateCompleted(candidate=target, attempt=attempt, status=status,
                                        summary=ev.CandidateSummary(failure_type=cause[:120], failure_sha=failure_sha)))
        return True

    def _ingest(self, state: RunState, target: str, attempt: int) -> bool:
        if target == BASELINE:
            result = self._baseline_result()
            self._delivery_rule()  # a baseline that cannot ground the library-health rule stops the cell here
            self.emit(ev.BaselineIngested(attempt=attempt, pool_sha=result.pool_sha256,
                                          policy_sample_sha=result.policy_sample_sha256, n_rows=result.n_rows))
            return True
        result = self._result(target)
        self.emit(ev.CandidateCompleted(candidate=target, attempt=attempt, status="ok",
                                        summary=ev.CandidateSummary(**result.summary.model_dump())))
        return True

    # ----- self-validation and ledger rows --------------------------------------------------------

    def _selfval(self, state: RunState) -> bool:
        progressed = False
        for name in state.unscored:
            table = {"status": "not_computed", "reason": "no sentinel is measured in this arm"}
            self.emit(ev.SelfvalComputed(candidate=name, selfval_sha=self.blobs.put_json(table)))
            progressed = True
        return progressed

    def _baseline_medians(self) -> dict[str, dict[str, float | None]]:
        raw = self._baseline_result().pool["raw"]
        return medians(raw, list(self.s.task.conditions), self.headline)

    def _baseline_bands(self) -> dict[str, dict[str, tuple[float, float]]]:
        return composition_bands(self._baseline_result().pool["raw"], list(self.s.task.conditions))

    def _rows(self, state: RunState) -> bool:
        progressed = False
        for name in state.unfrozen:
            candidate = state.candidates[name]
            if candidate.status == "ok" and candidate.selfval_sha is None:
                continue
            spec = self.blobs.get_json(candidate.recipe_sha)
            parent = candidate.parent
            parent_state = state.candidates.get(parent) if parent and parent != "initial" else None
            parent_row = (self.blobs.get_json(parent_state.ledger_row_sha)
                          if parent_state is not None and parent_state.ledger_row_sha else None)
            parent_recipe = (self.blobs.get_json(parent_state.recipe_sha)["recipe"] if parent_state is not None
                             else dict(self.s.recipe_ctx.initial_recipe))
            ok = candidate.status == "ok"
            failure = None
            if not ok:
                failure = self.blobs.get_json(candidate.completed.failure_sha) if candidate.completed.failure_sha \
                    else {"type": candidate.status, "message": ""}
            row = build_row(
                candidate={"id": name, "round": candidate.round, "slot": candidate.slot, "parent": parent,
                           "start": candidate.start, "repeat_of": candidate.repeat_of, "anchor": candidate.anchor,
                           "status": candidate.status},
                spec=spec, parent_recipe=parent_recipe, parent_row=parent_row,
                baseline_pool=self._baseline_medians(), baseline_bands=self._baseline_bands(),
                result=self._result(name).model_dump() if ok else None,
                failure=failure, conditions=list(self.s.task.conditions), headline=self.headline)
            tokens = count_tokens(render_row(visible_row(row)))
            self.emit(ev.LedgerRowFrozen(candidate=name, row_sha=self.blobs.put_json(row), tokens=tokens))
            progressed = True
        return progressed

    # ----- the critic -----------------------------------------------------------------------------

    def _critic(self, state: RunState) -> bool:
        if state.stop.accepted is not None or state.uningested or state.unscored or state.unfrozen:
            return False
        allocated = [r for r in state.rounds if r.status == "allocated"]
        if not allocated:
            return False
        rnd = allocated[-1]
        if rnd.critic_row_sha is not None or rnd.critic_unavailable:
            return False
        schema = critic_schema()
        members = [c for c in rnd.candidates if state.candidates[c].status != "forfeited"]
        tail = (f"## Review\nReview the candidates of round {rnd.index}: {', '.join(members)}. Raise concerns and "
                "request evidence; you advise and do not decide.")
        failure: Exception | None = None
        for _ in range(2):
            try:
                rendered = render_with_tail(state, self.blobs, arm=self.arm, header=self.s.header_text,
                                            instructions=CRITIC_SYSTEM, tail=tail,
                                            limit=self.s.control.llm.max_prompt_tokens, schema=schema, tools=())
                turn = run_critic(self.transport, instructions=rendered.instructions,
                                  input_items=rendered.input_items, schema=schema)
            except (BlindingViolation, PromptBudgetExceeded, RunTokenBudgetExceeded, LedgerCorruption):
                raise
            except Exception as exc:  # an advisory call must not stop the search
                failure = exc
                continue
            row = {"round": rnd.index, "concerns": list(turn.parsed["concerns"]),
                   "requested_evidence": list(turn.parsed["requested_evidence"]),
                   "rationale": turn.parsed["rationale"]}
            self.emit(ev.CriticAdvisory(
                round=rnd.index, row_sha=self.blobs.put_json(row), n_concerns=len(row["concerns"]),
                **_llm_fields(turn, rendered, self.blobs.put_json(schema), self.blobs.put_json(turn.parsed), None)))
            self._count(turn)
            return True
        self.emit(ev.CriticUnavailable(round=rnd.index, attempts=2,
                                       cause=f"{type(failure).__name__}: {failure}"[:200]))
        return True

    def _count(self, turn: AgentTurn) -> None:
        self.meter.add(turn.usage)

    # ----- decisions ------------------------------------------------------------------------------

    def _b_effective(self, state: RunState) -> int:
        proposals = self.s.control.proposals
        allocated = sum(1 for r in state.rounds if r.status == "allocated")
        fits = state.remaining_updates // self.s.recipe_ctx.steps_min
        if allocated >= proposals.rounds_max or fits < 1:
            return 0
        return min(state.B, fits)

    def _agent_count(self, round_index: int, b_effective: int) -> int:
        anchor = self.s.control.proposals.anchor_candidate and round_index == 1
        return b_effective - 1 if anchor else b_effective

    def _decide(self, state: RunState) -> bool:
        if state.stop.accepted is not None or state.baseline.status != "ingested":
            return False
        if state.uningested or state.unscored or state.unfrozen:
            return False
        if state.components_in_flight:
            return self._advance_components(state)
        pending = state.stop.pending
        if pending is not None:
            self.emit(ev.StopAccepted(round=pending.round, source=pending.source, outcome=pending.outcome,
                                      assignment=pending.assignment))
            return True
        for decision_id in state.decisions_awaiting_validation:
            self._validate_recorded(state, decision_id)
            return True
        for decision_id in state.decisions_awaiting_action:
            self._act(state, decision_id)
            return True
        rnd = state.open_round
        if rnd is None:
            if state.consecutive_skips > self.s.control.proposals.max_skipped_rounds:
                raise ControlError("too many consecutive rounds without a valid proposal")
            self.emit(ev.RoundOpened(round=state.next_round, b_effective=self._b_effective(state)))
            return True
        attempts = self.s.control.proposals.max_decision_attempts
        if rnd.attempts >= attempts:
            refused = state.stop.consecutive_rejections >= attempts
            accepted = self._accepted(state, rnd)
            if rnd.b_effective == 0 or (refused and state.rounds_done >= self.s.control.proposals.rounds_min):
                self._default_stop(state, rnd)
            elif accepted is not None:
                self._allocate(state, rnd, *accepted)
            else:
                self.emit(ev.RoundSkipped(round=rnd.index, attempts=rnd.attempts, reason_code="no_valid_decision"))
            return True
        self._ask(state, rnd)
        return True

    def _round_schema(self, state: RunState, rnd) -> dict[str, Any]:
        control, ctx = self.s.control, self.s.recipe_ctx
        synthesis = control.synthesis
        allow_components = (synthesis.enabled and rnd.b_effective > 0
                            and len(state.components) < synthesis.max_requests)
        primitives = self.s.primitives() if callable(self.s.primitives) else self.s.primitives
        return round_response_schema(
            conditions=list(self.s.task.conditions), reward_oracles=sorted(ctx.reward_eligible),
            aggregators=primitives.get("cross_oracle_aggregation", ()),
            utilities=primitives.get("per_oracle_utility", ()), penalties=primitives.get("per_sequence_penalty", ()),
            gates=primitives.get("hard_biological_gate", ()),
            max_candidates=max(self._agent_count(rnd.index, rnd.b_effective) - len(self._kept(state, rnd)), 0),
            stop_only=rnd.b_effective == 0,
            synthesis_roles=list(synthesis.roles) if allow_components else None,
            approved_references=list(self.s.approved_references), aggregator_form=ctx.initial_recipe.get("aggregator"))

    def _accepted(self, state: RunState, rnd, *, before: str | None = None) -> tuple[str, dict[str, Any]] | None:
        """The newest propose answer of this round before `before` (default: every answer), with what it accepted.

        It holds every candidate the round's refused answers accepted, the anchor included, or nothing when it accepted
        none.
        """
        recorded = rnd.decision_ids if before is None else rnd.decision_ids[:rnd.decision_ids.index(before)]
        for decision_id in reversed(recorded):
            decision = state.decisions[decision_id]
            if decision.kind == "propose":
                if decision.normalized_sha is None:
                    return None
                return decision_id, self.blobs.get_json(decision.normalized_sha)
        return None

    def _kept(self, state: RunState, rnd, *, before: str | None = None) -> list[dict[str, Any]]:
        """The agent's candidates that refused answers of this round already had accepted, so the retry asks only
        for the slots still open; when the attempts run out the round allocates them."""
        accepted = self._accepted(state, rnd, before=before)
        return [] if accepted is None else [entry for entry in accepted[1]["entries"]
                                            if not entry["resolved"]["anchor"]]

    def _ask(self, state: RunState, rnd) -> None:
        schema = self._round_schema(state, rnd)
        tools = retrieval_tools(list(self.s.task.conditions), self.headline)
        kept = self._kept(state, rnd)
        rendered = render(state, self.blobs, arm=self.arm, header=self.s.header_text,
                          instructions=self.s.instructions, control=self.s.control, success=self.s.success,
                          conditions=list(self.s.task.conditions), stop_only=rnd.b_effective == 0,
                          b_effective=self._agent_count(rnd.index, rnd.b_effective) - len(kept), schema=schema,
                          tools=tools, kept=[f"slot {entry['slot']} (parent {entry['resolved']['parent']}, "
                                             f"{entry['resolved']['start']}, {entry['resolved']['steps']} steps)"
                                             for entry in kept])
        retriever = Retriever(run_dir=self.s.run_dir, state=state, store=self.blobs, arm=self.arm,
                              definitions=self.s.definitions, visible_sources=self.visible,
                              conditions=list(self.s.task.conditions))
        try:
            turn = run_strategist(self.transport, instructions=rendered.instructions,
                                  input_items=rendered.input_items, schema=schema, tools=tools,
                                  execute_tool=retriever, max_tool_calls=self.s.control.llm.tool_calls_per_decision)
        except ControlError as exc:
            partial = getattr(exc, "partial_usage", None)
            if partial is not None:
                self.meter.add(partial)  # tokens a failed turn still cost count against the run's cap
            raise
        transcript_sha = self.blobs.put_json(list(turn.transcript)) if turn.transcript else None
        self.emit(ev.DecisionRecorded(
            decision_id=state.next_decision_id, round=rnd.index, kind=turn.parsed["kind"], attempt=rnd.attempts + 1,
            state_hash=state_hash(state),
            **_llm_fields(turn, rendered, self.blobs.put_json(schema), self.blobs.put_json(turn.parsed),
                          transcript_sha)))
        self._count(turn)

    # validation ---------------------------------------------------------------------------------

    def _problem(self, code: str, path: str = "", detail: str = "") -> ev.Problem:
        return ev.Problem(code=code, path=path, detail=detail[:300])

    def _parents(self, state: RunState) -> dict[str, dict[str, Any]]:
        parents = {"initial": rc.initial_parent_record(self.s.recipe_ctx)}
        for candidate in state.ordered_candidates():
            if candidate.status == "ok":
                end = next(r for r in self._result(candidate.id).checkpoints
                           if r.id == self._result(candidate.id).end_checkpoint_id)
                parents[candidate.id] = end.model_dump()
        return parents

    def _validate_recorded(self, state: RunState, decision_id: str) -> None:
        decision = state.decisions[decision_id]
        parsed = self.blobs.get_json(decision.response_sha)
        rnd = state.rounds[decision.round - 1]
        kind = parsed["kind"]
        if kind == "propose":
            ok, n_valid, problems, normalized = self._validate_propose(state, rnd, decision, parsed)
        elif kind == "stop":
            ok, n_valid, problems, normalized = self._validate_stop(state, rnd, parsed)
        else:
            ok, n_valid, problems, normalized = self._validate_component(state, rnd, parsed)
        self.emit(ev.DecisionValidated(
            decision_id=decision_id, ok=ok, n_valid=n_valid, problems=problems[:12],
            normalized_sha=None if normalized is None else self.blobs.put_json(normalized)))
        if kind == "stop" and not ok:
            self.emit(ev.StopRejected(round=rnd.index, decision_id=decision_id, source="agent", reasons=problems[:12]))

    def _validate_propose(self, state: RunState, rnd, decision, parsed):
        problems: list[ev.Problem] = []
        if rnd.b_effective == 0:
            return False, 0, [self._problem("propose_not_allowed", "kind", "no further candidates may be proposed")], None
        anchored = 1 if rnd.index == 1 and self.s.control.proposals.anchor_candidate else 0
        parents = self._parents(state)
        history: dict[str, str] = {}
        for candidate in state.ordered_candidates():
            if candidate.status != "forfeited" and candidate.fingerprint:
                history.setdefault(candidate.fingerprint, candidate.id)
        entries: list[dict[str, Any]] = []
        if anchored:
            anchor_resolved, anchor_problems = rc.resolve(
                rc.anchor_proposal(), self.s.recipe_ctx, parents, committed_updates=state.committed_updates,
                history=history, anchor=True)
            if anchor_problems:
                raise ControlError("the declared initial recipe does not resolve: "
                                   + "; ".join(f"{p.code} {p.detail}" for p in anchor_problems))
            entries.append({"slot": 1, "proposal": rc.anchor_proposal(), "resolved": _resolved_dict(anchor_resolved)})
        # Candidates accepted from this round's earlier refused answers occupy their slots and count as siblings.
        entries.extend(self._kept(state, rnd, before=decision.id))
        expected = self._agent_count(rnd.index, rnd.b_effective) - (len(entries) - anchored)
        resolved = [rc.Resolved(**entry["resolved"]) for entry in entries]
        proposals = list(parsed["candidates"])
        got = len(proposals)
        if got > expected:
            proposals = []  # more candidates than open slots: none is taken, so no slot number can overflow
        for index, proposal in enumerate(proposals):
            item, found = rc.resolve(proposal, self.s.recipe_ctx, parents, committed_updates=state.committed_updates,
                                     siblings=resolved, history=history)
            for problem in found:
                problems.append(self._problem(problem.code, f"candidates[{index}].{problem.path}".rstrip("."),
                                              problem.detail))
            if item is not None:
                resolved.append(item)
                entries.append({"slot": len(resolved), "proposal": proposal, "resolved": _resolved_dict(item)})
        if got != expected:
            problems.append(self._problem("candidate_count", "candidates",
                                          f"propose exactly {expected} candidates; got {got}"))
        last = rnd.attempts >= self.s.control.proposals.max_decision_attempts
        ok = not problems or (last and len(entries) >= 1)
        # A refused answer records what it got accepted, for the retry to build on.
        normalized = {"kind": "propose", "entries": entries} if ok or len(entries) > anchored else None
        return ok, len(entries), problems, normalized

    def _validate_stop(self, state: RunState, rnd, parsed):
        stop = parsed.get("stop")
        control = self.s.control
        if stop is None:
            return False, 0, [self._problem("stop_missing", "stop", "kind stop needs the stop object")], None
        problems: list[ev.Problem] = []
        if state.rounds_done < control.proposals.rounds_min:
            problems.append(self._problem(
                "stop_too_early", "kind", f"a stop is accepted after {control.proposals.rounds_min} completed rounds; "
                                          f"{state.rounds_done} are complete"))
        if state.committed_updates < control.stop.min_updates_fraction * state.header.budgets.total_updates:
            problems.append(self._problem("stop_underspent", "kind", "more of the update budget must be committed first"))
        evidence = evidence_of(state, self.blobs)
        conditions = list(self.s.task.conditions)
        assignment = {entry["condition"]: entry["candidate"] for entry in stop["assignment"]}
        if stop["outcome"] == "deliver":
            problems += self._lift(validate_assignment(
                assignment, evidence, conditions, require_qualified=control.stop.require_assignment_qualified))
        else:
            problems += self._lift(validate_no_qualified(evidence, conditions))
        ok = not problems
        normalized = {"kind": "stop", "outcome": "delivered" if stop["outcome"] == "deliver" else "no_qualified_candidate",
                      "assignment": assignment if stop["outcome"] == "deliver" else {},
                      "summary": stop["summary"]} if ok else None
        return ok, 0, problems, normalized

    def _validate_component(self, state: RunState, rnd, parsed):
        control = self.s.control
        component = parsed.get("component")
        problems = []
        if not control.synthesis.enabled:
            problems.append(self._problem("synthesis_off", "kind", "component synthesis is not available"))
        elif component is None:
            problems.append(self._problem("component_missing", "component", "kind develop_component needs the component"))
        elif len(state.components) >= control.synthesis.max_requests:
            problems.append(self._problem("component_budget", "component", "no component requests remain"))
        elif component["role"] not in control.synthesis.roles:
            problems.append(self._problem("component_role", "component.role", f"allowed: {list(control.synthesis.roles)}"))
        ok = not problems
        return ok, 1 if ok else 0, problems, ({"kind": "develop_component", "component": component} if ok else None)

    @staticmethod
    def _lift(problems):
        return list(problems)

    # actions ------------------------------------------------------------------------------------

    def _act(self, state: RunState, decision_id: str) -> None:
        decision = state.decisions[decision_id]
        normalized = self.blobs.get_json(decision.normalized_sha)
        rnd = state.rounds[decision.round - 1]
        if normalized["kind"] == "propose":
            self._allocate(state, rnd, decision_id, normalized)
        elif normalized["kind"] == "stop":
            summary_sha = self.blobs.put_json({"summary": normalized["summary"]})
            self.emit(ev.StopRequested(round=rnd.index, decision_id=decision_id, source="agent",
                                       outcome=normalized["outcome"], assignment=normalized["assignment"],
                                       summary_sha=summary_sha))
            self.emit(ev.StopAccepted(round=rnd.index, source="agent", outcome=normalized["outcome"],
                                      assignment=normalized["assignment"]))
        else:
            component = normalized["component"]
            request_sha = self.blobs.put_json(component)
            self.emit(ev.ComponentRequested(
                request_id=state.next_request_id, round=rnd.index, decision_id=decision_id, role=component["role"],
                name=component["name"], reference_candidate=component.get("reference_candidate"),
                request_sha=request_sha))

    def _allocate(self, state: RunState, rnd, decision_id: str, normalized: Mapping[str, Any]) -> None:
        candidates, used = [], set()
        for entry in normalized["entries"]:
            slot, item, proposal = entry["slot"], entry["resolved"], entry["proposal"]
            used.add(slot)
            spec_sha = self.blobs.put_json({
                "recipe": item["recipe"], "train": item["train"], "hypothesis": proposal["hypothesis"],
                "proposal": proposal, "fingerprint": item["fingerprint"]})
            ordinal = ordinal_of(rnd.index, slot, state.B)
            candidates.append(ev.AllocatedCandidate(
                id=candidate_id(ordinal), ordinal=ordinal, slot=slot,
                parent=None if item["parent"] == "initial" else item["parent"], start=item["start"],
                recipe_sha=spec_sha, fingerprint=item["fingerprint"], steps=item["steps"],
                repeat_of=item["repeat_of"], anchor=item["anchor"]))
        forfeited = [slot for slot in range(1, rnd.b_effective + 1) if slot not in used]
        self.emit(ev.CandidatesAllocated(round=rnd.index, decision_id=decision_id, candidates=candidates,
                                         forfeited_slots=forfeited))

    def _default_stop(self, state: RunState, rnd) -> None:
        evidence = evidence_of(state, self.blobs)
        best = best_assignment(self.s.success, evidence, list(self.s.task.conditions))
        outcome = "delivered" if best is not None else "no_qualified_candidate"
        assignment = best or {}
        self.emit(ev.StopRequested(round=rnd.index, decision_id=None, source="contract_default", outcome=outcome,
                                   assignment=assignment, summary_sha=None))
        self.emit(ev.StopAccepted(round=rnd.index, source="contract_default", outcome=outcome, assignment=assignment))

    def _advance_components(self, state: RunState) -> bool:
        if self.components is None:
            raise ControlError("a component is requested but no component runner is mounted")
        self.components.run(self, state, state.components_in_flight[0])
        return True

    # ----- delivery -------------------------------------------------------------------------------

    def _finish(self, state: RunState) -> bool:
        accepted = state.stop.accepted
        if accepted is None:
            return False
        if state.library is not None:
            self.emit(ev.RunFinished(status=state.library.status, reason_code=state.library.reason_code))
            return True
        run_dir = self.s.run_dir
        if accepted.outcome == "no_qualified_candidate":
            write_json(run_dir / "delivery.json", {"outcome": "no_qualified_candidate", "source": accepted.source,
                                                  "blind_audit": {"status": "not_run"}})
            self.emit(ev.LibraryFrozen(status="completed_no_qualified_candidate", library_sha=None,
                                       reason_code=None))
            return True
        if state.delivery is None:
            directories = {name: str(self.candidate_dir(name)) for name in set(accepted.assignment.values())}
            plan, health = self._delivery_rule()
            result = self.assemble(dict(accepted.assignment), directories, plan, run_dir, health=health)
            by_condition = result["by_condition"]
            pool_sha = file_sha256(run_dir / "delivery" / "pool.json")
            self.emit(ev.DeliveryAssembled(
                assignment=dict(accepted.assignment), pool_sha=pool_sha,
                plan_sha=sha256_hex(canonical_json(plan)),
                qualified={c: bool(info["qualified"]) for c, info in by_condition.items()},
                counts={c: int(info["selected"]) for c, info in by_condition.items()}))
            return True
        frozen = all(state.delivery.qualified.values()) and (run_dir / "final_sequences.parquet").exists()
        sha = file_sha256(run_dir / "final_sequences.parquet") if frozen else None
        self.emit(ev.LibraryFrozen(
            status="frozen" if frozen else "completed_no_qualified_candidate", library_sha=sha,
            reason_code=None if frozen else "delivery_unqualified"))
        return True


def _resolved_dict(item: rc.Resolved) -> dict[str, Any]:
    return {"parent": item.parent, "start": item.start, "steps": item.steps, "recipe": item.recipe,
            "train": item.train, "fingerprint": item.fingerprint, "repeat_of": item.repeat_of, "anchor": item.anchor}
