"""The files a controller and its workers exchange, and nothing else they share.

A controller writes one `JobSpec` per job and hands its path to a scheduler. The
worker reads it, writes its artifacts under `output_dir`, and writes
`worker_result.json` last. `JobResult` is that marker; `CandidateResult` and
`BaselineResult` are the summaries the controller ingests when it is `ok`.

Every model is `extra="forbid"` and carries `schema_version`, so a file written by
another version is refused instead of read as if it meant something else.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Any, Literal

import jsonschema
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from recital.actions import GENERATION_SCHEMA
from recital.control.health import HealthRule
from recital.control.jsonio import atomic_write, canonical_json, sha256_hex
from recital.dna_session import json_safe

SCHEMA_VERSION = 1

RESULT_NAME = "worker_result.json"
FAILURE_NAME = "failure.json"
CANDIDATE_RESULT_NAME = "candidate_result.json"
BASELINE_RESULT_NAME = "baseline_result.json"

BASELINE_ACTION_ID = "baseline_000000"
INITIAL_CHECKPOINT = "initial"

JobKind = Literal["baseline_pool", "train_candidate"]
Positive = Annotated[int, Field(ge=1)]
Count = Annotated[int, Field(ge=0)]
Sha = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

CANDIDATE_ID = re.compile(r"^cand_[0-9]{6}$")
# `<prefix>_<int>`: the integer is the ordinal `_seed` derives every seed from, and the
# whole string is a filename-safe training action identifier.
ACTION_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_]*_[0-9]+$")
# One scheduler comment: Slurm accepts this alphabet in `--comment`.
TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
TIME_LIMIT = re.compile(r"^(?:([0-9]+)-)?([0-9]{1,3}):([0-5][0-9]):([0-5][0-9])$")
MEMORY = re.compile(r"^[0-9]+[KMGT]$")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def action_ordinal(action_id: str) -> int:
    """The integer after the last underscore, which is what `_seed` parses."""
    return int(action_id.rsplit("_", 1)[-1])


class Resources(_Model):
    """What a job asks of the cluster. A local scheduler honours `time_limit` only."""

    partition: Annotated[str, StringConstraints(min_length=1)]
    gres: str | None
    cpus: Positive
    mem: str
    time_limit: str

    @model_validator(mode="after")
    def _well_formed(self) -> "Resources":
        if not MEMORY.match(self.mem):
            raise ValueError("mem must look like 24G")
        if not TIME_LIMIT.match(self.time_limit):
            raise ValueError("time_limit must be [D-]HH:MM:SS")
        if self.time_limit_seconds() < 1:
            raise ValueError("time_limit must be positive")
        return self

    def time_limit_seconds(self) -> int:
        days, hours, minutes, seconds = TIME_LIMIT.match(self.time_limit).groups()
        return ((int(days or 0) * 24 + int(hours)) * 60 + int(minutes)) * 60 + int(seconds)


class Candidate(_Model):
    """Which job this is. `seed` is the run's contract seed.

    Training and evaluation seeds are derived from `(seed, ordinal)` by the backend,
    never stored here, so a spec cannot carry a stream the backend would not
    reproduce. The worker refuses a spec whose `seed` is not the config's.
    """

    id: str
    ordinal: Count
    round: Count
    slot: Count
    action_id: str
    seed: Count


class CheckpointRecord(_Model):
    """A checkpoint as the controller registers it (`Experiment._register_training`).

    The initial checkpoint has no `schedule`, `kl_reference` or `training_summary`.
    """

    id: str
    path: str
    parent: str | None
    action_id: str | None
    step: Count
    lineage_updates: Count
    full_state_available: bool
    weights_available: bool
    recipe: dict[str, Any]
    schedule: dict[str, Any] | None = None
    kl_reference: str | None = None
    training_summary: dict[str, Any] | None = None
    measurement_refs: list[str] = Field(default_factory=list)


class TrainParameters(_Model):
    """Exactly what `materialize_action` hands the backend for a `train` action."""

    recipe: dict[str, Any]
    start_mode: Literal["initial_restart", "weights_new_stage", "full_state_continuation"]
    start_checkpoint: str
    additional_updates: Positive
    optimizer_policy: Literal["retain", "reset"]
    schedule_policy: dict[str, Any]
    reward_state_policy: Literal["reset_all", "inherit_all", "inherit_compatible"]
    randomness_policy: Literal["inherit_stream", "new_stream"]
    checkpoint_steps: list[Positive]
    kl_reference: str

    @model_validator(mode="after")
    def _consistent(self) -> "TrainParameters":
        steps = self.checkpoint_steps
        if steps != sorted(set(steps)) or steps[-1:] != [self.additional_updates]:
            raise ValueError("checkpoint_steps must increase and end at additional_updates")
        if self.start_mode == "initial_restart" and self.start_checkpoint != INITIAL_CHECKPOINT:
            raise ValueError("initial_restart must start from the initial checkpoint")
        return self


GENERATION_KEYS = ("count", "count_scope", "temperature", "top_p", "batch_size", "condition_counts")


class Evaluation(_Model):
    """The resolved evaluation of one pool.

    The selection plan is the contract's with its diversity floor derived, and `health` is the reference that floor
    and the library-health requirement come from. Both are absent for the baseline pool, which is what they are
    measured on, and present for every other pool.
    """

    generation_plan: dict[str, Any]
    scorers: list[str]
    selection_plan: dict[str, Any] | None
    health: HealthRule | None

    @model_validator(mode="after")
    def _resolved(self) -> "Evaluation":
        plan = self.generation_plan
        missing = [key for key in GENERATION_KEYS if key not in plan]
        if missing:
            raise ValueError(f"generation_plan is not resolved; it lacks {missing}")
        try:
            jsonschema.Draft202012Validator(GENERATION_SCHEMA).validate(plan)
        except jsonschema.ValidationError as exc:
            raise ValueError(f"generation_plan is invalid: {exc.message}") from exc
        if (self.selection_plan is None) != (self.health is None):
            raise ValueError("a selection plan and its library-health rule come together or not at all")
        if not self.scorers or len(set(self.scorers)) != len(self.scorers):
            raise ValueError("scorers must be nonempty and unique")
        return self


def _absolute(value: str, name: str) -> None:
    if not Path(value).is_absolute():
        raise ValueError(f"{name} must be an absolute path")


class JobSpec(_Model):
    schema_version: Literal[1] = SCHEMA_VERSION
    kind: JobKind
    run_dir: str
    config_path: str
    repo_root: str
    output_dir: str
    device: Literal["cpu", "cuda"]
    attempt: Positive
    submit_token: str
    candidate: Candidate
    train: TrainParameters | None
    parent_checkpoints: dict[str, CheckpointRecord]
    evaluation: Evaluation
    resources: Resources

    @model_validator(mode="after")
    def _consistent(self) -> "JobSpec":
        for name in ("run_dir", "config_path", "repo_root", "output_dir"):
            _absolute(getattr(self, name), name)
        run = Path(self.run_dir)
        if run not in Path(self.output_dir).parents:
            raise ValueError("output_dir must lie inside run_dir")
        if not TOKEN.match(self.submit_token):
            raise ValueError("submit_token must be 1-128 of [A-Za-z0-9._:-]")
        for key, record in self.parent_checkpoints.items():
            if record.id != key:
                raise ValueError(f"checkpoint record {record.id!r} is keyed as {key!r}")
        if INITIAL_CHECKPOINT not in self.parent_checkpoints:
            raise ValueError("parent_checkpoints must include the initial checkpoint")
        candidate = self.candidate
        if not ACTION_ID.match(candidate.action_id):
            raise ValueError("action_id must look like <prefix>_<int>")
        if self.kind == "baseline_pool":
            self._baseline_consistent()
        else:
            self._candidate_consistent()
        return self

    def _baseline_consistent(self) -> None:
        candidate = self.candidate
        if (candidate.id, candidate.action_id) != (BASELINE_ACTION_ID, BASELINE_ACTION_ID) or (
                candidate.ordinal, candidate.round, candidate.slot) != (0, 0, 0):
            raise ValueError(f"a baseline job is {BASELINE_ACTION_ID} with ordinal, round and slot 0")
        if self.train is not None:
            raise ValueError("a baseline job trains nothing")
        if set(self.parent_checkpoints) != {INITIAL_CHECKPOINT}:
            raise ValueError("a baseline job reads only the initial checkpoint")
        if self.evaluation.selection_plan is not None:
            raise ValueError("a baseline job selects no library: its pool is what the selection plan is derived from")

    def _candidate_consistent(self) -> None:
        candidate, train = self.candidate, self.train
        if train is None:
            raise ValueError("a train_candidate job carries its training parameters")
        if not CANDIDATE_ID.match(candidate.id):
            raise ValueError("candidate id must look like cand_000001")
        if candidate.ordinal < 1 or candidate.round < 1 or candidate.slot < 1:
            raise ValueError("ordinal, round and slot start at 1")
        if action_ordinal(candidate.id) != candidate.ordinal or action_ordinal(candidate.action_id) != candidate.ordinal:
            raise ValueError("candidate id and action_id must end in the ordinal")
        if self.evaluation.selection_plan is None:
            raise ValueError("every candidate is evaluated under the contract's selection plan")
        needed = {INITIAL_CHECKPOINT, train.start_checkpoint, train.kl_reference}
        absent = sorted(needed - set(self.parent_checkpoints))
        if absent:
            raise ValueError(f"parent_checkpoints lacks {absent}")
        extra = sorted(set(self.parent_checkpoints) - needed)
        if extra:
            raise ValueError(f"parent_checkpoints carries checkpoints the job does not use: {extra}")

    def digest(self) -> str:
        """sha256 of the canonical content; formatting of the file does not matter."""
        return sha256_hex(canonical_json(self.model_dump(mode="json")))


def write_json(path: str | Path, value: Any) -> None:
    """The artifact format of the run: sorted keys, indent 2, ASCII, trailing newline.

    Numpy values become Python ones and a non-finite float becomes null.
    """
    text = json.dumps(json_safe(value), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    atomic_write(path, text.encode("utf-8"))


def write_spec(path: str | Path, spec: JobSpec) -> None:
    write_json(path, spec.model_dump(mode="json"))


def read_spec(path: str | Path) -> JobSpec:
    return JobSpec.model_validate(json.loads(Path(path).read_text()))


class JobFailure(_Model):
    type: Annotated[str, StringConstraints(min_length=1)]
    message: str
    retryable: bool


class JobResult(_Model):
    """The completion marker, written last. `ok` means the result file is complete."""

    schema_version: Literal[1] = SCHEMA_VERSION
    kind: JobKind
    candidate_id: str
    attempt: Positive
    submit_token: str
    status: Literal["ok", "failed"]
    timings: dict[str, float]
    receipt_sha256: Sha | None
    pool_sha256: Sha | None
    failure: JobFailure | None

    @model_validator(mode="after")
    def _status_matches(self) -> "JobResult":
        if self.status == "failed":
            if self.failure is None:
                raise ValueError("a failed result names its failure")
            return self
        if self.failure is not None:
            raise ValueError("an ok result has no failure")
        if self.pool_sha256 is None:
            raise ValueError("an ok result names its pool")
        if (self.kind == "train_candidate") != (self.receipt_sha256 is not None):
            raise ValueError("an ok train_candidate result names its receipt, and a baseline has none")
        return self


class ConditionQualification(_Model):
    """One condition's share of the library's qualification."""

    qualified: bool
    count_met: bool
    selected: Count
    failing_requirements: list[str]


class CandidateSummary(_Model):
    """The fields of `events.CandidateSummary` a successful job can fill."""

    steps_done: Count | None
    wall_s: Annotated[float, Field(ge=0)] | None
    reward_first: float | None
    reward_last: float | None
    reward_std: float | None
    frac_zero_std: float | None
    kl_ref: float | None
    pool_sha: Sha
    evaluation_sha: Sha


class Cost(_Model):
    actual_updates: Count
    status: Literal["measured", "lower_bound"]


class CandidateResult(_Model):
    """`candidate_result.json`: everything the controller registers for one candidate."""

    schema_version: Literal[1] = SCHEMA_VERSION
    candidate_id: str
    action_id: str
    attempt: Positive
    end_checkpoint_id: str
    checkpoints: list[CheckpointRecord]
    receipt: dict[str, Any]
    cost: Cost
    summary: CandidateSummary
    training_evidence: dict[str, Any]
    pool: dict[str, Any]
    observed_sources: list[str]
    selection_sources: list[str]
    delivery: dict[str, Any]
    qualified: bool
    by_condition: dict[str, ConditionQualification]
    receipt_sha256: Sha
    pool_sha256: Sha
    evaluation_sha256: Sha


class BaselineResult(_Model):
    """`baseline_result.json`: the initial-policy pool and the reference samples."""

    schema_version: Literal[1] = SCHEMA_VERSION
    attempt: Positive
    pool: dict[str, Any]
    observed_sources: list[str]
    selection_sources: list[str]
    delivery: dict[str, Any] | None
    by_condition: dict[str, ConditionQualification] | None
    policy_sample_sha256: Sha
    policy_sample_count: Count
    composition_reference_sources: list[str]
    n_rows: Count
    pool_sha256: Sha
    evaluation_sha256: Sha


def file_sha256(path: str | Path) -> str:
    return sha256_hex(Path(path).read_bytes())


def read_result(directory: str | Path) -> JobResult | None:
    """The completion marker, or None while the job has not finished."""
    path = Path(directory) / RESULT_NAME
    if not path.exists():
        return None
    return JobResult.model_validate(json.loads(path.read_text()))


def read_failure(directory: str | Path) -> JobFailure | None:
    path = Path(directory) / FAILURE_NAME
    if not path.exists():
        return None
    return JobFailure.model_validate(json.loads(path.read_text()))
