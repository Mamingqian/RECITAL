"""One job per process: `python -m recital.control.worker --spec <path>`.

A worker reads its `JobSpec`, builds the task runtime without any blind oracle,
mounts the run's published extensions without a Coder, does the job, and writes
`worker_result.json` last. A repeat run of the same spec completes from whatever the
earlier one left on disk. It takes no lock the controller holds and sets nothing about
API keys; it imports no agent or LLM code.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

import yaml

from recital.control.backend import (
    SequenceExperimentBackend, candidate_checkpoint_records,
)
from recital.control.jobs import (
    BASELINE_RESULT_NAME, CANDIDATE_RESULT_NAME, FAILURE_NAME, RESULT_NAME,
    BaselineResult, CandidateResult, CandidateSummary, JobFailure, JobResult, JobSpec,
    file_sha256, read_spec, write_json,
)
from recital.control.errors import ScoringFailed
from recital.control.runtime_build import build_runtime, check_worker_runtime, mount_synthesis

# Errors of the storage layer that a second attempt can be expected to get past.
TRANSIENT_ERRNOS = frozenset({errno.ESTALE, errno.ETIMEDOUT})
MESSAGE_LIMIT = 4000


def is_retryable(exc: BaseException) -> bool:
    """True when `exc`, or what it was raised from, is a known transient OS error or a scoring failure.

    A scoring failure is retried because only the scoring repeats: the pool and the training behind it are on disk.
    """
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, ScoringFailed) or (isinstance(exc, OSError) and exc.errno in TRANSIENT_ERRNOS):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def _series_value(evidence: dict, name: str, key: str):
    series = (evidence.get("training_progress") or {}).get("series") or {}
    return (series.get(name) or {}).get(key)


def _training_wall_seconds(directory: Path) -> float | None:
    """Trainer-call wall time summed over every invocation the job needed."""
    ledger = directory / "training_calls.jsonl"
    if not ledger.exists():
        return None
    calls = [json.loads(line) for line in ledger.read_text().splitlines()]
    values = [call.get("measurements", {}).get("total_elapsed_seconds", {}).get("value") for call in calls]
    return float(sum(values)) if values and all(value is not None for value in values) else None


def _candidate_result(spec: JobSpec, backend: SequenceExperimentBackend, receipt: dict,
                      directory: Path) -> CandidateResult:
    evaluation = json.loads((directory / "evaluation.json").read_text())
    records, end_id = candidate_checkpoint_records(spec, receipt)
    evidence = backend.training_evidence(directory)
    pool_sha, evaluation_sha = file_sha256(directory / "pool.json"), file_sha256(directory / "evaluation.json")
    delivery = evaluation["delivery"]
    summary = CandidateSummary(
        steps_done=int(receipt["additional_updates_completed"]),
        wall_s=_training_wall_seconds(directory),
        reward_first=_series_value(evidence, "reward", "first"),
        reward_last=_series_value(evidence, "reward", "last"),
        reward_std=evidence.get("reward_std"), frac_zero_std=evidence.get("frac_reward_zero_std"),
        kl_ref=evidence.get("reference_policy_kl"),
        pool_sha=pool_sha, evaluation_sha=evaluation_sha,
    )
    return CandidateResult(
        candidate_id=spec.candidate.id, action_id=spec.candidate.action_id, attempt=spec.attempt,
        end_checkpoint_id=end_id, checkpoints=records, receipt=receipt,
        cost={"actual_updates": backend.actual_training_updates(directory),
              "status": backend.training_cost_status(directory)},
        summary=summary, training_evidence=evidence, pool=evaluation["pool"],
        observed_sources=evaluation["observed_sources"], selection_sources=evaluation["selection_sources"],
        delivery=delivery, qualified=bool(delivery["qualified"]),
        by_condition=backend.by_condition(delivery),
        receipt_sha256=file_sha256(directory / "training_receipt.json"),
        pool_sha256=pool_sha, evaluation_sha256=evaluation_sha,
    )


def _baseline_result(spec: JobSpec, backend: SequenceExperimentBackend, directory: Path) -> BaselineResult:
    evaluation = json.loads((directory / "evaluation.json").read_text())
    delivery = evaluation.get("delivery")
    sample_path = backend.output / "initial_policy_sample.json"
    return BaselineResult(
        attempt=spec.attempt, pool=evaluation["pool"], observed_sources=evaluation["observed_sources"],
        selection_sources=evaluation["selection_sources"], delivery=delivery,
        by_condition=backend.by_condition(delivery) if delivery is not None else None,
        policy_sample_sha256=file_sha256(sample_path),
        policy_sample_count=len(json.loads(sample_path.read_text())["sequences"]),
        composition_reference_sources=backend.reward_eligible_sources(),
        n_rows=evaluation["pool"]["raw_count"],
        pool_sha256=file_sha256(directory / "pool.json"),
        evaluation_sha256=file_sha256(directory / "evaluation.json"),
    )


def run_job(spec: JobSpec) -> JobResult:
    """Do the job; write its result file; return the (ok) marker for the caller to write."""
    clock = time.monotonic
    started = clock()
    run_dir, directory = Path(spec.run_dir), Path(spec.output_dir)
    config = yaml.safe_load(Path(spec.config_path).read_text())
    runtime = build_runtime(config, repo_root=spec.repo_root, output=run_dir, device=spec.device,
                            audit_oracle_factories={})
    check_worker_runtime(runtime)
    synthesis = mount_synthesis(runtime, run_dir)
    timings = {"build_s": clock() - started}
    try:
        backend = SequenceExperimentBackend(runtime, output=run_dir, device=spec.device, synthesis=synthesis)
        if spec.kind == "baseline_pool":
            phase = clock()
            backend.baseline_pool(spec)
            timings["evaluate_s"] = clock() - phase
            write_json(directory / BASELINE_RESULT_NAME,
                       _baseline_result(spec, backend, directory).model_dump(mode="json"))
            receipt_sha = None
        else:
            phase = clock()
            receipt = backend.train_candidate(spec)
            timings["train_s"] = clock() - phase
            phase = clock()
            backend.evaluate_pool(spec)
            timings["evaluate_s"] = clock() - phase
            write_json(directory / CANDIDATE_RESULT_NAME,
                       _candidate_result(spec, backend, receipt, directory).model_dump(mode="json"))
            receipt_sha = file_sha256(directory / "training_receipt.json")
    finally:
        synthesis.catalog.close()
    timings["total_s"] = clock() - started
    return JobResult(
        kind=spec.kind, candidate_id=spec.candidate.id, attempt=spec.attempt, submit_token=spec.submit_token,
        status="ok", timings=timings, receipt_sha256=receipt_sha,
        pool_sha256=file_sha256(directory / "pool.json"), failure=None)


def _host_record(spec: JobSpec) -> dict:
    """Where this attempt ran, so an analysis can see which hardware a candidate trained on."""
    record = {"host": socket.gethostname(), "attempt": spec.attempt, "device": spec.device,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "partition": os.environ.get("SLURM_JOB_PARTITION")}
    if spec.device == "cuda":
        import torch

        record.update(gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)))
    return record


def run(spec: JobSpec) -> int:
    """Run `spec`, leaving `worker_result.json` as the last file written. Exit status 0 if ok."""
    directory = Path(spec.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / f"host_a{spec.attempt}.json", _host_record(spec))
    # A marker or a result left by an earlier attempt must not be read as this one's.
    for name in (RESULT_NAME, FAILURE_NAME, CANDIDATE_RESULT_NAME, BASELINE_RESULT_NAME):
        (directory / name).unlink(missing_ok=True)
    started = time.monotonic()
    try:
        result = run_job(spec)
    except Exception as exc:
        traceback.print_exc()
        failure = JobFailure(type=type(exc).__name__, message=str(exc)[:MESSAGE_LIMIT],
                             retryable=is_retryable(exc))
        write_json(directory / FAILURE_NAME, failure.model_dump(mode="json"))
        result = JobResult(
            kind=spec.kind, candidate_id=spec.candidate.id, attempt=spec.attempt, submit_token=spec.submit_token,
            status="failed", timings={"total_s": time.monotonic() - started}, receipt_sha256=None,
            pool_sha256=None, failure=failure)
        write_json(directory / RESULT_NAME, result.model_dump(mode="json"))
        return 1
    write_json(directory / RESULT_NAME, result.model_dump(mode="json"))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run one control-layer job")
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(argv)
    return run(read_spec(args.spec))


if __name__ == "__main__":
    sys.exit(main())
