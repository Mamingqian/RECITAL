"""Slurm as a `Scheduler`.

`submit` writes `sbatch.sh` beside the spec and runs `sbatch --parsable` from the spec's
`repo_root`. Slurm sets `SLURM_SUBMIT_DIR` from the submit directory and the job puts its `src`
on `PYTHONPATH`, so a job runs the source tree its spec names with the interpreter that runs the
controller. The job's
`--comment` is the submit token; `find_by_token` reads it back from `squeue` and, for a
job that already left the queue, from `sacct`, so a controller that died after `sbatch`
and before recording the job id finds the job again.

`poll` asks `squeue -j` first and `sacct -X -j` for a job that is no longer live:

    squeue PENDING, CONFIGURING                       PENDING
    squeue RUNNING, COMPLETING, SUSPENDED             RUNNING
    sacct  COMPLETED with exit 0:0                    COMPLETED
    sacct  COMPLETED with any other exit or signal    FAILED   reason "exit <code>:<signal>"
    sacct  FAILED NODE_FAIL BOOT_FAIL OUT_OF_MEMORY
           DEADLINE PREEMPTED REVOKED                 FAILED   reason = the Slurm state
    sacct  TIMEOUT                                    TIMEOUT
    sacct  CANCELLED, "CANCELLED by <uid>"            CANCELLED
    no row, an unrecognised state                     UNKNOWN, then FAILED (below)
    a failed or timed out squeue/sacct call           UNKNOWN

`sacct` also reports PENDING and RUNNING jobs and maps them as `squeue` does. A job that
`squeue` still lists in an ended state is read from `sacct`; if accounting has no row for
it yet the answer is UNKNOWN and is not counted, because the job is not missing.

A job in neither `squeue` nor `sacct`, or in a state this table does not know, is UNKNOWN
for `lag_polls` consecutive polls, since accounting lags a job's end, and FAILED
("vanished", or "unrecognized state X") from the next poll on. A failed or timed out
command says nothing about the job: it returns UNKNOWN and neither counts nor resets. Any
definite answer resets the count. The counts live in this instance, so a restarted
controller starts every handle at zero.

`find_by_token` never answers None when Slurm did not answer: it raises
`SchedulerUnreachable`, because None means "no job holds this token" and the engine would
submit a second one. A token names one job for good, so a found handle is remembered and
later lookups of it ask Slurm nothing. A handle found in `squeue` or `sacct` carries the
spec path of the `sbatch.sh` under `roots` whose `--comment` is the token, or "" when
`roots` holds none.

`sbatch` retries a transient controller error `retries` times after 2 s, 4 s,
8 s, ... and then raises `SchedulerUnreachable`. A transient `sbatch` failure can follow a
job the controller accepted, so every retry first looks the token up and returns that job
if it exists. Any other `sbatch` error (an unknown partition, an invalid account) raises
`JobInfraError` at once with Slurm's message. Commands run without any `*_API_KEY`
variable, which `sbatch` would otherwise copy into the job; the worker needs none.
"""
from __future__ import annotations

import getpass
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Callable, Iterator, Sequence

from recital.control.errors import ControlError, JobInfraError, SchedulerUnreachable
from recital.control.jobs import TOKEN, Resources, read_spec
from recital.control.jsonio import atomic_write
from recital.control.scheduler import JobHandle, JobState, JobStatus

Runner = Callable[..., tuple[int, str, str]]

SCRIPT_NAME = "sbatch.sh"
QUERY_TIMEOUT = 60
BACKOFF_SECONDS = 2
JOB_NAME_TAIL = 32
ACCOUNTING_WINDOW = "now-14days"
# What a worker needs fixed for a reproducible run: set iteration order, no user site packages, no TensorFlow or Flax
# backends pulled in by transformers.
WORKER_ENVIRONMENT = (("PYTHONHASHSEED", "0"), ("PYTHONNOUSERSITE", "1"), ("TOKENIZERS_PARALLELISM", "false"),
                      ("TRANSFORMERS_NO_TF", "1"), ("USE_TF", "0"), ("USE_FLAX", "0"))

_PENDING = frozenset({"PENDING", "CONFIGURING"})
_RUNNING = frozenset({"RUNNING", "COMPLETING", "SUSPENDED"})
_FAILED = frozenset({"FAILED", "NODE_FAIL", "BOOT_FAIL", "OUT_OF_MEMORY", "DEADLINE", "PREEMPTED", "REVOKED"})
_NO_SUCH_JOB = re.compile(r"invalid job id", re.IGNORECASE)
_TRANSIENT = re.compile(
    r"timed out|unable to contact|temporarily|connection (?:refused|reset)|not responding|"
    r"zero bytes were transmitted|persistent connection|transport endpoint|try again",
    re.IGNORECASE)
_EXIT_CODE = re.compile(r"^([0-9]+):([0-9]+)$")


def run_command(argv: Sequence[str], cwd: str | Path | None = None, timeout: float = QUERY_TIMEOUT
                ) -> tuple[int, str, str]:
    """Run a Slurm command without `*_API_KEY` variables; a timeout is exit code 124."""
    env = {name: value for name, value in os.environ.items() if not name.endswith("_API_KEY")}
    try:
        done = subprocess.run(list(argv), cwd=cwd, env=env, stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "", f"{argv[0]} timed out after {timeout}s"
    return done.returncode, done.stdout, done.stderr


def render_script(spec_path: Path, resources: Resources, token: str) -> str:
    directives = [
        f"--job-name=recital-{token[-JOB_NAME_TAIL:]}",
        f"--partition={resources.partition}",
        *([f"--gres={resources.gres}"] if resources.gres else []),
        f"--cpus-per-task={resources.cpus}",
        f"--mem={resources.mem}",
        f"--time={resources.time_limit}",
        "--no-requeue",
        f"--comment={token}",
        f"--output={spec_path.parent}/slurm-%j.out",
    ]
    body = [
        "set -euo pipefail",
        'cd "${SLURM_SUBMIT_DIR}"',
        *(f"export {name}={value}" for name, value in WORKER_ENVIRONMENT),
        'export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"',
        f"exec {shlex.quote(sys.executable)} -u -m recital.control.worker --spec {shlex.quote(str(spec_path))}",
    ]
    return "\n".join(["#!/usr/bin/env bash", *(f"#SBATCH {line}" for line in directives), "", *body, ""])


def _first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def _failure(command: str, code: int, err: str) -> str:
    return f"{command} exit {code}: {_first_line(err)}"


def _live(state: str | None) -> JobState | None:
    if state in _PENDING:
        return JobState.PENDING
    if state in _RUNNING:
        return JobState.RUNNING
    return None


def _from_accounting(row: str) -> JobStatus:
    """The status a `State|ExitCode` row of `sacct -P` means."""
    fields = row.split("|")
    words = fields[0].split()
    state = words[0].rstrip("+").upper() if words else ""
    live = _live(state)
    if live is not None:
        return JobStatus(live)
    match = _EXIT_CODE.match(fields[1].strip()) if len(fields) > 1 else None
    code, signal = (int(match.group(1)), int(match.group(2))) if match else (None, None)
    if state == "COMPLETED" and code is not None:
        if (code, signal) == (0, 0):
            return JobStatus(JobState.COMPLETED, exit_code=0)
        return JobStatus(JobState.FAILED, exit_code=code, signal=signal or None, reason=f"exit {code}:{signal}")
    if state in _FAILED:
        return JobStatus(JobState.FAILED, exit_code=code, signal=signal or None, reason=state)
    if state == "TIMEOUT":
        return JobStatus(JobState.TIMEOUT, exit_code=code, signal=signal or None, reason=state)
    if state == "CANCELLED":
        return JobStatus(JobState.CANCELLED, exit_code=code, signal=signal or None, reason=state)
    return JobStatus(JobState.UNKNOWN, reason=f"unrecognized state {row!r}")


def _job_id(listing: str, token: str) -> str | None:
    """The id in an `id|comment` listing whose comment is exactly `token`; array and step ids never match."""
    for line in listing.splitlines():
        job_id, _, comment = line.partition("|")
        if comment.strip() == token and job_id.strip().isdigit():
            return job_id.strip()
    return None


class SlurmScheduler:
    """Runs each job as a Slurm batch job; see the module docstring for the contract."""

    name = "slurm"

    def __init__(self, roots: Path | str | Sequence[Path | str] | None = None, *, user: str | None = None,
                 lag_polls: int = 5, retries: int = 3, runner: Runner | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        paths = [] if roots is None else [roots] if isinstance(roots, (str, Path)) else list(roots)
        self.roots = [Path(root) for root in paths]
        self.user = user or getpass.getuser()
        self.lag_polls, self.retries = lag_polls, retries
        self.runner: Runner = runner or run_command
        self.sleep = sleep
        self._handles: dict[str, JobHandle] = {}
        self._unseen: dict[str, int] = {}

    # ----- submission -----------------------------------------------------------------------------

    def submit(self, spec_path: Path, resources: Resources, token: str) -> JobHandle:
        spec_path = Path(spec_path).resolve()
        if not TOKEN.match(token):
            raise ControlError(f"{token!r} is not a valid submit token")
        spec = read_spec(spec_path)
        if spec.submit_token != token:
            raise ControlError(f"spec {spec_path} was written for token {spec.submit_token}, not {token}")
        if self.find_by_token(token) is not None:
            raise ControlError(f"token {token} was already submitted")
        script = spec_path.with_name(SCRIPT_NAME)
        atomic_write(script, render_script(spec_path, resources, token).encode("utf-8"))
        handle = JobHandle("slurm", self._sbatch(script, token, spec.repo_root), token, str(spec_path))
        self._handles[token] = handle
        return handle

    def _attempts(self) -> Iterator[int]:
        """0 to `retries`, sleeping 2 s, 4 s, 8 s, ... before every attempt after the first."""
        for attempt in range(self.retries + 1):
            if attempt:
                self.sleep(BACKOFF_SECONDS * 2 ** (attempt - 1))
            yield attempt

    def _sbatch(self, script: Path, token: str, cwd: str) -> str:
        failure = ""
        for attempt in self._attempts():
            if attempt:
                try:
                    held = self.find_by_token(token)
                except SchedulerUnreachable as exc:
                    failure = str(exc)
                    continue
                if held is not None:
                    return held.job_id
            code, out, err = self.runner(["sbatch", "--parsable", str(script)], cwd=cwd, timeout=QUERY_TIMEOUT)
            if code == 0:
                job_id = _first_line(out).split(";")[0]
                if not job_id.isdigit():
                    raise JobInfraError(f"sbatch printed {out!r} for {script}, not a job id")
                return job_id
            if not _TRANSIENT.search(err):
                raise JobInfraError(f"sbatch refused {script}: {err.strip()}")
            failure = _failure("sbatch", code, err)
        raise SchedulerUnreachable(f"sbatch failed {self.retries + 1} times; last: {failure}")

    # ----- lookup ---------------------------------------------------------------------------------

    def find_by_token(self, token: str) -> JobHandle | None:
        if token in self._handles:
            return self._handles[token]
        job_id = _job_id(self._query(["squeue", "-u", self.user, "-h", "-o", "%i|%k"]), token)
        if job_id is None:
            job_id = _job_id(self._query(["sacct", "-X", "-n", "-P", "-u", self.user, "-S", ACCOUNTING_WINDOW,
                                          "--format=JobID,Comment"]), token)
        if job_id is None:
            return None
        handle = JobHandle("slurm", job_id, token, self._spec_for(token))
        self._handles[token] = handle
        return handle

    def _query(self, argv: list[str]) -> str:
        code, out, err = self.runner(argv, timeout=QUERY_TIMEOUT)
        if code != 0:
            raise SchedulerUnreachable(_failure(argv[0], code, err))
        return out

    def _spec_for(self, token: str) -> str:
        marker = f"#SBATCH --comment={token}\n"
        for root in self.roots:
            for pattern in (SCRIPT_NAME, f"*/{SCRIPT_NAME}", f"*/*/{SCRIPT_NAME}"):
                for script in sorted(root.glob(pattern)):
                    if marker in script.read_text():
                        return str(script.with_name("spec.json"))
        return ""

    # ----- status ---------------------------------------------------------------------------------

    def poll(self, handle: JobHandle) -> JobStatus:
        job_id = handle.job_id
        code, out, err = self.runner(["squeue", "-j", job_id, "-h", "-o", "%T"], timeout=QUERY_TIMEOUT)
        if code == 0:
            listed = _first_line(out).upper() or None
        elif _NO_SUCH_JOB.search(err):
            listed = None
        else:
            return JobStatus(JobState.UNKNOWN, reason=_failure("squeue", code, err))
        live = _live(listed)
        if live is not None:
            self._unseen.pop(job_id, None)
            return JobStatus(live)
        code, out, err = self.runner(["sacct", "-X", "-n", "-P", "-j", job_id, "--format=State,ExitCode"],
                                     timeout=QUERY_TIMEOUT)
        if code != 0:
            return JobStatus(JobState.UNKNOWN, reason=_failure("sacct", code, err))
        row = _first_line(out)
        if not row:
            if listed is not None:
                return JobStatus(JobState.UNKNOWN, reason=f"squeue lists {listed}; sacct has no record yet")
            return self._unseen_status(job_id, "vanished")
        status = _from_accounting(row)
        if status.state is JobState.UNKNOWN:
            return self._unseen_status(job_id, status.reason)
        self._unseen.pop(job_id, None)
        return status

    def _unseen_status(self, job_id: str, reason: str | None) -> JobStatus:
        count = self._unseen.get(job_id, 0) + 1
        self._unseen[job_id] = count
        return JobStatus(JobState.FAILED if count > self.lag_polls else JobState.UNKNOWN, reason=reason)
