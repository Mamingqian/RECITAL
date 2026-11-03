"""Where jobs run.

A `Scheduler` starts one worker per job spec and reports on it. The contract is
written so a controller that dies between deciding to submit and recording the
submission can find the job again by its token: a Slurm implementation passes the
token as `sbatch --comment=<token>` and answers `find_by_token` from
`squeue -u $USER -o "%i|%k"` (and `sacct` for a job that already left the queue).

The scheduler does not queue. How many jobs run at once, and in what order they are
submitted, is the engine's decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Mapping, Protocol, Sequence, runtime_checkable

from recital.control.errors import ControlError
from recital.control.jobs import TOKEN, Resources, write_json


class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class JobHandle:
    """Enough to find a job again: the scheduler's id, the token it was submitted under."""

    scheduler: str
    job_id: str
    token: str
    spec_path: str


@dataclass(frozen=True)
class JobStatus:
    """A job's state and, once it has ended, how.

    `reason` is the scheduler's own word for the state (a Slurm state such as
    NODE_FAIL or OUT_OF_MEMORY); `exit_code` and `signal` come from the process.
    `UNKNOWN` means the scheduler cannot say, which is not a failure: Slurm
    accounting lags a job's end.
    """

    state: JobState
    exit_code: int | None = None
    signal: int | None = None
    reason: str | None = None


@runtime_checkable
class Scheduler(Protocol):
    def submit(self, spec_path: Path, resources: Resources, token: str) -> JobHandle:
        """Start `python -m recital.control.worker --spec <spec_path>` under `token`.

        Raises if `token` was already submitted: a token names one job.
        """

    def poll(self, handle: JobHandle) -> JobStatus: ...

    def find_by_token(self, token: str) -> JobHandle | None:
        """The job submitted under `token`, or None if the scheduler holds none."""


_WRAPPER = """
exit_file=$1; limit=$2; shift 2
timeout --foreground -k 10 "$limit" "$@"
code=$?
printf '%s\\n' "$code" > "$exit_file.tmp" && mv "$exit_file.tmp" "$exit_file"
"""
_TIMEOUT_EXIT = 124


def _process_start(pid: int) -> tuple[str, str] | None:
    """(state, start time in clock ticks since boot) of a live process, else None."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields = text[text.rindex(")") + 2:].split()
    return fields[0], fields[19]


class LocalScheduler:
    """Runs each job as a detached process on this machine.

    The job is `<python> -m <module> --spec <path>` wrapped in a shell that applies
    `Resources.time_limit` with `timeout` and records the exit code in
    `<spec stem>.local.exit`. Beside the spec it keeps `<spec stem>.local.json` (pid,
    start time, token) and `<spec stem>.local.log` (stdout and stderr), so a new
    controller process finds and polls a job a dead one started. A spec sits at most
    two directories below a root (`<root>/<job>/<attempt>/spec.json`).
    """

    name = "local"

    def __init__(self, roots: Path | str | Sequence[Path | str], *, python: str = sys.executable,
                 module: str = "recital.control.worker", env: Mapping[str, str] | None = None):
        paths = [roots] if isinstance(roots, (str, Path)) else list(roots)
        self.roots = [Path(root) for root in paths]
        self.python, self.module = python, module
        self.env = dict(os.environ if env is None else env)
        self._children: dict[str, subprocess.Popen] = {}

    @staticmethod
    def _record_path(spec_path: Path) -> Path:
        return spec_path.with_name(f"{spec_path.stem}.local.json")

    @staticmethod
    def _exit_path(spec_path: Path) -> Path:
        return spec_path.with_name(f"{spec_path.stem}.local.exit")

    def _records(self):
        for root in self.roots:
            for pattern in ("*.local.json", "*/*.local.json", "*/*/*.local.json"):
                yield from sorted(root.glob(pattern))

    def submit(self, spec_path: Path, resources: Resources, token: str) -> JobHandle:
        spec_path = Path(spec_path).resolve()
        if not TOKEN.match(token):
            raise ControlError(f"{token!r} is not a valid submit token")
        if not spec_path.is_file():
            raise ControlError(f"no spec at {spec_path}")
        if self.find_by_token(token) is not None:
            raise ControlError(f"token {token} was already submitted")
        exit_path = self._exit_path(spec_path)
        exit_path.unlink(missing_ok=True)
        command = ["/bin/sh", "-c", _WRAPPER, "recital-job", str(exit_path), str(resources.time_limit_seconds()),
                   self.python, "-m", self.module, "--spec", str(spec_path)]
        with spec_path.with_name(f"{spec_path.stem}.local.log").open("ab") as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     env=self.env, start_new_session=True)
        started = _process_start(child.pid)
        write_json(self._record_path(spec_path), {
            "token": token, "pid": child.pid, "start_time": started[1] if started else None,
            "spec_path": str(spec_path),
        })
        self._children[token] = child
        return JobHandle("local", str(child.pid), token, str(spec_path))

    def find_by_token(self, token: str) -> JobHandle | None:
        for path in self._records():
            record = json.loads(path.read_text())
            if record["token"] == token:
                return JobHandle("local", str(record["pid"]), token, record["spec_path"])
        return None

    def _alive(self, record: dict) -> bool:
        child = self._children.get(record["token"])
        if child is not None:
            return child.poll() is None
        found = _process_start(record["pid"])
        return found is not None and found[0] not in ("Z", "X") and found[1] == record["start_time"]

    def poll(self, handle: JobHandle) -> JobStatus:
        spec_path = Path(handle.spec_path)
        try:
            record = json.loads(self._record_path(spec_path).read_text())
        except OSError:
            return JobStatus(JobState.UNKNOWN, reason="no local record")
        if record["token"] != handle.token:
            return JobStatus(JobState.UNKNOWN, reason="record belongs to another token")
        exit_path = self._exit_path(spec_path)
        if exit_path.exists():
            code = int(exit_path.read_text())
            if code == 0:
                return JobStatus(JobState.COMPLETED, exit_code=0)
            if code == _TIMEOUT_EXIT:
                return JobStatus(JobState.TIMEOUT, exit_code=code, reason="time limit")
            if code > 128:
                return JobStatus(JobState.FAILED, exit_code=code, signal=code - 128, reason="signal")
            return JobStatus(JobState.FAILED, exit_code=code)
        if self._alive(record):
            return JobStatus(JobState.RUNNING)
        # The wrapper died without recording an exit code: it was killed outright.
        return JobStatus(JobState.FAILED, signal=signal.SIGKILL, reason="process vanished")
