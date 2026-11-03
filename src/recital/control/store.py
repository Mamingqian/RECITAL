"""Append-only hash-chained event log and a content-addressed blob store."""
from __future__ import annotations

import fcntl
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from recital.control.errors import ControlError, LedgerCorruption
from recital.control.jsonio import atomic_write, canonical_json, fsync_dir, sha256_hex

GENESIS_HASH = "0" * 64
LOCK_NAME = "control.lock"
_RECORD_KEYS = frozenset({"seq", "ts", "type", "data", "prev", "hash"})
_SHA = re.compile(r"[0-9a-f]{64}")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def chain_hash(prev: str, seq: int, ts: str, type_: str, data: Mapping[str, Any]) -> str:
    """sha256(prev || canonical({seq, ts, type, data})), with prev as ASCII hex."""
    body = canonical_json({"seq": seq, "ts": ts, "type": type_, "data": data})
    return sha256_hex(prev.encode("ascii") + body)


@dataclass(frozen=True)
class Event:
    """One log record. `data` is shared with the log; treat it as read-only."""

    seq: int
    ts: str
    type: str
    data: dict[str, Any]
    prev: str
    hash: str

    def record(self) -> dict[str, Any]:
        return {"seq": self.seq, "ts": self.ts, "type": self.type, "data": self.data,
                "prev": self.prev, "hash": self.hash}


class EventLog:
    """One jsonl file, one writer.

    Every line is `{seq, ts, type, data, prev, hash}`. `seq` starts at 1 and grows by
    one; `prev` is the previous line's `hash` (all zeros for the first). Opening the
    log takes a non-blocking flock on `control.lock` in the log's directory, replays
    the whole chain, and drops a torn final line (unparseable, or missing its
    newline) left by a crash during an append. Any other inconsistency raises
    LedgerCorruption. Each append is written and fsynced before it returns.
    """

    def __init__(self, path: str | os.PathLike, *, clock: Callable[[], str] = utc_now):
        self.path = Path(path)
        self._clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = self._acquire_lock()
        self._fd: int | None = None
        try:
            self._events, self.repaired_bytes = self._load()
            self._size = self.path.stat().st_size if self.path.exists() else 0
            self._fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            fsync_dir(self.path.parent)
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self._events)

    @property
    def head_hash(self) -> str:
        return self._events[-1].hash if self._events else GENESIS_HASH

    def events(self, after: int = 0) -> Iterator[Event]:
        """Events with `seq > after`, from a snapshot taken at call time."""
        return iter(tuple(self._events[after:]))

    def append(self, type: str, data: Mapping[str, Any]) -> Event:
        if self._fd is None:
            raise ControlError(f"event log {self.path} is closed")
        if not isinstance(type, str) or not type:
            raise ControlError("an event needs a nonempty string type")
        data = json.loads(canonical_json(data))
        if not isinstance(data, dict):
            raise ControlError("event data must be a JSON object")
        ts = self._clock()
        if not isinstance(ts, str):
            raise ControlError("the event clock must return a string")
        seq, prev = len(self._events) + 1, self.head_hash
        event = Event(seq, ts, type, data, prev, chain_hash(prev, seq, ts, type, data))
        self._write(canonical_json(event.record()) + b"\n")
        self._events.append(event)
        return event

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def _acquire_lock(self):
        lock_path = self.path.parent / LOCK_NAME
        handle = lock_path.open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise ControlError(
                f"another controller holds {lock_path}; refusing to open {self.path}"
            ) from exc
        except BaseException:
            handle.close()
            raise
        return handle

    def _load(self) -> tuple[list[Event], int]:
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            return [], 0
        *complete, tail = raw.split(b"\n")
        events: list[Event] = []
        prev, good = GENESIS_HASH, 0
        for number, line in enumerate(complete, start=1):
            try:
                record = json.loads(line)
            except ValueError as exc:
                if number == len(complete) and not tail:
                    break
                raise LedgerCorruption(f"{self.path}: line {number} is not valid JSON") from exc
            event = self._verified(record, number, prev)
            events.append(event)
            prev, good = event.hash, good + len(line) + 1
        dropped = len(raw) - good
        if dropped:
            with open(self.path, "r+b") as handle:
                handle.truncate(good)
                handle.flush()
                os.fsync(handle.fileno())
            fsync_dir(self.path.parent)
        return events, dropped

    def _verified(self, record: Any, number: int, prev: str) -> Event:
        where = f"{self.path}: line {number}"
        if not isinstance(record, dict) or set(record) != _RECORD_KEYS:
            raise LedgerCorruption(f"{where} is not an event record")
        if type(record["seq"]) is not int or record["seq"] != number:
            raise LedgerCorruption(f"{where} has seq {record['seq']!r}, expected {number}")
        if not (isinstance(record["ts"], str) and isinstance(record["type"], str)
                and isinstance(record["data"], dict) and isinstance(record["hash"], str)):
            raise LedgerCorruption(f"{where} has a malformed field")
        if record["prev"] != prev:
            raise LedgerCorruption(f"{where} does not chain to the previous line")
        expected = chain_hash(prev, number, record["ts"], record["type"], record["data"])
        if record["hash"] != expected:
            raise LedgerCorruption(f"{where} fails its hash")
        return Event(number, record["ts"], record["type"], record["data"], prev, expected)

    def _write(self, line: bytes) -> None:
        try:
            view = memoryview(line)
            while view:
                view = view[os.write(self._fd, view):]
            os.fsync(self._fd)
        except BaseException:
            try:
                os.ftruncate(self._fd, self._size)
            except OSError:
                self.close()
            raise
        self._size += len(line)


def read_events(path: str | os.PathLike) -> list[Event]:
    """The verified events of a log, read without its lock and without repairing a torn tail.

    A controller may be appending while this runs; a final line it has not finished is ignored.
    """
    log = EventLog.__new__(EventLog)
    log.path = Path(path)
    try:
        raw = log.path.read_bytes()
    except FileNotFoundError:
        return []
    *complete, tail = raw.split(b"\n")
    events: list[Event] = []
    prev = GENESIS_HASH
    for number, line in enumerate(complete, start=1):
        try:
            record = json.loads(line)
        except ValueError as exc:
            if number == len(complete) and not tail:
                break
            raise LedgerCorruption(f"{log.path}: line {number} is not valid JSON") from exc
        event = log._verified(record, number, prev)
        events.append(event)
        prev = event.hash
    return events


class BlobStore:
    """Immutable bytes addressed by sha256, at `<root>/blobs/<sha[:2]>/<sha>`."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)

    def path_of(self, sha: str) -> Path:
        if not isinstance(sha, str) or not _SHA.fullmatch(sha):
            raise ControlError(f"not a sha256 digest: {sha!r}")
        return self.root / "blobs" / sha[:2] / sha

    def has(self, sha: str) -> bool:
        return self.path_of(sha).is_file()

    def put(self, data: bytes) -> str:
        sha = sha256_hex(data)
        if not self.has(sha):
            atomic_write(self.path_of(sha), data)
        return sha

    def get(self, sha: str) -> bytes:
        path = self.path_of(sha)
        try:
            data = path.read_bytes()
        except FileNotFoundError as exc:
            raise LedgerCorruption(f"blob {sha} is missing from {self.root}") from exc
        if sha256_hex(data) != sha:
            raise LedgerCorruption(f"blob {path} does not match its digest")
        return data

    def put_json(self, obj: Any) -> str:
        return self.put(canonical_json(obj))

    def get_json(self, sha: str) -> Any:
        return json.loads(self.get(sha))
