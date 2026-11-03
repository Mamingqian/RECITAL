"""Canonical JSON bytes, digests, and crash-safe file writes."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> bytes:
    """Sorted keys, no whitespace, UTF-8; NaN and Infinity raise ValueError."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fsync_dir(path: str | os.PathLike) -> None:
    """Make a directory entry durable. On a network filesystem this can fail with a stale handle after the data it
    protects is already safe, so a storage error here is not a failed write."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write(path: str | os.PathLike, data: bytes) -> None:
    """Write `data` to `path` so a reader sees the old file or the whole new one.

    The temporary name carries the pid and a random token, so concurrent writers of
    the same target never share a temporary file. The file and then its directory
    are fsynced around the rename.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    fsync_dir(target.parent)
