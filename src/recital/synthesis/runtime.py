"""Trusted bounded JSON transport to the bubblewrap supervisor."""
from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import threading
import time
from typing import Any

from .bundle import RuntimeBundle
from .models import IsolationError, RuntimeLimits, SynthesisError, canonical_json


class IsolatedWorker:
    def __init__(self, bundle: RuntimeBundle, config: dict[str, Any], limits: RuntimeLimits):
        self.limits = limits
        self._lock = threading.Lock()
        self.closed = False
        self.process = subprocess.Popen(
            bundle.command(max_message_bytes=limits.max_message_bytes),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env={"PATH": "/usr/bin:/bin"},
            close_fds=True, start_new_session=True, bufsize=0,
        )
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            os.set_blocking(stream.fileno(), False)
        try:
            response = self._exchange(config, limits.admission_seconds)
            if response != {"ok": True, "ready": True}:
                raise IsolationError("isolated supervisor failed its startup handshake")
        except BaseException:
            self.close()
            raise

    def _exchange(self, message: dict[str, Any], timeout: float) -> dict[str, Any]:
        data = canonical_json(message, max_bytes=self.limits.max_message_bytes - 1) + b"\n"
        if len(data) > self.limits.max_message_bytes:
            raise SynthesisError("isolated input exceeds byte limit")
        deadline = time.monotonic() + timeout
        received = bytearray()
        diagnostics = bytearray()
        sent = 0
        selector = selectors.DefaultSelector()
        selector.register(self.process.stdin, selectors.EVENT_WRITE, "input")
        selector.register(self.process.stdout, selectors.EVENT_READ, "output")
        selector.register(self.process.stderr, selectors.EVENT_READ, "error")
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise IsolationError("isolated supervisor wall timeout")
                events = selector.select(remaining)
                if not events:
                    raise IsolationError("isolated supervisor wall timeout")
                for key, _ in events:
                    if key.data == "input":
                        try:
                            sent += os.write(key.fd, data[sent:sent + 65536])
                        except BrokenPipeError as exc:
                            raise IsolationError("isolated supervisor closed its input") from exc
                        if sent == len(data):
                            selector.unregister(key.fileobj)
                    else:
                        chunk = os.read(key.fd, 65536)
                        if key.data == "error":
                            diagnostics.extend(chunk)
                            if len(diagnostics) > 16384:
                                raise IsolationError("isolated supervisor diagnostics exceeded limit")
                            if not chunk:
                                selector.unregister(key.fileobj)
                        else:
                            received.extend(chunk)
                            if len(received) > self.limits.max_message_bytes:
                                raise IsolationError("isolated supervisor output exceeded limit")
                            if b"\n" in received:
                                if not received.endswith(b"\n") or received.count(b"\n") != 1:
                                    raise IsolationError("invalid isolated response framing")
                                value = json.loads(received)
                                canonical_json(value)
                                if not isinstance(value, dict):
                                    raise IsolationError("invalid isolated response object")
                                return value
                            if not chunk:
                                detail = diagnostics.decode("utf-8", "replace")[:2000]
                                raise IsolationError(f"isolated supervisor exited: {detail}")
        finally:
            selector.close()

    def request(self, operation: str, **payload: Any) -> dict[str, Any]:
        with self._lock:
            if self.closed:
                raise IsolationError("isolated worker is closed")
            timeout = (self.limits.admission_seconds if operation == "admit"
                       else self.limits.wall_seconds)
            try:
                # Supervisor owns the execution deadline; small transport/reap grace.
                response = self._exchange({"op": operation, **payload}, timeout + 0.5)
            except BaseException:
                self.close()
                raise
            if response.get("ok") is not True:
                message = str(response.get("error", "isolated execution failed"))
                raise IsolationError(message) if response.get("isolation_failure") else SynthesisError(message)
            result = response.get("result")
            if type(result) is not dict:
                self.close()
                raise IsolationError("invalid isolated result")
            return result

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.process.wait(timeout=2)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()
