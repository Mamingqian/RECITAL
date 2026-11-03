"""Observed trainer-call costs; no prices, GPU estimates or budget formulas.

Counters describe events visible at their named boundary. Oracle attempts count
calls, including exceptions, and requested sequences (not unique sequences or
internal model forwards). Training candidates count deliveries to the reward
bridge. Sampler completions count rows actually returned by model.generate.
Opaque task closures, external calibration/SFT/agents and asynchronous GPU work
are outside this ledger. Host monotonic elapsed times are not GPU compute time.
"""
from __future__ import annotations

import contextlib
import copy
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Iterator


METRIC_UNITS = {
    "training_candidates": "candidates",
    "base_completions": "completions",
    "final_completions": "completions",
    "probe_completions": "completions",
    "base_cache_hits": "hits",
    "base_reused_candidates": "candidates",
    "oracle_call_attempts": "calls",
    "oracle_calls_returned": "calls",
    "oracle_calls_failed": "calls",
    "oracle_sequences_requested": "sequences",
    "oracle_elapsed_seconds": "seconds",
    "optimizer_steps": "optimizer_steps",
    "train_runtime_seconds": "seconds",
    "training_elapsed_seconds": "seconds",
    "total_elapsed_seconds": "seconds",
}


def _valid_number(value: Any) -> bool:
    if type(value) is int:
        return value >= 0
    return isinstance(value, float) and math.isfinite(value) and value >= 0


def _valid_metric(value: Any, unit: str) -> bool:
    return _valid_number(value) and (unit == "seconds" or type(value) is int)


def _entry(value: Any, unit: str, reason: str = "") -> dict[str, Any]:
    return {"value": value, "unit": unit,
            "status": "measured" if value is not None else "unknown", "reason": reason}


class TrainingCallCost:
    """One call's mutable ledger; ``snapshot`` returns detached JSON-safe data."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._started = clock()
        self._finished: float | None = None
        self._status = "running"
        self._failure_type: str | None = None
        self._phase = "setup"
        self._metrics = {name: _entry(0, unit) for name, unit in METRIC_UNITS.items()}
        self._metrics["train_runtime_seconds"] = _entry(None, "seconds", "Trainer output not available")
        self._metrics["total_elapsed_seconds"] = _entry(None, "seconds", "trainer call not finished")
        self._events: list[dict[str, Any]] = []
        self._step_start: int | None = None
        self._step_last: int | None = None

    def add(self, metric: str, value: int | float) -> None:
        if not _valid_metric(value, self._metrics[metric]["unit"]):
            self.incomplete(metric, "invalid observed count/time")
            return
        entry = self._metrics[metric]
        total = value if entry["value"] is None else entry["value"] + value
        if not _valid_metric(total, entry["unit"]):
            self.incomplete(metric, "aggregate overflow; previous finite subtotal retained")
            return
        entry["value"] = total

    def incomplete(self, metric: str, reason: str) -> None:
        entry = self._metrics[metric]
        entry["status"] = "partial" if entry["value"] is not None else "unknown"
        entry["reason"] = "; ".join(filter(None, [entry["reason"], reason]))

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        previous = self._phase
        self._phase = name
        try:
            yield
        finally:
            self._phase = previous

    def in_phase(self, fn: Callable[..., Any], phase: str) -> Callable[..., Any]:
        """Delegate without wrapping a second oracle counter around the call."""
        def scoped(*args, **kwargs):
            with self.phase(phase):
                return fn(*args, **kwargs)
        return scoped

    def wrap_oracles(self, oracles: Mapping[str, Callable[..., Any]]) -> dict[str, Any]:
        """Create new callable objects; leave the supplied registry and functions intact."""
        return {name: CountedOracle(name, fn, self) if callable(fn) else fn
                for name, fn in oracles.items()}

    def set_phase(self, phase: str) -> None:
        self._phase = phase

    def _oracle_started(self, name: str, args: tuple, kwargs: dict) -> dict[str, Any]:
        seqs = args[0] if args else kwargs.get("sequences", kwargs.get("seqs"))
        try:
            count = len(seqs) if not isinstance(seqs, (str, bytes)) else None
        except (TypeError, AttributeError):
            count = None  # Never consume an iterator or call the oracle twice to find its size.
        self.add("oracle_call_attempts", 1)
        if count is None:
            self.incomplete("oracle_sequences_requested", "oracle input has no observed sequence count")
        else:
            self.add("oracle_sequences_requested", count)
        event = {"oracle": name, "phase": self._phase, "attempt": len(self._events) + 1,
                 "sequences_requested": count, "outcome": "in_progress",
                 "elapsed_seconds": None, "exception_type": None}
        self._events.append(event)
        return event

    def observe_completions(self, phase: str, count: int) -> None:
        self.add(f"{phase}_completions", count)

    def observe_step(self, step: Any, *, initial: bool = False) -> None:
        if type(step) is not int or step < 0 or (self._step_last is not None and step < self._step_last):
            self.incomplete("optimizer_steps", "invalid/nonmonotonic TrainerState.global_step")
            return
        if initial:
            self._step_start = step
            self._metrics["optimizer_steps"] = _entry(0, "optimizer_steps")
        self._step_last = step
        if self._step_start is None:
            self.incomplete("optimizer_steps", "initial TrainerState.global_step is unavailable")
        else:
            self._metrics["optimizer_steps"]["value"] = step - self._step_start

    @contextlib.contextmanager
    def training(self) -> Iterator[None]:
        start = self._clock()
        self._metrics["optimizer_steps"] = _entry(None, "optimizer_steps")
        with self.phase("training"):
            try:
                yield
            except BaseException:
                self.incomplete("training_candidates", "failed training; only reward-bridge deliveries observed")
                self.incomplete("optimizer_steps", "failed training; only public callback step observations available")
                raise
            finally:
                elapsed = self._clock() - start
                self.add("training_elapsed_seconds", elapsed)
                if self._step_start is None:
                    self.incomplete("optimizer_steps", "Trainer did not expose an initial step")

    def observe_train_runtime(self, metrics: Mapping[str, Any] | None) -> None:
        runtime = (metrics or {}).get("train_runtime")
        self._metrics["train_runtime_seconds"] = _entry(
            runtime if _valid_number(runtime) else None, "seconds",
            "" if _valid_number(runtime) else "finite train_runtime missing from Trainer output",
        )

    def finish(self, *, failed: BaseException | None = None) -> None:
        self._finished = self._clock()
        self._status = "failed" if failed is not None else "completed"
        self._failure_type = type(failed).__name__ if failed is not None else None
        elapsed = self._finished - self._started
        self._metrics["total_elapsed_seconds"] = _entry(
            elapsed if _valid_number(elapsed) else None, "seconds",
            "" if _valid_number(elapsed) else "invalid monotonic elapsed time",
        )

    def prefix_snapshot(self) -> dict[str, Any]:
        """Measured elapsed time to this capture without settling the open call.

        This is host wall time, including callbacks already executed. It is not
        GPU utilization time and must not be summed with a later overlapping
        prefix or with separate callback elapsed times.
        """
        result = self.snapshot()
        end = self._finished if self._finished is not None else self._clock()
        result["measurements"]["total_elapsed_seconds"] = _entry(end - self._started, "seconds")
        result["capture_scope"] = "cumulative_trainer_call_prefix"
        return result

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy({
            "schema_version": 1, "scope": "trainer_call", "status": self._status,
            "process_scope": "calling_process_only; no distributed reduction",
            "failure_type": self._failure_type, "measurements": self._metrics,
            "optimizer_step_provenance": {
                "initial_global_step": self._step_start, "last_observed_global_step": self._step_last,
                "source": "public Trainer callbacks; steps within this call only",
            },
            "oracle_events": self._events,
            "coverage": {
                "training_candidates": "raw candidate rows received by reward bridge; includes invalid rows",
                "sampling_completions": "rows returned by shared sampler model.generate; cache reuse separate",
                "oracle_calls": "wrapped context.oracle_fns invocations; returned does not imply output validity",
                "elapsed": "host monotonic wall time; no GPU synchronization or utilization estimate",
                "excluded": ["SFT", "calibration", "runner/agent work", "opaque task closure calls",
                             "oracle internal model-forward counts", "GPU seconds", "dollars"],
            },
        })


@dataclass(frozen=True)
class CountedOracle:
    name: str
    fn: Callable[..., Any]
    cost: TrainingCallCost

    def __call__(self, *args, **kwargs):
        event = self.cost._oracle_started(self.name, args, kwargs)
        start = self.cost._clock()
        try:
            result = self.fn(*args, **kwargs)
        except BaseException as exc:
            self.cost.add("oracle_calls_failed", 1)
            event.update(outcome="raised", exception_type=type(exc).__name__)
            raise
        else:
            self.cost.add("oracle_calls_returned", 1)
            event["outcome"] = "returned"
            return result
        finally:
            elapsed = self.cost._clock() - start
            event["elapsed_seconds"] = elapsed if _valid_number(elapsed) else None
            self.cost.add("oracle_elapsed_seconds", elapsed)


def journaled_oracle_call(fn, sequences, *, name, phase, journal_path):
    """Append an observed external scoring call using the same trainer event schema.

    This covers scoring workers outside the trainer. A process killed before
    the finally block can leave unobserved work; completed-cache sizes are not
    substituted for an exact physical call count.
    """
    import json
    from pathlib import Path

    cost = TrainingCallCost()
    cost.set_phase(phase)
    oracle = CountedOracle(name, fn, cost)
    try:
        return oracle(sequences)
    finally:
        path = Path(journal_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            for event in cost.snapshot()["oracle_events"]:
                handle.write(json.dumps(event, allow_nan=False) + "\n")


def trainer_cost_callback(cost: TrainingCallCost):
    """Build a public HF callback lazily; aggregation itself needs no train stack."""
    from transformers import TrainerCallback

    class CostCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            cost.observe_step(state.global_step, initial=True)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            cost.observe_step(state.global_step)
            return control

        def on_train_end(self, args, state, control, **kwargs):
            cost.observe_step(state.global_step)
            return control

    return CostCallback()
