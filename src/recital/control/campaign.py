"""Several cells in one controller process under one cap on live jobs.

Each engine is stepped in turn. A cell that raises is stopped and recorded as needing attention; the others
continue, so one cell's bug never idles the cluster's other cells.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from .engine import Engine, StepResult
from .errors import LedgerCorruption
from .jobs import write_json
from .state import RunState


@dataclass
class Cell:
    name: str
    engine: Engine
    done: bool = False
    crashed: str | None = None


@dataclass
class JobGate:
    """The shared cap: an engine asks before it submits."""

    cap: int
    cells: Sequence[Cell] = field(default_factory=tuple)

    def bind(self, cells: Sequence[Cell]) -> None:
        self.cells = cells

    def live(self) -> int:
        total = 0
        for cell in self.cells:
            if cell.done:
                continue
            state = cell.engine.state()
            total += len(state.inflight) + (1 if state.baseline.status in ("submitted", "running") else 0)
        return total

    def allow(self) -> bool:
        return self.live() < self.cap


def _crash(cell: Cell, exc: Exception) -> None:
    cell.done, cell.crashed = True, f"{type(exc).__name__}: {exc}"
    if isinstance(exc, LedgerCorruption):
        write_json(cell.engine.s.run_dir / "needs_attention.json", {"type": type(exc).__name__, "message": str(exc)})
        return
    try:
        cell.engine._halt(exc)
    except Exception:  # a log that cannot be written must not take the other cells down with it
        write_json(cell.engine.s.run_dir / "needs_attention.json", {"type": type(exc).__name__, "message": str(exc)})


def run_campaign(cells: Sequence[Cell], *, poll_seconds: float, sleep: Callable[[float], None] = time.sleep,
                 should_stop: Callable[[], bool] = lambda: False) -> Mapping[str, RunState]:
    """Step every unfinished cell until all are terminal, or until `should_stop` returns True."""
    while True:
        progressed = False
        for cell in cells:
            if cell.done:
                continue
            try:
                result = cell.engine.step()
            except Exception as exc:  # a cell's bug must not stop the other cells
                _crash(cell, exc)
                continue
            if result is StepResult.DONE:
                cell.done = True
            elif result is StepResult.PROGRESS:
                progressed = True
        if all(cell.done for cell in cells) or should_stop():
            return {cell.name: cell.engine.state() for cell in cells}
        if not progressed:
            sleep(poll_seconds)
