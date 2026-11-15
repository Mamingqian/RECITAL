"""The initial-recipe (anchor) libraries of the main-line cells, scored with the terminal audit.

Every cell trains the declared initial recipe first (the candidate whose `anchor` flag is set). Its library,
selected from that candidate's pool with the cell's own delivery plan, is scored by
`scripts/analysis/anchor_audit.py` with the oracles and `audit_measure` of `recital.control.terminal_audit`.
That script writes `<ANCHOR_ROOT>/<cell>/terminal_audit/` in the layout of a cell's own `terminal_audit/`, and
`<cell>/anchor_record.json` (what was selected). The figure builders and `paper_numbers.py` read the files here;
nothing is written.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from common.control_cells import SCRATCH_ROOT

ANCHOR_ROOT = SCRATCH_ROOT / "outputs" / "anchor_audit_20261022"


def _read(path: Path) -> Any:
    return json.loads(path.read_text()) if path.is_file() else None


def summary(cell: str) -> dict[str, Any] | None:
    """The anchor library's audit summary, or None while the audit has not finished completely."""
    found = _read(ANCHOR_ROOT / cell / "terminal_audit" / "summary.json")
    return found if found is not None and found.get("status") == "complete" else None


def record(cell: str) -> dict[str, Any] | None:
    """What `anchor_audit.py` selected: the anchor candidate, the rows per condition and whether the library qualifies."""
    return _read(ANCHOR_ROOT / cell / "anchor_record.json")


def manifest(cell: str) -> dict[str, Any] | None:
    return _read(ANCHOR_ROOT / cell / "terminal_audit" / "manifest.json")


def source_values(cell: str, source: str) -> np.ndarray | None:
    """Per-row values of one blind source for the anchor library (rows in `manifest(cell)` order)."""
    found = _read(ANCHOR_ROOT / cell / "terminal_audit" / f"{source}.json")
    return None if found is None or found.get("status") != "complete" else np.asarray(found["values"], dtype=float)


def library_measurement(cell: str, name: str) -> np.ndarray | None:
    """A visible per-row measurement of the anchor library, looked up by row id in the assembled pool."""
    directory = ANCHOR_ROOT / cell / "anchor" / "delivery"
    pool, measured, rows = _read(directory / "pool.json"), _read(directory / "public_measurements.json"), manifest(cell)
    entry = ((measured or {}).get("measurements") or {}).get(name)
    if pool is None or entry is None or rows is None:
        return None
    by_row = dict(zip(pool["row_ids"], entry["values"]))
    return np.asarray([by_row.get(row_id) for row_id in rows["row_ids"]], dtype=float)
