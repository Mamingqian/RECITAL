"""Explicit design-condition and oracle-panel coordinates and CPU-only calibration checks.

Activity and MinGap are separate measurements. No target-activity floor has
been specified, so these helpers never infer activity qualification from a gap.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from recital.spec.cell_gap import CellGap, cell_gap_components


TARGET_CELLS = ("K562", "HepG2")
PROMPT_TO_COLUMN = {"<CELL_K562>": 0, "<CELL_HepG2>": 1}
MALINOIS_SOURCE_COLUMNS = {"K562": 0, "HepG2": 1, "SK-N-SH": 2}
ENFORMER_SOURCE_COLUMNS = {"K562": 0, "HepG2": 1}


def panel_scope(
    oracle_columns: Sequence[str],
    off_target_cells_by_target: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Validate named columns and all declared off-targets without dropping heads."""
    columns = list(oracle_columns)
    if len(set(columns)) != len(columns) or not set(TARGET_CELLS) <= set(columns):
        raise ValueError("oracle columns must be unique and contain both design cells")
    if set(off_target_cells_by_target) != set(TARGET_CELLS):
        raise ValueError("declare off-target panels for both design cells")
    for cell, others in off_target_cells_by_target.items():
        if (not others or len(set(others)) != len(others) or cell in others
                or not set(others) <= set(columns)):
            raise ValueError(f"invalid declared off-target panel for {cell}")
    return {
        "target_cells": list(TARGET_CELLS),
        "oracle_output_columns": columns,
        "off_target_cells_by_target": {cell: list(others) for cell, others in off_target_cells_by_target.items()},
        "gap_definition": "raw target head minus maximum raw declared off-target head",
    }


def panel_gap(
    values: Any, cells: Sequence[str], *, oracle_columns: Sequence[str],
    off_target_cells_by_target: Mapping[str, Sequence[str]], missing: str = "raise",
    label: str | None = None,
) -> CellGap:
    """Target head, strongest declared off-target head and gap for rows of a named panel.

    ``cells[i]`` is row ``i``'s target cell and the comparison is exactly the
    declared off-target list of that cell, never every other head. A row whose
    cell has no declared list has no gap: it is NaN under ``missing="nan"`` and an
    error under ``"raise"``.
    """
    columns = list(oracle_columns)
    cells = list(cells)
    array = np.asarray(values, dtype=float)
    declared = np.asarray([cell in off_target_cells_by_target for cell in cells], dtype=bool)
    if missing == "raise" and not declared.all():
        raise ValueError(f"no declared off-target panel for cells "
                         f"{sorted({cell for cell, ok in zip(cells, declared) if not ok})}")
    kept = np.flatnonzero(declared)
    present = {cell for cell in cells if cell in off_target_cells_by_target}
    parts = cell_gap_components(
        array[kept], np.asarray([columns.index(cells[row]) for row in kept], dtype=int),
        {columns.index(cell): [columns.index(other) for other in off_target_cells_by_target[cell]]
         for cell in present},
        missing=missing, label=label,
    )
    filled = [np.full(len(cells), np.nan) for _ in parts]
    for column, part in zip(filled, parts):
        column[kept] = part
    return CellGap(*filled)


def score_arrays(
    values: Any, *, target_cells: Sequence[str], oracle: str,
    oracle_columns: Sequence[str],
    off_target_cells_by_target: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Report finite raw activity/gap arrays in explicitly declared coordinates.

    ``target_cells`` follows row order; it is never an oracle column order.
    Oracle columns can include GM12878 or other declared off-target cells.
    Oracles reserved for the terminal blind evaluation are excluded.
    """
    if oracle == "chrombpnet":
        raise ValueError("audit oracle is excluded from engineering score inputs")
    scope = panel_scope(oracle_columns, off_target_cells_by_target)
    targets = list(target_cells)
    unknown = set(targets) - set(TARGET_CELLS)
    if unknown:
        raise ValueError(f"unsupported target cells: {sorted(unknown)}")
    array = np.asarray(values, dtype=float)
    if array.shape != (len(targets), len(oracle_columns)) or not np.isfinite(array).all():
        raise ValueError(f"{oracle} requires finite scores shaped ({len(targets)}, {len(oracle_columns)})")
    target, off_target, gap = panel_gap(
        array, targets, oracle_columns=oracle_columns,
        off_target_cells_by_target=off_target_cells_by_target, missing="raise", label=oracle,
    )
    return {
        "oracle": oracle,
        "scope": scope,
        "target_cells": targets,
        "target_activity_raw": target.tolist(),
        "off_target_activity_raw": off_target.tolist(),
        "mingap_raw": gap.tolist(),
        "target_activity_floor_by_cell": dict.fromkeys(TARGET_CELLS),
        "activity_qualification": None,
        "qualification_status": "undefined_target_activity_floor",
    }


