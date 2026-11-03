"""Per-row target-minus-strongest-off-target gap over a multi-cell oracle output.

Every consumer of the gap -- the reward input, the public measurements, the
terminal blind evaluator, library selection, calibration and the monitor -- computes
it here, so the off-target set and the treatment of a missing head are stated once.

Off-target set. ``off_target_columns=None`` compares against every column except
the row's target, which is what ``contrastive_mingap`` does. A caller with a
declared panel passes ``{target column: [off-target columns]}`` and the gap is
taken against exactly those heads. The two coincide for a two-column panel and
differ for a three-column one: ChromBPNet emits K562, HepG2, SK-N-SH, and the
declared off-target of K562 is HepG2 alone. Each caller therefore names its own
columns; ``reward_input: cell_gap`` has only the target columns bound per batch
and so uses every other column, which is the declaration for each two-head
reward oracle.

Missing evidence. With no declared off-target set a NaN head is an absent head and
is skipped as long as one other head is finite; +/-inf is never valid. A declared
off-target set is complete: one non-finite declared head makes the row's gap
missing instead of shrinking the comparison set. A non-finite target, or an
overflowing difference, also makes the gap missing. ``missing="raise"`` turns the
first missing row into a ``ValueError``; ``missing="nan"`` returns NaN there.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, NamedTuple

import numpy as np

from recital.targets import OracleTargetObjective, ScalarTarget

CELL_GAP_UNITS = "raw target-minus-max-offtarget gap"


class CellGap(NamedTuple):
    """Per-row raw target head, strongest off-target head and their difference."""

    target: np.ndarray
    off_target: np.ndarray
    gap: np.ndarray


def _declared_columns(others: Sequence[int], target: int, n_cells: int) -> list[int]:
    columns = [int(column) for column in others]
    if (not columns or len(set(columns)) != len(columns) or target in columns
            or min(columns) < 0 or max(columns) >= n_cells):
        raise ValueError(f"invalid declared off-target columns {columns} for target column {target}")
    return columns


def cell_gap_components(
    values,
    target_columns: Sequence[int],
    off_target_columns: Mapping[int, Sequence[int]] | None = None,
    *,
    missing: str = "raise",
    label: str | None = None,
) -> CellGap:
    """Target head, strongest off-target head and gap for each row of ``values``.

    ``values`` is ``(n_rows, n_cells)``; ``target_columns[i]`` is the column that
    holds row ``i``'s target cell. ``off_target_columns`` maps a target column to
    the columns it is compared against, or is ``None`` for every other column. A
    row whose target column has no entry has no declared off-target and is missing.
    ``target`` is the raw target-column value, ``off_target`` is NaN wherever the
    off-target side is missing, and ``gap`` is NaN wherever either side is.
    ``label`` names the source in error messages.
    """
    if missing not in ("raise", "nan"):
        raise ValueError("missing must be 'raise' or 'nan'")
    where = f"{label}: " if label else ""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f"{where}cell_gap expects (n_rows, n_cells) values with at least two cell columns")
    n_rows, n_cells = array.shape
    targets = np.asarray(target_columns)
    if targets.shape != (n_rows,):
        raise ValueError(f"{where}target_columns must have length n_rows")
    if targets.dtype.kind not in "iu" or np.any(targets < 0) or np.any(targets >= n_cells):
        raise ValueError(f"{where}target_columns must name valid cell columns")

    rows = np.arange(n_rows)
    target = array[rows, targets]
    finite = np.isfinite(array)
    if off_target_columns is None:
        compared = np.ones(array.shape, dtype=bool)
        compared[rows, targets] = False
        usable = compared & finite
        off_complete = usable.any(axis=1) & ~np.isinf(array).any(axis=1)
    else:
        compared = np.zeros(array.shape, dtype=bool)
        declared = np.zeros(n_rows, dtype=bool)
        for column in np.unique(targets).tolist():
            others = off_target_columns.get(column)
            if others is None:
                continue
            selected = np.flatnonzero(targets == column)
            compared[np.ix_(selected, _declared_columns(others, column, n_cells))] = True
            declared[selected] = True
        usable = compared
        off_complete = declared & (finite | ~compared).all(axis=1)

    with np.errstate(over="ignore", invalid="ignore"):
        strongest = np.max(np.where(usable, array, -np.inf), axis=1)
        gap = target - strongest
    complete = off_complete & np.isfinite(target)
    overflow = complete & ~np.isfinite(gap)
    if missing == "raise":
        if not complete.all():
            raise ValueError(f"{where}missing finite target or off-target evidence")
        if overflow.any():
            raise ValueError(f"{where}nonfinite target-minus-off-target gap")
    return CellGap(
        target=target,
        off_target=np.where(off_complete, strongest, np.nan),
        gap=np.where(complete & ~overflow, gap, np.nan),
    )


def cell_gap(
    values,
    target_columns: Sequence[int],
    off_target_columns: Mapping[int, Sequence[int]] | None = None,
    *,
    missing: str = "raise",
    label: str | None = None,
) -> np.ndarray:
    """Per-row ``target - strongest off-target``; see :func:`cell_gap_components`."""
    return cell_gap_components(
        values, target_columns, off_target_columns, missing=missing, label=label
    ).gap


def cell_gap_target_objective() -> OracleTargetObjective:
    """The framework-owned goal of a gap reward input: more gap is better.

    Reference 0 and scale 1 make the declared utility the raw gap itself, so a
    utility that normalizes it starts from the gap's own coordinate.
    """
    return OracleTargetObjective(target=ScalarTarget(
        kind="maximize", units=CELL_GAP_UNITS, scale=1.0, coordinate="raw", reference=0.0,
    ))


@dataclass
class CellGapInput:
    """Reward input that reduces each oracle's per-cell output to one gap per row.

    ``target_indices`` holds each row's target column and is bound per batch by
    the trainer, exactly as it is for ``contrastive_mingap``. A gap that cannot be
    formed is an error: missing target evidence never becomes a neutral reward.

    The input also owns the goal its gaps are measured against (``target``) and the
    utility an oracle gets when a recipe names none (``default_utility``): gaps of
    different oracles live on different scales, so averaging them raw would let one
    oracle dominate, which is what ``contrastive_mingap(normalize=True)`` avoids.
    """

    target_indices: list[int] = field(default_factory=list)
    default_utility: ClassVar[dict[str, str]] = {"name": "gap_ema_zscore"}

    @staticmethod
    def target() -> OracleTargetObjective:
        return cell_gap_target_objective()

    def __call__(self, raw: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {name: cell_gap(values, self.target_indices, missing="raise", label=name)
                for name, values in raw.items()}


REWARD_INPUTS = {"cell_gap": CellGapInput}
