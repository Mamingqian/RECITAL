"""Compact, measured optimization evidence, distinct from held-out evaluation."""
from __future__ import annotations

import math
from typing import Any

# Only training scalars: never serialize arbitrary scorer columns or sequences.
TRAINING_SCALARS = (
    "reward", "reward_std", "kl", "loss", "learning_rate", "grad_norm",
    "frac_reward_zero_std", "clip_ratio/region_mean", "entropy",
    "completions/mean_length", "completions/clipped_ratio",
)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def training_progress(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Summarize the full logged series without inventing validation curves."""
    if isinstance(trajectory.get("training_progress"), dict):
        return trajectory["training_progress"]
    logs = trajectory.get("training_log") or []
    rows = [row for row in logs if isinstance(row, dict) and finite_number(row.get("step"))]
    result: dict[str, Any] = {
        "population": "changing generated training batches; not fixed-library validation",
        "optimizer_steps_completed": trajectory.get("optimizer_steps_completed"),
        "logged_rows": len(rows),
        "status": "measured" if rows else "unavailable",
        "series": {},
    }
    if not rows:
        result["reason"] = "No step-indexed training telemetry was supplied; a terminal score does not establish a plateau."
    for name in TRAINING_SCALARS:
        present = [(row["step"], row[name]) for row in rows if name in row]
        values = sorted([(step, value) for step, value in present if finite_number(value)], key=lambda item: item[0])
        if not present:
            continue
        summary = {"n_finite": len(values), "n_nonfinite": len(present) - len(values)}
        if values:
            width = min(5, len(values))
            peak = max(values, key=lambda item: item[1])
            trough = min(values, key=lambda item: item[1])
            summary.update(
                first_step=values[0][0], last_step=values[-1][0],
                first=values[0][1], last=values[-1][1],
                first_window_mean=sum(v for _, v in values[:width]) / width,
                last_window_mean=sum(v for _, v in values[-width:]) / width,
                window_points=width, maximum=peak[1], maximum_step=peak[0],
                minimum=trough[1], minimum_step=trough[0],
            )
        result["series"][name] = summary
    if rows and not result["series"]:
        result.update(status="unavailable", reason="No supported training scalars were recorded.")
    return result
