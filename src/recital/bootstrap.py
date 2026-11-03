"""Percentile bootstrap intervals for the blind numbers.

`median_bootstrap` resamples one library; `worst_stratum_bootstrap` resamples within each stratum and re-takes the
minimum of the stratum means. Neither imputes a missing or non-finite value.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def median_bootstrap(
    values: Sequence[float],
    *,
    draws: int = 2_000,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Percentile interval for the median of one library.

    The CRE blind number is the median ChromBPNet MinGap over the sequences of one
    cell at one iteration; resampling that library gives the interval around it.
    """
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"point_estimate": None, "reason": "no finite observations"}
    rng = np.random.default_rng(seed)
    draws_idx = rng.integers(0, array.size, size=(draws, array.size))
    medians = np.median(array[draws_idx], axis=1)
    alpha = (1.0 - float(level)) / 2.0
    low, high = np.quantile(medians, [alpha, 1.0 - alpha])
    return {
        "point_estimate": float(np.median(array)),
        "ci_low": float(low),
        "ci_high": float(high),
        "level": float(level),
        "draws": int(draws),
        "seed": int(seed),
        "n": int(array.size),
        "statistic": "median",
    }


def worst_stratum_bootstrap(
    per_stratum_values: Mapping[str, Sequence[float]],
    *,
    draws: int = 2_000,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Percentile interval for ``min`` over strata of the per-stratum mean.

    The Loop-seq blind number is the worst of three group hit rates.

    It is NOISY. Each group hit rate is estimated from a finite library, so the
    statistic varies between independent evaluations of an identical policy.

    It is BIASED. A minimum over three noisy means is pulled below the minimum of
    the true means, and the bias grows as the groups get noisier. Resampling within
    each stratum and re-taking the minimum reproduces both effects, which a normal
    approximation would misrepresent. The point estimate is the plug-in minimum.
    """
    strata = sorted(per_stratum_values)
    columns = [
        np.asarray(per_stratum_values[name], dtype=float) for name in strata
    ]
    columns = [column[np.isfinite(column)] for column in columns]
    if not columns or any(column.size == 0 for column in columns):
        return {
            "point_estimate": None,
            "reason": "a stratum has no finite observations",
            "strata": strata,
        }

    rng = np.random.default_rng(seed)
    point = float(min(float(column.mean()) for column in columns))
    worst = np.empty(draws, dtype=float)
    for draw in range(draws):
        worst[draw] = min(
            float(
                column[rng.integers(0, column.size, column.size)].mean()
            )
            for column in columns
        )
    alpha = (1.0 - float(level)) / 2.0
    low, high = np.quantile(worst, [alpha, 1.0 - alpha])
    return {
        "point_estimate": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "level": float(level),
        "draws": int(draws),
        "seed": int(seed),
        "bootstrap_mean": float(worst.mean()),
        # point - bootstrap_mean estimates the downward bias of min-over-strata.
        "bias_estimate": float(point - worst.mean()),
        "n_per_stratum": {
            name: int(column.size) for name, column in zip(strata, columns, strict=True)
        },
        "statistic": "min over strata of the stratum mean",
    }


__all__ = [
    "median_bootstrap",
    "worst_stratum_bootstrap",
]
