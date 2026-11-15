"""A digest of one training run by reward component, built from the files the trainer wrote.

`training_statistics.json` holds, for every reward call of a candidate's training, the mean of each oracle, the mean, maximum and
non-zero count of each penalty, the gate violation fraction and the reward spread and clipping per condition. A ledger row shows
five training scalars of it. The digest splits the run into equal windows and reports each quantity per window, and flags a
penalty whose mean is a small share of the reward spread, as a fact about the run; what a small share implies for the next
edit is left to the reader.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

WINDOWS = 5
# A penalty is small when it subtracts, on average, less than this share of the within-batch reward spread, or is non-zero in
# fewer than this share of the samples.
SMALL_SHARE = 0.02
SMALL_NONZERO = 0.05


@dataclass(frozen=True)
class Flag:
    kind: str           # `small_penalty`
    name: str
    share: float        # mean subtracted over the mean within-batch reward spread
    nonzero: float      # share of the samples with a non-zero penalty


def _windows(n: int, k: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, n, k + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(k)]


def _series(history: Sequence[Mapping[str, Any]], pick, k: int) -> list[float | None]:
    out: list[float | None] = []
    for start, stop in _windows(len(history), k):
        values = [v for v in (pick(entry) for entry in history[start:stop]) if v is not None]
        out.append(float(np.mean(values)) if values else None)
    return out


def summarize(statistics: Mapping[str, Any], log_history: Sequence[Mapping[str, Any]] = (), *, windows: int = WINDOWS,
              ) -> dict[str, Any]:
    """The per-window series of one candidate's training and the whole-run penalty statistics."""
    history = statistics["reward_execution_history"]
    composition = lambda entry: entry.get("composition") or {}
    oracles = sorted({name for entry in history[:50] for name in (entry.get("training_oracles") or {})})
    found: dict[str, Any] = {
        "calls": len(history),
        "oracle_mean": {name: _series(history, lambda e, n=name: ((e.get("training_oracles") or {}).get(n) or {}).get("mean"),
                                      windows) for name in oracles},
        "reward_mean": _series(history, lambda e: (composition(e).get("aggregation") or {}).get("mean"), windows),
        "reward_spread": _series(history, lambda e: composition(e).get("reward_std"), windows),
        "gate_violation": _series(history, lambda e: composition(e).get("gate_violation_fraction"), windows),
        "penalties": {}, "conditions": {}}
    execution = statistics.get("reward_execution") or {}
    for index, penalty in (execution.get("penalties") or {}).items():
        found["penalties"][penalty["name"]] = {
            "mean": _series(history, lambda e, i=index: (((composition(e).get("penalties") or {}).get(i)) or {}).get("mean"),
                            windows),
            "maximum": penalty["maximum"], "nonzero": penalty.get("nonzero_fraction")}
    for condition in sorted((composition(history[0]).get("by_condition") or {}) if history else ()):
        by_condition = lambda e, c=condition: ((composition(e).get("by_condition") or {}).get(c)) or {}
        found["conditions"][condition] = {
            "reward_spread": _series(history, lambda e: by_condition(e).get("reward_std"), windows),
            "lower_clip": _series(history, lambda e: by_condition(e).get("lower_clip_fraction"), windows)}
    for key in ("kl", "entropy"):
        picked = [entry[key] for entry in log_history if key in entry]
        found[key] = [picked[i] for i in np.linspace(0, len(picked) - 1, min(windows, len(picked))).astype(int)] if picked else []
    return found


def flags(summary: Mapping[str, Any]) -> list[Flag]:
    """The penalties whose mean over the run is a small share of the reward spread."""
    spread = [x for x in summary["reward_spread"] if x is not None]
    scale = float(np.mean(spread)) if spread else 0.0
    found = []
    for name, penalty in summary["penalties"].items():
        means = [x for x in penalty["mean"] if x is not None]
        share = float(np.mean(means)) / scale if means and scale > 0 else 0.0
        nonzero = penalty["nonzero"] if penalty["nonzero"] is not None else 1.0
        if share < SMALL_SHARE or nonzero < SMALL_NONZERO:
            found.append(Flag("small_penalty", name, share, float(nonzero)))
    return found


def _list(values: Sequence[float | None]) -> str:
    return "[" + ", ".join("-" if v is None else f"{v:.3g}" for v in values) + "]"


def render(summary: Mapping[str, Any], found: Sequence[Flag], *, token_cap: int = 250, count_tokens=None) -> str:
    """The digest as text; over `token_cap` tokens the per-condition lines go first."""
    def build(with_conditions: bool) -> str:
        lines = [f"reflection ({summary['calls']} reward calls, {WINDOWS} windows, first to last):"]
        lines += [f" oracle {name} mean {_list(series)}" for name, series in summary["oracle_mean"].items()]
        lines.append(f" total reward mean {_list(summary['reward_mean'])} within-batch spread {_list(summary['reward_spread'])}")
        for name, penalty in summary["penalties"].items():
            lines.append(f" penalty {name} subtracts on average {_list(penalty['mean'])}, maximum {penalty['maximum']:.3g}, "
                         f"non-zero in {penalty['nonzero']:.0%} of samples" if penalty["nonzero"] is not None else
                         f" penalty {name} subtracts on average {_list(penalty['mean'])}, maximum {penalty['maximum']:.3g}")
        if any(v for v in summary["gate_violation"] if v):
            lines.append(f" gate violation fraction {_list(summary['gate_violation'])}")
        if with_conditions:
            for condition, values in summary["conditions"].items():
                lines.append(f" condition {condition}: reward spread {_list(values['reward_spread'])} "
                             f"lower clip {_list(values['lower_clip'])}")
        if summary["kl"]:
            lines.append(f" kl {_list(summary['kl'])} entropy {_list(summary['entropy'])}")
        lines += [f" note: penalty {f.name} subtracts {f.share:.1%} of the within-batch reward spread on average and is "
                  f"non-zero in {f.nonzero:.0%} of samples" for f in found]
        return "\n".join(lines)

    text = build(True)
    if count_tokens is not None and count_tokens(text) > token_cap:
        text = build(False)
    return text
