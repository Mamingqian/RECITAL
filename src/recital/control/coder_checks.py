"""Property checks for a component the Coder wrote, and the choice among several drafts of it.

The model writes its own unit test, and a self-written test can pass a wrong primitive. These checks are written by the
framework, per role, and run on the primitive as a callable; none of them reads a score of any oracle. A failed check
carries the concrete case (the input, what the primitive returned, what the property requires), which is what a repair
prompt shows. `choose` picks among drafts that pass by their agreement on a probe batch, a cheap proxy that needs no
model call and no GPU.

Roles and the call each check makes:

  hard_biological_gate        call(sequences) -> bool array, True where a sequence violates
  per_sequence_penalty        call(sequences) -> finite non-negative array
  per_oracle_utility          call(raw, target=target, conditioning=None) -> finite array
  cross_oracle_aggregation    call({oracle: utilities}) -> finite array
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

GATE, PENALTY = "hard_biological_gate", "per_sequence_penalty"
UTILITY, AGGREGATION = "per_oracle_utility", "cross_oracle_aggregation"
MAXIMIZE = {"kind": "maximize", "coordinate": "raw", "units": "score", "scale": 1.0, "reference": 0.0}
ROWS_CHECKED = 12


@dataclass(frozen=True)
class Check:
    """One property. `blocking` checks reject a draft; the others are reported (a gate that flags nothing, a flat utility)."""

    name: str
    passed: bool
    blocking: bool
    detail: str = ""


def _try(function: Callable[[], Any]) -> tuple[Any, str]:
    try:
        return function(), ""
    except Exception as exc:  # a primitive that raises on a valid input has failed the property, whatever it raised
        return None, f"{type(exc).__name__}: {str(exc)[:200]}"


def _same(left: np.ndarray, right: np.ndarray) -> bool:
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        return False
    if left.dtype == bool or right.dtype == bool:
        return bool((left == right).all())
    return bool(np.allclose(left.astype(float), right.astype(float), rtol=0, atol=1e-9, equal_nan=False))


def _shape_and_range(role: str, out: Any, n: int) -> Check:
    array = np.asarray(out)
    if array.shape != (n,):
        return Check("shape", False, True, f"returned shape {array.shape}, the role requires ({n},)")
    if role == GATE:
        ok = array.dtype == bool
        return Check("dtype_bool", ok, True, "" if ok else f"returned dtype {array.dtype}, a gate returns bool")
    values = array.astype(float)
    if not np.isfinite(values).all():
        return Check("finite", False, True, f"non-finite value at row {int(np.argmax(~np.isfinite(values)))}")
    if role == PENALTY and (values < 0).any():
        row = int(np.argmax(values < 0))
        return Check("non_negative", False, True, f"row {row} has penalty {values[row]}, a penalty is non-negative")
    return Check("shape_and_range", True, True)


def _sequence_checks(role: str, call: Callable[..., Any], sequences: Sequence[str],
                     conditioning: Sequence[str] | None = None, batch_level: bool = False) -> list[Check]:
    batch = list(sequences)
    n = len(batch)
    labels = None if conditioning is None else list(conditioning)
    if labels is not None and len(labels) != n:
        raise ValueError("conditioning must hold one label per sequence")

    def invoke(rows: Sequence[int]) -> Any:
        """The primitive on the rows `rows` of the batch, with their condition labels when the primitive takes them."""
        picked = [batch[i] for i in rows]
        return call(picked) if labels is None else call(picked, conditioning=[labels[i] for i in rows])

    everything = list(range(n))
    first, error = _try(lambda: invoke(everything))
    if error:
        return [Check("runs", False, True, error)]
    checks = [_shape_and_range(role, first, n)]
    if not checks[0].passed:
        return checks
    first = np.asarray(first)
    second, error = _try(lambda: invoke(everything))
    checks.append(Check("deterministic", error == "" and _same(first, np.asarray(second)), True,
                        error or "two calls on the same batch returned different values"))
    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    shuffled, error = _try(lambda: invoke([int(i) for i in order]))
    ok = error == "" and _same(np.asarray(shuffled), first[order])
    checks.append(Check("order_equivariant", ok, True, error or "shuffling the batch changed a row's value"))
    if batch_level:
        checks.extend(_collapse_check(invoke, batch))
    else:
        rows = [int(i) for i in rng.choice(n, size=min(ROWS_CHECKED, n), replace=False)]
        for row in rows:
            single, error = _try(lambda: invoke([row]))
            if error or not _same(np.asarray(single), first[[row]]):
                checks.append(Check("row_independent", False, True, error or (
                    f"sequence {batch[row]!r} alone gives {np.asarray(single).tolist()}, inside the batch {first[[row]].tolist()}")))
                break
        else:
            checks.append(Check("row_independent", True, True))
        duplicated, error = _try(lambda: invoke([0, 0]))
        checks.append(Check("duplicates_equal", error == "" and len(set(np.asarray(duplicated).tolist())) == 1, True,
                            error or f"the same sequence twice gave {np.asarray(duplicated).tolist()}"))
    empty, error = _try(lambda: invoke([]))
    checks.append(Check("empty_batch", error == "" and np.asarray(empty).shape == (0,), True,
                        error or f"an empty batch returned shape {np.asarray(empty).shape}"))
    values = first.astype(float)
    if role == GATE:
        rate = float(first.mean())
        checks.append(Check("not_all_flagged", rate < 1.0, True,
                            "" if rate < 1.0 else "the gate flags every sequence of the probe batch, so no row would earn reward"))
        checks.append(Check("flags_something", rate > 0.0, False,
                            "" if rate > 0.0 else "the gate flags no sequence of the probe batch"))
    else:
        checks.append(Check("not_constant", bool(np.ptp(values) > 0), False,
                            "" if np.ptp(values) > 0 else "the penalty is the same for every sequence of the probe batch"))
    return checks


def _distance(target: Mapping[str, Any], condition: str | None, grid: np.ndarray) -> np.ndarray | None:
    """How far each raw value is from satisfying the target (0 inside an interval), or None for a kind this does not know."""
    declared = target["by_condition"][condition] if "by_condition" in target else target
    kind = target["kind"]
    if kind == "maximize":
        return -grid
    if kind == "minimize":
        return grid
    if kind == "interval":
        lower = -np.inf if declared.get("lower") is None else float(declared["lower"])
        upper = np.inf if declared.get("upper") is None else float(declared["upper"])
        return np.maximum(np.maximum(lower - grid, 0.0), grid - upper)
    return None


def _collapse_check(invoke: Callable[[Sequence[int]], Any], batch: Sequence[str]) -> list[Check]:
    """For a penalty that depends on the whole batch: a batch of one sequence repeated must be penalized more than a batch of
    distinct sequences (a term that measures redundancy has to see the collapse it exists to prevent)."""
    distinct: dict[str, int] = {}
    for index, sequence in enumerate(batch):
        distinct.setdefault(sequence, index)
    rows = list(distinct.values())[:32]
    if len(rows) < 2:
        return [Check("collapse_penalized", True, False, "the probe batch has no two distinct sequences")]
    collapsed, error = _try(lambda: invoke([rows[0]] * len(rows)))
    varied, other = _try(lambda: invoke(rows))
    if error or other:
        return [Check("collapse_penalized", False, True, error or other)]
    higher = float(np.mean(np.asarray(collapsed, dtype=float))) > float(np.mean(np.asarray(varied, dtype=float))) + 1e-9
    return [Check("collapse_penalized", higher, True, "" if higher else (
        f"{len(rows)} copies of one sequence get mean penalty {float(np.mean(np.asarray(collapsed, dtype=float))):.4g}, "
        f"{len(rows)} distinct sequences get {float(np.mean(np.asarray(varied, dtype=float))):.4g}; a batch-level term must "
        "penalize the collapsed batch more"))]


def _utility_checks(call: Callable[..., Any], target: Mapping[str, Any], condition: str | None) -> list[Check]:
    grid = np.linspace(-3.0, 3.0, 61)
    labels = None if condition is None else [condition] * len(grid)
    out, error = _try(lambda: call(grid, target=target, conditioning=labels))
    if error:
        return [Check("runs", False, True, error)]
    checks = [_shape_and_range(UTILITY, out, len(grid))]
    if not checks[0].passed:
        return checks
    values = np.asarray(out, dtype=float)
    again, error = _try(lambda: call(grid, target=target, conditioning=labels))
    checks.append(Check("deterministic", error == "" and _same(values, np.asarray(again)), True,
                        error or "two calls on the same batch returned different values"))
    distance = _distance(target, condition, grid)
    if distance is None:
        checks.append(Check("moves_toward_target", True, False, f"no direction check for a {target['kind']} target"))
    else:
        step_d, step_u = np.diff(distance), np.diff(values)
        wrong = ((step_d > 1e-12) & (step_u > 1e-9)) | ((step_d < -1e-12) & (step_u < -1e-9))
        first_wrong = int(np.argmax(wrong)) if wrong.any() else 0
        checks.append(Check("moves_toward_target", not wrong.any(), True,
                            "" if not wrong.any() else (
                                f"going from raw {grid[first_wrong]:.2f} to {grid[first_wrong + 1]:.2f} moves "
                                f"{'away from' if step_d[first_wrong] > 0 else 'toward'} the {target['kind']} target "
                                f"but the utility {'rises' if step_u[first_wrong] > 0 else 'falls'}")))
    checks.append(Check("has_gradient", bool(np.ptp(values) > 1e-9), True,
                        "" if np.ptp(values) > 1e-9 else "the utility is constant over raw in [-3, 3], so it gives no learning signal"))
    half, error = _try(lambda: call(grid[:30], target=target, conditioning=None if labels is None else labels[:30]))
    checks.append(Check("batch_independent", error == "" and _same(np.asarray(half), values[:30]), False,
                        error or "the utility of the first 30 rows changes when the other rows are removed "
                        "(acceptable only for a running normalizer)"))
    empty, error = _try(lambda: call(np.asarray([], dtype=float), target=target, conditioning=None if labels is None else []))
    checks.append(Check("empty_batch", error == "" and np.asarray(empty).shape == (0,), True,
                        error or f"an empty batch returned shape {np.asarray(empty).shape}"))
    return checks


def _aggregation_checks(call: Callable[[Mapping[str, np.ndarray]], Any], oracles: Sequence[str]) -> list[Check]:
    rng = np.random.default_rng(0)
    base = {name: rng.normal(size=64) for name in oracles}
    out, error = _try(lambda: call(base))
    if error:
        return [Check("runs", False, True, error)]
    checks = [_shape_and_range(AGGREGATION, out, 64)]
    if not checks[0].passed:
        return checks
    values = np.asarray(out, dtype=float)
    again, error = _try(lambda: call(base))
    checks.append(Check("deterministic", error == "" and _same(values, np.asarray(again)), True,
                        error or "two calls on the same utilities returned different values"))
    violations = []
    for name in oracles:
        raised = {**base, name: base[name] + 0.5}
        higher, error = _try(lambda: call(raised))
        if error or (np.asarray(higher, dtype=float) < values - 1e-9).any():
            violations.append(name)
    checks.append(Check("monotone_in_each_oracle", not violations, True,
                        "" if not violations else f"raising the utility of {violations[0]} lowered the aggregate on some row"))
    if len(oracles) > 1:
        reordered = dict(reversed(list(base.items())))
        flipped, error = _try(lambda: call(reordered))
        checks.append(Check("oracle_order_invariant", error == "" and _same(np.asarray(flipped), values), False,
                            error or "listing the oracles in another order changed the aggregate (acceptable for declared weights)"))
    perm = rng.permutation(64)
    shuffled, error = _try(lambda: call({name: array[perm] for name, array in base.items()}))
    checks.append(Check("order_equivariant", error == "" and _same(np.asarray(shuffled), values[perm]), True,
                        error or "shuffling the rows changed a row's aggregate"))
    half, error = _try(lambda: call({name: array[:32] for name, array in base.items()}))
    checks.append(Check("row_independent", error == "" and _same(np.asarray(half), values[:32]), True,
                        error or "the aggregate of the first 32 rows changes when the other rows are removed"))
    return checks


def battery(role: str, call: Callable[..., Any], *, sequences: Sequence[str] = (), oracles: Sequence[str] = ("a", "b", "c"),
            target: Mapping[str, Any] = MAXIMIZE, condition: str | None = None,
            conditioning: Sequence[str] | None = None, batch_level: bool = False) -> list[Check]:
    """Every property of `role` on `call`.

    `sequences` is the probe batch of a gate or a penalty and `conditioning` the condition label of each of its rows (the
    call then takes `conditioning=`); `oracles` are the approved oracle names of an aggregation; `target` and `condition`
    are the target a utility is called with and the condition label of its rows. `batch_level` declares a penalty whose value
    for a row depends on the other rows of the batch (a redundancy or diversity term): the row-independence and duplicate
    checks, which it violates by design, are replaced by `collapse_penalized`."""
    if role in (GATE, PENALTY):
        return _sequence_checks(role, call, sequences, conditioning, batch_level and role == PENALTY)
    if role == UTILITY:
        return _utility_checks(call, target, condition)
    if role == AGGREGATION:
        return _aggregation_checks(call, oracles)
    raise ValueError(f"no checks for the role {role!r}")


def passed(checks: Sequence[Check]) -> bool:
    """True when no blocking check failed."""
    return all(check.passed for check in checks if check.blocking)


def feedback(checks: Sequence[Check]) -> str:
    """The failed blocking checks, each with its concrete case: what a repair prompt shows."""
    failed = [check for check in checks if check.blocking and not check.passed]
    return "\n".join(f"- property `{check.name}` failed: {check.detail}" for check in failed)


def agreement(role: str, left: np.ndarray, right: np.ndarray, mode: str = "rank") -> float:
    """How much two drafts' outputs on the same inputs agree: the share of equal flags for a gate; otherwise, in mode `rank`,
    the rank correlation of the values (1.0 when they are equal; 0.0 when one is constant and the other is not), and in mode
    `value` one minus the mean absolute difference over the mean absolute value (floored at 0), which also sees a difference of
    scale that a rank correlation cannot."""
    left, right = np.asarray(left), np.asarray(right)
    if role == GATE:
        return float((left.astype(bool) == right.astype(bool)).mean())
    left, right = left.astype(float), right.astype(float)
    if np.allclose(left, right, rtol=0, atol=1e-9):
        return 1.0
    if mode == "value":
        scale = float(np.mean(np.abs(np.concatenate([left, right]))))
        return max(0.0, 1.0 - float(np.mean(np.abs(left - right))) / scale) if scale > 0 else 0.0
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return 0.0
    ranks = lambda values: np.argsort(np.argsort(values, kind="stable"), kind="stable").astype(float)  # noqa: E731
    return float(np.corrcoef(ranks(left), ranks(right))[0, 1])


def choose(role: str, outputs: Mapping[Any, np.ndarray], mode: str = "rank") -> tuple[Any, dict[Any, float]]:
    """The draft whose outputs agree most with the other drafts' (the medoid), and each draft's mean agreement.

    `outputs` maps a draft to its output on the probe batch, for the drafts that passed. One draft is chosen as it stands;
    ties go to the earlier draft (the mapping's order)."""
    if not outputs:
        raise ValueError("no draft to choose from")
    keys = list(outputs)
    if len(keys) == 1:
        return keys[0], {keys[0]: 1.0}
    mean = {key: float(np.mean([agreement(role, outputs[key], outputs[other], mode) for other in keys if other != key]))
            for key in keys}
    best = max(keys, key=lambda key: (mean[key], -keys.index(key)))
    return best, mean


def probe_sequences(training: Sequence[str], n_real: int = 300, seed: int = 0, n_noise: int = 40) -> list[str]:
    """The probe batch of a gate or a penalty: real sequences evenly spaced through `training`, degenerate ones that trip
    each kind of gate, and random ones."""
    rng = np.random.default_rng(seed)
    real = [training[i] for i in np.linspace(0, len(training) - 1, min(n_real, len(training)), dtype=int)]
    length = len(real[0]) if real else 50
    alphabet = sorted({character for sequence in real for character in sequence}) or list("ACGT")
    noise = lambda n: "".join(rng.choice(alphabet, n))  # noqa: E731
    first, second = alphabet[0], alphabet[min(1, len(alphabet) - 1)]
    degenerate = [first * length, second * length, (first + second) * (length // 2) + first * (length % 2),
                  first * (length // 2) + noise(length - length // 2), second * 7 + noise(max(length - 7, 0)),
                  (first + second + second) * (length // 3) + second * (length % 3)]
    return real + degenerate + [noise(length) for _ in range(n_noise)]


def probe_output(role: str, call: Callable[..., Any], *, sequences: Sequence[str] = (), oracles: Sequence[str] = ("a", "b", "c"),
                 target: Mapping[str, Any] = MAXIMIZE, condition: str | None = None) -> np.ndarray:
    """What `call` returns on the probe inputs of `battery`: the quantity `choose` compares between drafts."""
    if role in (GATE, PENALTY):
        return np.asarray(call(list(sequences)))
    if role == UTILITY:
        grid = np.linspace(-3.0, 3.0, 61)
        return np.asarray(call(grid, target=target, conditioning=None if condition is None else [condition] * len(grid)), dtype=float)
    rng = np.random.default_rng(0)
    return np.asarray(call({name: rng.normal(size=64) for name in oracles}), dtype=float)
