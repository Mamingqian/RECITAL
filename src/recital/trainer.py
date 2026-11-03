"""GRPO/DAPO trainer abstraction.

The framework calls `trainer.run(spec, reward_fn, context)` and gets back a
trajectory dict that the Diagnostician parses. Concrete trainers:

  - MockTrainer:        deterministic, no compute; used for dry runs.
  - TRLDAPOTrainer:     real GRPO/DAPO via the TRL library (requires torch + trl).
                        Hyperparameters (lr, KL, group_size, steps) are read
                        directly from the spec.

Generation uses the adapter-declared vocabulary. Raw decoded candidates are
retained, validated, and scored without sequence repair.
"""
from __future__ import annotations

import inspect
import json
import logging
import random
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from recital.randomness import evaluation_mode, isolated_rng

LOG = logging.getLogger(__name__)

# Base-policy reference libraries, keyed by published checkpoint and sampler.
# A different parent policy produces a different snapshot identity; cache reuse
# never follows parameter count or model architecture alone.
_BASE_LIBRARY_CACHE: dict[tuple, list[str]] = {}


class CandidateTrainingError(RuntimeError):
    """A candidate-local failure safe to persist as a nontraining outcome."""


class CandidateRewardExecutionError(CandidateTrainingError):
    """Reward-DSL construction/execution failed for the proposed candidate."""


def _candidate_reward_build(factory: Callable[[], Any], *, stage: str) -> Any:
    try:
        return factory()
    except CandidateTrainingError:
        raise
    except Exception as exc:  # noqa: BLE001 - reward DSL is the candidate boundary
        raise CandidateRewardExecutionError(
            f"candidate reward construction failed during {stage}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _kl_window_summary(kl_windows: list[float]) -> tuple[float | None, float | None]:
    """Median and maximum of the logged KL windows; (None, None) when none is logged or any is nonfinite.

    Each logged value is TRL's mean over one logging window of an unbounded per-token estimator, so the
    middle window stands for the run and the maximum keeps a spike visible.
    """
    if not kl_windows or not np.isfinite(kl_windows).all():
        return None, None
    return float(np.median(kl_windows)), float(np.max(kl_windows))


def _reward_execution_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compact optimization-only telemetry; full call records stay in artifacts."""
    total = sum(r["candidate_count"] for r in records)
    valid = sum(r["composition"].get("count", 0) for r in records)
    summary = {
        "population": "generated_training_candidates",
        "reward_calls": len(records), "candidate_count": total,
        "invalid_count": sum(r["invalid_count"] for r in records),
        "valid_composition_count": valid,
        "by_prompt": {},
    }
    # Keep component activation visible to agents. Reference-library screening
    # can activate an exact-copy penalty even when it never affects training.
    for role in ("penalties", "gates"):
        components = {}
        indices = sorted({idx for row in records for idx in row["composition"].get(role, {})})
        for idx in indices:
            rows = [(row["composition"].get("count", 0), row["composition"][role][idx])
                    for row in records if idx in row["composition"].get(role, {})]
            count = sum(n for n, _ in rows)
            if not count:
                continue
            item = {"candidate_count": count}
            if role == "penalties":
                nonzero = sum(value["nonzero_count"] for _, value in rows)
                item.update(name=rows[0][1]["name"], nonzero_count=nonzero,
                            nonzero_fraction=nonzero / count,
                            mean=sum(n * value["mean"] for n, value in rows) / count,
                            maximum=max(value["maximum"] for _, value in rows))
            else:
                violations = sum(round(n * value["violation_fraction"]) for n, value in rows)
                item.update(violation_count=violations, violation_fraction=violations / count)
            components[idx] = item
        summary[role] = components
    aggregation_rows = [(r["composition"].get("count", 0), r["composition"]["aggregation"])
                        for r in records if r["composition"].get("aggregation") and r["composition"].get("count", 0)]
    if aggregation_rows:
        n = sum(count for count, _ in aggregation_rows)
        summary["aggregation"] = {"name": aggregation_rows[0][1]["name"], "candidate_count": n,
            "mean": sum(count * value["mean"] for count, value in aggregation_rows) / n,
            "mean_within_batch_std": sum(count * value["std"] for count, value in aggregation_rows) / n}
        components = {}
        for name in sorted({name for _, value in aggregation_rows for name in value.get("components", {})}):
            values = [(count, value["components"][name]) for count, value in aggregation_rows
                      if name in value.get("components", {})]
            count = sum(n for n, _ in values)
            components[name] = {
                "weight": values[-1][1]["weight"], "candidate_count": count,
                "first_batch_mean": values[0][1]["mean"], "last_batch_mean": values[-1][1]["mean"],
                "mean": sum(n * value["mean"] for n, value in values) / count,
                "mean_absolute": sum(n * value["mean_absolute"] for n, value in values) / count,
                "nonzero_count": sum(value["nonzero_count"] for _, value in values),
            }
        if components:
            summary["aggregation"]["components"] = components
    summary["training_oracles"] = {}
    for name in sorted({name for r in records for name in r.get("training_oracles", {})}):
        scored = [(r.get("reward_call"), r["training_oracles"][name]) for r in records
                  if name in r.get("training_oracles", {})
                  and r["training_oracles"][name]["mean"] is not None]
        if scored:
            values = [value for _, value in scored]
            summary["training_oracles"][name] = {"candidate_count": sum(v["count"] for v in values),
                # "first" is the first batch with a readable mean, which is not
                # necessarily batch 1; say which one it was rather than implying it.
                "first_batch_mean": values[0]["mean"], "last_batch_mean": values[-1]["mean"],
                "first_batch_reward_call": scored[0][0],
                "last_batch_reward_call": scored[-1][0],
                "batches_recorded": len(values), "batches_seen": len(records),
                "mean": (sum(v["count"] * np.asarray(v["mean"]) for v in values)
                         / sum(v["count"] for v in values)).tolist()}
    for key in ("lower_clip_fraction", "gate_violation_fraction"):
        summary[key] = (sum(r["composition"].get("count", 0) * r["composition"].get(key, 0.0)
                            for r in records) / valid) if valid else None
    # Over the batches that measured a ceiling: a batch whose utilities declare none says nothing about it, so it is
    # left out of the weighting and not counted as zero.
    ceiling_rows = [(r["composition"]["count"], r["composition"]["upper_clip_fraction"]) for r in records
                    if r["composition"].get("upper_clip_fraction") is not None and r["composition"].get("count", 0)]
    summary["upper_clip_fraction"] = (sum(n * share for n, share in ceiling_rows) / sum(n for n, _ in ceiling_rows)
                                      if ceiling_rows else None)
    for prompt in sorted({p for r in records for p in r["by_prompt"]}):
        rows = [r["by_prompt"][prompt] for r in records if prompt in r["by_prompt"]]
        count = sum(row["count"] for row in rows)
        summary["by_prompt"][prompt] = {
            "candidate_count": count,
            "mean_within_batch_reward_std": sum(row["count"] * row["reward_std"] for row in rows) / count,
        }
    return summary


def _validate_runtime_primitives(
    spec: dict[str, Any], context: TrainContext, reward_fn: Any = None,
) -> None:
    """Resolve run-local constructors through context, never through JSON specs."""
    resolver = context.extras.get("primitive_resolver")
    if resolver is None:
        return
    if not callable(resolver):
        raise TypeError("primitive_resolver must be callable")
    from recital.spec.composer import validate_spec

    problems = validate_spec(
        spec, available_oracles=list(context.oracle_fns), primitive_resolver=resolver,
        known_gate_names=set(getattr(reward_fn, "_gate_registry", {}) or {}),
    )
    if problems:
        raise CandidateTrainingError("runtime reward validation failed: " + "; ".join(problems))


def _candidate_reward_call(
    reward_fn: Callable[..., Any],
    *,
    sequences: list[str],
    oracle_outputs: dict[str, np.ndarray],
    expected_n: int,
    stage: str,
    conditioning: list[str] | None = None,
) -> np.ndarray:
    try:
        kwargs = {"sequences": sequences, "oracle_outputs": oracle_outputs}
        if conditioning is not None:
            kwargs["conditioning"] = conditioning
        return _require_finite_reward_vector(
            reward_fn(**kwargs),
            expected_n=expected_n,
            stage=stage,
        )
    except CandidateTrainingError:
        raise
    except Exception as exc:  # noqa: BLE001 - reward DSL is the candidate boundary
        raise CandidateRewardExecutionError(
            f"candidate reward execution failed during {stage}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _seed_evaluation_rng(seed: int) -> dict[str, int]:
    """Reset generation RNGs immediately before the final evaluation library."""
    if isinstance(seed, bool):
        raise ValueError("evaluation_seed must be an integer, not bool")
    value = int(seed)
    if value < 0:
        raise ValueError("evaluation_seed must be nonnegative")
    random.seed(value)
    numpy_seed = value % (2**32)
    np.random.seed(numpy_seed)
    applied = {"python_random": value, "numpy": numpy_seed}
    try:
        import torch

        torch.manual_seed(value)
        applied["torch"] = value
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)
            applied["torch_cuda"] = value
    except ImportError:
        pass
    return applied


@dataclass
class TrainContext:
    """Anything the trainer needs that isn't part of the reward specification."""

    policy: Any
    train_data: Any
    valid_data: Any | None = None
    conditioning_prompts: list[str] = field(default_factory=list)
    token_alphabet: str = "ACGT"
    oracle_fns: dict[str, Callable[[list[str]], np.ndarray]] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


# THE list of protocol conditions that read a reserved oracle ONLINE.
_SENTINEL_ONLINE_CONDITIONS = frozenset(
    {
        "self_validating",
        "shuffled_sentinel",
        "heldout_early_stopping",
    }
)


def _checkpoint_layout(
    spec: dict[str, Any], context: TrainContext, base_dir: Path
) -> tuple[Path, int, bool]:
    """Mid-run checkpoint directory, save cadence, and resume flag.

    Default (no extras): no mid-run saves and no subdirectory. Opted in via
    ``checkpoint_save_steps`` / ``resume_from_checkpoint`` extras: checkpoints
    isolate per iteration (the loop's runtime transaction tag) so a checkpoint
    only ever resumes the (iteration, spec) pair that produced it; direct
    fixed-spec callers fall back to a digest of the semantic spec fields.
    """
    save_every = int(context.extras.get("checkpoint_save_steps") or 0)
    resume_mid_run = bool(context.extras.get("resume_from_checkpoint"))
    if save_every <= 0 and not resume_mid_run:
        return base_dir, 0, False
    transaction = spec.get("_framework_iteration_transaction") or {}
    iteration_idx = transaction.get("outer_iteration")
    if iteration_idx is not None:
        tag = f"iter{int(iteration_idx):03d}"
    else:
        semantic = {
            key: spec.get(key)
            for key in (
                "enabled_oracles",
                "held_out_oracles",
                "transforms",
                "aggregator",
                "shape",
                "penalties",
                "gates",
                "hyperparameters",
                "reward_clip_low",
                "reward_clip_high",
            )
        }
        mix = 2166136261
        for byte in json.dumps(semantic, sort_keys=True, default=str).encode():
            mix ^= byte
            mix = (mix * 16777619) & 0xFFFFFFFF
        tag = f"spec_{mix:08x}"
    return base_dir / tag, save_every, resume_mid_run


def _latest_checkpoint(ckpt_dir: Path) -> str | None:
    """Newest completed HF checkpoint, skipping interrupted checkpoint writes.

    Transformers creates the directory before saving weights, optimizer and
    RNG, then writes ``trainer_state.json`` last. A numbered directory alone
    can therefore be the half-written checkpoint from a preempted process.
    Use that final state file as the completion signal without loading large
    tensors or adding a second checkpoint format.
    """
    existing = sorted(
        (
            p
            for p in ckpt_dir.glob("checkpoint-*")
            if p.is_dir() and p.name.rsplit("-", 1)[-1].isdigit()
        ),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]),
        reverse=True,
    )
    for path in existing:
        try:
            state = json.loads((path / "trainer_state.json").read_text())
            step = int(path.name.rsplit("-", 1)[-1])
            if not isinstance(state, dict) or type(state.get("global_step")) is not int or (
                state["global_step"] != step
            ):
                raise ValueError("trainer state does not match the checkpoint step")
        except (OSError, ValueError) as exc:
            LOG.warning("Skipping incomplete checkpoint %s: %s", path, exc)
            continue
        return str(path)
    return None


def _oracle_roles(
    spec: dict[str, Any], context: TrainContext
) -> tuple[list[str], list[str], set[str]]:
    """Return validated enabled, held-out, and framework-private audit roles.

    This is a trainer-side security boundary, not merely a convenience check.
    Direct callers can bypass the Critic/``validate_spec`` path, so the trainer
    must never fall back from an empty enabled list to scoring every registered
    oracle.  Audit sources may be registered for a terminal harness, but are
    forbidden from every runtime scoring path declared here.
    """
    enabled = [str(name) for name in (spec.get("enabled_oracles") or [])]
    held_out = [str(name) for name in (spec.get("held_out_oracles") or [])]
    runtime = spec.get("_framework_runtime") or {}
    audit = {
        str(name)
        for name in (
            list(context.extras.get("audit_oracles") or [])
            + list(runtime.get("audit_oracles") or [])
        )
    }
    if not enabled:
        raise ValueError("spec.enabled_oracles must be non-empty")
    for role, names in (("enabled_oracles", enabled), ("held_out_oracles", held_out)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"{role} contains duplicate source(s): {duplicates}")
    overlap = sorted(set(enabled) & set(held_out))
    if overlap:
        raise ValueError(
            "enabled_oracles and held_out_oracles must be disjoint; "
            f"overlap={overlap}"
        )
    audit_overlap = sorted((set(enabled) | set(held_out)) & audit)
    if audit_overlap:
        raise ValueError(
            "runtime enabled/held-out roles contain framework-private audit "
            f"source(s): {audit_overlap}"
        )
    reward_pool = {
        str(name) for name in (runtime.get("reward_oracles") or [])
    }
    sentinel_pool = {
        str(name) for name in (runtime.get("sentinel_oracles") or [])
    }
    if reward_pool:
        invalid_enabled = sorted(set(enabled) - reward_pool)
        if invalid_enabled:
            raise ValueError(
                "enabled_oracles contains source(s) outside the frozen reward "
                f"pool: {invalid_enabled}"
            )
    if sentinel_pool:
        invalid_held_out = sorted(set(held_out) - sentinel_pool)
        if invalid_held_out:
            raise ValueError(
                "held_out_oracles contains source(s) outside the frozen sentinel "
                f"pool: {invalid_held_out}"
            )
    missing_enabled = sorted(set(enabled) - set(context.oracle_fns))
    if missing_enabled:
        raise KeyError(f"enabled oracle scorer(s) are not registered: {missing_enabled}")
    return enabled, held_out, audit


def _protocol_condition(spec: dict[str, Any], context: TrainContext) -> str:
    """Resolve the protocol condition and reject contradictory caller state."""
    runtime_condition = (spec.get("_framework_runtime") or {}).get(
        "protocol_condition"
    )
    context_condition = context.extras.get("protocol_condition")
    if (
        runtime_condition is not None
        and context_condition is not None
        and str(runtime_condition) != str(context_condition)
    ):
        raise ValueError(
            "protocol condition mismatch between loop runtime metadata and "
            f"TrainContext: {runtime_condition!r} != {context_condition!r}"
        )
    return str(context_condition or runtime_condition or "legacy")


def _evaluation_oracle_keys(
    spec: dict[str, Any], context: TrainContext
) -> tuple[list[str], list[str]]:
    """Return (all evaluation keys, held-out keys actually scored online)."""
    enabled, held_out, audit = _oracle_roles(spec, context)
    condition = _protocol_condition(spec, context)
    score_held_out = condition == "legacy" or condition in _SENTINEL_ONLINE_CONDITIONS
    scored_held_out = held_out if score_held_out else []
    diagnostics = [
        str(name) for name in (context.extras.get("diagnostic_oracles") or [])
    ]
    declared_sentinels = {
        str(name)
        for name in (
            (spec.get("_framework_runtime") or {}).get("sentinel_oracles") or []
        )
    }
    private_diagnostics = sorted(set(diagnostics) & audit)
    if private_diagnostics:
        raise ValueError(
            "diagnostic_oracles contains framework-private audit source(s): "
            f"{private_diagnostics}"
        )
    # A no-self-validation control must not score a sentinel through the generic
    # diagnostic path, which a task's full oracle panel in ``diagnostic_oracles``
    # would otherwise open.
    diagnostics = [
        name
        for name in diagnostics
        if name not in declared_sentinels or name in set(scored_held_out)
    ]
    if not score_held_out:
        diagnostics = [name for name in diagnostics if name not in set(held_out)]
    keys = list(dict.fromkeys([*enabled, *scored_held_out, *diagnostics]))
    return keys, scored_held_out


def _required_evaluation_oracles(
    spec: dict[str, Any],
    context: TrainContext,
    *,
    scored_held_out: list[str],
) -> set[str]:
    """Return scorers whose final-evaluation output must fail closed."""
    enabled, _, _ = _oracle_roles(spec, context)
    declared = (
        context.extras.get("required_held_out_oracles")
        or (spec.get("_framework_runtime") or {}).get(
            "required_held_out_oracles"
        )
        or scored_held_out
    )
    required_held_out = set(str(name) for name in declared) & set(scored_held_out)
    return set(enabled) | required_held_out


def _validated_oracle_output(
    *,
    name: str,
    values: Any,
    expected_n: int,
    expected_widths: dict[str, Any] | None = None,
    required_finite_columns: dict[str, Any] | None = None,
    require_finite: bool,
    stage: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate one scorer result at the framework-owned trust boundary.

    A task may declare an exact multi-output width and a subset of columns that
    must be finite. In the absence of such a declaration, every value from a
    required scorer must be finite.
    """
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{stage} oracle {name!r} did not return a numeric array"
        ) from exc

    widths = expected_widths or {}
    raw_width = widths.get(name)
    if raw_width is not None:
        if isinstance(raw_width, bool):
            raise ValueError(f"invalid expected width for oracle {name!r}")
        width = int(raw_width)
        if width < 1 or width != raw_width:
            raise ValueError(f"invalid expected width for oracle {name!r}")
        expected_shape = (expected_n,) if width == 1 else (expected_n, width)
        if array.shape != expected_shape:
            raise ValueError(
                f"{stage} oracle {name!r} must return shape "
                f"{expected_shape}, got {array.shape}"
            )
    elif array.ndim not in {1, 2} or array.shape[0] != expected_n:
        raise ValueError(
            f"{stage} oracle {name!r} must return one row per sequence "
            f"({expected_n}), got shape {array.shape}"
        )

    columns_by_oracle = required_finite_columns or {}
    declared_columns = columns_by_oracle.get(name)
    if declared_columns is None:
        finite_view = array
        checked_columns: list[int] | str = "all" if require_finite else []
    else:
        if not isinstance(declared_columns, (list, tuple)):
            raise ValueError(
                f"required finite columns for oracle {name!r} must be a list"
            )
        columns = [int(column) for column in declared_columns]
        if len(columns) != len(set(columns)) or any(column < 0 for column in columns):
            raise ValueError(
                f"required finite columns for oracle {name!r} are invalid"
            )
        width = 1 if array.ndim == 1 else int(array.shape[1])
        if any(column >= width for column in columns):
            raise ValueError(
                f"required finite columns for oracle {name!r} exceed width {width}"
            )
        finite_view = array if array.ndim == 1 else array[:, columns]
        checked_columns = columns

    required_finite_fraction = (
        float(np.mean(np.isfinite(finite_view))) if finite_view.size else 0.0
    )
    if require_finite and required_finite_fraction != 1.0:
        bad = int(np.size(finite_view) - np.count_nonzero(np.isfinite(finite_view)))
        raise FloatingPointError(
            f"{stage} oracle {name!r} has {bad} nonfinite required value(s); "
            f"columns={checked_columns}"
        )
    return array, {
        "status": "ok",
        "finite_fraction": (
            float(np.mean(np.isfinite(array))) if array.size else 0.0
        ),
        "required_finite_fraction": required_finite_fraction,
        "required_finite_columns": checked_columns,
        "shape": list(array.shape),
    }


# Every evaluated library -- the final library and the base-policy reference --
# is drawn with these. Training does NOT truncate: `gen_kwargs`
# sets only suppress_tokens/min_new_tokens/use_cache, so TRL samples at top_p=1.0.
# The evaluation default matches training. A task may declare truncation through
# context.extras["sampling_top_p"].
EVALUATION_SAMPLING_TOP_P = 1.0
EVALUATION_SAMPLING_TEMPERATURE = 1.0


def _target_indices(
    prompts: list[Any], conditioning_prompts: list[str]
) -> list[int]:
    """Map prompts to declared strata, failing closed on an unknown prompt."""
    normalized = [str(prompt).strip() for prompt in conditioning_prompts]
    if len(set(normalized)) != len(normalized):
        raise ValueError("conditioning_prompts must be unique after whitespace normalization")
    prompt_to_index = {prompt: index for index, prompt in enumerate(normalized)}
    observed = [str(prompt).strip() for prompt in prompts]
    unknown = sorted(set(observed) - set(prompt_to_index))
    if unknown:
        raise ValueError(
            "generated batch contains prompt(s) outside conditioning_prompts: "
            f"{unknown}"
        )
    return [prompt_to_index[prompt] for prompt in observed]


def _oracle_target_columns(prompts, conditioning_prompts, mapping=None) -> list[int]:
    """Resolve oracle columns independently of the active prompt subset/order."""
    local = _target_indices(list(prompts), list(conditioning_prompts))
    if mapping is None:
        return local  # Without a mapping, columns follow the ordering of the active panel.
    normalized = {str(prompt).strip(): column for prompt, column in mapping.items()}
    if len(normalized) != len(mapping) or any(
        type(column) is not int or column < 0 for column in normalized.values()
    ):
        raise ValueError("oracle_target_indices requires unique prompts and nonnegative integer columns")
    missing = {str(prompt).strip() for prompt in conditioning_prompts} - normalized.keys()
    if missing:
        raise ValueError(f"oracle_target_indices omits active prompts: {sorted(missing)}")
    return [normalized[str(prompt).strip()] for prompt in prompts]


def _column_nanmean(values: np.ndarray) -> list | None:
    """Per-column mean ignoring NaN; None only when a column is entirely NaN."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(np.where(np.isfinite(values), values, np.nan), axis=0)
    return None if not np.isfinite(np.asarray(mean)).any() else np.asarray(mean).tolist()


def _valid_reward_std(rewards, prompts, label, valid) -> float:
    """Dispersion of the rewards the COMPOSITION produced, excluding invalid fills."""
    selected = (np.asarray(prompts) == label) & np.asarray(valid, dtype=bool)
    return float(np.std(np.asarray(rewards)[selected])) if selected.any() else 0.0


def _default_invalid_penalty() -> float:
    """Single source for the reward floor; see spec.composer for why it matters."""
    from recital.spec.composer import DEFAULT_INVALID_PENALTY

    return DEFAULT_INVALID_PENALTY


def _rebind_conditioning(rfn, prompts, conditioning_prompts, oracle_target_indices=None):
    """Point a composed reward at this batch's strata WITHOUT rebuilding it.

    Rebuilding is the obvious alternative and it is wrong: every stateful
    component (EmaZScore, contrastive_mingap's running moments) starts from an
    empty state, so `decay=0.99` degrades to standardizing the single batch in
    hand. Only the target indices vary per batch; the composition does not.
    """
    columns = list(conditioning_prompts or [])
    ti = _target_indices(list(prompts), columns)
    if rfn._reward_input is not None:
        rfn._reward_input.target_indices = _oracle_target_columns(
            list(prompts), columns, oracle_target_indices
        )
    if rfn._aggregator is not None and hasattr(rfn._aggregator, "target_indices"):
        rfn._aggregator.target_indices = _oracle_target_columns(
            list(prompts), columns, oracle_target_indices
        )
    if rfn._shape is not None and hasattr(rfn._shape, "target_indices"):
        rfn._shape.target_indices = ti
    for component in (*rfn._penalties, *rfn._gates):
        if hasattr(component, "target_indices"):
            component.target_indices = ti
    return rfn


def _auxiliary_telemetry(
    *,
    spec: dict[str, Any],
    context: TrainContext,
    sequences: list[str],
    conditioning: list[str] | None,
    oracle_outputs: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Build task summary and complexity telemetry outside reward aggregation."""
    from recital.sequence_complexity import (
        sequence_complexity_metrics,
        stratify_sequence_complexity_metrics,
    )

    complexity_k = int(
        context.extras.get("complexity_k")
        or (8 if sequences and len(sequences[0]) >= 100 else 3)
    )
    at_run_min = int(context.extras.get("at_run_min") or 5)
    training_set = context.extras.get("training_set") or []
    complexity = sequence_complexity_metrics(
        sequences,
        training_set=training_set,
        k=complexity_k,
        at_run_min=at_run_min,
    )
    if conditioning and len(conditioning) == len(sequences):
        complexity["by_stratum"] = stratify_sequence_complexity_metrics(
            complexity,
            sequences,
            [str(value) for value in conditioning],
        )

    # gc_hacking compares the generated GC distribution against the corpus the
    # policy was pretrained on. Without this key it returns "no train_gc_mean"
    # and passes vacuously.
    train_gc_mean: float | None = None
    if training_set:
        gc_fractions = [
            sum(1 for base in str(seq).upper() if base in ("G", "C")) / len(seq)
            for seq in training_set
            if seq
        ]
        if gc_fractions:
            train_gc_mean = float(np.mean(gc_fractions))

    telemetry: dict[str, Any] = {
        "protocol_condition": _protocol_condition(spec, context),
        "train_gc_mean": train_gc_mean,
        "required_held_out_oracles": list(
            context.extras.get("required_held_out_oracles")
            or (spec.get("_framework_runtime") or {}).get(
                "required_held_out_oracles"
            )
            or spec.get("held_out_oracles")
            or []
        ),
        "sequence_complexity": complexity,
    }
    return telemetry


# ---------------------------------------------------------------------------
# MockTrainer: deterministic, no compute. Exercises orchestration without a GPU.
# ---------------------------------------------------------------------------

@dataclass
class MockTrainer:
    """Deterministic stand-in. Generates sequences from `context.train_data` and
    scores them via `reward_fn`, ignoring any actual optimization.
    """

    rng_seed: int = 0
    samples_per_iter: int = 64

    def run(
        self,
        *,
        spec: dict[str, Any],
        reward_fn: Callable[..., np.ndarray],
        context: TrainContext,
    ) -> dict[str, Any]:
        _validate_runtime_primitives(spec, context, reward_fn)
        rng = np.random.default_rng(self.rng_seed)
        L = 50
        seqs = [
            "".join(rng.choice(list(context.token_alphabet), size=L))
            for _ in range(self.samples_per_iter)
        ]
        cond = (
            list(rng.choice(context.conditioning_prompts, size=self.samples_per_iter))
            if context.conditioning_prompts
            else None
        )
        evaluation_keys, scored_held_out = _evaluation_oracle_keys(spec, context)
        required = _required_evaluation_oracles(
            spec, context, scored_held_out=scored_held_out
        )
        oracle_outputs: dict[str, np.ndarray] = {}
        for oracle in evaluation_keys:
            if oracle not in context.oracle_fns:
                if oracle in required:
                    raise KeyError(
                        f"required evaluation oracle is not registered: {oracle}"
                    )
                continue
            array, _ = _validated_oracle_output(
                name=oracle,
                values=context.oracle_fns[oracle](seqs),
                expected_n=len(seqs),
                expected_widths=context.extras.get("oracle_expected_widths"),
                required_finite_columns=context.extras.get(
                    "oracle_required_finite_columns"
                ),
                require_finite=oracle in required,
                stage="mock final evaluation",
            )
            oracle_outputs[oracle] = array

        spec_patched = _bind_prompt_conditioning(
            spec, cond, context.conditioning_prompts, context.extras.get("group_intervals"),
            oracle_target_indices=context.extras.get("oracle_target_indices"),
            primitive_resolver=context.extras.get("primitive_resolver")
            or getattr(reward_fn, "_primitive_resolver", None),
        )
        if spec_patched is not spec:
            from recital.spec.composer import ComposedRewardFn, RewardSpec

            reward_fn = _candidate_reward_build(
                lambda: ComposedRewardFn(
                    RewardSpec.from_dict(spec_patched),
                    gate_registry=getattr(reward_fn, "_gate_registry", None),
                    training_set=getattr(reward_fn, "_training_set", None),
                    reference_libraries=getattr(reward_fn, "_reference_libraries", None),
                    primitive_resolver=context.extras.get("primitive_resolver")
                    or getattr(reward_fn, "_primitive_resolver", None),
                ),
                stage="mock conditional reward binding",
            )
        rewards = (
            _candidate_reward_call(
                reward_fn,
                sequences=seqs,
                oracle_outputs=oracle_outputs,
                expected_n=len(seqs),
                stage="mock final evaluation",
                conditioning=(
                    _reward_conditioning(
                        cond or [], context.conditioning_prompts,
                        context.extras.get("condition_names"),
                    )
                    if _requires_reward_conditioning(spec, reward_fn, context.extras.get("primitive_resolver")) else None
                ),
            )
            if oracle_outputs
            else np.zeros(len(seqs))
        )
        trajectory = {
            "sequences": seqs,
            "conditioning": cond,
            "oracle_outputs": oracle_outputs,
            "heldout_oracle_outputs": {
                oracle: oracle_outputs[oracle]
                for oracle in scored_held_out
                if oracle in oracle_outputs
            },
            "enabled_oracles": list(spec.get("enabled_oracles") or []),
            "reward": np.asarray(rewards, dtype=float),
            "base_policy_kl": 0.5,
            "policy": context.policy,
        }
        trajectory.update(
            _auxiliary_telemetry(
                spec=spec,
                context=context,
                sequences=seqs,
                conditioning=cond,
                oracle_outputs=oracle_outputs,
            )
        )
        return trajectory


# ---------------------------------------------------------------------------
# TRLDAPOTrainer: real GRPO/DAPO via TRL.
# ---------------------------------------------------------------------------


def _sequence_validity(
    sequences: list[str], *, alphabet: str, sequence_length: int | None = None,
    min_length: int = 1, max_length: int | None = None, codec: Any = None,
) -> dict[str, Any]:
    """Assess raw candidates; never normalize, strip, truncate, or pad them."""
    if not alphabet:
        raise ValueError("sequence alphabet must be nonempty")
    if sequence_length is not None:
        min_length = max_length = int(sequence_length)
    if min_length < 0 or (max_length is not None and max_length < min_length):
        raise ValueError("invalid sequence length bounds")
    allowed = set(alphabet)
    reasons = []
    for sequence in sequences:
        failures = []
        if not isinstance(sequence, str):
            raise TypeError("decoded candidates must be strings")
        if not set(sequence) <= allowed:
            failures.append("alphabet")
        if len(sequence) < min_length or (max_length is not None and len(sequence) > max_length):
            failures.append("length")
        if codec is not None:
            try:
                codec.validate(sequence)
            except ValueError as exc:
                failures.append(f"codec: {exc}")
        reasons.append(failures)
    valid = np.asarray([not reason for reason in reasons], dtype=bool)
    return {
        "valid_mask": valid.tolist(), "reasons": reasons,
        "count": len(sequences), "valid_count": int(valid.sum()),
        "invalid_count": int((~valid).sum()),
        "invalid_fraction": float((~valid).mean()) if len(valid) else 0.0,
        "alphabet": alphabet, "min_length": min_length, "max_length": max_length,
        "raw_sequences_preserved": True,
    }


def _score_valid_sequences(
    *, name: str, scorer: Callable, sequences: list[str], valid_mask: np.ndarray,
    expected_widths=None, required_finite_columns=None, require_finite: bool,
    stage: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Score only valid rows and scatter into the unchanged candidate denominator."""
    valid_sequences = [s for s, valid in zip(sequences, valid_mask, strict=True) if valid]
    if valid_sequences:
        # Scorer loading and inference may draw randomness even when their
        # predictions are deterministic. They cannot consume the policy stream.
        with isolated_rng():
            scores = scorer(valid_sequences)
        measured, status = _validated_oracle_output(
            name=name, values=scores, expected_n=len(valid_sequences),
            expected_widths=expected_widths, required_finite_columns=required_finite_columns,
            require_finite=require_finite, stage=stage,
        )
        shape = (len(sequences), *measured.shape[1:])
    else:
        width = (expected_widths or {}).get(name, 1)
        if isinstance(width, bool) or int(width) != width or width < 1:
            raise ValueError(f"invalid expected width for oracle {name!r}")
        shape = (len(sequences),) if width == 1 else (len(sequences), int(width))
        measured = np.empty((0, *shape[1:]))
        status = {"status": "not_measured", "reason": "no_valid_sequences"}
    values = np.full(shape, np.nan, dtype=float)
    values[valid_mask] = measured
    status.update({
        "shape": list(values.shape), "candidate_count": len(sequences),
        "measured_count": len(valid_sequences),
        "invalid_sequence_count": len(sequences) - len(valid_sequences),
        "valid_sequence_mask": valid_mask.tolist(),
    })
    return values, status


def _raw_completion_text(tokenizer: Any, token_rows: Any) -> list[str]:
    """Remove the EOS terminator and its padding, retaining other special tokens."""
    eos = tokenizer.eos_token_id
    rows = []
    for row in token_rows:
        ids = [int(token) for token in row]
        if eos is not None and eos in ids:
            ids = ids[:ids.index(eos)]
        rows.append(ids)
    return tokenizer.batch_decode(
        rows, skip_special_tokens=False, clean_up_tokenization_spaces=False,
    )


def _reward_conditioning(prompts, conditioning_prompts, condition_names=None):
    indices = _target_indices(list(prompts), list(conditioning_prompts))
    names = list(conditioning_prompts if condition_names is None else condition_names)
    if len(names) != len(conditioning_prompts) or len(set(names)) != len(names):
        raise ValueError("condition_names must uniquely name every conditioning prompt")
    return [names[index] for index in indices]


def _publish_reference_checkpoint(model, architecture: type, output_dir: Path) -> dict[str, Any]:
    """Publish the supplied policy through its own safe, immutable model interface.

    Content-addressed snapshots let successive candidates use different parent
    policies without overwriting a reference. TRL never reloads a raw .bin.
    """
    architecture_name = f"{architecture.__module__}.{architecture.__qualname__}"
    with tempfile.TemporaryDirectory(prefix=".reference-", dir=output_dir) as temporary:
        stage = Path(temporary) / "model"
        model.save_pretrained(stage, safe_serialization=True)
        if not list(stage.glob("*.safetensors")):
            raise ValueError("model adapter must publish reference weights as safetensors")
        records = [
            {
                "path": str(path.relative_to(stage)), "size": path.stat().st_size,
            }
            for path in sorted(stage.rglob("*")) if path.is_file()
        ]
        total = sum(item["size"] for item in records)
        destination = output_dir.resolve() / "reference_models" / f"model_{len(records)}_{total}"
        manifest = {
            "path": str(destination), "architecture": architecture_name,
            "source_model_id": str(getattr(model.config, "_name_or_path", "")),
            "files": records,
        }
        if destination.exists():
            for record in records:
                path = destination / record["path"]
                if path.stat().st_size != record["size"]:
                    raise ValueError(f"published reference checkpoint changed: {path}")
        else:
            (stage / "reference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            destination.parent.mkdir(parents=True, exist_ok=True)
            stage.rename(destination)
    return manifest


def _check_reference_tensor_parity(model, reference) -> int:
    import torch

    expected, observed = model.state_dict(), reference.state_dict()
    if expected.keys() != observed.keys():
        raise ValueError("loaded KL reference has different tensor names from the supplied policy")
    for name, value in expected.items():
        other = observed[name]
        if value.dtype != other.dtype or not torch.equal(value.detach().cpu(), other.detach().cpu()):
            raise ValueError(f"loaded KL reference differs from the supplied policy: {name}")
    return len(expected)


def _require_finite_reward_vector(
    values: Any,
    *,
    expected_n: int,
    stage: str,
) -> np.ndarray:
    rewards = np.asarray(values, dtype=float)
    if rewards.shape != (expected_n,):
        raise ValueError(
            f"{stage} reward must have shape ({expected_n},), got {rewards.shape}"
        )
    nonfinite = np.flatnonzero(~np.isfinite(rewards))
    if nonfinite.size:
        preview = nonfinite[:10].astype(int).tolist()
        raise FloatingPointError(
            f"{stage} reward contains {nonfinite.size} nonfinite value(s); "
            f"first indices={preview}"
        )
    return rewards


def _optimizer_step_provenance(
    trainer: Any, *, configured_steps: int
) -> tuple[int, str]:
    """Return the measured optimizer-step count and its provenance.

    Confirmatory trajectory AUCs use cumulative optimizer steps on the x-axis.
    The configured maximum is not evidence that all steps completed (training can
    stop or fail early), so the real TRL state is authoritative and absence is a
    hard error rather than a silent fallback to ``configured_steps``.
    """
    state = getattr(trainer, "state", None)
    measured = getattr(state, "global_step", None)
    if isinstance(measured, bool):
        raise RuntimeError("trainer.state.global_step must be an integer, not bool")
    try:
        completed = int(measured)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "trainer.state.global_step is unavailable; cannot prove completed "
            "optimizer steps"
        ) from exc
    if completed < 1:
        raise RuntimeError(
            "trainer.state.global_step is non-positive; no completed optimizer "
            "step can be recorded"
        )
    if completed > int(configured_steps):
        raise RuntimeError(
            "measured optimizer steps exceed the configured maximum: "
            f"{completed} > {configured_steps}"
        )
    return completed, "trainer_state.global_step"


def _bind_prompt_conditioning(spec_dict, prompts, conditioning_prompts, group_intervals=None,
                             oracle_target_indices=None, primitive_resolver=None):
    """Bind per-sequence conditioning into the spec from each completion's prompt.

    Returns a patched copy of ``spec_dict`` with per-sequence ``target_indices``
    set on a ``reward_input``, a ``contrastive_mingap`` aggregator and/or a
    ``target_interval`` shape (the latter also gets the per-group ``intervals``).
    Returns the spec unchanged if none of them is in use or there are no prompts.
    """
    from dataclasses import fields, is_dataclass
    from recital.spec.registry import (
        ROLE_CROSS_ORACLE_AGG, ROLE_HARD_GATE, ROLE_PER_SEQ_PENALTY,
        ROLE_PER_SEQ_SHAPING, get_primitive,
    )

    resolve = primitive_resolver or get_primitive

    def accepts_indices(role, reference):
        if not isinstance(reference, dict) or not reference.get("name"):
            return False
        try:
            constructor = resolve(role, reference["name"])
        except KeyError:
            # Admission/validation owns missing names. Binding does not fall
            # back to a global constructor when a scoped resolver rejects one.
            return False
        return is_dataclass(constructor) and any(f.name == "target_indices" for f in fields(constructor))

    agg = spec_dict.get("aggregator") or {}
    shape = spec_dict.get("shape") or {}
    reward_input = spec_dict.get("reward_input")
    needs_agg = agg.get("name") == "contrastive_mingap" or accepts_indices(ROLE_CROSS_ORACLE_AGG, agg)
    needs_shape = shape.get("name") == "target_interval" or accepts_indices(ROLE_PER_SEQ_SHAPING, shape)
    conditional_lists = {
        name: [accepts_indices(role, ref) for ref in (spec_dict.get(name) or [])]
        for name, role in (("penalties", ROLE_PER_SEQ_PENALTY), ("gates", ROLE_HARD_GATE))
    }
    if not (reward_input or needs_agg or needs_shape or any(any(v) for v in conditional_lists.values())) or not prompts or not conditioning_prompts:
        return spec_dict
    target_indices = _target_indices(list(prompts), list(conditioning_prompts))
    patched = dict(spec_dict)
    if reward_input:
        patched["reward_input"] = {**reward_input, "target_indices": _oracle_target_columns(
            prompts, conditioning_prompts, oracle_target_indices
        )}
    if needs_agg:
        patched["aggregator"] = {**agg, "target_indices": _oracle_target_columns(
            prompts, conditioning_prompts, oracle_target_indices
        )}
    if needs_shape:
        new_shape = {**shape, "target_indices": target_indices}
        # Inject the task's per-group bands unless the spec already supplies a
        # NON-EMPTY one (an echoed empty list must not block injection, else the
        # target_interval shaping silently returns all zeros).
        if group_intervals is not None and not shape.get("intervals"):
            new_shape["intervals"] = group_intervals
        patched["shape"] = new_shape
    for name, needs in conditional_lists.items():
        if any(needs):
            patched[name] = [
                {**ref, "target_indices": target_indices} if required else dict(ref)
                for ref, required in zip(spec_dict[name], needs, strict=True)
            ]
    return patched


def _requires_reward_conditioning(spec, reward_fn=None, primitive_resolver=None):
    """Explicit context protocol for scoped proxies and registered penalties."""
    if spec.get("target_objectives"):
        return True
    from recital.spec.registry import ROLE_HARD_GATE, ROLE_PER_SEQ_PENALTY, get_primitive

    resolve = primitive_resolver or getattr(reward_fn, "_primitive_resolver", None) or get_primitive
    for field_name, role in (("penalties", ROLE_PER_SEQ_PENALTY), ("gates", ROLE_HARD_GATE)):
        for ref in (spec.get(field_name) or []):
            if not isinstance(ref, dict) or not ref.get("name"):
                continue
            try:
                constructor = resolve(role, ref["name"])
            except KeyError:
                continue
            if getattr(constructor, "requires_conditioning", False):
                return True
    return any(getattr(p, "requires_conditioning", False) or hasattr(p, "evaluate_context")
               for name in ("_penalties", "_gates") for p in getattr(reward_fn, name, []))


def _checkpoint_selection_cost_callback(selector: Any, execution_cost: Any) -> Any:
    """Observe selection through public callbacks, after cost/selector events.

    Each JSON-frozen prefix links measured costs to the selector's original
    artifact binding; it does not rewrite that binding or call task evaluators.
    Opaque evaluator costs remain outside the trainer ledger. Prefix snapshots
    are cumulative within this call and must not be summed across checkpoints.
    """
    from transformers import TrainerCallback
    from recital.training.callback_state import CheckpointStateMixin

    class SelectionCostCallback(CheckpointStateMixin, TrainerCallback):
        def __init__(self):
            self._scheduled = frozenset(selector.summary()["scheduled_steps"])
            self._snapshots: dict[int, str] = {}
            self._start_step = 0

        def state_dict(self):
            return {"snapshots": [json.loads(value) for value in self._snapshots.values()]}

        def load_state_dict(self, state, *, checkpoint_step):
            snapshots = state["snapshots"] if state is not None else []
            if any(item["actual_step"] > checkpoint_step for item in snapshots):
                raise ValueError("checkpoint cost observations exceed the saved optimizer step")
            self._snapshots = {item["actual_step"]: json.dumps(item, allow_nan=False) for item in snapshots}
            self._start_step = checkpoint_step

        def on_step_end(self, args, state, control, **kwargs):
            step = state.global_step
            if step not in self._scheduled or step in self._snapshots:
                return control
            report = selector.summary()
            record = next((record for record in report["records"]
                           if record["scheduled_step"] == record["actual_step"] == step), None)
            if record is None:
                return control  # No evaluated checkpoint; never fabricate a binding.
            snapshot = execution_cost.prefix_snapshot()
            # training() only settles this interval when train() returns/raises.
            # Its initial zero is not a measured live training duration.
            snapshot["measurements"]["training_elapsed_seconds"] = {
                "value": None, "unit": "seconds", "status": "unknown",
                "reason": "training interval is still open at this checkpoint",
            }
            payload = {
                "schema_version": 1, "scope": "trainer_call_prefix",
                "cumulative": True, "capture_boundary": "after_selection_on_step_end",
                "scheduled_step": step, "actual_step": step,
                "trainer_call_start_step": self._start_step,
                "selection_checkpoint": record.get("artifacts", {}).get("checkpoint"),
                "selection_library": record.get("artifacts", {}).get("library"),
                "execution_cost": snapshot,
            }
            # Store immutable bytes-as-text; every public summary is detached.
            self._snapshots[step] = json.dumps(payload, allow_nan=False)
            return control

        def summary(self):
            report = selector.summary()
            snapshots = [json.loads(value) for value in self._snapshots.values()]
            report["execution_cost_snapshots"] = snapshots
            report["first_qualification_execution_cost"] = next((
                item for item in snapshots
                if item["actual_step"] == report["first_qualification_step"]
            ), None)
            return report

    return SelectionCostCallback()


@dataclass
class TRLDAPOTrainer:
    """Wraps trl.GRPOTrainer with the DAPO decoupled-clip variant.

    Hyperparameters are read directly from spec['hyperparameters'].
    Token-alphabet constraint is enforced by a LogitsProcessor that masks
    tokens outside `context.token_alphabet` to -inf before sampling.

    An optional ``context.extras['checkpoint_selection_callback']`` receives
    the actual policy through the public HF callback API in this one training
    invocation. Its evidence is returned as ``trajectory['checkpoint_selection']``
    and retained in ``last_checkpoint_selection`` on failure. Selecting a bound
    checkpoint does not replace the final policy or final evaluation library.
    The summary adds ``execution_cost_snapshots`` and
    ``first_qualification_execution_cost`` for measured trainer-call prefixes;
    opaque task evaluation work and unsettled live durations remain excluded.

    ``run(..., perform_final_evaluation=True)`` performs final evaluation. False
    requires a selector scheduled through the declared final optimizer step;
    it returns training/selection evidence without final library/score fields,
    with ``evaluation_performed=False`` and ``eval_checkpoint='not_evaluated'``.
    """

    use_dapo: bool = True
    max_completion_length: int = 50
    output_dir: str | Path | None = None
    bf16: bool = False
    use_cpu: bool = False
    last_execution_cost: dict[str, Any] | None = field(default=None, init=False, repr=False)
    last_checkpoint_selection: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _active_checkpoint_selection: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._import_error: str | None = None
        try:
            import torch  # noqa: F401
            import trl  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            self._import_error = repr(exc)

    @property
    def available(self) -> bool:
        return self._import_error is None

    def _build_dataset(
        self,
        conditioning_prompts: list[str],
        samples_per_prompt: int,
        repeats=None,
    ):
        """One row per (prompt, generation slot). `repeats` is an optional
        {prompt_index: int} oversampling map (by position in conditioning_prompts):
        a hard group can be repeated so it gets proportionally more GRPO updates.
        Each prompt's rows stay contiguous and in multiples of samples_per_prompt
        so TRL's num_generations grouping remains within-prompt. Default
        (None/empty) = 1x for every prompt (uniform)."""
        from datasets import Dataset

        # TRL repeats each dataset row num_generations times. The rows here
        # control sampling allocation, not a guarantee that each optimizer step
        # contains every condition (distributed/accumulated batches can differ).
        repeats = repeats or {}
        rows = []
        for i, prompt in enumerate(conditioning_prompts):
            for _ in range(max(int(repeats.get(i, 1)), 1)):
                rows.append({"prompt": prompt})
        return Dataset.from_list(rows)

    @staticmethod
    def _legal_token_ids(tokenizer, alphabet: str, codec=None) -> list[int]:
        if codec is not None:
            legal = list(codec.legal_token_ids(include_eos=True))
        else:
            legal = []
            special = set(tokenizer.all_special_ids)
            for character in alphabet:
                ids = tokenizer.encode(character, add_special_tokens=False)
                if len(ids) != 1 or ids[0] in special:
                    raise ValueError(
                        "alphabet characters must encode as single ordinary tokens; "
                        "supply a sequence codec for a subword tokenizer"
                    )
                decoded = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
                if decoded != character:
                    raise ValueError("alphabet token does not round-trip; supply a sequence codec")
                legal.extend(ids)
            if len(set(legal)) != len(alphabet):
                raise ValueError("alphabet characters must have distinct token IDs")
            if tokenizer.eos_token_id is not None:
                legal.append(tokenizer.eos_token_id)
        if not legal or any(isinstance(i, bool) or int(i) != i or i < 0 for i in legal):
            raise ValueError("codec must provide nonnegative legal token IDs")
        return sorted(set(int(i) for i in legal))

    def _build_legal_alphabet_processor(self, tokenizer, alphabet: str, codec=None):
        import torch
        from transformers import LogitsProcessor

        legal_tensor = torch.tensor(self._legal_token_ids(tokenizer, alphabet, codec))

        class _LegalAlphabet(LogitsProcessor):
            def __call__(self, input_ids, scores):
                if int(legal_tensor.max()) >= scores.shape[-1]:
                    raise ValueError("legal token ID exceeds the model vocabulary")
                mask = torch.full_like(scores, float("-inf"))
                mask[..., legal_tensor.to(scores.device)] = 0.0
                return scores + mask

        return _LegalAlphabet()

    def _make_reward_funcs(
        self,
        spec_dict: dict[str, Any],
        reward_fn: Callable[..., np.ndarray],
        oracle_fns: dict[str, Callable[[list[str]], np.ndarray]],
        enabled: list[str],
        conditioning_prompts: list[str] | None = None,
        group_intervals: list | None = None,
        token_alphabet: str = "ACGT",
        oracle_expected_widths: dict[str, Any] | None = None,
        oracle_required_finite_columns: dict[str, Any] | None = None,
        completion_prefix: str = "",
        sequence_length: int | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        codec: Any = None,
        tokenizer: Any = None,
        condition_names: list[str] | None = None,
        oracle_target_indices: dict[str, int] | None = None,
        execution_cost: Any = None,
        primitive_resolver: Callable[[str, str], Callable] | None = None,
    ):
        """Wrap the framework reward into TRL's `reward_funcs` signature.

        Only the spec's ENABLED oracles are queried each step, so the agent
        controls which (possibly slow) oracles run online; unselected oracles are
        never scored during training. For conditional parts (a ``reward_input``,
        a ``contrastive_mingap`` aggregator and/or a ``target_interval`` shape) the
        per-sequence target is bound from each completion's conditioning prompt
        (its position in ``conditioning_prompts``) per batch.
        """
        if sequence_length is None and min_length is None and max_length is None and codec is None:
            sequence_length = len(completion_prefix) + self.max_completion_length
        min_length = 1 if min_length is None else int(min_length)
        invalid_penalty = float(spec_dict.get("invalid_penalty", _default_invalid_penalty()))
        if not np.isfinite(invalid_penalty):
            raise ValueError("invalid_penalty must be finite")
        if not enabled:
            raise ValueError("enabled oracle list must be non-empty")
        missing = sorted(set(enabled) - set(oracle_fns))
        if missing:
            raise KeyError(f"enabled oracle scorer(s) are not registered: {missing}")
        scored = {o: oracle_fns[o] for o in enabled}
        # For conditional specs (reward_input / contrastive_mingap / target_interval)
        # the per-sequence target_indices change every batch but the rest of the pipeline
        # (incl. STATEFUL transforms like ema_zscore) must persist across steps.
        # So build the reward fn ONCE with the static parts injected (group
        # intervals) and mutate only target_indices per batch — rebuilding it each
        # step would reset every EmaZScore's running mean/var, degrading decay=0.99
        # to single-batch z-scoring.
        shape0 = spec_dict.get("shape") or {}
        bound_base = _bind_prompt_conditioning(
            spec_dict, conditioning_prompts, conditioning_prompts, group_intervals,
            oracle_target_indices=oracle_target_indices,
            primitive_resolver=primitive_resolver or getattr(reward_fn, "_primitive_resolver", None),
        )
        is_conditional = bound_base is not spec_dict
        needs_conditioning = _requires_reward_conditioning(spec_dict, reward_fn, primitive_resolver)
        cached_rfn = None
        if is_conditional and conditioning_prompts:
            base = dict(bound_base)
            if (
                shape0.get("name") == "target_interval"
                and group_intervals is not None
                and not shape0.get("intervals")
            ):
                base["shape"] = {**shape0, "intervals": group_intervals}
            from recital.spec.composer import ComposedRewardFn, RewardSpec

            cached_rfn = _candidate_reward_build(
                lambda: ComposedRewardFn(
                    RewardSpec.from_dict(base),
                    gate_registry=getattr(reward_fn, "_gate_registry", None),
                    training_set=getattr(reward_fn, "_training_set", None),
                    reference_libraries=getattr(reward_fn, "_reference_libraries", None),
                    primitive_resolver=primitive_resolver
                    or getattr(reward_fn, "_primitive_resolver", None),
                ),
                stage="conditional training reward binding",
            )

        def _trl_reward(prompts, completions, completion_ids=None, **_kwargs):
            observed_count = len(completion_ids) if completion_ids is not None else len(completions)
            _trl_reward.candidate_count += observed_count
            if execution_cost is not None:
                execution_cost.add("training_candidates", observed_count)
            decoded = (
                _raw_completion_text(tokenizer, completion_ids)
                if tokenizer is not None and completion_ids is not None
                else completions
            )
            if any(not isinstance(c, str) for c in decoded):
                raise TypeError("sequence training requires plain-text completions")
            seqs = [completion_prefix + c for c in decoded]
            if len(prompts) != len(seqs):
                raise ValueError("one conditioning prompt is required per completion")
            validity = _sequence_validity(
                seqs, alphabet=token_alphabet, sequence_length=sequence_length,
                min_length=min_length, max_length=max_length, codec=codec,
            )
            _trl_reward.invalid_count += validity["invalid_count"]
            valid = np.asarray(validity["valid_mask"], dtype=bool)
            valid_prompts = [p for p, keep in zip(prompts, valid, strict=True) if keep]
            # Validate the full prompt list, including invalid candidates.
            if conditioning_prompts:
                _oracle_target_columns(list(prompts), list(conditioning_prompts), oracle_target_indices)
            oracle_outputs = {}
            for oracle, scorer in scored.items():
                values, _ = _score_valid_sequences(
                    name=oracle, scorer=scorer, sequences=seqs, valid_mask=valid,
                    expected_widths=oracle_expected_widths,
                    required_finite_columns=oracle_required_finite_columns,
                    require_finite=True, stage="training",
                )
                oracle_outputs[oracle] = values
            rewards = np.full(len(seqs), invalid_penalty)
            batch_diagnostics = {}
            if valid.any():
                rfn = reward_fn
                if cached_rfn is not None:
                    rfn = _rebind_conditioning(
                        cached_rfn, valid_prompts, conditioning_prompts,
                        oracle_target_indices,
                    )
                rewards[valid] = _candidate_reward_call(
                    rfn, sequences=[s for s, keep in zip(seqs, valid, strict=True) if keep],
                    oracle_outputs={name: values[valid] for name, values in oracle_outputs.items()},
                    expected_n=int(valid.sum()), stage="training",
                    conditioning=(
                        _reward_conditioning(valid_prompts, conditioning_prompts or [], condition_names)
                        if needs_conditioning else None
                    ),
                )
                batch_diagnostics = dict(getattr(rfn, "last_diagnostics", {}))
            _trl_reward.execution_history.append({
                "reward_call": len(_trl_reward.execution_history) + 1,
                "candidate_count": len(seqs), "invalid_count": validity["invalid_count"],
                "composition": batch_diagnostics,
                # nanmean, not a whole-array finiteness test. An oracle with a
                # BY-DESIGN NaN column (Agarwal has no SK-N-SH head) fails an
                # `np.isfinite(...).all()` test and would have its entire recorded
                # training telemetry dropped -- silently, since None rows are
                # then filtered out of the summary.
                "training_oracles": {
                    name: {
                        "count": int(valid.sum()),
                        "mean": _column_nanmean(values[valid]) if valid.any() else None,
                    }
                    for name, values in oracle_outputs.items()
                },
                # Valid rows only. Filling invalid rows with invalid_penalty and
                # then taking the std over the FULL array mixes two populations:
                # on a reward spanning [-1.5, 0] a handful of -2.0 entries
                # dominate. `composition.reward_std` next to it is valid-only.
                "by_prompt": {
                    label: {
                        "count": sum(p == label for p in prompts),
                        "reward_std": _valid_reward_std(rewards, prompts, label, valid),
                    }
                    for label in sorted(set(prompts))
                },
            })
            _trl_reward.last_batch = {
                "sequences": seqs, "conditioning": list(prompts),
                "sequence_validity": validity, "oracle_outputs": oracle_outputs,
            }
            return rewards.tolist()

        _trl_reward.execution_history = []
        _trl_reward.candidate_count = 0
        _trl_reward.invalid_count = 0
        _trl_reward.__name__ = "recital_reward"
        # Conditional recipes execute their bound clone, whose normalizers
        # differ from the input reward's unused instances. Persist that exact
        # callable's state when HF saves/resumes a training checkpoint.
        active_reward = cached_rfn if cached_rfn is not None else reward_fn
        for name in ("state_dict", "load_state_dict"):
            method = getattr(active_reward, name, None)
            if callable(method):
                setattr(_trl_reward, name, method)
        # The final evaluation must score with THIS object, not a fresh clone:
        # its normalizers carry the training run's accumulated state, and that
        # state is what the reward level means. See `_rebind_conditioning`.
        _trl_reward.composed_reward = active_reward
        _trl_reward.oracle_target_indices = oracle_target_indices
        return _trl_reward

    def run(
        self,
        *,
        spec: dict[str, Any],
        reward_fn: Callable[..., np.ndarray],
        context: TrainContext,
        perform_final_evaluation: bool = True,
    ) -> dict[str, Any]:
        from recital.costs import TrainingCallCost

        execution_cost = TrainingCallCost()
        self.last_execution_cost = None
        self.last_checkpoint_selection = None
        self._active_checkpoint_selection = None
        try:
            resolver = spec.get("_framework_primitive_resolver")
            if resolver is not None:
                if not callable(resolver):
                    raise TypeError("framework primitive resolver must be callable")
                context = replace(context, extras={**context.extras, "primitive_resolver": resolver})
            checkpoint_context = spec.get("_framework_checkpoint_context")
            if checkpoint_context is not None:
                if not isinstance(checkpoint_context, dict):
                    raise TypeError("framework checkpoint context must be a mapping")
                context = replace(context, extras={**context.extras, **checkpoint_context})
            allocation = spec.get("_framework_search_budget")
            if allocation is not None and not context.extras.get("training_action"):
                # One uninterrupted invocation, with delivery from its archived
                # scheduled libraries. A default argument cannot cause a redraw.
                perform_final_evaluation = allocation.get("perform_final_evaluation", False)
                if type(perform_final_evaluation) is not bool:
                    raise ValueError("allocated evaluation mode must be boolean")
                callback = context.extras.get("checkpoint_selection_callback")
                if perform_final_evaluation and callback is not None:
                    raise ValueError("final-library budget cannot add checkpoint evaluations")
                if not perform_final_evaluation and (callback is None or callback.summary()["scheduled_steps"] != allocation["scheduled_steps"]):
                    raise ValueError("budgeted training requires the allocated checkpoint schedule")
            _validate_runtime_primitives(spec, context, reward_fn)
            counted_context = replace(
                context, oracle_fns=execution_cost.wrap_oracles(context.oracle_fns),
                extras=dict(context.extras),
            )
            trajectory = self._run_with_cost(
                spec=spec, reward_fn=reward_fn, context=counted_context,
                execution_cost=execution_cost,
                perform_final_evaluation=perform_final_evaluation,
            )
        except BaseException as exc:
            self._capture_checkpoint_selection()
            execution_cost.finish(failed=exc)
            self.last_execution_cost = execution_cost.snapshot()
            if isinstance(exc, CandidateTrainingError):
                # Preserve the failed call's measured spending and any already
                # archived checkpoints across the generic trainer_fn boundary.
                exc.execution_cost = self.last_execution_cost
                exc.checkpoint_selection = self.last_checkpoint_selection
            raise
        self._capture_checkpoint_selection()
        if self.last_checkpoint_selection is not None:
            trajectory["checkpoint_selection"] = self.last_checkpoint_selection
            trajectory["checkpoint_selection_applied_to_policy"] = False
            trajectory["checkpoint_measurements"] = [
                {"step": record["scheduled_step"], "measurements": record["measurements"]}
                for record in self.last_checkpoint_selection.get("records", [])
            ]
        execution_cost.finish()
        self.last_execution_cost = execution_cost.snapshot()
        trajectory["execution_cost"] = execution_cost.snapshot()
        return trajectory

    def _capture_checkpoint_selection(self) -> None:
        """Retain selection evidence even if training or final evaluation raises."""
        callback = self._active_checkpoint_selection
        self._active_checkpoint_selection = None
        if callback is None:
            return
        try:
            self.last_checkpoint_selection = callback.summary()
        except Exception as exc:  # noqa: BLE001 - do not replace the training exception
            self.last_checkpoint_selection = {
                "status": "unknown", "error": f"{type(exc).__name__}: {exc}",
                "reason": "checkpoint selection summary unavailable",
            }

    @staticmethod
    def _validate_checkpoint_selection(
        spec: dict[str, Any], context: TrainContext, steps: int,
    ) -> Any:
        """Validate callback access separately from ordinary diagnostic access.

        Sources outside the evaluation oracle keys, and every audit source, stay
        private to the callback's SuccessSpec even when their callables are not
        mounted.
        """
        callback = context.extras.get("checkpoint_selection_callback")
        if callback is None:
            return None
        from recital.success import SuccessSpec
        from recital.training.checkpoint_selection import CheckpointSelectionCallback

        if not isinstance(callback, CheckpointSelectionCallback):
            raise TypeError("checkpoint_selection_callback must be a CheckpointSelectionCallback instance")
        summary = callback.summary()
        if (summary["records"] or summary["actual_steps_completed"] != 0) and not context.extras.get("resume_from_checkpoint"):
            raise ValueError("checkpoint selection needs a fresh callback for each trainer call")
        if summary["scheduled_steps"][-1] > steps:
            raise ValueError("checkpoint selection schedule exceeds the declared optimizer-step budget")
        available, _ = _evaluation_oracle_keys(spec, context)
        _, _, audit = _oracle_roles(spec, context)
        selection_spec = SuccessSpec.model_validate(summary["success_spec"])
        private = set(context.oracle_fns) - set(available)
        selection_spec.validate_information_access(
            known_sources=set(context.oracle_fns), audit_sources=audit,
            private_sources=private,
        )
        return callback

    def _run_with_cost(
        self,
        *,
        spec: dict[str, Any],
        reward_fn: Callable[..., np.ndarray],
        context: TrainContext,
        execution_cost: Any,
        perform_final_evaluation: bool = True,
    ) -> dict[str, Any]:
        if type(perform_final_evaluation) is not bool:
            raise TypeError("perform_final_evaluation must be a boolean")
        if "perform_final_evaluation" in context.extras:
            raise TypeError("perform_final_evaluation is a public run() keyword, not a context.extras setting")
        if self._import_error is not None:
            raise RuntimeError(
                f"TRLDAPOTrainer requires torch + trl + transformers: {self._import_error}"
            )

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
        from recital.costs import trainer_cost_callback

        hp = spec.get("hyperparameters") or {}
        lr = float(hp.get("lr", 1e-4))
        kl = float(hp.get("kl", 0.02))
        group_size = int(hp.get("group_size", 8))
        steps = int(hp.get("steps", 100))
        action = None
        action_request = context.extras.get("training_action")
        if action_request is not None:
            from recital.training.action_state import TrainingAction

            action = TrainingAction(
                action_request, spec=spec,
                output_dir=Path(self.output_dir or tempfile.mkdtemp(prefix="rd_trl_")),
                seed=int(context.extras.get("seed", 0)),
                data_signature={
                    "prompts": list(context.conditioning_prompts),
                    "condition_repeats": hp.get("condition_repeats") or context.extras.get("condition_repeats") or {},
                    "group_size": group_size,
                    "generation_kwargs": context.extras.get("generation_kwargs") or {},
                    "group_intervals": context.extras.get("group_intervals"),
                    "condition_names": context.extras.get("condition_names"),
                    "oracle_target_indices": context.extras.get("oracle_target_indices"),
                    "completion_prefix": context.extras.get("completion_prefix", ""),
                    "num_iterations": hp.get("num_iterations", 1),
                    "steps_per_generation": hp.get("steps_per_generation"),
                    "gradient_accumulation_steps": hp.get("gradient_accumulation_steps", 1),
                    "max_completion_length": self.max_completion_length,
                    "min_new_tokens": context.extras.get("min_new_tokens"),
                    "token_alphabet": context.token_alphabet,
                },
            )
            steps = action.end_step
            lr = action.schedule["start_lr"]
        selection_cb = self._validate_checkpoint_selection(spec, context, steps)
        if not perform_final_evaluation and action is None:
            if selection_cb is None:
                raise ValueError("perform_final_evaluation=False requires a checkpoint selection callback")
            if selection_cb.summary()["scheduled_steps"][-1] != steps:
                raise ValueError("skipping final evaluation requires a selection schedule ending at the step budget")
        seed = action.seed if action is not None else int(context.extras.get("seed", 0))
        codec = context.extras.get("sequence_codec", context.extras.get("codec"))

        # Model adapters own architecture/tokenizer behavior. No instance methods,
        # classes, transformer trust resolvers, or TRL factories are replaced here.
        load_kwargs = dict(context.extras.get("model_init_kwargs") or {})
        architecture = context.extras.get("model_architecture", AutoModelForCausalLM)
        def load_action_model(source, model_class):
            # Task adapters own raw .bin checkpoint normalization as well as
            # architecture. Loading a raw Hyena .bin through HF alone can
            # round its vocabulary and reject trained tensors.
            loader = context.extras.get("model_loader")
            if loader is not None:
                return loader(str(source))
            return model_class.from_pretrained(source, **load_kwargs)

        if action is not None:
            if "model_architecture" not in context.extras and not isinstance(context.policy, (str, Path)):
                architecture = type(context.policy)
            policy_source = action.initial_reference if action.mode == "initial_restart" else str(action.parent_path)
            tokenizer = context.extras.get("tokenizer")
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(
                    action.initial_reference, **dict(context.extras.get("tokenizer_init_kwargs") or {}))
            model = load_action_model(policy_source, architecture)
            if not isinstance(context.policy, (str, Path)) and getattr(context.policy, "generation_config", None) is not None:
                import copy
                model.generation_config = copy.deepcopy(context.policy.generation_config)
        elif isinstance(context.policy, str):
            tokenizer = context.extras.get("tokenizer")
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(
                    context.policy, **dict(context.extras.get("tokenizer_init_kwargs") or {})
                )
            model = architecture.from_pretrained(context.policy, **load_kwargs)
        else:
            model = context.policy
            tokenizer = context.extras.get("tokenizer")
        if tokenizer is None or tokenizer.pad_token_id is None:
            raise ValueError("model adapter must provide a tokenizer with a declared pad token")
        if "attention_mask" not in (tokenizer.model_input_names or []):
            raise ValueError("model adapter tokenizer must return attention_mask")
        if not callable(getattr(model, "generate", None)) or model.generation_config is None:
            raise ValueError("model adapter must implement the standard causal generation interface")
        min_new = int(context.extras.get("min_new_tokens", self.max_completion_length))
        if not 0 <= min_new <= self.max_completion_length:
            raise ValueError("min_new_tokens must be within the completion token budget")
        reference_architecture = context.extras.get("reference_model_architecture", type(model))
        if not isinstance(reference_architecture, type) or not callable(getattr(reference_architecture, "from_pretrained", None)):
            raise ValueError("reference_model_architecture must be a loadable model class")
        reference_kwargs = {**load_kwargs, **dict(context.extras.get("reference_model_init_kwargs") or {})}
        reference_kwargs.update({"architecture": reference_architecture, "device_map": None})

        # The base-policy reference library is part of every real run and can
        # contain hundreds of completions.  Keep it on the allocated accelerator
        # when one is available; otherwise a GPU run silently spends most of
        # its time doing HyenaDNA generation on CPU before TRL ever starts.
        if self.use_cpu:
            model = model.to("cpu")
        elif torch.cuda.is_available():
            model = model.to("cuda")

        # 2. Tiny dataset: group_size rows per conditioning prompt, with optional
        # per-group oversampling (condition_repeats) so hard groups get more GRPO
        # updates. Precedence: spec hyperparameter (agent-tunable) > task default
        # (context.extras) > uniform. Keys are prompt indices; JSON keys are strings.
        samples_per_prompt = max(group_size, 1)
        _repeats_raw = hp.get("condition_repeats") or context.extras.get("condition_repeats") or {}
        repeats = {int(k): int(v) for k, v in dict(_repeats_raw).items()}
        if repeats:
            LOG.info("condition_repeats (per-group prompt oversampling): %s", repeats)
        train_ds = self._build_dataset(
            context.conditioning_prompts or [""], samples_per_prompt, repeats=repeats
        )

        legal_ids = self._legal_token_ids(tokenizer, context.token_alphabet, codec)
        out_emb = model.get_output_embeddings()
        n_ids = int(out_emb.weight.shape[0])
        if max(legal_ids) >= n_ids:
            raise ValueError("legal token ID exceeds the model output vocabulary")
        gen_kwargs = {
            **dict(context.extras.get("generation_kwargs") or {}),
            "suppress_tokens": [i for i in range(n_ids) if i not in legal_ids],
            "min_new_tokens": min_new,
            "use_cache": bool(getattr(model.config, "use_cache", True)),
        }

        # 3. GRPOConfig with spec hyperparameters.
        # Allocate one generation group per declared prompt in the local batch;
        # this remains a multiple of num_generations. Actual condition allocation
        # depends on the dataset sampler and is not guaranteed by batch size.
        n_prompts = max(len(context.conditioning_prompts or [""]), 1)
        per_device = n_prompts * samples_per_prompt
        out_dir = Path(self.output_dir or tempfile.mkdtemp(prefix="rd_trl_"))
        # Mid-run checkpointing (opt-in via extras) is isolated per
        # iteration/spec; a completed checkpoint resumes into a no-op train()
        # that still reloads the trained weights -- the crash-after-training-
        # before-eval recovery path.
        if action is None:
            out_dir, save_every, resume_mid_run = _checkpoint_layout(spec, context, out_dir)
        else:
            out_dir, save_every, resume_mid_run = action.out_dir, 0, False
        out_dir.mkdir(parents=True, exist_ok=True)
        reference_model = model
        if action is not None:
            reference_model = load_action_model(action.kl_reference, reference_architecture).to("cpu")
        reference_snapshot = _publish_reference_checkpoint(reference_model, reference_architecture, out_dir)
        cfg_kwargs = dict(
            output_dir=str(out_dir),
            per_device_train_batch_size=per_device,
            learning_rate=lr,
            beta=kl,
            max_steps=steps,
            num_generations=group_size,
            max_completion_length=self.max_completion_length,
            generation_kwargs=gen_kwargs,
            loss_type="dapo" if self.use_dapo else "grpo",
            bf16=self.bf16,
            use_cpu=self.use_cpu or not torch.cuda.is_available(),
            remove_unused_columns=False,
            report_to=[],
            logging_steps=max(1, steps // 10),
            save_steps=save_every if save_every > 0 else steps + 1,
            # Checkpoint selection may promote a later or earlier qualified
            # state after the full budget. Keep every scheduled checkpoint;
            # deleting older states here makes the frozen selector unable to
            # inspect or recover the best eligible candidate.
            save_total_limit=None,
            save_safetensors=True,
            seed=seed,
            data_seed=seed,
        )
        if action is not None:
            cfg_kwargs.update(
                save_strategy="no", load_best_model_at_end=False,
                ignore_data_skip=bool(action.resume_path and not action.data_compatible),
            )
            for name in ("num_iterations", "steps_per_generation", "gradient_accumulation_steps",
                         "adam_beta1", "adam_beta2", "adam_epsilon", "weight_decay", "max_grad_norm"):
                if name in hp:
                    cfg_kwargs[name] = hp[name]
        # ``save_safetensors`` is accepted by some Transformers/TRL versions, but
        # recent TRL versions do not take it in ``GRPOConfig`` (safetensors is the
        # default).  Pass the flag only where the runtime supports it, so startup
        # does not fail.
        if "save_safetensors" not in inspect.signature(GRPOConfig).parameters:
            cfg_kwargs.pop("save_safetensors", None)
        # TRL uses this official factory hook for the reference even when the
        # policy is already instantiated (TRL 0.26.2, GRPOTrainer.__init__).
        cfg_kwargs["model_init_kwargs"] = reference_kwargs
        cfg = GRPOConfig(**cfg_kwargs)

        # 4. Reward function bridge. Only the spec's enabled oracles are scored
        # online (the agent decides which, including any slow ones); the prompt
        # drives contrastive_mingap target indices for conditional multi-cell tasks.
        enabled, _, _ = _oracle_roles(spec, context)
        trl_reward = self._make_reward_funcs(
            spec, reward_fn, context.oracle_fns, enabled, context.conditioning_prompts,
            group_intervals=context.extras.get("group_intervals"),
            token_alphabet=context.token_alphabet,
            oracle_expected_widths=context.extras.get("oracle_expected_widths"),
            oracle_required_finite_columns=context.extras.get(
                "oracle_required_finite_columns"
            ),
            completion_prefix=str(context.extras.get("completion_prefix") or ""),
            sequence_length=context.extras.get("sequence_length"),
            min_length=context.extras.get("min_sequence_length"),
            max_length=context.extras.get("max_sequence_length"),
            codec=codec, tokenizer=tokenizer,
            condition_names=context.extras.get("condition_names"),
            oracle_target_indices=context.extras.get("oracle_target_indices"),
            execution_cost=execution_cost,
            primitive_resolver=context.extras.get("primitive_resolver"),
        )

        # 5. Run training.
        callbacks: list[Any] = [trainer_cost_callback(execution_cost)]
        from recital.training.reward_state import reward_state_callback

        reward_state_cb = reward_state_callback(trl_reward)
        if reward_state_cb is not None:
            callbacks.append(reward_state_cb)
        if action is not None:
            action.restore_reward(trl_reward)

        # Sample the BASE policy before GRPO touches it. This is the reference the
        # oracle-free degeneracy monitor calibrates against, and it is the right
        # reference: the question is not "are these sequences less complex than
        # natural DNA" (every RL policy's are, and so is the SFT policy's), it is
        # "did optimizing this reward make them less complex than the policy we
        # started from". That difference is Goodhart; the other is a property of
        # the pretrained model. Every iteration re-inits from the same frozen SFT
        # checkpoint, so this library is identical across iterations -- sample it
        # once per process and reuse.
        #
        # Sampling here advances the global RNGs, and the determinism the framework
        # relies on -- identical (spec, seed) => bit-identical retraining, which is
        # what makes protocol conditions comparable at all -- would break if that
        # leaked into training. It cannot, for two reasons that both have to
        # keep holding:
        #
        #   1. GRPOTrainer is constructed BELOW, and transformers.Trainer.__init__
        #      calls set_seed(args.seed) (and TRL calls set_seed again after it)
        #      before any training RNG is drawn. Keep this call above that line.
        #   2. No PEFT. TRL runs get_peft_model() BEFORE super().__init__(), so LoRA
        #      initialization draws from the global RNG while it is still carrying
        #      whatever this sampling left behind. Enable PEFT and every run's
        #      adapters silently diverge -- no error, just incomparable runs.
        #      The check below refuses to let that happen quietly.
        if (spec.get("hyperparameters") or {}).get("peft_config") is not None:
            raise ValueError(
                "PEFT is not supported here: TRL initializes LoRA before the trainer "
                "reseeds, so the base-policy reference sampling above would perturb "
                "adapter init and break bit-identical retraining across arms"
            )
        base_sampling_metadata: dict[str, Any] = {}
        baseline_model = model
        baseline_identity = reference_snapshot["path"]
        if action is not None:
            baseline_identity = action.initial_reference
            if context.extras.get("base_policy_sequences") is None:
                baseline_model = load_action_model(action.initial_reference, reference_architecture)
                baseline_model = baseline_model.to(model.device)
        base_policy_sequences = self._base_policy_library(
            model=baseline_model,
            tokenizer=tokenizer,
            context=context,
            samples_per_prompt=int(
                context.extras.get("base_library_samples_per_prompt") or 128
            ),
            seed=int(context.extras.get("baseline_seed", 0)) if action is not None else seed,
            reference_identity=baseline_identity,
            sampling_metadata=base_sampling_metadata,
            execution_cost=execution_cost,
        )
        if action is not None:
            del baseline_model

        original_model_id = model.config._name_or_path
        model.config._name_or_path = reference_snapshot["path"]
        try:
            trainer_cls = GRPOTrainer
            trainer_extra = {}
            if action is not None:
                from recital.training.action_state import action_trainer_class
                trainer_cls = action_trainer_class(GRPOTrainer)
                trainer_extra["action"] = action
            trainer = trainer_cls(
                model=model,
                reward_funcs=trl_reward,
                args=cfg,
                train_dataset=train_ds,
                processing_class=tokenizer,
                callbacks=callbacks,
                **trainer_extra,
            )
        finally:
            model.config._name_or_path = original_model_id
        selection_cost_cb = None
        if selection_cb is not None:
            # Public HF API: attach to this same GRPOTrainer, with its existing
            # model/optimizer/scheduler/reference. Selection never loads weights.
            trainer.add_callback(selection_cb)
            selection_cost_cb = _checkpoint_selection_cost_callback(selection_cb, execution_cost)
            trainer.add_callback(selection_cost_cb)
            self._active_checkpoint_selection = selection_cost_cb
        if trainer.ref_model is not None:
            reference_snapshot["matched_tensors"] = _check_reference_tensor_parity(
                reference_model, trainer.ref_model,
            )
            reference_snapshot["tensor_parity_verified"] = True
        else:
            reference_snapshot["tensor_parity_verified"] = False
            reference_snapshot["reference_not_instantiated"] = "TRL reference disabled"
        if action is not None:
            del reference_model
        LOG.info("TRLDAPOTrainer starting: steps=%d lr=%.0e kl=%.3f group=%d",
                 steps, lr, kl, group_size)
        resume_path: str | None = None
        if action is not None:
            resume_path = str(action.resume_path) if action.resume_path else None
            if resume_path and action.mode == "crash_recovery":
                for callback in (selection_cb, selection_cost_cb):
                    if callback is not None:
                        callback.restore(resume_path)
        elif resume_mid_run:
            resume_path = _latest_checkpoint(out_dir)
            if resume_path:
                LOG.info("Resuming mid-run from %s", resume_path)
                if reward_state_cb is not None:
                    reward_state_cb.restore(resume_path)
                for callback in (selection_cb, selection_cost_cb):
                    if callback is not None:
                        callback.restore(resume_path)
        # Architecture is a runtime factory class during construction. Once
        # policy/reference loading is complete, persist its qualified name so
        # standard GRPOConfig JSON and training-argument serialization stay valid.
        reference_name = f"{reference_architecture.__module__}.{reference_architecture.__qualname__}"
        cfg.model_init_kwargs = {**reference_kwargs, "architecture": reference_name}
        with execution_cost.training():
            train_output = trainer.train(resume_from_checkpoint=resume_path)
        execution_cost.observe_train_runtime(train_output.metrics if train_output else None)
        LOG.info("TRLDAPOTrainer finished: %s", train_output.metrics if train_output else None)
        optimizer_steps_completed, optimizer_steps_source = (
            _optimizer_step_provenance(trainer, configured_steps=steps)
        )
        action_receipt = None
        if action is not None:
            if trainer.state.global_step != action.end_step:
                raise RuntimeError("training action ended before its declared additional_updates")
            end_path = action.out_dir / f"checkpoint-{trainer.state.global_step}"
            if not (end_path / "training_action_state.json").exists():
                trainer._save_checkpoint(trainer.model, trial=None)
            action_receipt = action.receipt(trainer.state.global_step, execution_cost=execution_cost.snapshot())
            from recital.training.action_state import _json_write
            _json_write(action.out_dir / "training_completed.json", action_receipt)
            completed_callback = context.extras.get("training_completed_callback")
            if completed_callback is not None:
                completed_callback(action_receipt)
            optimizer_steps_completed = action_receipt["additional_updates_completed"]
            optimizer_steps_source = "training_action_delta"

        # 6. Common training evidence is independent of final evaluation.
        training_log = list(getattr(trainer.state, "log_history", []) or [])
        if action is not None:
            # HF correctly retains ancestor history for recovery, but a new
            # action can change the recipe and KL reference. Its summary must
            # not mix measurements from those different scales/references.
            training_log = [row for row in training_log if isinstance(row, dict)
                            and isinstance(row.get("step"), (int, float))
                            and action.start_step < row["step"] <= trainer.state.global_step]
        # Real KL to the base policy: TRL logs per-step 'kl' in state.log_history;
        # train_output.metrics has NO kl key.
        kl_hist = [
            float(h["kl"]) for h in training_log
            if isinstance(h, dict) and "kl" in h
        ]
        # Retain every logged step. A large first observation is evidence, not
        # an automatically discarded transient. Reference parity is verified
        # before training; missing/nonfinite telemetry remains unavailable.
        base_policy_kl, base_policy_kl_max = _kl_window_summary(kl_hist)
        # GRPO advantage-variance signal: TRL logs 'frac_reward_zero_std' per step
        # (fraction of generation groups whose completions all got the SAME reward,
        # so the within-group advantage is identically 0 and contributes no
        # gradient). When this is high the policy CANNOT learn no matter the lr --
        # the fix is more generations per prompt (group_size) or hotter sampling,
        # not a higher lr. Summaries retain all logged observations.
        def _avg_metric(key: str) -> float | None:
            vals = [
                float(h[key]) for h in training_log
                if isinstance(h, dict) and key in h
            ]
            return float(np.mean(vals)) if vals and np.isfinite(vals).all() else None

        from recital.spec.composer import (
            DEFAULT_INVALID_PENALTY,
            DEFAULT_REWARD_CLIP_LOW,
        )

        frac_reward_zero_std = _avg_metric("frac_reward_zero_std")
        reward_std = _avg_metric("reward_std")
        traj: dict[str, Any] = {
            "enabled_oracles": enabled,
            "training_sequence_validity": {
                "candidate_count": trl_reward.candidate_count,
                "invalid_count": trl_reward.invalid_count,
            },
            "reference_model_architecture": reference_name,
            "reference_checkpoint": reference_snapshot,
            "training_configuration": cfg.to_dict(),
            # The spec records neither constant -- they are absent keys, not
            # nulls -- so the recorded spec alone is insufficient to rebuild the
            # reward. Record what this run actually trained against.
            "effective_reward_constants": {
                "invalid_penalty": float(
                    spec.get("invalid_penalty", DEFAULT_INVALID_PENALTY)
                ),
                "reward_clip_low": float(
                    spec.get("reward_clip_low", DEFAULT_REWARD_CLIP_LOW)
                ),
                "declared_in_spec": sorted(
                    key for key in ("invalid_penalty", "reward_clip_low")
                    if key in spec
                ),
            },
            "base_policy_sampling": base_sampling_metadata,
            "training_log": training_log,
            "reward_execution": _reward_execution_summary(trl_reward.execution_history),
            "reward_execution_history": list(trl_reward.execution_history),
            "base_policy_kl": base_policy_kl,
            "base_policy_kl_max": base_policy_kl_max,
            "frac_reward_zero_std": frac_reward_zero_std,
            "reward_std": reward_std,
            "group_size": group_size,
            "condition_repeats": {
                str(index): int(count)
                for index, count in sorted(repeats.items())
            },
            "seed": seed,
            # Training-set sample for identity/reuse diagnostics (identity_drift,
            # seen-sequence detection); supplied by the runner via extras.
            "training_set": context.extras.get("training_set") or [],
            "policy": trainer.model,
            "train_metrics": dict(train_output.metrics) if train_output else {},
            "optimizer_steps_completed": optimizer_steps_completed,
            "optimizer_steps_source": optimizer_steps_source,
            "optimizer_steps_configured": action.additional_updates if action is not None else steps,
            "evaluation_performed": False,
            "eval_checkpoint": "not_evaluated",
            "checkpoint_dir": str(out_dir) if (save_every > 0 or resume_mid_run or action is not None) else None,
            "checkpoint_resumed_from": resume_path,
            "training_start": "resume_same_candidate" if resume_path else "configured_initial_policy",
            "reward_state_resume": (
                reward_state_cb.restore_status if reward_state_cb is not None else "not_exposed"
            ),
        }
        if action_receipt is not None:
            traj["training_action_receipt"] = action_receipt
            traj["training_start"] = action.mode
            traj["reward_state_resume"] = f"training_action:{action.reward_policy}"
            traj["reference_policy_kl"] = base_policy_kl
            traj["reference_policy_kl_max"] = base_policy_kl_max
            traj["kl_reference"] = action.kl_reference
            traj["initial_reference"] = action.initial_reference
            if action.kl_reference != action.initial_reference:
                traj["base_policy_kl"] = None
                traj["base_policy_kl_max"] = None
                traj["base_policy_kl_unavailable_reason"] = "KL was measured against the declared stage reference, not initial SFT"
        traj["base_policy_sequences"] = base_policy_sequences
        if perform_final_evaluation:
            # Final diagnostics must not advance the training RNG or leave the
            # policy in a different mode when the caller only needs selection.
            with isolated_rng(), evaluation_mode(trainer.model):
                traj.update(self._final_evaluation(
                    spec=spec, reward_fn=reward_fn, context=context,
                    trainer=trainer, tokenizer=tokenizer, execution_cost=execution_cost,
                    samples_per_prompt=samples_per_prompt, seed=seed,
                    eval_checkpoint="final",
                    trained_reward=getattr(trl_reward, "composed_reward", None),
                ))
        from recital.decision_evidence import training_progress
        traj["training_progress"] = training_progress(traj)
        return traj

    def _final_evaluation(
        self, *, spec: dict[str, Any], reward_fn: Callable[..., np.ndarray],
        context: TrainContext, trainer: Any, tokenizer: Any, execution_cost: Any,
        samples_per_prompt: int, seed: int, eval_checkpoint: str,
        trained_reward: Any = None,
    ) -> dict[str, Any]:
        """Generate and score a final library only when explicitly performed."""
        execution_cost.set_phase("final")
        codec = context.extras.get("sequence_codec", context.extras.get("codec"))
        # Generate a final EVAL batch for the Diagnostician/monitors. This must be
        # large enough to measure per-group hit rates with low variance: at
        # group_size (e.g. 8) samples/group the rates are 1/8-quantized noise. Use
        # eval_samples_per_prompt (default 128 -> 384 total for 3 groups).
        # Configurable via context.extras['eval_samples_per_prompt'].
        eval_spp = int(
            context.extras.get("eval_samples_per_prompt")
            or max(samples_per_prompt, 128)
        )
        evaluation_seed_source = (
            "context.extras.evaluation_seed"
            if context.extras.get("evaluation_seed") is not None
            else "trainer_seed_fallback"
        )
        evaluation_seed = int(context.extras.get("evaluation_seed", seed))
        evaluation_seed_applied = _seed_evaluation_rng(evaluation_seed)
        evaluation_sampling_metadata: dict[str, Any] = {}
        gen_seqs, gen_cond = self._sample_final_batch(
            trainer=trainer,
            tokenizer=tokenizer,
            conditioning_prompts=context.conditioning_prompts or [""],
            samples_per_prompt=eval_spp,
            alphabet=context.token_alphabet,
            min_new_tokens=context.extras.get("min_new_tokens"),
            sampling_temperature=float(context.extras.get("sampling_temperature", EVALUATION_SAMPLING_TEMPERATURE)),
            sampling_top_p=float(context.extras.get("sampling_top_p", EVALUATION_SAMPLING_TOP_P)),
            completion_prefix=str(context.extras.get("completion_prefix") or ""),
            codec=context.extras.get("sequence_codec", context.extras.get("codec")),
            sampling_metadata=evaluation_sampling_metadata,
            execution_cost=execution_cost,
            cost_phase="final",
            **self._sampling_options(context),
        )
        # Score the enabled oracles plus any diagnostic-only oracles the task wants
        # available to its monitors (e.g. the Malinois MinGap monitor), so the
        # Diagnostician has what it needs even for oracles not used in the reward.
        from recital.spec.composer import ComposedRewardFn, RewardSpec

        evaluation_keys, scored_held_out = _evaluation_oracle_keys(spec, context)
        final_keys = set(evaluation_keys)
        required_final = _required_evaluation_oracles(
            spec, context, scored_held_out=scored_held_out
        )
        prefix = str(context.extras.get("completion_prefix") or "")
        sequence_length = context.extras.get("sequence_length")
        min_length = context.extras.get("min_sequence_length")
        max_length = context.extras.get("max_sequence_length")
        if sequence_length is None and min_length is None and max_length is None and codec is None:
            sequence_length = len(prefix) + self.max_completion_length
        validity = _sequence_validity(
            gen_seqs, alphabet=context.token_alphabet, sequence_length=sequence_length,
            min_length=1 if min_length is None else int(min_length),
            max_length=max_length, codec=codec,
        )
        valid_mask = np.asarray(validity["valid_mask"], dtype=bool)
        valid_sequences = [s for s, keep in zip(gen_seqs, valid_mask, strict=True) if keep]
        valid_conditions = [p for p, keep in zip(gen_cond, valid_mask, strict=True) if keep]
        oracle_outputs: dict[str, np.ndarray] = {}
        oracle_status: dict[str, dict[str, Any]] = {}
        for o in final_keys:
            if o not in context.oracle_fns:
                oracle_status[o] = {"status": "unregistered"}
                if o in required_final:
                    raise KeyError(
                        f"required final-evaluation oracle is not registered: {o}"
                    )
                continue
            try:
                values, status = _score_valid_sequences(
                    name=o, scorer=context.oracle_fns[o],
                    sequences=gen_seqs, valid_mask=valid_mask,
                    expected_widths=context.extras.get("oracle_expected_widths"),
                    required_finite_columns=context.extras.get(
                        "oracle_required_finite_columns"
                    ),
                    require_finite=o in required_final,
                    stage="final evaluation",
                )
                oracle_outputs[o] = values
                oracle_status[o] = status
            except Exception as exc:  # noqa: BLE001
                oracle_status[o] = {"status": "error", "error": repr(exc)}
                if o in required_final:
                    raise RuntimeError(
                        "required oracle failed at the final-evaluation scorer "
                        f"boundary: {o}"
                    ) from exc
                LOG.warning(
                    "Optional diagnostic oracle %s failed on final batch: %r", o, exc
                )
        # Filling zeros or retaining a malformed scorer output can manufacture a
        # clean-looking trajectory and silently influence outer selection.
        missing_required = sorted(required_final - set(oracle_outputs))
        if missing_required:
            raise RuntimeError(
                "required oracle(s) failed on the final evaluation batch: "
                f"{missing_required}"
            )
        rfn = reward_fn
        _patched = _bind_prompt_conditioning(
            spec, valid_conditions, context.conditioning_prompts, context.extras.get("group_intervals"),
            oracle_target_indices=context.extras.get("oracle_target_indices"),
            primitive_resolver=context.extras.get("primitive_resolver")
            or getattr(reward_fn, "_primitive_resolver", None),
        )
        if _patched is not spec:
            # Reuse the object training ran on. A fresh ComposedRewardFn here
            # would z-score the final library against ITSELF -- a rank, not a
            # level -- because its EmaZScore/contrastive_mingap state is empty.
            if trained_reward is not None:
                rfn = _rebind_conditioning(
                    trained_reward, valid_conditions, context.conditioning_prompts,
                    context.extras.get("oracle_target_indices"),
                )
            else:
                rfn = _candidate_reward_build(
                    lambda: ComposedRewardFn(
                        RewardSpec.from_dict(_patched),
                        gate_registry=getattr(reward_fn, "_gate_registry", None),
                        training_set=getattr(reward_fn, "_training_set", None),
                        reference_libraries=getattr(reward_fn, "_reference_libraries", None),
                        primitive_resolver=context.extras.get("primitive_resolver")
                        or getattr(reward_fn, "_primitive_resolver", None),
                    ),
                    stage="final conditional reward binding",
                )
        invalid_penalty = float(spec.get("invalid_penalty", _default_invalid_penalty()))
        if not np.isfinite(invalid_penalty):
            raise ValueError("invalid_penalty must be finite")
        final_rewards = np.full(len(gen_seqs), invalid_penalty)
        if valid_mask.any():
            final_rewards[valid_mask] = _candidate_reward_call(
                rfn, sequences=valid_sequences,
                oracle_outputs={name: values[valid_mask] for name, values in oracle_outputs.items()},
                expected_n=len(valid_sequences), stage="final evaluation",
                conditioning=(
                    _reward_conditioning(valid_conditions, context.conditioning_prompts, context.extras.get("condition_names"))
                    if _requires_reward_conditioning(spec, rfn, context.extras.get("primitive_resolver")) else None
                ),
            )
        heldout_oracle_outputs = {
            k: oracle_outputs[k] for k in scored_held_out if k in oracle_outputs
        }
        result = {
            "evaluation_performed": True, "eval_checkpoint": eval_checkpoint,
            "sequences": gen_seqs, "conditioning": gen_cond,
            "oracle_outputs": oracle_outputs,
            "heldout_oracle_outputs": heldout_oracle_outputs,
            "oracle_status": oracle_status, "sequence_validity": validity,
            "invalid_fraction": validity["invalid_fraction"],
            "evaluation_sampling": evaluation_sampling_metadata,
            "reward": np.asarray(final_rewards, dtype=float),
            "evaluation_seed": evaluation_seed,
            "evaluation_seed_source": evaluation_seed_source,
            "evaluation_seed_applied": evaluation_seed_applied,
        }
        result.update(
            _auxiliary_telemetry(
                spec=spec,
                context=context,
                sequences=gen_seqs,
                conditioning=gen_cond,
                oracle_outputs=oracle_outputs,
            )
        )
        return result

    def _base_policy_library(
        self,
        *,
        model,
        tokenizer,
        context: TrainContext,
        samples_per_prompt: int,
        seed: int,
        reference_identity: str | None = None,
        sampling_metadata: dict[str, Any] | None = None,
        execution_cost: Any = None,
    ) -> list[str]:
        """Sequences from the untrained SFT policy, cached for the process.

        Cheap by construction: one generation pass per run, reused by every
        iteration, because every iteration re-inits from the same checkpoint.
        """
        import types as _types

        prompts = list(context.conditioning_prompts or [""])
        supplied = context.extras.get("base_policy_sequences")
        if supplied is not None:
            if (not isinstance(supplied, (list, tuple)) or not supplied
                    or any(not isinstance(s, str) or not s or set(s) - set(context.token_alphabet) for s in supplied)):
                raise ValueError("base_policy_sequences must contain valid nonempty sequence strings")
            if execution_cost is not None:
                execution_cost.add("base_cache_hits", 1)
                execution_cost.add("base_reused_candidates", len(supplied))
            if sampling_metadata is not None:
                sampling_metadata.update(cache_hit=True, source="task_supplied_reference_library", count=len(supplied))
            return list(supplied)
        key = (
            reference_identity, tuple(prompts), samples_per_prompt, seed,
            self.max_completion_length, context.token_alphabet,
            context.extras.get("min_new_tokens"), context.extras.get("completion_prefix", ""),
            context.extras.get("sampling_temperature", EVALUATION_SAMPLING_TEMPERATURE), context.extras.get("sampling_top_p", EVALUATION_SAMPLING_TOP_P),
            tuple(self._sampling_options(context).items()),
        )
        cached = _BASE_LIBRARY_CACHE.get(key) if reference_identity is not None else None
        if cached is not None:
            if execution_cost is not None:
                execution_cost.add("base_cache_hits", 1)
                execution_cost.add("base_reused_candidates", len(cached))
            if sampling_metadata is not None:
                sampling_metadata.update({
                    "cache_hit": True, "reference_identity": reference_identity,
                    "sampling_options": self._sampling_options(context),
                    "samples_per_prompt": samples_per_prompt,
                })
            return cached

        _seed_evaluation_rng(seed)
        sequences, _ = self._sample_final_batch(
            trainer=_types.SimpleNamespace(model=model),
            tokenizer=tokenizer,
            conditioning_prompts=prompts,
            samples_per_prompt=samples_per_prompt,
            alphabet=context.token_alphabet,
            min_new_tokens=context.extras.get("min_new_tokens"),
            sampling_temperature=float(context.extras.get("sampling_temperature", EVALUATION_SAMPLING_TEMPERATURE)),
            sampling_top_p=float(context.extras.get("sampling_top_p", EVALUATION_SAMPLING_TOP_P)),
            completion_prefix=str(context.extras.get("completion_prefix") or ""),
            codec=context.extras.get("sequence_codec", context.extras.get("codec")),
            sampling_metadata=sampling_metadata,
            execution_cost=execution_cost,
            cost_phase="base",
            **self._sampling_options(context),
        )
        model.train()
        if sampling_metadata is not None:
            sampling_metadata["cache_hit"] = False
        if reference_identity is not None:
            _BASE_LIBRARY_CACHE[key] = sequences
        LOG.info("Sampled base-policy reference library: n=%d", len(sequences))
        return sequences

    @staticmethod
    def _sampling_options(context: TrainContext) -> dict[str, Any]:
        """Task-supplied options shared by base and final generation."""
        return {
            name: context.extras[name]
            for name in (
                "sampling_add_special_tokens", "sampling_batch_size", "sampling_use_cache",
                "sampling_repetition_penalty",
            )
            if name in context.extras
        }

    def _sample_final_batch(
        self,
        *,
        trainer,
        tokenizer,
        conditioning_prompts: list[str],
        samples_per_prompt: int,
        alphabet: str,
        min_new_tokens: int | None = None,
        sampling_temperature: float = EVALUATION_SAMPLING_TEMPERATURE,
        sampling_top_p: float = EVALUATION_SAMPLING_TOP_P,
        completion_prefix: str = "",
        codec: Any = None,
        sampling_add_special_tokens: bool | None = None,
        sampling_batch_size: int | None = None,
        sampling_use_cache: bool | None = None,
        sampling_repetition_penalty: float | None = None,
        sampling_metadata: dict[str, Any] | None = None,
        execution_cost: Any = None,
        cost_phase: str | None = None,
    ) -> tuple[list[str], list[str]]:
        """Generate raw completions in condition order, with explicit opt-in settings.

        None preserves the tokenizer/model defaults and one batch per condition.
        Atomic-prompt tasks must explicitly pass
        add_special_tokens=False; the task-owned tokenizer must ALSO implement
        the correct default for TRL's separate training-generation entry point.
        Chunk size affects random draws and is part of reproducible sampling.
        """
        import torch

        if type(samples_per_prompt) is not int or samples_per_prompt < 1:
            raise ValueError("samples_per_prompt must be a positive integer")
        if sampling_batch_size is not None and (
            type(sampling_batch_size) is not int or sampling_batch_size < 1
        ):
            raise ValueError("sampling_batch_size must be a positive integer")
        for name, value in (("sampling_add_special_tokens", sampling_add_special_tokens),
                            ("sampling_use_cache", sampling_use_cache)):
            if value is not None and type(value) is not bool:
                raise ValueError(f"{name} must be a boolean or None")
        if sampling_repetition_penalty is not None and (
            isinstance(sampling_repetition_penalty, bool)
            or not np.isfinite(sampling_repetition_penalty) or sampling_repetition_penalty <= 0
        ):
            raise ValueError("sampling_repetition_penalty must be finite and positive")
        model = trainer.model.eval()
        device = next(model.parameters()).device
        all_seqs: list[str] = []
        all_cond: list[str] = []
        if min_new_tokens is None:
            min_new_tokens = self.max_completion_length
        if not 0 <= int(min_new_tokens) <= self.max_completion_length:
            raise ValueError("min_new_tokens must be within the completion token budget")
        batch_size = sampling_batch_size or samples_per_prompt
        tokenizer_options = {} if sampling_add_special_tokens is None else {
            "add_special_tokens": sampling_add_special_tokens,
        }
        generation_options = {
            "max_new_tokens": self.max_completion_length,
            "min_new_tokens": min_new_tokens,
            "do_sample": True,
            "temperature": sampling_temperature,
            "top_p": sampling_top_p,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if sampling_use_cache is not None:
            generation_options["use_cache"] = sampling_use_cache
        if sampling_repetition_penalty is not None:
            generation_options["repetition_penalty"] = sampling_repetition_penalty
        legal_proc = self._build_legal_alphabet_processor(tokenizer, alphabet, codec)
        if sampling_metadata is not None:
            sampling_metadata.clear()
            generation_config = getattr(model, "generation_config", None)
            sampling_metadata.update({
                "tokenizer_class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
                "tokenizer_options": tokenizer_options,
                "tokenizer_default_preserved": sampling_add_special_tokens is None,
                "samples_per_prompt": samples_per_prompt,
                "batch_size": batch_size,
                "sample_order": "condition_then_chunk",
                "rng_policy": "caller_global_rng; chunk size affects draws",
                "generation_options": generation_options,
                "model_generation_config": (
                    generation_config.to_dict() if generation_config is not None else None
                ),
                "alphabet": alphabet,
                "batches": [],
            })
        for prompt in conditioning_prompts:
            for start in range(0, samples_per_prompt, batch_size):
                count = min(batch_size, samples_per_prompt - start)
                inputs = tokenizer(
                    [prompt] * count, return_tensors="pt", padding=True, **tokenizer_options,
                ).to(device)
                if sampling_add_special_tokens is False:
                    if "attention_mask" not in inputs:
                        inputs["attention_mask"] = inputs["input_ids"].ne(tokenizer.pad_token_id).long()
                    if inputs["input_ids"].shape[1] == 0 or not inputs["attention_mask"].all():
                        raise ValueError("atomic-prompt sampling requires nonempty, unpadded prompts")
                if sampling_metadata is not None:
                    sampling_metadata["batches"].append({
                        "prompt": prompt, "sample_start": start, "batch_size": count,
                        "prompt_token_ids": inputs["input_ids"][0].tolist(),
                    })
                try:
                    with torch.no_grad():
                        out = model.generate(
                            **inputs, **generation_options, logits_processor=[legal_proc],
                        )
                except BaseException:
                    if execution_cost is not None and cost_phase is not None:
                        execution_cost.incomplete(
                            f"{cost_phase}_completions", "generation failed before returning all output rows",
                        )
                    raise
                if execution_cost is not None and cost_phase is not None:
                    execution_cost.observe_completions(cost_phase, int(out.shape[0]))
                prompt_len = inputs["input_ids"].shape[1]
                decoded = _raw_completion_text(tokenizer, out[:, prompt_len:])
                if completion_prefix:
                    decoded = [completion_prefix + text for text in decoded]
                all_seqs.extend(decoded)
                all_cond.extend([prompt] * len(decoded))
        return all_seqs, all_cond
