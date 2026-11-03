"""Contract-owned Loop-seq measurements, independent of diagnostic flags.

Raw generated rows are the denominator. Invalid DNA is never repaired or dropped.
Oracle scores stay in each predictor's declared coordinates. References must be
hash-pinned inputs; an absent calibration remains unavailable, never a looser
threshold inferred from the candidate being evaluated.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from math import log2
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from recital.targets import OracleTargetObjective


def canonical_dna(sequence: str) -> str:
    """Canonical identity under reverse complement (legal DNA only)."""
    reverse = sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    return min(sequence, reverse)


def _entropy(sequence: str) -> float:
    counts = Counter(sequence[i : i + 2] for i in range(len(sequence) - 1))
    total = sum(counts.values())
    return -sum((n / total) * log2(n / total) for n in counts.values()) if total else 0.0


def _measurement(
    value: float | None, count: int, *, status: str = "measured", reason: str | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {"value": value, "count": count, "status": status}
    if reason:
        result["reason"] = reason
    return result


def interval_objective(entry: Mapping[str, Any]) -> OracleTargetObjective:
    """The half-open low/mid/high bands of one candidate-pool entry, in its raw coordinates.

    Low is below `low_mid_boundary`, mid is from it up to `mid_high_boundary`, high is at or above that. The
    online oracles' training and qualification targets and the terminal blind evaluator's bands are built here.
    """
    low = float(entry["low_mid_boundary"])
    high = float(entry["mid_high_boundary"])
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError(f"invalid raw boundaries for {entry.get('id')}")
    return OracleTargetObjective.from_dict(
        {
            "kind": "interval",
            "coordinate": "raw",
            "units": entry["score_units"],
            "scale": entry["target_scale"],
            "by_condition": {
                "low": {"lower": None, "upper": low},
                "mid": {"lower": low, "upper": high},
                "high": {"lower": high, "upper": None},
            },
        }
    )


@dataclass
class LoopseqSuccessEvaluator:
    """Task-specific measurement adapter; the shared core owns qualification."""

    groups: tuple[str, ...]
    conditioning_groups: dict[str, str]
    sequence_length: int
    oracle_boundaries: dict[str, tuple[float, float]]
    kmer_size: int = 3
    training_identities: frozenset[str] | None = None
    quality_thresholds: dict[str, dict[str, float]] | None = None
    reference_status: dict[str, dict[str, Any]] = field(default_factory=dict)
    oracle_objectives: dict[str, OracleTargetObjective] = field(default_factory=dict)

    @classmethod
    def from_contract(
        cls, contract: Mapping[str, Any], *, repo_root: Path, audit_oracles: set[str] | None = None
    ) -> "LoopseqSuccessEvaluator | None":
        if contract.get("success") is None:
            return None
        config = (contract.get("data") or {}).get("success_measurement")
        if not isinstance(config, dict):
            raise ValueError("a success contract requires data.success_measurement")
        groups = tuple(config["groups"])
        if not groups or len(groups) != len(set(groups)) or set(groups) != {"low", "mid", "high"}:
            raise ValueError("Loop-seq requires the explicit low/mid/high groups")
        mapping = dict(config["conditioning_groups"])
        if set(mapping.values()) != set(groups):
            raise ValueError("conditioning_groups must cover every expected group")
        length = int(config["sequence_length"])
        kmer_size = int(config["kmer_size"])
        if length < 2 or not 1 <= kmer_size <= length:
            raise ValueError("invalid sequence_length or kmer_size")
        pool = {entry["id"]: entry for entry in contract["oracles"]["candidate_pool"]}
        blind = set(audit_oracles or ()) | {
            key
            for key, entry in pool.items()
            if entry.get("use_for") in {"audit", "optional_audit"}
        }
        online = list(config["online_oracles"])
        if not online or len(set(online)) != len(online) or set(online) & blind:
            raise ValueError(
                "success online_oracles must be unique, nonempty, and exclude blind oracles"
            )
        boundaries = {}
        objectives = {}
        for oracle in online:
            entry = pool[oracle]
            objectives[oracle] = interval_objective(entry)
            boundaries[oracle] = (float(entry["low_mid_boundary"]), float(entry["mid_high_boundary"]))
        evaluator = cls(groups, mapping, length, boundaries, kmer_size)
        evaluator.oracle_objectives.update(objectives)
        required_reference_metrics = {item["metric"] for item in (
            *contract["success"].get("requirements", []),
            *contract["success"].get("ranking", []),
        )}
        evaluator._load_references(config, Path(repo_root), required_metrics=required_reference_metrics)
        supported = {
            "generated_count",
            "legal_ratio",
            "condition_valid_ratio",
            "group_balance_ratio",
            "rc_unique_ratio",
            "train_exact_overlap",
            "entropy_pass_ratio",
            "kmer_pass_ratio",
            "joint_hit",
            "joint_target_quality_ratio",
            *(f"oracle_hit/{name}" for name in online),
            *(f"oracle_margin/{name}" for name in online),
        }
        for item in [
            *contract["success"].get("requirements", []),
            *contract["success"].get("ranking", []),
        ]:
            if item["metric"] not in supported:
                raise ValueError(f"Loop-seq cannot measure success metric {item['metric']!r}")
            expected = (
                {"all"}
                if item["metric"] in {"condition_valid_ratio", "group_balance_ratio"}
                else set(groups)
            )
            if not set(item["strata"]).issubset(expected):
                raise ValueError(f"invalid strata for success metric {item['metric']!r}")
            metric = item["metric"]
            sources = (
                [metric.split("/", 1)[1]]
                if metric.startswith(("oracle_hit/", "oracle_margin/"))
                else online
                if metric in {"joint_hit", "joint_target_quality_ratio"}
                else []
            )
            if set(contract["success"].get("metric_sources", {}).get(metric, ())) != set(sources):
                raise ValueError(f"success metric_sources misrepresents {metric!r}")
        return evaluator

    def reward_targets(self, enabled_oracles: list[str]) -> dict[str, Any]:
        """Compile the input target into the composer's typed objective API."""
        return {oracle: self.oracle_objectives[oracle].to_dict() for oracle in enabled_oracles}

    def _reference_path(self, record: Any, root: Path, name: str) -> Path | None:
        if not isinstance(record, dict) or not record.get("path"):
            self.reference_status[name] = {
                "status": "unavailable",
                "reason": "input reference is pending",
            }
            return None
        path = Path(record["path"]).expanduser()
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            self.reference_status[name] = {
                "status": "unavailable",
                "reason": "declared reference file is absent",
            }
            return None
        self.reference_status[name] = {
            "status": "measured",
            "path": str(path),
        }
        return path

    def _load_references(self, config: dict[str, Any], root: Path, *, required_metrics: set[str]) -> None:
        training = config.get("training_reference")
        path = (self._reference_path(training, root, "training")
                if training is not None or "train_exact_overlap" in required_metrics else None)
        if path is not None:
            import pandas as pd

            columns = [training["sequence_column"], training["split_column"]]
            frame = pd.read_parquet(path, columns=columns)
            seqs = frame.loc[
                frame[training["split_column"]] == training["split_value"],
                training["sequence_column"],
            ].tolist()
            if not seqs or not all(self._legal(seq) for seq in seqs):
                raise ValueError("full training reference is empty or contains invalid DNA")
            self.training_identities = frozenset(canonical_dna(seq) for seq in seqs)
            self.reference_status["training"]["n_rows"] = len(seqs)
        quality = config.get("quality_reference")
        path = (self._reference_path(quality, root, "quality")
                if quality is not None or required_metrics & {"entropy_pass_ratio", "kmer_pass_ratio", "joint_target_quality_ratio"}
                else None)
        if path is not None:
            payload = json.loads(path.read_text())
            if (
                payload.get("schema_version") != 1
                or payload.get("sequence_length") != self.sequence_length
                or payload.get("kmer_size") != self.kmer_size
            ):
                raise ValueError("quality reference representation differs from success contract")
            thresholds = payload.get("strata") or {}
            for group in self.groups:
                for metric, upper in (
                    ("dinucleotide_entropy_min", 4),
                    ("unique_kmer_min", self.sequence_length - self.kmer_size + 1),
                ):
                    value = (thresholds.get(group) or {}).get(metric)
                    if value is None or not np.isfinite(value) or not 0 <= value <= upper:
                        raise ValueError(f"quality reference lacks a valid {group}/{metric}")
            self.quality_thresholds = thresholds

    def _legal(self, sequence: Any) -> bool:
        return (
            isinstance(sequence, str)
            and len(sequence) == self.sequence_length
            and set(sequence) <= set("ACGT")
        )

    def _condition_group(self, condition) -> str | None:
        if not isinstance(condition, str):
            return None
        group = self.conditioning_groups.get(condition, condition)
        return group if group in self.groups else None

    def evaluate(self, trajectory: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
        """Evaluate every requested group, even if the batch omitted that group."""
        sequences = list(trajectory.get("sequences") or [])
        conditioning = list(trajectory.get("conditioning") or [])
        n = len(sequences)
        if len(conditioning) != n:
            raise ValueError("success measurement requires aligned raw sequences and conditioning")
        groups = [self._condition_group(label) for label in conditioning]
        legal = np.array([self._legal(seq) for seq in sequences], dtype=bool)
        result: dict[str, dict[str, dict[str, Any]]] = {}
        outputs = trajectory.get("oracle_outputs") or {}
        hits: dict[str, np.ndarray] = {}
        depths: dict[str, np.ndarray] = {}
        problems: dict[str, tuple[str, str]] = {}
        for oracle, (low, high) in self.oracle_boundaries.items():
            raw = outputs.get(oracle)
            if raw is None:
                problems[oracle] = ("unavailable", "required online oracle has no raw scores")
                continue
            try:
                values = np.asarray(raw, dtype=float)
            except (TypeError, ValueError):
                problems[oracle] = ("error", "oracle scores are not numeric")
                continue
            if values.shape != (n,):
                problems[oracle] = ("error", "oracle scores do not align with raw rows")
                continue
            # Use the same half-open raw-coordinate targets as training. Unknown
            # conditions remain non-hits and fail condition_valid_ratio below.
            mask = np.zeros(n, dtype=bool)
            # How far inside its requested band each row sits, in declared target
            # scales. `hit` alone reads 1.0 for every in-band row, so it cannot
            # order two qualified deliveries; the depth behind it still separates them.
            depth = np.full(n, np.nan)
            for group in self.groups:
                indices = np.array(
                    [i for i, value in enumerate(groups) if value == group], dtype=int
                )
                target = self.oracle_objectives[oracle].by_condition[group]
                mapped = target.evaluate(values[indices])
                mask[indices] = mapped.hit
                if mapped.margin is not None:
                    depth[indices] = mapped.margin
            hits[oracle] = legal & np.isfinite(values) & mask
            depths[oracle] = depth
            # Non-finite scores on legal rows mean the required measurement is
            # incomplete; scores on invalid rows are unnecessary (those rows fail).
            problems[oracle] = (
                ("partial", "non-finite oracle scores on legal DNA")
                if np.any(legal & ~np.isfinite(values))
                else ("", "")
            )

        def record(
            metric: str,
            group: str,
            value: float | None,
            count: int,
            status: str = "measured",
            reason: str | None = None,
        ) -> None:
            if count == 0 and status == "measured":
                status, value, reason = "unavailable", None, "expected group has no generated rows"
            result.setdefault(metric, {})[group] = _measurement(
                value, count, status=status, reason=reason
            )

        record(
            "condition_valid_ratio",
            "all",
            sum(group in self.groups for group in groups) / n if n else None,
            n,
        )
        sizes = [groups.count(group) for group in self.groups]
        record(
            "group_balance_ratio",
            "all",
            min(sizes) / max(sizes) if max(sizes, default=0) else 0.0,
            n,
        )
        for group in self.groups:
            indices = np.array([i for i, value in enumerate(groups) if value == group], dtype=int)
            count = len(indices)
            denominator = count or 1
            record("generated_count", group, float(count), count)
            record("legal_ratio", group, float(legal[indices].sum() / denominator), count)
            identities = [canonical_dna(sequences[i]) for i in indices if legal[i]]
            record("rc_unique_ratio", group, len(set(identities)) / denominator, count)
            if self.training_identities is None:
                record(
                    "train_exact_overlap",
                    group,
                    None,
                    count,
                    "unavailable",
                    self.reference_status.get("training", {}).get(
                        "reason", "full training reference is pending"
                    ),
                )
            else:
                record(
                    "train_exact_overlap",
                    group,
                    sum(identity in self.training_identities for identity in identities)
                    / denominator,
                    count,
                )
            group_hits = []
            joint_problem = None
            for oracle in self.oracle_boundaries:
                status, reason = problems.get(oracle, ("", ""))
                if status == "partial":
                    raw = np.asarray(outputs[oracle], dtype=float)
                    status = (
                        "unavailable" if np.any(legal[indices] & ~np.isfinite(raw[indices])) else ""
                    )
                if status:
                    record(f"oracle_hit/{oracle}", group, None, count, status, reason)
                    record(f"oracle_margin/{oracle}", group, None, count, status, reason)
                    joint_problem = (status, reason)
                else:
                    group_hits.append(hits[oracle][indices])
                    record(
                        f"oracle_hit/{oracle}",
                        group,
                        float(hits[oracle][indices].sum() / denominator),
                        count,
                    )
                    # Median over the rows that carry a depth at all. An illegal or
                    # non-finite row has no position relative to the band; it is
                    # already failing legal_ratio and does not get a stand-in value.
                    inside = indices[legal[indices] & np.isfinite(depths[oracle][indices])]
                    record(
                        f"oracle_margin/{oracle}",
                        group,
                        float(np.median(depths[oracle][inside])) if inside.size else None,
                        count,
                        *(("measured", None) if inside.size
                          else ("unavailable", "no legal row carries a band depth")),
                    )
            joint = np.logical_and.reduce(group_hits) if group_hits else np.zeros(count, dtype=bool)
            if joint_problem:
                record("joint_hit", group, None, count, *joint_problem)
            else:
                record("joint_hit", group, float(joint.sum() / denominator), count)
            if self.quality_thresholds is None:
                for metric in (
                    "entropy_pass_ratio",
                    "kmer_pass_ratio",
                    "joint_target_quality_ratio",
                ):
                    record(
                        metric,
                        group,
                        None,
                        count,
                        "unavailable",
                        self.reference_status.get("quality", {}).get(
                            "reason", "conditional quality reference is pending"
                        ),
                    )
                continue
            thresholds = self.quality_thresholds[group]
            entropy_ok = np.array(
                [
                    legal[i] and _entropy(sequences[i]) >= thresholds["dinucleotide_entropy_min"]
                    for i in indices
                ],
                dtype=bool,
            )
            kmer_ok = np.array(
                [
                    legal[i]
                    and len(
                        {
                            sequences[i][j : j + self.kmer_size]
                            for j in range(len(sequences[i]) - self.kmer_size + 1)
                        }
                    )
                    >= thresholds["unique_kmer_min"]
                    for i in indices
                ],
                dtype=bool,
            )
            record("entropy_pass_ratio", group, float(entropy_ok.sum() / denominator), count)
            record("kmer_pass_ratio", group, float(kmer_ok.sum() / denominator), count)
            if joint_problem:
                record("joint_target_quality_ratio", group, None, count, *joint_problem)
            else:
                record(
                    "joint_target_quality_ratio",
                    group,
                    float((joint & entropy_ok & kmer_ok).sum() / denominator),
                    count,
                )
        return result
