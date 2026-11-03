"""CM measurements injected into the shared resumable checkpoint selector.

The task supplies sampling and biological scorers. The shared callback owns
timing, RNG isolation, checkpoint selection and recovery. No terminal predictor
is loaded here, and evaluation never changes the training reward or optimizer.
"""
from __future__ import annotations

import copy
import gc
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from recital.costs import journaled_oracle_call
from recital.sequence_complexity import sequence_complexity_metrics
from recital.training.checkpoint_selection import CheckpointSelectionCallback, path_reference
from recital.terminal_eligibility import EvidenceEligibility

from .evaluation import sample_policy, select_library
from .adapter import protocol_condition
from .oracles import load_oracle
from .qualification import write_json
from .data import save_table


AUDIT = ("audit_2020_esm", "audit_2020_t5")


def eligible_pool(table: pd.DataFrame, *, min_distance: float,
                  distance_ceiling: float | None = None) -> tuple[pd.DataFrame, str]:
    """Unique, novel, non-training candidates at or past the declared distance floor.

    Eligibility is the floor. The cm_novelty ceiling does not appear here: it is a
    reward term, and passing it raises.

    The control layer's delivery maximum (contract delivery.max_natural_distance) is
    applied by its selection plan, not here.

    The second return value is always the empty string.
    """
    eligible = (table.valid & ~table.duplicate & ~table.training_copy
                & table.natural_distance.ge(min_distance))
    if distance_ceiling is not None:
        raise ValueError(
            "the cm_novelty ceiling is a reward term, not an eligibility bound; "
            "pass it through oracles.novelty, not eligible_pool"
        )
    return table[eligible & np.isfinite(table.function_2026_esm)], ""


def candidate_measurements(table: pd.DataFrame, delivery: dict, *, count: int,
                           min_distance: float, score_names=()) -> dict:
    """Common selection measurements and a T5-independent search proxy."""
    measurements = {
        "cm_delivered": {"CM": {"value": float(delivery["selected_count"]), "count": len(table)}},
        "cm_selection_score": {"CM": {"value": delivery["selection_score_median"],
                               "count": delivery["selected_count"]}},
    }
    # The hidden-feedback proxy is ESM/sequence-only. Common final selection
    # can apply a family requirement without revealing its reserved evidence.
    pool, veto = eligible_pool(table, min_distance=min_distance)
    primary = pool.nlargest(count, "function_2026_esm")
    measurements.update({
        "cm_eligible": {"CM": {"value": float(len(pool)), "count": len(table), "reason": veto}},
        "cm_primary": {"CM": {"value": float(primary.function_2026_esm.median())
                                if len(primary) else None, "count": len(primary)}},
    })
    for name in dict.fromkeys(("function_2026_esm", "function_2026_t5", "cm_max", "cm_poet",
                              "natural_distance", *score_names)):
        if name not in table:
            continue
        values = table.loc[table.valid & np.isfinite(table[name]), name]
        measurements[name] = {"CM": {"value": float(values.median()) if len(values) else None,
                                      "count": len(values)}}
    return measurements


class CMCheckpointEvaluator:
    def __init__(self, *, root: Path, output: Path, contract: dict, family, tokenizer,
                 oracle_fns: dict, evaluation_seed: int, device: str, diagnostic_oracles=()):
        self.root, self.output = Path(root), Path(output)
        self.contract, self.family, self.tokenizer = contract, family, tokenizer
        roles = contract["oracles"]["source_roles"]["primary"]
        self.online_names = tuple(dict.fromkeys(roles["reward_pool"] + roles["sentinel_pool"]))
        self.audit_names = tuple(roles.get("audit_oracles", AUDIT))
        if set(self.online_names) & set(self.audit_names):
            raise ValueError("CM checkpoint sources must exclude terminal predictors")
        if protocol_condition(contract) == "no_self_validation":
            if contract["delivery"].get("selection_aggregation") != "primary":
                raise ValueError("CM no_self_validation requires primary-only checkpoint selection")
            if set(diagnostic_oracles) - set(roles["reward_pool"]):
                raise ValueError("CM retired diagnostics must come from the reward pool")
            self.online_names = tuple(dict.fromkeys([*oracle_fns, *diagnostic_oracles]))
        self.oracle_fns = dict(oracle_fns)
        self.seed, self.device = evaluation_seed, device
        self.current_directory = self.output
        self.calibration = {name: joblib.load(self.root / "oracles" / f"{name}.joblib")["calibration"]
                            for name in ("function_2026_esm", "function_2026_t5")}

    def score(self, name: str, sequences: list[str]) -> np.ndarray:
        if name not in self.online_names:
            raise ValueError(f"CM checkpoints cannot access {name}")
        if not sequences:
            return np.empty(0, dtype=float)
        existing = name in self.oracle_fns
        oracle = self.oracle_fns.get(name)
        if oracle is None:
            oracle = load_oracle(self.root, name, device=self.device if name in (
                "function_2026_esm", "function_2026_t5", "cm_poet") else "cpu",
                family=self.family, poet_config=self.contract["oracles"]["poet"],
                novelty_config=self.contract["oracles"].get("novelty"))
        try:
            if name == "cm_poet":
                indices = [i for i, sequence in enumerate(sequences) if sequence not in oracle.context]
                values = np.full(len(sequences), np.nan)
                if indices:
                    values[indices] = journaled_oracle_call(oracle,
                        [sequences[i] for i in indices], name=name, phase="checkpoint_evaluation",
                        journal_path=self.current_directory / "scoring_calls.jsonl")
                return values
            return np.asarray(journaled_oracle_call(oracle, sequences, name=name,
                phase="checkpoint_evaluation", journal_path=self.current_directory / "scoring_calls.jsonl"))
        finally:
            if not existing:
                del oracle
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def evaluate(self, model, step: int) -> dict:
        directory = self.output / "checkpoints" / f"step_{step}"
        directory.mkdir(parents=True, exist_ok=True)
        self.current_directory = directory
        settings, delivery = self.contract["framework"]["checkpoint_evaluation"], self.contract["delivery"]
        sequences = sample_policy(model, self.tokenizer, count=settings["sample_count"], seed=self.seed)
        save_table(directory / "sequences.csv", pd.DataFrame({"sequence": sequences}))
        calibrators = tuple((lambda values, c=self.calibration[name]: values * c["slope"] + c["intercept"])
                            for name in ("function_2026_esm", "function_2026_t5"))
        kwargs = {}
        requirement = delivery.get("family_requirement")
        if requirement:
            kwargs = {"family_requirement": requirement,
                      "family_score": lambda ss: self.score(requirement["oracle"], ss)}
        table, summary = select_library(sequences, self.family,
            lambda ss: self.score("function_2026_esm", ss),
            (lambda ss: self.score("function_2026_t5", ss)) if "function_2026_t5" in self.online_names
            else (lambda ss: np.full(len(ss), np.nan)),
            count=delivery["count"], min_distance=delivery["min_natural_distance"],
            score_names=("function_2026_esm", "function_2026_t5"), calibrators=calibrators,
            ranking=delivery["selection_aggregation"], **kwargs)
        valid = table.loc[table.valid, "sequence"].tolist()
        for name in self.online_names:
            if name in {"function_2026_esm", "function_2026_t5"}:
                continue
            values = (table.loc[table.valid, "family_score"].to_numpy()
                      if requirement and requirement["oracle"] == name else self.score(name, valid))
            table.loc[table.valid, name] = values
        save_table(directory / "candidates.csv", table)
        write_json(directory / "delivery_summary.json", summary)
        write_json(directory / "poet_summary.json", {
            "scored_count": int(np.isfinite(table.cm_poet).sum()) if "cm_poet" in table else 0,
            "phase": "checkpoint_evaluation", "measured": "cm_poet" in self.online_names})
        measurements = candidate_measurements(table, summary, count=delivery["count"],
            min_distance=delivery["min_natural_distance"], score_names=self.online_names)
        write_json(directory / "measurements.json", measurements)
        return {"measurements": measurements, "library_count": len(table),
                "library": path_reference(directory / "candidates.csv"),
                "raw_artifacts": {"measurements": path_reference(directory / "measurements.json")}}

    def write_checkpoint(self, model, step: int, *, evaluation: dict) -> dict:
        directory = self.output / "checkpoints" / f"step_{step}" / "model"
        model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        return {"step": step, "checkpoint": path_reference(directory)}

    def callback(self, success_spec) -> CheckpointSelectionCallback:
        return CheckpointSelectionCallback(evaluator=self.evaluate, checkpoint_writer=self.write_checkpoint,
            success_spec=success_spec,
            scheduled_steps=self.contract["framework"]["checkpoint_evaluation"]["steps"],
            stop_when_qualified=False, unknown_policy="continue",
            online_sources=self.online_names, forbidden_sources=self.audit_names,
            evidence_policy=EvidenceEligibility(**self.contract["framework"].get("terminal_eligibility", {})),
            require_heldout=protocol_condition(self.contract) == "self_validating",
            required_held_out_oracles=self.contract["oracles"]["source_roles"]["primary"].get("required_sentinels", ()))

    def publish_final_evaluation(self, trajectory: dict) -> None:
        """Expose the last scheduled pool; shared selection retains every checkpoint.

        Agent feedback is filtered by the engine. Do not replace its input pool
        with one chosen using hidden T5 evidence when sentinel feedback is hidden.
        """
        records = trajectory["checkpoint_selection"]["records"]
        last_step = self.contract["framework"]["checkpoint_evaluation"]["steps"][-1]
        if not records or records[-1]["actual_step"] != last_step:
            raise RuntimeError("CM checkpoint evaluation did not complete its final scheduled measurement")
        directory = self.output / "checkpoints" / f"step_{last_step}"
        for name in ("candidates.csv", "delivery_summary.json", "poet_summary.json"):
            shutil.copyfile(directory / name, self.output / name)
        table = pd.read_csv(directory / "candidates.csv")
        # Checkpoint evaluation replaces final evaluation, so the shared trainer's
        # auxiliary telemetry never runs. Compute the sequence-complexity statistics
        # (nearest_training_hamming, run lengths, k-mer reuse) task-side instead of
        # changing the general trainer's gating. k=3 rather than the k=8 default: over
        # a 20-letter alphabet and 80-96 residues, 8-mers are almost all unique and
        # carry no signal.
        complexity = sequence_complexity_metrics(
            table.sequence.tolist(), training_set=self.family.training, k=3, at_run_min=5)
        trajectory.update(sequence_complexity=complexity,
            sequences=table.sequence.tolist(), conditioning=["CM"] * len(table),
            oracle_outputs={"function_2026_esm": [float(v) if np.isfinite(v) else None
                                                 for v in table.function_2026_esm]},
            candidate_measurements=copy.deepcopy(records[-1]["measurements"]),
            checkpoint_measurements=[{"step": r["actual_step"], "measurements": copy.deepcopy(r["measurements"])}
                                     for r in records],
            evaluation_performed=False,
            evaluation_timing="during training at declared optimizer steps; optimizer continues unchanged",
            evaluation_checkpoint_step=last_step)
