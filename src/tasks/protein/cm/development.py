"""Prior-scale calibration and recipe training on the shared CM trainer."""
from __future__ import annotations

import gc
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import ALPHABET
from .family import NATURAL_FAMILY_REFERENCE, FamilyReference, PottsModel
from .predictors import RegressionOracle
from .qualification import write_json


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class CalibratedPrior:
    """Comparable utility scales from the same unlabeled SFT reference library."""

    def __init__(self, models: dict, mode: str, calibration: dict):
        self.models, self.mode, self.calibration = models, mode, calibration

    def __call__(self, sequences: list[str]) -> np.ndarray:
        normalized = {}
        for name, model in self.models.items():
            values = model(sequences)
            c = self.calibration["sources"][name]
            normalized[name] = (values - c["mean"]) / c["std"]
        if self.mode in normalized:
            value = normalized[self.mode]
        elif self.mode == "mean":
            value = (normalized["bm"] + normalized["me"]) / 2
        elif self.mode == "max":
            value = np.maximum(normalized["bm"], normalized["me"])
        else:
            raise ValueError(f"Unknown prior mode {self.mode}")
        c = self.calibration["composites"][self.mode]
        return (value - c["mean"]) / c["std"] * self.calibration["function_std"]


def prepare_prior_scale(root: Path, pilot_root: Path, *, device: str) -> dict:
    output = root / "development"
    output.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(pilot_root / "baselines/sft_candidates.csv")
    sequences = [s for s in table.sequence if 80 <= len(s) <= 96 and set(s) <= set(ALPHABET)]
    family = FamilyReference(root)
    function = RegressionOracle(root, "function_2026_esm", device=device)
    target = function(sequences)
    del function
    release_cuda()
    values, result = {}, {"sources": {}, "composites": {}, "n": len(sequences),
                          "function_std": max(float(np.std(target)), 1e-6),
                          "reference": "common pilot SFT candidates, no target labels used"}
    for short, name in (("bm", "bmDCA"), ("me", "meDCA")):
        scores = PottsModel(root, name, family)(sequences)
        mean, std = float(np.mean(scores)), max(float(np.std(scores)), 1e-6)
        result["sources"][short] = {"mean": mean, "std": std}
        values[short] = (scores - mean) / std
    values["mean"] = (values["bm"] + values["me"]) / 2
    values["max"] = np.maximum(values["bm"], values["me"])
    result["bm_me_pearson_on_sft"] = float(np.corrcoef(values["bm"], values["me"])[0, 1])
    for name, scores in values.items():
        result["composites"][name] = {"mean": float(scores.mean()), "std": max(float(scores.std()), 1e-6)}
    write_json(output / "prior_scale.json", result)
    return result


def train_recipe(root: Path, *, config: dict, spec: dict, seed: int, device: str,
                 output: Path, family, oracle_fns: dict, resume: bool = True,
                 evaluation_seed: int | None = None, primitive_resolver=None,
                 started: float | None = None) -> dict:
    """Train one CM recipe with the shared CM trainer.

    Default runs score online held-out models after this process exits. An
    explicit checkpoint contract mounts task measurements during training;
    those scores remain outside the RL reward.
    """
    from recital.spec.composer import ComposedRewardFn, RewardSpec
    from recital.trainer import TRLDAPOTrainer, TrainContext
    from tasks.protein.progen import ProGen3CausalLM, ProGenSequenceCodec
    from .runner import load_tokenizer, _write_json

    started = time.monotonic() if started is None else started
    output.mkdir(parents=True, exist_ok=True)
    training_spec = copy.deepcopy(spec)
    training_spec["held_out_oracles"] = []
    checkpoint = Path(config.get("sft_checkpoint") or root / "sft/model")
    tokenizer = load_tokenizer(checkpoint)
    codec = ProGenSequenceCodec(tokenizer)
    if not (output / "trajectory.json").exists():
        model = ProGen3CausalLM.from_pretrained(checkpoint)
        model.generation_config.forced_eos_token_id = tokenizer.eos_token_id
        model.generation_config.do_sample = True
        model.generation_config.top_k = 0
        trainer = TRLDAPOTrainer(max_completion_length=97, output_dir=output / "training",
                                 bf16=False, use_cpu=device == "cpu")
        context = TrainContext(policy=model, train_data=None, conditioning_prompts=["<bos>1"],
            token_alphabet=codec.representation.alphabet, oracle_fns=oracle_fns, extras={
                "tokenizer": tokenizer, "sequence_codec": codec, "seed": seed,
                "evaluation_seed": seed + 100000 if evaluation_seed is None else evaluation_seed,
                "model_architecture": ProGen3CausalLM, "reference_model_architecture": ProGen3CausalLM,
                "min_new_tokens": 80, "min_sequence_length": 80, "max_sequence_length": 96,
                "generation_kwargs": {"forced_eos_token_id": tokenizer.eos_token_id, "top_k": 0},
                "eval_samples_per_prompt": config["evaluation_candidates"],
                "base_library_samples_per_prompt": 32, "sampling_batch_size": 32,
                "sampling_add_special_tokens": False, "sampling_top_p": 1., "sampling_temperature": 1.,
                "training_set": family.training,
                "checkpoint_save_steps": config.get("checkpoint_save_steps", 64),
                "resume_from_checkpoint": resume,
                "primitive_resolver": primitive_resolver,
            })
        if config.get("reference_library"):
            context.extras["base_policy_sequences"] = pd.read_csv(config["reference_library"]).sequence.tolist()
        checkpoint_evaluator = None
        if config.get("checkpoint_contract"):
            from .checkpoints import CMCheckpointEvaluator

            checkpoint_evaluator = CMCheckpointEvaluator(root=root, output=output,
                contract=config["checkpoint_contract"], family=family, tokenizer=tokenizer,
                oracle_fns=oracle_fns, evaluation_seed=context.extras["evaluation_seed"], device=device,
                diagnostic_oracles=config.get("diagnostic_oracles", ()))
            training_spec["held_out_oracles"] = list(spec.get("held_out_oracles", []))
            context.oracle_fns = {**oracle_fns, **{
                name: (lambda sequences, name=name: checkpoint_evaluator.score(name, sequences))
                for name in checkpoint_evaluator.online_names if name not in oracle_fns}}
            context.extras.update(
                diagnostic_oracles=[name for name in checkpoint_evaluator.online_names if name not in spec["enabled_oracles"]
                                    and name not in spec.get("held_out_oracles", [])],
                audit_oracles=list(checkpoint_evaluator.audit_names),
                checkpoint_selection_callback=checkpoint_evaluator.callback(config["checkpoint_success"]),
            )
        reward = ComposedRewardFn(RewardSpec.from_dict(spec), training_set=family.training,
                                 primitive_resolver=primitive_resolver,
                                 reference_libraries={NATURAL_FAMILY_REFERENCE: family.sequences})
        try:
            trajectory = trainer.run(spec=training_spec, reward_fn=reward, context=context,
                                     perform_final_evaluation=checkpoint_evaluator is None)
        finally:
            # Preserve failed/resumed calls as well as the successful segment.
            # A resumed global step count does not describe total spent compute.
            if trainer.last_execution_cost is not None:
                with (output / "training_calls.jsonl").open("a") as handle:
                    handle.write(json.dumps(trainer.last_execution_cost) + "\n")
        if checkpoint_evaluator is not None:
            checkpoint_evaluator.publish_final_evaluation(trajectory)
        trajectory["policy"].save_pretrained(output / "model")
        tokenizer.save_pretrained(output / "model")
        trajectory["deferred_held_out_oracles"] = list(spec.get("held_out_oracles", []))
        _write_json(output / "trajectory.json", trajectory)
        trajectory.pop("policy").to("cpu")
        del trainer, model, context
        release_cuda()
    else:
        trajectory = json.loads((output / "trajectory.json").read_text())
    if trajectory["optimizer_steps_completed"] != config["steps"]:
        raise RuntimeError("Development run did not complete its configured update budget")
    write_json(output / "training_stage.json", {"elapsed_seconds": time.monotonic() - started})
    return {"status": "trained", "output": str(output), "steps": config["steps"]}


