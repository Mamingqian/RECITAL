"""Explicit training stages on top of HF's existing checkpoint machinery.

An action's stop step and its learning-rate horizon are different quantities.
HF owns model/optimizer/scaler/RNG persistence; this module binds those files to
the accepted action and saves GRPO's otherwise transient rollout state.
"""
from __future__ import annotations

import copy
import json
import os
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


ACTION_STATE = "training_action_state.json"
ROLLOUT_STATE = "training_action_runtime.pt"


def _json_write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def public_spec(spec: dict) -> dict:
    return json.loads(json.dumps({key: value for key, value in spec.items()
                                 if not key.startswith("_framework_")}, allow_nan=False))


def load_action_checkpoint(path: str | Path) -> dict:
    path = Path(path)
    record = json.loads((path / ACTION_STATE).read_text())
    state = json.loads((path / "trainer_state.json").read_text())
    if record.get("format") != "training_action_v1" or record.get("hf_step") != state.get("global_step"):
        raise ValueError("training-action checkpoint and HF state disagree")
    required = ["optimizer.pt", "scheduler.pt", "rng_state.pth", ROLLOUT_STATE]
    if record.get("scaler_available"):
        required.append("scaler.pt")
    if any(not (path / name).is_file() for name in required):
        raise ValueError("checkpoint does not contain complete training state")
    if not list(path.glob("*.safetensors")):
        raise ValueError("checkpoint has no saved policy weights")
    return record


def latest_action_checkpoint(directory: str | Path) -> str | None:
    """An interrupted HF save is not a complete action checkpoint."""
    candidates = [path for path in Path(directory).glob("checkpoint-*")
                  if path.name.rsplit("-", 1)[-1].isdigit()]
    for path in sorted(candidates, key=lambda value: int(value.name.rsplit("-", 1)[-1]), reverse=True):
        try:
            load_action_checkpoint(path)
        except (OSError, ValueError, KeyError):
            continue
        return str(path)
    return None


def _component_definitions(spec: dict) -> dict:
    """State compatibility follows component inputs as well as its name."""
    upstream = {key: spec.get(key) for key in
                ("enabled_oracles", "target_objectives", "oracle_utilities", "reward_input")}
    definitions = {}
    for name, value in (spec.get("oracle_utilities") or {}).items():
        definitions[f"utility/{name}"] = {"component": value, "inputs": upstream}
    for index, value in enumerate(spec.get("transforms") or []):
        definitions[f"transform/{index}"] = {"component": value, "inputs": copy.deepcopy(upstream)}
        upstream[f"transform/{index}"] = value
    for name in ("aggregator", "shape"):
        definitions[name] = {"component": spec.get(name), "inputs": copy.deepcopy(upstream)}
        upstream[name] = spec.get(name)
    for field, label in (("penalties", "penalty"), ("gates", "gate")):
        for index, value in enumerate(spec.get(field) or []):
            definitions[f"{label}/{index}"] = {"component": value, "inputs": copy.deepcopy(upstream)}
        upstream[field] = spec.get(field)
    return definitions


def restore_reward_state(reward, checkpoint: Path | None, old_spec: dict | None,
                         spec: dict, policy: str) -> dict:
    fresh = reward.state_dict() if callable(getattr(reward, "state_dict", None)) else {}
    if policy == "reset_all" or checkpoint is None:
        return {"retained": [], "reset": sorted(fresh)}
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    callback = state.get("stateful_callbacks", {}).get("RewardStateCallback")
    saved = (callback or {}).get("attributes", {}).get("reward_state")
    if saved is None:
        if fresh:
            raise ValueError("full continuation requires saved reward normalization state")
        saved = {}
    old_definitions, definitions = _component_definitions(old_spec or {}), _component_definitions(spec)
    retained = [key for key in fresh if key in saved and
                old_definitions.get(key) == definitions.get(key)]
    if policy == "inherit_all" and (set(retained) != set(fresh) or set(saved) != set(fresh)):
        raise ValueError("reward state is incompatible with inherit_all; select reset_all or inherit_compatible")
    restored = {key: copy.deepcopy(saved[key] if key in retained else value)
                for key, value in fresh.items()}
    if callable(getattr(reward, "load_state_dict", None)):
        reward.load_state_dict(restored)
    return {"retained": sorted(retained), "reset": sorted(set(fresh) - set(retained))}


class TrainingAction:
    """Materialized state transition; independent of the agent and task adapter."""

    def __init__(self, request: dict, *, spec: dict, output_dir: Path,
                 seed: int, data_signature: dict):
        if not isinstance(request, dict):
            raise TypeError("training_action must be a mapping")
        self.request = copy.deepcopy(request)
        self.mode = request.get("start_mode", "initial_restart")
        if self.mode not in {"initial_restart", "weights_new_stage", "full_state_continuation", "crash_recovery"}:
            raise ValueError("unsupported training action start_mode")
        self.action_id = request.get("action_id")
        if not isinstance(self.action_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.action_id):
            raise ValueError("training action_id must be a nonempty filename-safe identifier")
        self.spec = public_spec(spec)
        self.seed = seed
        self.data_signature = json.loads(json.dumps(data_signature, allow_nan=False))
        self.parent_path = Path(request["start_checkpoint"]).resolve() if request.get("start_checkpoint") else None
        self.parent = None
        if self.mode in {"full_state_continuation", "crash_recovery"}:
            if self.parent_path is None:
                raise ValueError("state continuation requires start_checkpoint")
            self.parent = load_action_checkpoint(self.parent_path)
        elif self.mode == "weights_new_stage":
            if self.parent_path is None or not self.parent_path.is_dir():
                raise ValueError("weights_new_stage requires an existing checkpoint")
            if (self.parent_path / ACTION_STATE).exists():
                self.parent = load_action_checkpoint(self.parent_path)
        elif self.parent_path is not None:
            raise ValueError("initial_restart uses initial_reference, not start_checkpoint")
        self.out_dir = Path(output_dir) / "actions" / self.action_id
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.mode != "crash_recovery" and any(self.out_dir.glob(f"checkpoint-*/{ACTION_STATE}")):
            raise ValueError("this action already has committed checkpoints; restore it using crash_recovery")
        self.checkpoints: list[dict] = []
        self.reward_transition: dict = {}
        self.runtime_restored = False
        self.lr_records: list[dict] = []
        self.data_cursor: dict = {}
        if self.mode == "crash_recovery":
            parent = self.parent
            if parent["action"]["action_id"] != self.action_id or parent["spec"] != self.spec:
                raise ValueError("crash recovery must restore the same action and recipe")
            if parent["data_signature"] != self.data_signature:
                raise ValueError("crash recovery cannot change training data or sampling")
            original = parent["action"]
            for key, value in request.items():
                if key not in {"start_mode", "start_checkpoint"} and key in original and value != original[key]:
                    raise ValueError(f"crash recovery cannot change {key}")
            self.request = copy.deepcopy(original)
            self.seed = parent["seed"]
            self.start_step = parent["action_start_step"]
            self.end_step = parent["action_end_step"]
            self.additional_updates = parent["action"]["additional_updates"]
            self.lineage_start = parent["lineage_start"]
            self.schedule = copy.deepcopy(parent["schedule"])
            self.optimizer_policy = "retain"
            self.reward_policy = "inherit_all"
            self.randomness_policy = "inherit_stream"
        else:
            self.additional_updates = _integer(request.get("additional_updates"), "additional_updates", 1)
            self.start_step = (self.parent["hf_step"] if self.mode == "full_state_continuation" else 0)
            self.end_step = self.start_step + self.additional_updates
            self.lineage_start = (self.parent["lineage_step"] if self.parent is not None else 0)
            self.optimizer_policy = request.get("optimizer_policy", "retain" if self.mode == "full_state_continuation" else "reset")
            self.reward_policy = request.get("reward_state_policy", "inherit_compatible" if self.mode == "full_state_continuation" else "reset_all")
            self.randomness_policy = request.get("randomness_policy", "inherit_stream" if self.mode == "full_state_continuation" else "new_stream")
            policy = request.get("schedule_policy") or {}
            kind = policy.get("kind", "new_segment")
            if kind == "continue_segment":
                if self.mode != "full_state_continuation":
                    raise ValueError("continue_segment requires full_state_continuation")
                self.schedule = copy.deepcopy(self.parent["schedule"])
                old_lr = (self.parent["spec"].get("hyperparameters") or {}).get("lr", 1e-4)
                if (spec.get("hyperparameters") or {}).get("lr", 1e-4) != old_lr:
                    raise ValueError("changing LR requires a new schedule segment")
                for key in ("horizon", "start_lr", "type", "warmup_updates"):
                    if key in policy and policy[key] != self.schedule[key]:
                        raise ValueError(f"continue_segment cannot change schedule {key}; append a new segment")
                if self.end_step > self.schedule["start_step"] + self.schedule["horizon"]:
                    raise ValueError("action exceeds remaining schedule segment; append a new segment")
            elif kind == "new_segment":
                horizon = _integer(policy.get("horizon", self.additional_updates), "schedule horizon", 1)
                if horizon < self.additional_updates:
                    raise ValueError("schedule horizon cannot be shorter than this action")
                self.schedule = {"segment_id": self.action_id, "start_step": self.start_step,
                                 "horizon": horizon, "type": policy.get("type", "linear"),
                                 "start_lr": float(policy.get("start_lr", (spec.get("hyperparameters") or {}).get("lr", 1e-4))),
                                 "warmup_updates": _integer(policy.get("warmup_updates", 0), "warmup_updates")}
                if not 0 < self.schedule["start_lr"] < float("inf"):
                    raise ValueError("new schedule start_lr must be finite and positive")
                if self.schedule["warmup_updates"] >= horizon:
                    raise ValueError("warmup_updates must be shorter than the horizon")
            else:
                raise ValueError("schedule_policy.kind must be continue_segment or new_segment")
        if self.optimizer_policy not in {"retain", "reset"}:
            raise ValueError("optimizer_policy must be retain or reset")
        if self.optimizer_policy == "retain" and self.mode not in {"full_state_continuation", "crash_recovery"}:
            raise ValueError("optimizer retention requires full_state_continuation")
        if self.reward_policy not in {"reset_all", "inherit_all", "inherit_compatible"}:
            raise ValueError("unsupported reward_state_policy")
        if self.randomness_policy not in {"inherit_stream", "new_stream"}:
            raise ValueError("unsupported randomness_policy")
        if self.randomness_policy == "inherit_stream" and self.parent is None:
            raise ValueError("inherit_stream requires a complete historical state")
        self.kl_reference = str(Path(self.request["kl_reference"]).resolve())
        self.initial_reference = str(Path(self.request["initial_reference"]).resolve())
        self.local_save_steps = set(self.request.get("checkpoint_steps") or []) | {self.additional_updates}
        for step in self.local_save_steps:
            _integer(step, "checkpoint_steps", 1)
            if step > self.additional_updates:
                raise ValueError("checkpoint_steps are local to this action")
        self.resume_path = (self.parent_path if self.mode in {"full_state_continuation", "crash_recovery"} else None)
        self.resume_step = self.parent["hf_step"] if self.resume_path else 0
        old_reward = {key: value for key, value in (self.parent or {}).get("spec", {}).items() if key != "hyperparameters"}
        new_reward = {key: value for key, value in self.spec.items() if key != "hyperparameters"}
        self.data_compatible = bool(self.resume_path and self.parent
                                   and self.parent["data_signature"] == self.data_signature)
        self.cache_compatible = bool(self.data_compatible and old_reward == new_reward
                                     and self.parent["action"]["kl_reference"] == self.kl_reference
                                     and self.randomness_policy == "inherit_stream"
                                     )
        if self.mode == "crash_recovery":
            self.cache_compatible = True
            self.data_compatible = True
        self.request.update(action_id=self.action_id, additional_updates=self.additional_updates,
                            kl_reference=self.kl_reference, initial_reference=self.initial_reference)
        if self.parent and self.randomness_policy == "inherit_stream":
            self.seed = self.parent["seed"]

    def restore_reward(self, reward) -> None:
        self.reward_transition = restore_reward_state(
            reward, self.parent_path, (self.parent or {}).get("spec"), self.spec, self.reward_policy)
        if self.reward_transition["reset"] and self.mode != "crash_recovery":
            self.cache_compatible = False

    def receipt(self, step: int, *, execution_cost=None) -> dict:
        checkpoint = self.out_dir / f"checkpoint-{step}"
        # A process may die after saving an intermediate node, before the outer
        # engine registers the completed action. Rebuild this action's complete
        # node list from durable manifests, including a zero-work recovery.
        roots = {self.out_dir}
        if self.mode == "crash_recovery" and self.parent_path is not None:
            roots.add(self.parent_path.parent)
        nodes = {}
        for root in sorted(roots):
            for path in root.glob("checkpoint-*"):
                try:
                    record = load_action_checkpoint(path)
                except (OSError, ValueError, KeyError):
                    continue
                if (record["action"]["action_id"] != self.action_id or record["spec"] != self.spec
                        or not self.start_step < record["hf_step"] <= step):
                    continue
                nodes[record["hf_step"]] = {
                    "path": str(path), "step": record["hf_step"],
                    "local_step": record["hf_step"] - self.start_step,
                    "lineage_step": record["lineage_step"], "full_state_available": True,
                    "weights_available": True, "schedule": copy.deepcopy(record["schedule"]),
                }
        self.checkpoints = [nodes[key] for key in sorted(nodes)]
        return {"action_id": self.action_id, "start_mode": self.mode,
                "start_checkpoint": str(self.parent_path) if self.parent_path else None,
                "action_start_step": self.start_step, "action_end_step": step,
                "additional_updates_completed": step - self.start_step,
                "updates_this_invocation": step - self.resume_step,
                "lineage_step": self.lineage_start + step - self.start_step,
                "end_checkpoint": str(checkpoint), "checkpoints": copy.deepcopy(self.checkpoints),
                "schedule": copy.deepcopy(self.schedule), "effective_lr": copy.deepcopy(self.lr_records),
                "kl_reference": self.kl_reference, "initial_reference": self.initial_reference,
                "state_policy": {"optimizer": self.optimizer_policy, "reward": self.reward_transition,
                                 "randomness": self.randomness_policy,
                                 "data_cursor": ("retained" if self.data_compatible else
                                                 "epoch_boundary_restart" if self.resume_path else "stage_start"),
                                 "rollout_cache": "retained" if self.cache_compatible else "reset"},
                "data_cursor_position": copy.deepcopy(self.data_cursor),
                "actual_updates_spent": (execution_cost or {}).get("measurements", {}).get("optimizer_steps", {}).get("value", step - self.resume_step),
                "execution_cost": copy.deepcopy(execution_cost)}


def _plain_state(value):
    """Torch weights-only loading supports ordinary containers and tensors."""
    if isinstance(value, dict):
        return {key: _plain_state(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, deque)):
        return [_plain_state(item) for item in value]
    if hasattr(value, "detach"):
        return value.detach().cpu()
    return value


def action_trainer_class(base):
    """Use HF's state loading, with explicit component and schedule overrides."""
    import torch
    from transformers import TrainerCallback, get_scheduler, set_seed
    from trl.trainer.utils import RepeatSampler

    class EpochRepeatSampler(RepeatSampler):
        # An epoch-addressable permutation survives HF's skip-to-global-step
        # recovery without replaying all earlier epochs to advance a generator.
        def set_epoch(self, epoch):
            # Accelerate's skip_first_batches wraps the same sampler in a new
            # DataLoaderShard whose iteration counter starts at zero. Preserve
            # HF's actual resume epoch through that first wrapper iteration.
            self.epoch = max(int(epoch), getattr(self, "minimum_epoch", 0))

        def __iter__(self):
            if self.shuffle:
                self.generator.manual_seed(int(self.seed or 0) + getattr(self, "epoch", 0))
            return super().__iter__()

    class SaveActionSteps(TrainerCallback):
        def __init__(self, action, trainer):
            self.action = action
            self.trainer = trainer
            self.applied_lr = None

        def on_train_begin(self, args, state, control, train_dataloader=None, **kwargs):
            action = self.action
            batches = len(train_dataloader)
            updates = max((batches + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps, 1)
            # HF preserves the global training epoch even when ignore_data_skip
            # restarts at that epoch's first batch. A new RNG seed alone does
            # not move the data cursor; a new weights stage starts at epoch 0.
            action.data_cursor = {
                "batches_per_epoch": batches, "updates_per_epoch": updates,
                "resume_epoch_index": state.global_step // updates if action.resume_path else 0,
                "resume_batches_skipped": ((state.global_step % updates) * args.gradient_accumulation_steps
                                           if action.resume_path and not args.ignore_data_skip else 0),
                "sampling_seed": action.seed, "final_hf_epoch": state.epoch,
            }
            self.trainer.action_sampler.minimum_epoch = action.data_cursor["resume_epoch_index"]
            if action.randomness_policy == "inherit_stream" and action.resume_path is None:
                self.trainer._load_rng_state(str(action.parent_path))
            return control

        def on_step_begin(self, args, state, control, optimizer=None, **kwargs):
            self.applied_lr = [group["lr"] for group in optimizer.param_groups]
            return control

        def on_step_end(self, args, state, control, optimizer=None, **kwargs):
            action = self.action
            action.data_cursor["final_hf_epoch"] = state.epoch
            action.lr_records.append({"step": state.global_step, "applied_lr": self.applied_lr,
                                      "lr": [group["lr"] for group in optimizer.param_groups]})
            if state.global_step - action.start_step in action.local_save_steps:
                control.should_save = True
            return control

    class ActionGRPOTrainer(base):
        def __init__(self, *args, action, **kwargs):
            self.training_action = action
            self._restored_rollout = False
            super().__init__(*args, **kwargs)
            self.add_callback(SaveActionSteps(action, self))

        def _get_train_sampler(self, dataset=None):
            sampler = super()._get_train_sampler(dataset)
            if len(sampler) == 0:
                raise ValueError("training data allocation is smaller than one generation batch")
            self.action_sampler = EpochRepeatSampler(
                sampler.data_source, sampler.mini_repeat_count, sampler.batch_size,
                sampler.repeat_count, sampler.shuffle, sampler.seed)
            return self.action_sampler

        def create_scheduler(self, num_training_steps, optimizer=None):
            if self.lr_scheduler is None:
                schedule = self.training_action.schedule
                self.lr_scheduler = get_scheduler(schedule["type"], optimizer=optimizer or self.optimizer,
                    num_warmup_steps=schedule["warmup_updates"], num_training_steps=schedule["horizon"])
                self._created_lr_scheduler = True
            return self.lr_scheduler

        def _load_optimizer_and_scheduler(self, checkpoint):
            if checkpoint is None:
                return
            action = self.training_action
            fresh_groups = [{key: value for key, value in group.items() if key != "params"}
                            for group in self.optimizer.param_groups]
            fresh_scheduler = copy.deepcopy(self.lr_scheduler.state_dict())
            if action.optimizer_policy == "retain":
                super()._load_optimizer_and_scheduler(checkpoint)
            else:
                self.lr_scheduler.load_state_dict(torch.load(Path(checkpoint) / "scheduler.pt", weights_only=True))
            same_segment = (action.mode == "crash_recovery" or
                            action.schedule["segment_id"] == action.parent["schedule"]["segment_id"])
            if not same_segment:
                self.lr_scheduler.load_state_dict(fresh_scheduler)
            if action.mode != "crash_recovery":
                for group, fresh in zip(self.optimizer.param_groups, fresh_groups, strict=True):
                    group.update(fresh)
                for group, lr in zip(self.optimizer.param_groups, self.lr_scheduler.get_last_lr(), strict=True):
                    group["lr"] = lr

        def _load_rng_state(self, checkpoint):
            action = self.training_action
            if action.randomness_policy == "inherit_stream":
                super()._load_rng_state(checkpoint)
            else:
                set_seed(action.seed)

        def _load_scaler(self, checkpoint):
            if self.training_action.optimizer_policy == "retain":
                super()._load_scaler(checkpoint)

        def _prepare_inputs(self, inputs):
            action = self.training_action
            if not self._restored_rollout:
                if action.resume_path and action.cache_compatible:
                    payload = torch.load(action.resume_path / ROLLOUT_STATE, map_location=self.args.device, weights_only=True)
                    self._step = payload["grpo_step"]
                    self._buffered_inputs = payload["buffered_inputs"]
                    self._total_train_tokens = payload.get("total_train_tokens", 0)
                    self._metrics = {phase: defaultdict(list, values) for phase, values in payload.get("metrics", {}).items()}
                    if not self._metrics:
                        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
                    for name, values in payload.get("logs", {}).items():
                        if name == "rewards":
                            self._logs[name].update({key: deque(items, maxlen=self.args.generation_batch_size)
                                                     for key, items in values.items()})
                        else:
                            self._logs[name] = deque(values, maxlen=self.args.generation_batch_size)
                self._restored_rollout = True
            return super()._prepare_inputs(inputs)

        def _save_checkpoint(self, model, trial):
            super()._save_checkpoint(model, trial)
            action = self.training_action
            directory = Path(self._get_output_dir(trial=trial)) / f"checkpoint-{self.state.global_step}"
            payload = {"grpo_step": self._step, "buffered_inputs": self._buffered_inputs,
                       "total_train_tokens": getattr(self, "_total_train_tokens", 0),
                       "metrics": self._metrics, "logs": self._logs}
            temporary = directory / (ROLLOUT_STATE + ".tmp")
            torch.save(_plain_state(payload), temporary)
            os.replace(temporary, directory / ROLLOUT_STATE)
            record = {"format": "training_action_v1", "action": action.request,
                      "hf_step": self.state.global_step, "action_start_step": action.start_step,
                      "action_end_step": action.end_step, "lineage_start": action.lineage_start,
                      "lineage_step": action.lineage_start + self.state.global_step - action.start_step,
                      "schedule": action.schedule, "spec": action.spec, "seed": action.seed,
                      "data_signature": action.data_signature,
                      "data_cursor_position": action.data_cursor,
                      "scaler_available": (directory / "scaler.pt").exists()}
            _json_write(directory / ACTION_STATE, record)
            item = {"path": str(directory), "step": self.state.global_step,
                    "local_step": self.state.global_step - action.start_step,
                    "lineage_step": record["lineage_step"], "full_state_available": True,
                    "weights_available": True, "schedule": copy.deepcopy(action.schedule)}
            action.checkpoints = [entry for entry in action.checkpoints if entry["step"] != item["step"]] + [item]

    return ActionGRPOTrainer
