"""Persist mutable reward normalization with the Trainer's checkpoint state."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from transformers import TrainerCallback
from transformers.trainer_callback import ExportableState

LOG = logging.getLogger(__name__)


class RewardStateCallback(TrainerCallback, ExportableState):
    """Use HF's final trainer-state write, not a separate checkpoint sidecar.

    The runtime callable stays attached to its existing oracle/primitive
    capabilities. Only its JSON state is restored; never deserialize code or
    replace the reward callable. A checkpoint with no record for this callback
    resumes with an explicit warning when the current reward is stateful.
    """

    def __init__(self, reward: Any):
        self.reward = reward
        self.restore_status = "not_resumed"

    def state(self) -> dict[str, Any]:
        # Validate and detach nested state before Transformers writes the JSON.
        payload = json.loads(json.dumps(self.reward.state_dict(), allow_nan=False))
        attributes = {"reward_state": payload}
        composed = getattr(self.reward, "composed_reward", self.reward)
        spec = getattr(composed, "spec", None)
        if callable(getattr(spec, "to_dict", None)):
            from recital.spec.composer import reward_measurement_spec
            attributes["reward_spec"] = reward_measurement_spec(spec.to_dict())
        return {"args": {}, "attributes": attributes}

    def restore(self, checkpoint: str | Path) -> None:
        saved = json.loads((Path(checkpoint) / "trainer_state.json").read_text())
        callbacks = saved.get("stateful_callbacks", {})
        record = callbacks.get(type(self).__name__)
        if record is None:
            if self.reward.state_dict():
                LOG.warning(
                    "Legacy checkpoint %s has no reward normalization state; "
                    "continuing with fresh reward statistics. This resume is "
                    "not equivalent to uninterrupted training.", checkpoint,
                )
                self.restore_status = "legacy_missing_reward_state"
            else:
                self.restore_status = "stateless_reward"
            return
        if not isinstance(record, dict) or not isinstance(record.get("attributes"), dict) or (
            "reward_state" not in record["attributes"]
        ):
            raise ValueError("checkpoint has malformed reward state")
        self.reward.load_state_dict(record["attributes"]["reward_state"])
        self.restore_status = "restored"

    def on_train_begin(self, args, state, control, **kwargs):
        # A trainer_state.json saved without this callback has no entry for it.
        # Register it after HF loads that state so the next normal checkpoint
        # includes reward state.
        state.stateful_callbacks[type(self).__name__] = self.state()
        return control


def reward_state_callback(reward: Any) -> RewardStateCallback | None:
    save = callable(getattr(reward, "state_dict", None))
    restore = callable(getattr(reward, "load_state_dict", None))
    if save != restore:
        raise TypeError("a stateful reward must implement both state methods")
    return RewardStateCallback(reward) if save else None
