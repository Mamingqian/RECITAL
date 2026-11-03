"""Restore small callback records alongside HF optimizer and RNG checkpoints."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from transformers.trainer_callback import ExportableState

LOG = logging.getLogger(__name__)


class CheckpointStateMixin(ExportableState):
    """Persist JSON data while retaining the caller's live evaluator functions."""

    def state(self):
        payload = json.loads(json.dumps(self.state_dict(), allow_nan=False))
        return {"args": {}, "attributes": {"callback_state": payload}}

    def restore(self, checkpoint):
        saved = json.loads((Path(checkpoint) / "trainer_state.json").read_text())
        step = saved["global_step"]
        record = saved.get("stateful_callbacks", {}).get(type(self).__name__)
        if record is None:
            LOG.warning("Checkpoint %s has no %s state; earlier observations are unavailable",
                        checkpoint, type(self).__name__)
            payload = None
            self.restore_status = "legacy_missing_callback_state"
        else:
            if not isinstance(record, dict) or not isinstance(record.get("attributes"), dict) or (
                "callback_state" not in record["attributes"]
            ):
                raise ValueError(f"Malformed {type(self).__name__} checkpoint state")
            payload = record["attributes"]["callback_state"]
            self.restore_status = "restored"
        self.load_state_dict(payload, checkpoint_step=step)

    def on_train_begin(self, args, state, control, **kwargs):
        # A resumed HF state may lack this callback's entry; include it in future saves.
        state.stateful_callbacks[type(self).__name__] = self.state()
        if getattr(self, "_stop_evidence", None) is not None:
            control.should_training_stop = True
        return control
