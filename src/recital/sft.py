"""Typed supervised-fine-tuning (SFT) recipe for sequence policies.

:class:`SFTRecipe` records the data, conditioning, and checkpoint provenance
needed to reproduce an SFT policy. The shipped tasks have qualified
checkpoints and task-local loaders; the recipe describes them and never starts
training.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping


SFTStatus = Literal["reuse_existing", "train", "unknown"]


@dataclass(frozen=True)
class SFTRecipe:
    """Reproducible metadata for the policy SFT stage.

    ``checkpoint`` and ``data_source`` intentionally remain strings rather than
    resolved ``Path`` objects so a recipe can be serialized into a run plan and
    replayed from a different checkout.  Consumers can call
    :meth:`checkpoint_path` with the repository root when they need a local
    path.
    """

    name: str
    status: SFTStatus = "unknown"
    checkpoint: str | None = None
    data_source: str | None = None
    sequence_column: str = "sequence"
    condition_column: str | None = None
    split_column: str | None = "split"
    train_split: str = "train"
    conditioning_prompts: tuple[str, ...] = ()
    alphabet: str = "ACGT"
    sequence_length: int = 0
    init: str = "completion_only"
    epochs: int | None = None
    seed: int | None = None
    output_dir: str | None = None
    loader: str | None = None
    trainer: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("SFT recipe name must be non-empty")
        if self.status not in {"reuse_existing", "train", "unknown"}:
            raise ValueError(f"unsupported SFT status: {self.status!r}")
        if self.status == "reuse_existing" and not self.checkpoint:
            raise ValueError("reuse_existing SFT recipes require a checkpoint")
        if self.status == "train" and not self.data_source:
            raise ValueError("train SFT recipes require a data_source")
        if self.sequence_length < 0:
            raise ValueError("SFT sequence_length cannot be negative")
        if not self.alphabet:
            raise ValueError("SFT alphabet must be non-empty")
        if self.epochs is not None and self.epochs < 1:
            raise ValueError("SFT epochs must be positive when provided")
        prompts = tuple(str(prompt) for prompt in self.conditioning_prompts)
        if len(set(prompts)) != len(prompts):
            raise ValueError("SFT conditioning_prompts must be unique")
        object.__setattr__(self, "conditioning_prompts", prompts)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_contract(
        cls,
        contract: Mapping[str, Any],
        *,
        name: str | None = None,
        sequence_length: int = 0,
        sequence_column: str = "sequence",
        condition_column: str | None = None,
        loader: str | None = None,
        trainer: str | None = None,
        epochs: int | None = None,
        status: SFTStatus | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "SFTRecipe":
        """Create a recipe from a task contract without importing model code."""
        task = contract.get("task") or {}
        data = contract.get("data") or {}
        base_models = contract.get("base_models") or []
        base_model = base_models[0] if base_models else {}
        checkpoint = str(base_model.get("model") or "") or None
        prompts = tuple(str(item) for item in data.get("conditioning_prompts") or ())
        contract_length = data.get("max_completion_length") or sequence_length
        merged_metadata = dict(metadata or {})
        if base_model.get("id"):
            merged_metadata.setdefault("base_model_id", str(base_model["id"]))
        if base_model.get("notes"):
            merged_metadata.setdefault("base_model_notes", str(base_model["notes"]))
        return cls(
            name=name or f"{task.get('name', 'sequence')}_sft",
            status=status or ("reuse_existing" if checkpoint else "unknown"),
            checkpoint=checkpoint,
            data_source=(
                str(data.get("unified_parquet") or data.get("source") or "") or None
            ),
            sequence_column=sequence_column,
            condition_column=condition_column,
            split_column=str(data.get("split_column") or "split"),
            train_split="train",
            conditioning_prompts=prompts,
            alphabet=str(data.get("token_alphabet") or "ACGT"),
            sequence_length=int(contract_length or 0),
            init=str(base_model.get("init") or "completion_only"),
            epochs=epochs,
            loader=loader,
            trainer=trainer,
            metadata=merged_metadata,
        )

    def checkpoint_path(self, repo_root: str | Path = ".") -> Path | None:
        """Return the checkpoint path resolved against ``repo_root``."""
        if not self.checkpoint:
            return None
        path = Path(self.checkpoint).expanduser()
        if not path.is_absolute():
            path = Path(repo_root).expanduser().resolve() / path
        return path

    def data_path(self, repo_root: str | Path = ".") -> Path | None:
        """Return the data path resolved against ``repo_root`` when declared."""
        if not self.data_source:
            return None
        path = Path(self.data_source).expanduser()
        if not path.is_absolute():
            path = Path(repo_root).expanduser().resolve() / path
        return path

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly recipe representation."""
        payload = asdict(self)
        payload["conditioning_prompts"] = list(self.conditioning_prompts)
        payload["metadata"] = dict(self.metadata)
        return payload


__all__ = [
    "SFTRecipe",
    "SFTStatus",
]
