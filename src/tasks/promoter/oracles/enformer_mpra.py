"""DNA-Diffusion's Enformer model fine-tuned on Malinois MPRA data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tasks.promoter.assets import ENFORMER_MPRA_CHECKPOINT
from tasks.promoter.score_contract import ENFORMER_SOURCE_COLUMNS, TARGET_CELLS


@dataclass
class EnformerMPRAOracle:
    """Return K562 and HepG2 activity from the released five-head model.

    The released checkpoint contains a standard Enformer backbone, attention
    pooling over its final embeddings, and a five-output Malinois MPRA head.
    Matching against the authors' released prediction table identifies output
    columns 0 and 1 as K562 and HepG2, respectively.
    """

    checkpoint_path: str | Path | None = None
    device: str = "cpu"
    batch_size: int = 32
    _modules: tuple[Any, Any, Any] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.checkpoint_path = Path(
            self.checkpoint_path or ENFORMER_MPRA_CHECKPOINT
        )

    def _load(self) -> None:
        if self._modules is not None:
            return
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Enformer-MPRA checkpoint not found: {self.checkpoint_path}"
            )

        import torch
        from enformer_pytorch import Enformer
        from torch import nn

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        state = checkpoint.get("state_dict")
        if not isinstance(state, dict):
            raise ValueError("Enformer-MPRA checkpoint has no state_dict")

        backbone = Enformer.from_hparams(target_length=-1)
        backbone.load_state_dict(
            {
                key.removeprefix("backbone.model."): value
                for key, value in state.items()
                if key.startswith("backbone.model.")
            },
            strict=True,
            assign=True,
        )

        attention_pool = nn.Linear(3072, 1)
        attention_pool.load_state_dict(
            {
                key.removeprefix("backbone.attention_pool."): value
                for key, value in state.items()
                if key.startswith("backbone.attention_pool.")
            },
            strict=True,
            assign=True,
        )

        head = nn.Linear(3072, 5)
        head.load_state_dict(
            {
                key.removeprefix("model.MalinoisMPRA."): value
                for key, value in state.items()
                if key.startswith("model.MalinoisMPRA.")
            },
            strict=True,
            assign=True,
        )

        self._modules = tuple(
            module.to(self.device).eval()
            for module in (backbone, attention_pool, head)
        )

    def __call__(self, seqs: list[str]) -> np.ndarray:
        """Return an array with columns ``[K562, HepG2]``."""
        if not seqs:
            return np.empty((0, 2), dtype=float)
        normalized = [str(seq).upper() for seq in seqs]
        invalid = [
            index
            for index, seq in enumerate(normalized)
            if len(seq) != 200 or set(seq) - set("ACGT")
        ]
        if invalid:
            raise ValueError(
                "Enformer-MPRA requires 200bp A/C/G/T sequences; invalid rows "
                f"{invalid[:10]}"
            )

        self._load()
        backbone, attention_pool, head = self._modules

        import torch
        from enformer_pytorch.data import str_to_one_hot

        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(normalized), self.batch_size):
                one_hot = str_to_one_hot(
                    normalized[start : start + self.batch_size]
                ).to(self.device)
                embeddings = backbone(one_hot, return_only_embeddings=True)
                weights = torch.softmax(attention_pool(embeddings), dim=1)
                pooled = torch.sum(weights * embeddings, dim=1)
                columns = [ENFORMER_SOURCE_COLUMNS[cell] for cell in TARGET_CELLS]
                chunks.append(head(pooled)[:, columns].float().cpu().numpy())
        output = np.concatenate(chunks, axis=0).astype(float, copy=False)
        if output.shape != (len(seqs), 2) or not np.isfinite(output).all():
            raise ValueError(
                "Enformer-MPRA returned invalid output with shape "
                f"{output.shape}"
            )
        return output
