from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


DNA_ALPHABET = "ACGT"
DNA_TO_INDEX = {base: idx for idx, base in enumerate(DNA_ALPHABET)}
RC_CHANNEL_INDEX = torch.tensor([3, 2, 1, 0], dtype=torch.long)


def one_hot_encode_sequence(sequence: str, length: int | None = None) -> np.ndarray:
    seq = sequence.upper()
    if length is None:
        length = len(seq)
    arr = np.zeros((4, length), dtype=np.float32)
    for pos, base in enumerate(seq[:length]):
        if base in DNA_TO_INDEX:
            arr[DNA_TO_INDEX[base], pos] = 1.0
    return arr


def reverse_complement_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected [B, 4, L], got shape {tuple(x.shape)}")
    return x.index_select(dim=1, index=RC_CHANNEL_INDEX.to(x.device)).flip(dims=[2])


class SequenceRegressionDataset(Dataset):
    def __init__(self, sequences: list[str], targets: np.ndarray):
        self.sequences = sequences
        self.targets = targets.astype(np.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sequence = self.sequences[index]
        encoded = one_hot_encode_sequence(sequence)
        target = self.targets[index]
        return {
            "inputs": torch.from_numpy(encoded),
            "targets": torch.tensor(target, dtype=torch.float32),
        }


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CyclizabilityCNN(nn.Module):
    def __init__(self, hidden_channels: list[int], kernel_sizes: list[int], dropout: float = 0.2):
        super().__init__()
        if len(hidden_channels) != len(kernel_sizes):
            raise ValueError("hidden_channels and kernel_sizes must have the same length")

        blocks = []
        in_channels = 4
        for out_channels, kernel_size in zip(hidden_channels, kernel_sizes, strict=True):
            blocks.append(ConvBlock(in_channels, out_channels, kernel_size, dropout))
            in_channels = out_channels
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels[-1] * 2, hidden_channels[-1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels[-1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled_mean = features.mean(dim=2)
        pooled_max = features.max(dim=2).values
        pooled = torch.cat([pooled_mean, pooled_max], dim=1)
        return self.head(pooled).squeeze(-1)


def evaluate_with_rc(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    pred_fw = model(inputs)
    pred_rc = model(reverse_complement_tensor(inputs))
    return 0.5 * (pred_fw + pred_rc)

class CyclizabilityEnsemble:
    def __init__(self, models: list[CyclizabilityCNN], device: str = "cpu"):
        self.models = models
        self.device = device
        for model in self.models:
            model.to(device)
            model.eval()

    @classmethod
    def from_dir(cls, directory: str | Path, device: str = "cpu") -> "CyclizabilityEnsemble":
        directory = Path(directory)
        members = sorted(directory.glob("member_*.pt"))
        if not members:
            raise FileNotFoundError(f"No primary scorer checkpoints found in {directory}")
        models = []
        for path in members:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model = CyclizabilityCNN(
                hidden_channels=checkpoint["model_config"]["hidden_channels"],
                kernel_sizes=checkpoint["model_config"]["kernel_sizes"],
                dropout=checkpoint["model_config"]["dropout"],
            )
            model.load_state_dict(checkpoint["state_dict"])
            models.append(model)
        return cls(models=models, device=device)

    def predict(self, sequences: list[str], batch_size: int = 512) -> np.ndarray:
        dummy_targets = np.zeros(len(sequences), dtype=np.float32)
        dataset = SequenceRegressionDataset(sequences, dummy_targets)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        ensemble_predictions = []
        for model in self.models:
            preds = []
            with torch.no_grad():
                for batch in loader:
                    inputs = batch["inputs"].to(self.device)
                    pred = evaluate_with_rc(model, inputs).cpu().numpy()
                    preds.append(pred)
            ensemble_predictions.append(np.concatenate(preds, axis=0))
        return np.mean(np.stack(ensemble_predictions, axis=0), axis=0)
