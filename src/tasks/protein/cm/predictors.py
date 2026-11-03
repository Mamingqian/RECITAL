"""Frozen ESM features and small supervised CM function predictors."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from .data import ALPHABET


class ESMFeatures:
    def __init__(self, path: Path, device: str = "cpu", batch_size: int = 32):
        from transformers import AutoTokenizer, EsmModel
        self.device, self.batch_size = device, batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.model = EsmModel.from_pretrained(str(path), local_files_only=True,
                                             add_pooling_layer=False).to(device).eval()
        self.model.requires_grad_(False)
        self.cache: dict[str, np.ndarray] = {}

    def __call__(self, sequences: list[str]) -> np.ndarray:
        if not sequences:
            return np.empty((0, self.model.config.hidden_size), dtype=np.float32)
        if any(not s or not set(s) <= set(ALPHABET) for s in sequences):
            raise ValueError("ESM CM inputs must be nonempty standard-AA sequences")
        missing = list(dict.fromkeys(s for s in sequences if s not in self.cache))
        with torch.inference_mode():
            for start in range(0, len(missing), self.batch_size):
                chunk = missing[start:start + self.batch_size]
                batch = self.tokenizer(chunk, padding=True, return_tensors="pt")
                batch = {k: v.to(self.device) for k, v in batch.items()}
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].bool()
                for token in self.tokenizer.all_special_ids:
                    mask &= batch["input_ids"].ne(token)
                if mask.sum(1).tolist() != [len(s) for s in chunk]:
                    raise ValueError("ESM tokenizer did not produce one token per residue")
                pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp_min(1).unsqueeze(-1)
                values = pooled.float().cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite ESM embeddings")
                for sequence, feature in zip(chunk, values):
                    self.cache[sequence] = feature
        return np.stack([self.cache[s] for s in sequences])


class ProtT5Features:
    """Frozen official encoder; average residues, excluding EOS and padding."""

    def __init__(self, path: Path, device: str = "cpu", batch_size: int = 8):
        from transformers import T5EncoderModel, T5Tokenizer

        self.device, self.batch_size = device, batch_size
        self.tokenizer = T5Tokenizer.from_pretrained(str(path), local_files_only=True,
                                                    do_lower_case=False)
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self.model = T5EncoderModel.from_pretrained(
            str(path), local_files_only=True, torch_dtype=dtype,
        ).to(device).eval()
        self.model.requires_grad_(False)
        self.cache: dict[str, np.ndarray] = {}

    def __call__(self, sequences: list[str]) -> np.ndarray:
        if not sequences:
            return np.empty((0, self.model.config.d_model), dtype=np.float32)
        if any(not s or not set(s) <= set(ALPHABET) for s in sequences):
            raise ValueError("ProtT5 CM inputs must be nonempty standard-AA sequences")
        missing = list(dict.fromkeys(s for s in sequences if s not in self.cache))
        with torch.inference_mode():
            for start in range(0, len(missing), self.batch_size):
                chunk = missing[start:start + self.batch_size]
                batch = self.tokenizer([" ".join(s) for s in chunk], padding=True,
                                       return_tensors="pt", return_special_tokens_mask=True)
                mask = batch.pop("special_tokens_mask").eq(0) & batch["attention_mask"].bool()
                if mask.sum(1).tolist() != [len(s) for s in chunk]:
                    raise ValueError("ProtT5 tokenizer did not produce one token per residue")
                batch = {k: v.to(self.device) for k, v in batch.items()}
                hidden = self.model(**batch).last_hidden_state.float()
                mask = mask.to(self.device).unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                values = pooled.cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite ProtT5 embeddings")
                self.cache.update(zip(chunk, values))
        return np.stack([self.cache[s] for s in sequences])


class RegressionOracle:
    """Dataset-specific regression head with a reusable frozen encoder."""

    def __init__(self, root: Path, name: str, *, device: str = "cpu", encoder=None):
        self.name = name
        self.head = joblib.load(root / "oracles" / f"{name}.joblib")
        feature_model = self.head["feature_model"]
        if feature_model not in {"prot-t5", "esm2-650m"}:
            raise ValueError(f"Unsupported CM regression encoder: {feature_model}")
        encoder_cls = ProtT5Features if feature_model == "prot-t5" else ESMFeatures
        self.encoder = encoder or encoder_cls(root / "assets" / feature_model, device=device)

    def __call__(self, sequences: list[str]) -> np.ndarray:
        if not sequences:
            return np.empty(0)
        features = self.encoder(sequences)
        if (features.shape != (len(sequences), self.head["scaler"].n_features_in_)
                or not np.isfinite(features).all()):
            raise ValueError(f"Invalid encoder features for {self.name}")
        x = self.head["scaler"].transform(features)
        prediction = np.stack([h.predict(x) for h in self.head["heads"]]).mean(0)
        if prediction.shape != (len(sequences),) or not np.isfinite(prediction).all():
            raise ValueError(f"Invalid function predictions for {self.name}")
        return prediction

    def calibrated(self, predictions: np.ndarray) -> np.ndarray:
        calibration = self.head["calibration"]
        return predictions * calibration["slope"] + calibration["intercept"]


def encode_sequences(sequences: list[str], length: int = 96) -> torch.Tensor:
    vocabulary = {aa: i for i, aa in enumerate(ALPHABET)}
    result = torch.zeros((len(sequences), len(ALPHABET), length))
    for i, sequence in enumerate(sequences):
        if not sequence or len(sequence) > length or not set(sequence) <= set(ALPHABET):
            raise ValueError("CNN requires nonempty standard-AA sequences of at most 96 residues")
        for j, aa in enumerate(sequence):
            result[i, vocabulary[aa], j] = 1
    return result


class FunctionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(20, 32, 5, padding=2), nn.GELU(),
            nn.Conv1d(32, 32, 5, padding=2), nn.GELU(),
            nn.Flatten(), nn.Dropout(0.2), nn.Linear(32 * 96, 64),
            nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


def evaluation_metrics(target: np.ndarray, prediction: np.ndarray,
                       functional: np.ndarray, k: int = 128) -> dict:
    n = len(target)
    if not n:
        return {"n": 0}
    indices = np.argsort(-prediction, kind="stable")[:min(k, n)]
    correlation = float(spearmanr(target, prediction).statistic) if n > 2 else float("nan")
    return {
        "n": n, "spearman": correlation if np.isfinite(correlation) else None,
        "rmse": float(np.sqrt(mean_squared_error(target, prediction))),
        "auc": float(roc_auc_score(functional, prediction)) if len(set(functional)) == 2 else None,
        "top_k": len(indices), "top_k_functional_fraction": float(functional[indices].mean()),
        "background_functional_fraction": float(functional.mean()),
        "top_k_measured_median": float(np.median(target[indices])),
    }


def train_predictors(root: Path, *, device: str, seed: int = 1101,
                     cnn_epochs: int = 120) -> dict:
    root = Path(root)
    output = root / "oracles"
    output.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(root / "data/function.csv")
    sequences = table.sequence.tolist()
    y = table.target.to_numpy(dtype=np.float32)
    train = np.flatnonzero(table.split.eq("train"))
    valid = np.flatnonzero(table.split.eq("valid"))
    test = np.flatnonzero(table.split.eq("test"))
    predictions: dict[str, np.ndarray] = {}
    for name in ("esm2-35m", "esm2-650m"):
        feature_path = output / f"{name}_features.npz"
        if feature_path.exists():
            stored = np.load(feature_path, allow_pickle=False)
            if list(stored["sequences"]) != sequences:
                raise ValueError("Cached embedding rows do not match function.csv")
            features = stored["features"]
        else:
            encoder = ESMFeatures(root / "assets" / name, device=device)
            features = encoder(sequences)
            np.savez_compressed(feature_path, features=features, sequences=np.asarray(sequences))
            del encoder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        scaler = StandardScaler().fit(features[train])
        x = scaler.transform(features)
        # Hyperparameters are chosen only with development measurements.
        candidates = []
        for alpha in (10., 100., 1000., 10000.):
            head = Ridge(alpha=alpha).fit(x[train], y[train])
            candidates.append((mean_squared_error(y[valid], head.predict(x[valid])), alpha))
        alpha = min(candidates)[1]
        heads = []
        for member in range(3):
            rng = np.random.default_rng(seed + member)
            indices = rng.choice(train, len(train), replace=True)
            heads.append(Ridge(alpha=alpha).fit(x[indices], y[indices]))
        predictions[name] = np.stack([head.predict(x) for head in heads]).mean(0)
        joblib.dump({"scaler": scaler, "heads": heads, "alpha": alpha,
                     "feature_model": name, "target": "M9c Norm R.E."}, output / f"{name}.joblib")
        print(f"Fitted {name} heads, alpha={alpha}", flush=True)

    inputs = encode_sequences(sequences).to(device)
    labels = torch.as_tensor(y, device=device)
    cnn_predictions = []
    for member in range(3):
        torch.manual_seed(seed + member)
        rng = np.random.default_rng(seed + member)
        model = FunctionCNN().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        best_loss, best_state, stale = float("inf"), None, 0
        history = []
        for epoch in range(cnn_epochs):
            model.train()
            indices = rng.permutation(train)
            for start in range(0, len(indices), 64):
                batch = indices[start:start + 64]
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(inputs[batch]), labels[batch])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            model.eval()
            with torch.inference_mode():
                val_loss = float(nn.functional.mse_loss(model(inputs[valid]), labels[valid]))
            history.append(val_loss)
            if val_loss < best_loss:
                best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
            if stale >= 15:
                break
        model.load_state_dict(best_state)
        model.eval()
        with torch.inference_mode():
            pred = torch.cat([model(inputs[i:i + 128]) for i in range(0, len(inputs), 128)])
        cnn_predictions.append(pred.cpu().numpy())
        torch.save({"state_dict": {k: v.cpu() for k, v in best_state.items()},
                    "valid_loss": best_loss, "history": history, "seed": seed + member},
                   output / f"cnn_{member}.pt")
        print(f"Fitted CNN member {member}, epochs={len(history)}, valid_mse={best_loss:.4f}", flush=True)
    predictions["cnn"] = np.stack(cnn_predictions).mean(0)
    result = {}
    functional = table.functional.to_numpy(dtype=float)
    for name, pred in predictions.items():
        result[name] = {}
        for split, indices in (("valid", valid), ("test", test)):
            result[name][split] = evaluation_metrics(y[indices], pred[indices], functional[indices])
            artificial = indices[table.artificial.to_numpy()[indices]]
            result[name][split + "_artificial"] = evaluation_metrics(
                y[artificial], pred[artificial], functional[artificial])
        table[f"prediction_{name}"] = pred
    result["target"] = "M9c Norm R.E."
    result["test_usage"] = "Evaluation only; no model/epoch/hyperparameter selection"
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    table.to_csv(output / "measured_predictions.csv", index=False)
    return result
