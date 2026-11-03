"""Public CM assets and sequence-disjoint supervised/SFT tables."""
from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
RAW_URL = "https://raw.githubusercontent.com/robertonetti/entropy-sparse-DCA/main/"
MEDIA_URL = "https://media.githubusercontent.com/media/robertonetti/entropy-sparse-DCA/main/"
EXPERIMENT_FILES = (
    "bmDCA_edDCA_eaDCA_CM_REdata.csv", "meDCA_CM_REdata.csv",
)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    temporary.replace(destination)
    return destination


def read_fasta(text: str) -> list[tuple[str, str]]:
    """Only a line-start '>' begins a record; public CM headers contain others."""
    records, name, parts = [], None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(parts).upper()))
            name, parts = line[1:], []
        elif line.strip():
            if name is None:
                raise ValueError("FASTA sequence precedes its header")
            parts.append(line.strip())
    if name is not None:
        records.append((name, "".join(parts).upper()))
    return records


def write_fasta(path: Path, sequences: list[str]) -> None:
    path.write_text("".join(f">seq_{i}\n{s}\n" for i, s in enumerate(sequences)))


def prepare_data(root: Path, *, seed: int = 1101) -> dict:
    """Aggregate repeat measurements without splitting a sequence across folds."""
    root = Path(root)
    assets = root / "assets"
    target = root / "data"
    target.mkdir(parents=True, exist_ok=True)
    archive = download(MEDIA_URL + "data/naturals/CM_130530_MC.zip", assets / "natural.zip")
    with zipfile.ZipFile(archive) as zf:
        fasta_name = next(n for n in zf.namelist() if n.endswith(".fasta"))
        naturals = read_fasta(zf.read(fasta_name).decode())
    if not naturals or {len(s) for _, s in naturals} != {96}:
        raise ValueError("Expected the published 96-column CM alignment")
    # Unknown residues retain their explicit meaning in the reference alignment;
    # they are never converted into a made-up residue in SFT sequences.
    write_fasta(target / "natural_aligned.fasta", [s for _, s in naturals])
    natural_sequences = {s.replace("-", "") for _, s in naturals}

    frames = []
    for batch, name in enumerate(EXPERIMENT_FILES):
        path = download(RAW_URL + "data/experiment/" + name, assets / name)
        frame = pd.read_csv(path)
        frame["batch"] = batch
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    raw["sequence"] = raw["Sequence"].str.replace("-", "", regex=False)
    raw["target"] = pd.to_numeric(raw["M9c Norm R.E."], errors="coerce")
    raw["raw_enrichment"] = pd.to_numeric(raw["M9c R.E."], errors="coerce")
    raw["functional"] = raw["Null Mode 3Sigma Classification"].eq("Functional").astype(float)
    raw["artificial"] = raw["Group"].str.contains("DCA")
    raw["test_source"] = raw["Group"].str.startswith("meDCA")
    raw.to_csv(target / "measurements.csv", index=False)
    # Standards are controls, not extra training proteins. Read coverage and
    # uncertainty are retained in measurements.csv for analysis.
    eligible = raw[raw["Group"].ne("Standard Curve") & np.isfinite(raw.target)].copy()
    eligible = eligible[eligible.sequence.map(lambda s: bool(s) and set(s) <= set(ALPHABET))]
    rows = []
    for sequence, group in eligible.groupby("sequence", sort=True):
        rows.append({
            "sequence": sequence,
            "aligned": group.Sequence.iloc[0],
            "target": float(group.target.mean()),
            "raw_enrichment": float(group.raw_enrichment.mean()),
            "functional": bool(group.functional.mean() >= 0.5),
            "artificial": bool(group.artificial.any()),
            "test_source": bool(group.test_source.any()),
            "sources": "|".join(sorted(set(group.Group))),
            "measurement_count": len(group),
        })
    table = pd.DataFrame(rows)
    table["split"] = np.where(table.test_source, "test", "train")
    rng = np.random.default_rng(seed)
    # Stratification retains positives and negatives in development. Grouping
    # above makes repeat/batch duplicates inseparable.
    for _, group in table[table.split.eq("train")].groupby(["artificial", "functional"]):
        indices = rng.permutation(group.index.to_numpy())
        table.loc[indices[:max(1, round(0.15 * len(indices)))], "split"] = "valid"
    table.to_csv(target / "function.csv", index=False)

    excluded = set(table.loc[table.split.eq("test"), "sequence"])
    positive = set(table.loc[
        table.split.eq("train") & table.artificial & table.functional, "sequence"
    ])
    sft = sorted(s for s in natural_sequences | positive
                 if 80 <= len(s) <= 96 and set(s) <= set(ALPHABET) and s not in excluded)
    rng.shuffle(sft)
    n_valid = max(1, round(0.1 * len(sft)))
    sft_table = pd.DataFrame({"sequence": sft,
                              "split": ["valid"] * n_valid + ["train"] * (len(sft) - n_valid)})
    sft_table.to_csv(target / "sft.csv", index=False)
    # The whole family stays in the novelty reference even when part of it is
    # held out for SFT loss. Function-training examples are separately recorded.
    write_fasta(target / "training_sequences.fasta", sorted(
        set(sft_table.loc[sft_table.split.eq("train"), "sequence"])
        | set(table.loc[table.split.eq("train"), "sequence"])
    ))
    report = {
        "seed": seed, "natural_msa_count": len(naturals),
        "measured_rows": len(raw), "measured_unique_sequences": raw.sequence.nunique(),
        "function_split_counts": table.split.value_counts().to_dict(),
        "positive_counts": table.groupby("split").functional.sum().to_dict(),
        "sft_split_counts": sft_table.split.value_counts().to_dict(),
        "sft_test_overlap": len(set(sft) & excluded),
        "target": "published M9c Norm R.E.; replicate mean per sequence",
        "functionality": "published Null Mode 3Sigma Classification",
        "test": "meDCA artificial sequences; shared natural controls are not test designs",
        "sft_note": "natural family membership is used without assuming measured functionality",
    }
    (target / "summary.json").write_text(json.dumps(report, indent=2))
    for name in ("bmDCA", "eaDCA", "edDCA", "meDCA"):
        download(MEDIA_URL + f"models/tested_models/{name}_params.zip", assets / f"{name}_params.zip")
    return report


def download_models(root: Path) -> None:
    """Download the public model weights into ``root/assets``."""
    from huggingface_hub import snapshot_download
    models = {
        "progen3-112m": "Profluent-Bio/progen3-112m",
        "esm2-35m": "facebook/esm2_t12_35M_UR50D",
        "esm2-650m": "facebook/esm2_t33_650M_UR50D",
        # The encoder behind function_2026_t5 and audit_2020_t5: the encoder-only
        # half-precision release, which predictors.ProtT5Features loads as
        # T5EncoderModel in float16 on cuda.
        "prot-t5": "Rostlab/prot_t5_xl_half_uniref50-enc",
    }
    for name, repo_id in models.items():
        snapshot_download(repo_id, local_dir=Path(root) / "assets" / name,
                          allow_patterns=["*.json", "vocab.txt", "spiece.model", "model.safetensors",
                                         "pytorch_model.bin"],
                          max_workers=3)
