"""Paths and small import helpers for the cell-specific CRE assets.

Mirrors tasks.loopseq.assets: the trained checkpoints, oracle weights and
datasets for the CRE task are multi-hundred-MB to multi-GB artifacts kept
outside Git, under ``backends/promoter_rl`` by default. Override the root with
the PROMOTER_ASSET_ROOT environment variable.

The policy loader and task entrypoints live in this repository; the
checkpoints and the scorer implementations live in the backend asset tree, so
scorer imports are resolved through one helper below (``import_scorer``).
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

# Package root: <repository>/backends/promoter_rl by default.
_PKG_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROMOTER = _PKG_ROOT / "backends" / "promoter_rl"
PROMOTER_ROOT = Path(os.environ.get("PROMOTER_ASSET_ROOT", str(_DEFAULT_PROMOTER)))


def resolve_asset_path(value: str | Path, *, repo_root: str | Path | None = None) -> Path:
    """Rebase a contract/backend path onto the active promoter asset root."""

    checkout_root = Path(repo_root or _PKG_ROOT).expanduser().resolve()
    configured = os.environ.get("PROMOTER_ASSET_ROOT")
    root = (
        Path(configured).expanduser().resolve()
        if configured
        else (checkout_root / "backends" / "promoter_rl").resolve()
    )
    checkout_backend = checkout_root / "backends" / "promoter_rl"
    path = Path(value).expanduser()

    def _suffix(candidate: Path, base: Path) -> Path | None:
        try:
            return candidate.relative_to(base)
        except ValueError:
            return None

    if path.is_absolute():
        suffix = _suffix(path, checkout_backend)
        if suffix is not None:
            return (root / suffix).resolve()
        try:
            suffix = path.resolve().relative_to(checkout_backend.resolve())
        except ValueError:
            suffix = None
        if suffix is not None:
            return (root / suffix).resolve()
        return path.resolve()

    parts = path.parts
    if parts[:2] == ("backends", "promoter_rl"):
        return root.joinpath(*parts[2:]).resolve()
    if parts and parts[0] == root.name:
        return root.joinpath(*parts[1:]).resolve()
    if parts and parts[0] in {"data", "outputs", "tools", "configs", "dbs"}:
        return (root / path).resolve()

    candidate = checkout_root / path
    if candidate.exists():
        return candidate.resolve()
    return (root / path).resolve()

# promoter_rl source tree + boda2 (needed to import the real oracle scorers).
PROMOTER_SRC = Path(os.environ.get("PROMOTER_SOURCE_ROOT", PROMOTER_ROOT / "src"))
BODA2_ROOT = PROMOTER_ROOT / "tools" / "boda2"

# Cell-type-conditioned HyenaDNA-large SFT policy (the RL starting point).
# Tokens baked into the checkpoint: <CELL_K562>=12, <CELL_HepG2>=13, <CELL_SKNSH>=14.
SFT_POLICY_DIR = PROMOTER_ROOT / "outputs" / "sft_cell_only" / "checkpoint-62500"

# Reward / panel oracle weights.
MALINOIS_ARTIFACT_DIR = PROMOTER_ROOT / "tools" / "malinois_weights" / "artifacts"
MALINOIS_AGARWAL_DIR = PROMOTER_ROOT / "tools" / "malinois_agarwal_weights"
MALINOIS_AGARWAL_WEIGHTS = MALINOIS_AGARWAL_DIR / "best_model.pt"
CHROMBPNET_MODELS_DIR = PROMOTER_ROOT / "tools" / "chrombpnet_models"
# On-disk per-cell-type subdir names under CHROMBPNET_MODELS_DIR (note: SKNSH, not SK-N-SH).
CHROMBPNET_CELL_DIRS = ("K562", "HepG2", "SKNSH")

# Enformer backbone fine-tuned on the Malinois MPRA data and released with
# DNA-Diffusion.
_DEFAULT_ENFORMER_MPRA = (
    _PKG_ROOT
    / "backends"
    / "enformer_mpra"
    / "mpra_model_best_epoch_07_val_MalinoisMPRA_mean_SpearmanR_0_73185.ckpt"
)
ENFORMER_MPRA_CHECKPOINT = Path(
    os.environ.get("ENFORMER_MPRA_CHECKPOINT", str(_DEFAULT_ENFORMER_MPRA))
)


def add_promoter_src_to_path() -> None:
    """Make promoter_rl's scorer code importable (promoter_rl.scorers.*).

    The boda2 library (required by the Malinois scorer) is added by the scorer
    itself, but its package root must be importable first.
    """
    for p in (PROMOTER_SRC, BODA2_ROOT):
        sp = str(p)
        if p.exists() and sp not in sys.path:
            sys.path.insert(0, sp)


def import_scorer(name: str):
    """Import one backend scorer through the task asset boundary."""
    add_promoter_src_to_path()
    return importlib.import_module(f"promoter_rl.scorers.{name}")
