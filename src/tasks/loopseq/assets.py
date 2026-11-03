"""Portable paths to the large Loop-seq assets kept outside Git.

The trained checkpoints, the unified dataset, the DNABERT pretrain weights and
the DNAcycP2 virtualenv are multi-hundred-MB to multi-GB artifacts that do not
belong in Git. The repository-local ``backends/loopseq_rl`` symlink is the
default stable entry point; ``LOOPSEQ_ASSET_ROOT`` can override it.
"""
from __future__ import annotations

import os
from pathlib import Path

# Package root: <repository>/backends/loopseq_rl by default.
_PKG_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_LOOPSEQ = _PKG_ROOT / "backends" / "loopseq_rl"
LOOPSEQ_ROOT = Path(os.environ.get("LOOPSEQ_ASSET_ROOT", str(_DEFAULT_LOOPSEQ)))

# Internal CNN cyclizability ensemble (training oracle, held-out Pearson ~0.845).
CNN_SCORER_DIR = LOOPSEQ_ROOT / "outputs" / "scorer"

# MIXBend: DNABERT-6 fine-tune + local pretrain assets.
MIXBEND_REPO = LOOPSEQ_ROOT / "MIXBend"
MIXBEND_DNABERT_DIR = MIXBEND_REPO / "pretrain_model" / "dnabert"
MIXBEND_PARAMS = MIXBEND_REPO / "params" / "MIXBend.json"
MIXBEND_FINETUNED_PT = LOOPSEQ_ROOT / "outputs" / "external_models" / "mixbend_v1" / "best.pt"

# BendNet: Keras bendability CNN weights.
BENDNET_REPO = LOOPSEQ_ROOT
BENDNET_WEIGHTS = LOOPSEQ_ROOT / "outputs" / "external_models" / "bendnet_v1" / "best.weights.h5"

# DNAcycP2: subprocess into an isolated Python 3.11 virtualenv.
DNACYCP2_REPO = LOOPSEQ_ROOT / "external" / "DNAcycP2-Python"

# Group-conditioned HyenaDNA SFT policy (the RL starting point).
SFT_POLICY_DIR = LOOPSEQ_ROOT / "outputs" / "sft_group3_v1"
