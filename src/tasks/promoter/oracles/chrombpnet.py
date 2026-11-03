"""ChromBPNet chromatin-accessibility predictor wrapper.

Scores a 200-bp element in the external scorer's deterministic 2,114-bp
synthetic GC-matched context (target GC 0.42; Python RNG seed 42). This runtime
does not use a reference genome or a dinucleotide-preserving/genomic context.
Per-cell-type models cover K562, HepG2, and SK-N-SH.

Output: (n_seqs, 3) array in canonical cell order (K562, HepG2, SK-N-SH).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from tasks.promoter import assets


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance_path(path: Path) -> str:
    repo_root = Path(__file__).resolve().parents[4]
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


@dataclass
class ChromBPNetOracle:
    weights_dir: str | Path | None = None
    cell_types: tuple[str, ...] = ("K562", "HepG2", "SK-N-SH")
    context_bp: int = 2114
    device: str = "cpu"
    batch_size: int = 64
    _scorer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if int(self.context_bp) != 2114:
            raise ValueError("frozen ChromBPNet context length must be 2,114 bp")
        if tuple(self.cell_types) != ("K562", "HepG2", "SK-N-SH"):
            raise ValueError("frozen ChromBPNet cell order must remain canonical")
        self.weights_dir = (
            Path(self.weights_dir)
            if self.weights_dir is not None
            else assets.CHROMBPNET_MODELS_DIR
        )

    def _load(self) -> None:
        """Lazy-load ChromBPNet scorer on first call."""
        if self._scorer is not None:
            return

        if self.weights_dir is None or not self.weights_dir.exists():
            raise ValueError(
                f"ChromBPNetOracle requires weights_dir; got {self.weights_dir}"
            )

        # Add promoter_rl to path so we can import the real scorer
        assets.add_promoter_src_to_path()

        from promoter_rl.scorers.chrombpnet import ChromBPNetScorer

        # Build model paths keyed by the ON-DISK subdir names (K562, HepG2, SKNSH);
        # these become the predict() dict keys. Their order lines up positionally
        # with the canonical RD order (K562, HepG2, SK-N-SH).
        # Use the bias-corrected model (chrombpnet_nobias.h5); plain chrombpnet.h5
        # bundles the bias model and is not loadable by BPNet.from_keras (this is
        # the file promoter_rl itself always loads).
        self._dir_names = assets.CHROMBPNET_CELL_DIRS
        model_paths = {
            dn: str(self.weights_dir / dn / "models" / "chrombpnet_nobias.h5")
            for dn in self._dir_names
        }

        self._scorer = ChromBPNetScorer.from_h5_models(
            model_paths=model_paths,
            device=self.device,
            # This deliberately selects the scorer's deterministic synthetic
            # GC-matched fallback. It is neither genomic nor dinucleotide-shuffled.
            genome_fa=None,
        )

    def provenance(self) -> dict[str, Any]:
        """Return exact wrapper/scorer/context provenance for terminal blind-evaluation packets."""
        wrapper = Path(__file__)
        external = assets.PROMOTER_SRC / "promoter_rl/scorers/chrombpnet.py"
        if not external.is_file():
            raise FileNotFoundError(f"external ChromBPNet scorer is absent: {external}")
        weights = []
        model_tree = hashlib.sha256()
        total_size = 0
        for directory in assets.CHROMBPNET_CELL_DIRS:
            path = self.weights_dir / directory / "models" / "chrombpnet_nobias.h5"
            if not path.is_file():
                raise FileNotFoundError(f"ChromBPNet model is absent: {path}")
            relative = path.relative_to(self.weights_dir).as_posix()
            digest = _sha256(path)
            size = path.stat().st_size
            model_tree.update(relative.encode())
            model_tree.update(b"\0")
            model_tree.update(digest.encode())
            model_tree.update(b"\n")
            total_size += size
            weights.append(
                {
                    "cell_directory": directory,
                    "path": _provenance_path(path),
                    "relative_path": relative,
                    "size_bytes": size,
                    "sha256": digest,
                }
            )
        return {
            "wrapper": {
                "path": _provenance_path(wrapper),
                "size_bytes": wrapper.stat().st_size,
                "sha256": _sha256(wrapper),
            },
            "external_scorer": {
                "path": _provenance_path(external),
                "size_bytes": external.stat().st_size,
                "sha256": _sha256(external),
            },
            "model_tree": {
                "path": _provenance_path(Path(self.weights_dir)),
                "n_files": len(weights),
                "total_size_bytes": total_size,
                "tree_sha256": model_tree.hexdigest(),
            },
            "model_weights": weights,
            "context": self.context_provenance(),
        }

    def context_provenance(self) -> dict[str, Any]:
        """Describe the frozen primary blind-evaluation context without loading assets."""
        return {
            "algorithm": "deterministic_gc_matched_synthetic_flanks_v1",
            "input_length_bp": int(self.context_bp),
            "insert_length_bp": 200,
            "flank_gc_target": 0.42,
            "python_random_seed": 42,
            "genome_fasta": None,
            "same_flanks_for_every_sequence": True,
            "dinucleotide_preserving": False,
            "genomic_context": False,
        }

    def __call__(self, seqs: list[str]) -> np.ndarray:
        """Predict chromatin accessibility for sequences.

        Returns (n_seqs, 3) with columns in canonical order (K562, HepG2, SK-N-SH),
        containing predicted log-scale accessibility counts per cell type.
        """
        if not seqs:
            return np.empty((0, len(self.cell_types)), dtype=float)
        self._load()
        preds_dict = self._scorer.predict(seqs, batch_size=self.batch_size)
        # Select by the on-disk dir names (== canonical order positionally).
        output = np.stack(
            [np.asarray(preds_dict[dn], dtype=float) for dn in self._dir_names],
            axis=1,
        )
        expected = (len(seqs), len(self.cell_types))
        if output.shape != expected:
            raise ValueError(
                f"ChromBPNet scorer returned shape {output.shape}; expected {expected}"
            )
        return output
