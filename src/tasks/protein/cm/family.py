"""CM family distances, alignment to published coordinates, and Potts scoring."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
from Bio.Align import PairwiseAligner
from recital.sequence_distance import nearest_normalized

from .data import ALPHABET, read_fasta

POTTS_ALPHABET = "-" + ALPHABET

# The binding id a reward specification names to measure distance from the
# natural AroQ family. The recipe names a library; the adapter binds its rows.
NATURAL_FAMILY_REFERENCE = "cm_natural_family"


class FamilyReference:
    def __init__(self, root: Path):
        records = read_fasta((root / "data/natural_aligned.fasta").read_text())
        self.aligned = [s for _, s in records if set(s) <= set(POTTS_ALPHABET)]
        self.sequences = [s.replace("-", "") for s in self.aligned]
        self.known = dict(zip(self.sequences, self.aligned))
        self.training = [s for _, s in read_fasta((root / "data/training_sequences.fasta").read_text())]
        self.training_set = set(self.training)
        self.cache: dict[str, dict] = {}
        self.aligner = PairwiseAligner(mode="global", match_score=2., mismatch_score=-1.,
                                      open_gap_score=-4., extend_gap_score=-0.5)

    def describe(self, sequence: str) -> dict:
        if sequence in self.cache:
            return self.cache[sequence]
        # Normalize edit distance by max sequence length. Gaps in the source
        # FASTA are never counted as generated amino acids.
        distance, index = nearest_normalized(sequence, self.sequences)
        train_distance, _ = nearest_normalized(sequence, self.training)
        if sequence in self.known:
            projected, overflow = self.known[sequence], 0
        else:
            anchor = self.aligned[index]
            positions = [i for i, aa in enumerate(anchor) if aa != "-"]
            alignment = self.aligner.align(self.sequences[index], sequence)[0]
            result = ["-"] * len(anchor)
            overflow = 0
            previous = -1
            pairs = alignment.indices.T.tolist()
            for column, (reference_position, query_position) in enumerate(pairs):
                if reference_position >= 0:
                    previous = reference_position
                    if query_position >= 0:
                        result[positions[reference_position]] = sequence[query_position]
                elif query_position >= 0:
                    following = next((r for r, _ in pairs[column + 1:] if r >= 0), len(positions))
                    lo = positions[previous] + 1 if previous >= 0 else 0
                    hi = positions[following] if following < len(positions) else len(anchor)
                    free = [j for j in range(lo, hi) if anchor[j] == "-" and result[j] == "-"]
                    if free:
                        result[free[0]] = sequence[query_position]
                    else:
                        overflow += 1
            projected = "".join(result)
        value = {"natural_distance": float(distance), "training_distance": float(train_distance),
                 "training_copy": sequence in self.training_set, "aligned": projected,
                 "unmapped_residues": overflow}
        self.cache[sequence] = value
        return value

    def distances(self, sequences: list[str]) -> np.ndarray:
        return np.asarray([self.describe(s)["natural_distance"] for s in sequences])


class PottsModel:
    def __init__(self, root: Path, name: str = "bmDCA", family: FamilyReference | None = None):
        self.name = name
        self.family = family or FamilyReference(root)
        path = root / "assets" / f"{name}.npz"
        if path.exists():
            data = np.load(path)
            self.h, self.j = data["h"], data["j"]
        else:
            self.h = np.zeros((96, 21), dtype=np.float32)
            self.j = np.zeros((96, 21, 96, 21), dtype=np.float32)
            codes = {aa: i for i, aa in enumerate(POTTS_ALPHABET)}
            with zipfile.ZipFile(root / "assets" / f"{name}_params.zip") as zf:
                member = next(n for n in zf.namelist() if n.endswith(".dat"))
                with zf.open(member) as stream:
                    for line in io.TextIOWrapper(stream):
                        fields = line.split()
                        if fields[0] == "J":
                            _, i, k, a, b, value = fields
                            i, k, a, b = int(i), int(k), codes[a], codes[b]
                            self.j[i, a, k, b] = self.j[k, b, i, a] = float(value)
                        elif fields[0] == "h":
                            _, i, a, value = fields
                            self.h[int(i), codes[a]] = float(value)
                        else:
                            raise ValueError(f"Unknown Potts parameter record: {fields[0]}")
            np.savez_compressed(path, h=self.h, j=self.j)
        natural = self.raw_scores(self.encode(self.family.aligned))
        self.center, self.scale = float(natural.mean()), max(float(natural.std()), 1e-6)

    @staticmethod
    def encode(sequences: list[str]) -> np.ndarray:
        codes = {aa: i for i, aa in enumerate(POTTS_ALPHABET)}
        return np.asarray([[codes[a] for a in s] for s in sequences], dtype=np.int64)

    def raw_scores(self, states: np.ndarray) -> np.ndarray:
        positions = np.arange(96)
        result = []
        for start in range(0, len(states), 32):
            x = states[start:start + 32]
            fields = self.h[positions[None, :], x].sum(1)
            coupling = self.j[positions[None, :, None], x[:, :, None],
                              positions[None, None, :], x[:, None, :]].sum((1, 2)) * 0.5
            result.extend(fields + coupling)
        return np.asarray(result)

    def __call__(self, sequences: list[str]) -> np.ndarray:
        if not sequences:
            return np.empty(0)
        descriptions = [self.family.describe(s) for s in sequences]
        raw = self.raw_scores(self.encode([d["aligned"] for d in descriptions]))
        # Residues outside the alignment are retained in the sequence and
        # charged explicitly, rather than silently discarded by the prior.
        overflow = np.asarray([d["unmapped_residues"] for d in descriptions])
        # Preserve ordering far from the natural family: clipping the score would
        # make distant sequences indistinguishable to this prior.
        return (raw - self.center) / self.scale - overflow / 10.

    def sample(self, count: int, *, seed: int, burn_in_sweeps: int = 200,
               thinning_sweeps: int = 20, chains: int = 64) -> tuple[list[str], list[dict]]:
        """Metropolis sampling with all 96 aligned sites free, including gaps."""
        rng = np.random.default_rng(seed)
        natural = self.encode(self.family.aligned)
        states = natural[rng.integers(len(natural), size=chains)].copy()
        batches = (count + chains - 1) // chains
        total_sweeps = burn_in_sweeps + batches * thinning_sweeps
        samples, trace = [], []
        rows, positions = np.arange(chains), np.arange(96)
        for sweep in range(total_sweeps):
            accepted = 0
            for _ in range(96):
                sites = rng.integers(96, size=chains)
                old = states[rows, sites]
                new = rng.integers(21, size=chains)
                delta = self.h[sites, new] - self.h[sites, old]
                delta += (self.j[sites[:, None], new[:, None], positions, states]
                          - self.j[sites[:, None], old[:, None], positions, states]).sum(1)
                accept = np.log(rng.random(chains)) < delta
                states[rows[accept], sites[accept]] = new[accept]
                accepted += int(accept.sum())
            if (sweep + 1) % 20 == 0:
                trace.append({"sweep": sweep + 1, "mean_log_score": float(self.raw_scores(states).mean()),
                              "acceptance": accepted / (96 * chains)})
            if sweep + 1 > burn_in_sweeps and (sweep + 1 - burn_in_sweeps) % thinning_sweeps == 0:
                samples.extend("".join(POTTS_ALPHABET[a] for a in row).replace("-", "") for row in states)
        return samples[:count], trace
