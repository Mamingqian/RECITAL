"""JASPAR MEME-format PWM parser and a fast, dependency-light motif scanner.

Standalone (it needs no FIMO binary). This
implements the standard log-odds PWM scan (Wasserman & Sandelin 2004 style)
with an *exact* p-value threshold per motif, computed by dynamic-programming
convolution of the per-column score distribution under a background model
(the same idea MOODS/FIMO use to calibrate significance thresholds), not a
crude "% of max score" heuristic.

Scanning is vectorized: sequences are one-hot encoded once, then scored
against ALL motifs of a given width in a single BLAS matrix-multiply (windows
reshaped to (n_windows, 4*w) times a stacked motif matrix (4*w, n_motifs_w)).
Motifs are grouped by width so this reduces ~900 motifs x 2 strands scans to
a couple dozen big matmuls instead of ~1800 small ones.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BASES = "ACGT"
BASE_IDX = {b: i for i, b in enumerate(BASES)}
COMPLEMENT_IDX = {0: 3, 1: 2, 2: 1, 3: 0}  # A<->T, C<->G


@dataclass
class Motif:
    motif_id: str
    name: str
    pwm: np.ndarray  # (w, 4) probabilities, columns order A,C,G,T
    width: int = field(init=False)

    def __post_init__(self) -> None:
        self.width = self.pwm.shape[0]


def parse_meme(path: str | Path) -> list[Motif]:
    """Parse a MEME minimal-format motif file (JASPAR2024 export)."""
    text = Path(path).read_text()
    motifs: list[Motif] = []
    blocks = text.split("\nMOTIF ")
    for raw in blocks[1:]:
        lines = raw.splitlines()
        header = lines[0].split()
        motif_id = header[0]
        name = " ".join(header[1:]) if len(header) > 1 else motif_id
        mat_idx = next(
            i for i, line in enumerate(lines) if line.startswith("letter-probability matrix")
        )
        m = re.search(r"w=\s*(\d+)", lines[mat_idx])
        if m is None:
            continue
        w = int(m.group(1))
        rows = []
        for j in range(w):
            vals = [float(x) for x in lines[mat_idx + 1 + j].split()]
            rows.append(vals[:4])
        pwm = np.asarray(rows, dtype=np.float64)
        pwm = pwm / pwm.sum(axis=1, keepdims=True).clip(min=1e-9)
        motifs.append(Motif(motif_id=motif_id, name=name, pwm=pwm))
    return motifs


def pwm_to_logodds(
    pwm: np.ndarray, background: np.ndarray, pseudocount: float = 1e-3
) -> np.ndarray:
    """log2-odds matrix (w,4) relative to a background base composition."""
    p = (pwm + pseudocount) / (1.0 + 4.0 * pseudocount)
    return np.log2(p / background[None, :])


def reverse_complement_logodds(logodds: np.ndarray) -> np.ndarray:
    """log-odds matrix for scanning the reverse-complement strand, expressed
    so it can be applied to the SAME (forward) one-hot encoding of the
    sequence. Standard trick: reverse the column (position) order, and
    permute the base axis so that base b at the reversed position scores as
    if it were complement(b) at the original position
    (complement: A(0)<->T(3), C(1)<->G(2))."""
    rc = logodds[::-1, :].copy()
    perm = [3, 2, 1, 0]  # new_row[b] = old_row[complement(b)]
    rc = rc[:, perm]
    return rc


def score_pvalue_threshold(
    logodds: np.ndarray,
    background: np.ndarray,
    target_p: float = 1e-4,
    scale: int = 50,
) -> tuple[float, float]:
    """Exact-DP p-value threshold (MOODS/FIMO-style): returns (score_threshold,
    achieved_p) such that P(background score >= score_threshold) <= target_p
    approximately (achieved_p is the true tail mass at the returned integer
    threshold, reported for transparency)."""
    scaled = np.round(logodds * scale).astype(np.int64)  # (w,4)
    w = scaled.shape[0]
    cur: dict[int, float] = {0: 1.0}
    for pos in range(w):
        nxt: dict[int, float] = {}
        for b in range(4):
            s = int(scaled[pos, b])
            p = background[b]
            if p <= 0:
                continue
            for sc, prob in cur.items():
                key = sc + s
                nxt[key] = nxt.get(key, 0.0) + prob * p
        cur = nxt
    scores = np.array(sorted(cur.keys()))[::-1]
    probs = np.array([cur[s] for s in scores])
    cum = np.cumsum(probs)
    idx = int(np.searchsorted(cum, target_p))
    idx = min(idx, len(scores) - 1)
    return float(scores[idx]) / scale, float(cum[idx])


def one_hot_encode(sequences: list[str]) -> np.ndarray:
    """(n, L, 4) float32 one-hot; sequences must be same length, ACGT only
    (N or other letters -> all-zero column, contributes 0 to every motif's
    log-odds sum only if background is uniform; we instead map unknown bases
    to a uniform 0.25 row so they never trivially win, see below)."""
    lengths = {len(s) for s in sequences}
    if len(lengths) != 1:
        raise ValueError(f"one_hot_encode requires equal-length sequences, got {lengths}")
    L = lengths.pop()
    n = len(sequences)
    arr = np.zeros((n, L, 4), dtype=np.float32)
    for i, s in enumerate(sequences):
        su = s.upper()
        for j, ch in enumerate(su):
            k = BASE_IDX.get(ch)
            if k is not None:
                arr[i, j, k] = 1.0
            else:
                arr[i, j, :] = 0.25  # N or ambiguous base: neutral
    return arr


def sliding_scores(onehot: np.ndarray, logodds_stack: np.ndarray) -> np.ndarray:
    """onehot: (n, L, 4). logodds_stack: (n_motifs, w, 4), ALL SAME WIDTH w.
    Returns (n, n_windows, n_motifs) score array via a single BLAS matmul.
    """
    n, L, _ = onehot.shape
    n_motifs, w, _ = logodds_stack.shape
    if L < w:
        return np.empty((n, 0, n_motifs), dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(onehot, window_shape=w, axis=1)
    # windows shape: (n, L-w+1, 4, w)  -> transpose to (n, n_windows, w, 4)
    windows = np.transpose(windows, (0, 1, 3, 2))
    n_windows = windows.shape[1]
    flat = windows.reshape(n * n_windows, w * 4)
    weight = logodds_stack.reshape(n_motifs, w * 4).T  # (w*4, n_motifs)
    scores = flat @ weight  # (n*n_windows, n_motifs)
    return scores.reshape(n, n_windows, n_motifs)
