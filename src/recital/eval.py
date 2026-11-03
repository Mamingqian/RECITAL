"""Sequence statistics that belong to no single task."""
from __future__ import annotations

import numpy as np


def gc_content(seqs: list[str]) -> np.ndarray:
    out = np.zeros(len(seqs), dtype=float)
    for i, s in enumerate(seqs):
        if not s:
            continue
        out[i] = sum(1 for c in s if c in ("G", "C")) / len(s)
    return out
