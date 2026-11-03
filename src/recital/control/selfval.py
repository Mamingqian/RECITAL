"""The per-row scalar each task family exposes for a source, named by the contract's `selfval.scalar`."""
from __future__ import annotations

SCALAR_TEMPLATES = {
    "cre_cell_gap": "mingap/{source}",
    "cm_raw": "{source}",
    "loopseq_margin": "margin/{source}",
}


def scalar_name(scalar: str, source: str) -> str:
    try:
        return SCALAR_TEMPLATES[scalar].format(source=source)
    except KeyError:
        raise ValueError(f"unknown self-validation scalar {scalar!r}; known: {sorted(SCALAR_TEMPLATES)}") from None
