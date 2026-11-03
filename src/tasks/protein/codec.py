"""The residue alphabet of the protein tasks."""
from __future__ import annotations


# The usual twenty canonical residues, in a stable order.  Keeping this as a
# string makes the representation serializable and makes it obvious that the
# protein tasks are not accidentally using the DNA alphabet.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
# Explicit name used by task oracles and contracts.
PROTEIN_ALPHABET = AMINO_ACIDS


__all__ = [
    "AMINO_ACIDS",
    "PROTEIN_ALPHABET",
]
