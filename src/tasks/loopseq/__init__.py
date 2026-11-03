"""Loop-seq task package.

Importing any part of this package registers the nucleotide reward primitives
its contracts name; the shared composer does not carry them.
"""
from tasks.dna import primitives as _dna_primitives  # noqa: F401
