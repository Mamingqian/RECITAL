"""Reward primitives shared by the nucleotide tasks (CRE promoter, Loop-seq).

Nothing here is imported by ``src/recital``: the framework owns the six
pipeline ROLES and the alphabet-independent primitives, while the composition
rules that only mean something over A/C/G/T live beside the tasks that declare
that alphabet. Both DNA adapters import ``tasks.dna.primitives`` so the names
their contracts reference are registered before a spec is validated.
"""
