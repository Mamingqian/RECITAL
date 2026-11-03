"""Promoter task oracle wrappers.

All return a (n_seqs, n_cells) array of predicted activity. Cell types are
fixed per oracle and the framework's contrastive_mingap primitive picks the
target index per sequence.
"""
from tasks.promoter.oracles.agarwal_lentimpra import AgarwalLentiMPRAOracle
from tasks.promoter.oracles.chrombpnet import ChromBPNetOracle
from tasks.promoter.oracles.enformer_mpra import EnformerMPRAOracle
from tasks.promoter.oracles.grammar_prior import GrammarPriorOracle
from tasks.promoter.oracles.malinois import MalinoisOracle

__all__ = [
    "MalinoisOracle",
    "ChromBPNetOracle",
    "AgarwalLentiMPRAOracle",
    "EnformerMPRAOracle",
    "GrammarPriorOracle",
]
