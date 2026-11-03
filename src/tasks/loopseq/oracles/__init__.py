"""Loop-seq oracle wrappers.

All oracles expose a unified signature `(seqs: list[str]) -> np.ndarray`. The
agent's `enabled_oracles` field picks which subset participates in the reward.
"""
from tasks.loopseq.oracles.bendnet import BendNetOracle
from tasks.loopseq.oracles.cnn_ensemble import CNNEnsembleOracle
from tasks.loopseq.oracles.dnacycp2 import DNAcycP2Oracle
from tasks.loopseq.oracles.mixbend import MIXBendOracle

__all__ = [
    "CNNEnsembleOracle",
    "DNAcycP2Oracle",
    "MIXBendOracle",
    "BendNetOracle",
]
