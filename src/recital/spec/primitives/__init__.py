"""Importing this package registers the modality-agnostic base primitives.

Layout:
  ema_zscore.py      -> per_oracle_normalization
  utilities.py       -> per_oracle_utility (target_default, target_margin,
                        gap_ema_zscore)
  aggregators.py     -> cross_oracle_aggregation (weighted_sum, trimmed_mean,
                        contrastive_mingap)
  shaping.py         -> per_sequence_shaping (target_distance, interval_reward,
                        target_interval)
  penalties.py       -> per_sequence_penalty (seen_penalty, reference_distance)

Nothing here reads composition: a primitive whose thresholds only mean something
over a particular alphabet is registered from its task package instead
(``tasks.dna.primitives``, ``tasks.protein.primitives``), and there is no
hard_biological_gate module because every gate is one of those.
A task may still inject gates through the composer's ``gate_registry``.
"""
from . import ema_zscore  # noqa: F401
from . import utilities  # noqa: F401
from . import aggregators  # noqa: F401
from . import shaping  # noqa: F401
from . import penalties  # noqa: F401
