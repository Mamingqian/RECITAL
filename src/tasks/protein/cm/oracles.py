"""Qualified CM scorers, shared by training and online evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from recital.sequence_distance import bound_shortfall

from .development import CalibratedPrior
from .family import FamilyReference, PottsModel
from .predictors import RegressionOracle


FUNCTION_MODELS = {
    "function_2026_esm", "function_2026_t5", "audit_2020_esm", "audit_2020_t5",
}

# Default floor: penalize only sequences closer to the natural family than this
# normalized whole-sequence edit distance.
NOVELTY_FLOOR = .2


def novelty_scorer(family, config: dict | None = None):
    """Distance-shortfall novelty term with explicitly declared bounds.

    The term is zero for any sequence at or past ``floor`` distance from the
    natural family and negative only below it. ``ceiling`` adds the symmetric
    upper bound as a declared parameter: distance past it is penalized.

    ``ceiling`` defaults to None, which gives the one-sided term.

    This is the framework's ``reference_distance`` penalty wearing an oracle's
    signature: same metric, same two-sided shortfall, negated because a reward
    source reports utility while a penalty reports cost. It keeps the oracle form
    because cm_novelty is a declared reward-pool SOURCE that the source-role
    protocol tracks, and it keeps ``family.distances`` because that path is
    memoised per sequence across the novelty term, the Potts prior and delivery.
    """
    settings = dict(config or {})
    unknown = set(settings) - {"floor", "ceiling"}
    if unknown:
        raise ValueError(f"unknown cm_novelty parameters: {sorted(unknown)}")
    floor, ceiling = settings.get("floor", NOVELTY_FLOOR), settings.get("ceiling")
    if floor is None or isinstance(floor, bool) or not np.isfinite(float(floor)):
        raise ValueError("cm_novelty floor must be a finite number")
    floor = float(floor)
    if ceiling is not None:
        if isinstance(ceiling, bool) or not np.isfinite(float(ceiling)):
            raise ValueError("cm_novelty ceiling must be a finite number or null")
        ceiling = float(ceiling)
        if ceiling < floor:
            raise ValueError("cm_novelty ceiling must not be below its floor")

    def score(sequences):
        distances = np.asarray(family.distances(sequences), dtype=float)
        return -bound_shortfall(distances, floor=floor, ceiling=ceiling)

    return score


def load_oracle(root: Path, name: str, *, device: str = "cuda",
                family: FamilyReference | None = None, poet_config: dict | None = None,
                novelty_config: dict | None = None):
    """Load a named scorer; callers control its reward/feedback/terminal role.

    Large encoders should be evaluated in separate worker processes. No fitted
    head, calibration or context is changed by loading a scorer.
    """
    if name in FUNCTION_MODELS:
        return RegressionOracle(root, name, device=device)
    if name == "cm_poet":
        from .poet import PoETOracle
        return PoETOracle(root, device=device, **(poet_config or {}))
    family = family or FamilyReference(root)
    if name == "cm_novelty":
        return novelty_scorer(family, novelty_config)
    if name in {"cm_bm", "cm_me", "cm_mean", "cm_max"}:
        mode = name.removeprefix("cm_")
        models = {short: PottsModel(root, short + "DCA", family)
                  for short in ("bm", "me") if mode in {short, "mean", "max"}}
        scale = json.loads((root / "development/prior_scale.json").read_text())
        return CalibratedPrior(models, mode, scale)
    raise ValueError(f"Unknown current CM oracle: {name}")


def candidate_seeds(run_seed: int, iteration: int) -> tuple[int, int]:
    """Independent training/evaluation streams, identical across paired methods."""
    if run_seed < 0 or iteration < 1:
        raise ValueError("CM run seed must be nonnegative and candidate iteration positive")
    return tuple(int(np.random.SeedSequence([run_seed, iteration, stream]).generate_state(1)[0])
                 for stream in (0, 1))


