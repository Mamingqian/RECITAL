"""Loop-seq conditioning-group helpers shared by the baselines and the run validator."""

from __future__ import annotations

from typing import Any


def group_key(conditioning: Any) -> str:
    """Map a conditioning prompt to an interval key (``low`` | ``mid`` | ``high``).

    The policy is prompted with tokens like ``<CYC_GROUP_LOW>``, but the target
    intervals are keyed by the bare group name. Bare names pass through, and a
    label that is not one of the three prompt tokens is never matched by
    substring: ``"slow"`` stays ``"slow"``.
    """
    value = str(conditioning).strip().lower()
    return {
        "<cyc_group_low>": "low",
        "<cyc_group_mid>": "mid",
        "<cyc_group_high>": "high",
    }.get(value, value)
