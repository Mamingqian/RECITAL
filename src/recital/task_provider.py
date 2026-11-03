"""The objective descriptor each task declares for its headline measurement.

A task supplies one `ObjectiveDescriptor` (the continuous ranking goal; qualification stays separate), so a new
task declares its objective instead of editing framework code.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveDescriptor:
    """Names the statistics that carry a task's headline objective and how its per-stratum values read.

    A new task is added by declaring one descriptor, not by branching on stat keys in framework code.
    """

    name: str
    # Statistic that carries the objective (e.g. "group_hit_rate_below_target").
    stat_key: str
    # Key within that statistic holding the worst-over-strata scalar (e.g. "worst_hit_rate").
    worst_key: str
    # Key within that statistic holding the per-stratum mapping (e.g. "per_group_hit_rate").
    per_group_key: str
    # True when per-stratum values live in [0,1] (a hit rate).
    bounded_unit_interval: bool = False
    # True ranks by the MEAN per-stratum value (MinGap, so partial de-flooring is rewarded);
    # False by the worst-over-strata scalar (hit rate).
    guidance_uses_mean_per_group: bool = False
    # Loop-seq maximizes an in-band rate, whose per-sequence reward still uses
    # the declared interval shape. CRE MinGap uses no setpoint shape.
    allow_setpoint_shape: bool = False
    # "maximize" means the reported aggregate is ranked upward with no
    # qualification threshold. A task may separately allow an interval shape for
    # per-sequence in-band utility (Loop-seq); that shape is not a finish gate.
    objective_kind: str = "maximize"


# Built-in descriptors for the two shipped DNA tasks. A third task supplies its
# own continuous maximize descriptor.
HIT_RATE_OBJECTIVE = ObjectiveDescriptor(
    name="group_hit_rate",
    stat_key="group_hit_rate_below_target",
    worst_key="worst_hit_rate",
    per_group_key="per_group_hit_rate",
    bounded_unit_interval=True,
    guidance_uses_mean_per_group=False,
    allow_setpoint_shape=True,
    objective_kind="maximize",
)
# CRE's delivery estimand is the median MinGap per target cell.
CRE_MINGAP_OBJECTIVE = ObjectiveDescriptor(
    name="cre_mingap",
    stat_key="mingap_below_target",
    worst_key="worst_median_mingap",
    per_group_key="median_mingap_per_cell",
    bounded_unit_interval=False,
    guidance_uses_mean_per_group=False,
    objective_kind="maximize",
)
