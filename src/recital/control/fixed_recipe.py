"""A no-search control: the declared initial recipe trained once for a fixed number of updates.

The cell is an ordinary control cell with one round and one candidate. The Strategist is scripted (it proposes the
declared initial recipe, every field left to its default, for `steps` updates), the Coder is off, nothing else is
edited, and the cell's own library-selection plan, health floor, delivery bound and terminal blind evaluation apply, so the
library is delivered by the rule the searches use. The learning-rate schedule of the one candidate spans its own length
(`recipes._training`: a new segment of horizon max(steps, initial_schedule_horizon)).

An anchor candidate cannot share a round with a single proposal slot (`ControlContract`: an anchor leaves no slot for the
agent when B is 1), so the declared initial recipe is the one proposed candidate and `anchor_candidate` is off.

The scripted Strategist stops with the one candidate assigned to every condition and gives that stop again when the
engine refuses it. A candidate that does not qualify therefore ends the cell by the engine's own protocol: after
`max_decision_attempts` refused stops the contract default accepts `no_qualified_candidate` and the cell finishes
`completed_no_qualified_candidate`. A restarted controller builds a new script and `ScriptedStrategist` answers each
request by its kind, so a resumed cell is answered the same as an uninterrupted one.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Mapping

from .drill import ScriptedStrategist
from .launch import Prepared, prepare
from .scheduler import Scheduler

# Job time limit = (steps * TRAIN_SECONDS_PER_STEP * time_factor + FIXED_JOB_SECONDS) (engine.time_limit). A job takes
# about 5 s per update including evaluation, so the search factor of 1.6 would end a 4,000-update job after 5.3 h;
# 2.5 gives 7.9 h for 4,000 updates and 15.5 h for 8,192.
TIME_FACTOR = 2.5


def fixed_recipe_config(config: Mapping[str, Any], steps: int) -> dict[str, Any]:
    """`config` (a cell config of a control campaign) as a one-candidate cell of `steps` updates, the Coder off."""
    out = copy.deepcopy(dict(config))
    control = out["control"]
    control["proposals"] = {**control["proposals"], "candidates_per_round": 1, "candidate_budget": 1, "rounds_min": 1,
                            "rounds_max": 1, "anchor_candidate": False}
    control["budget"] = {"steps": {"min": steps, "default": steps, "max": steps}}
    control["synthesis"] = {**control["synthesis"], "enabled": False, "roles": [], "max_requests": 0}
    out["scheduler"] = {**out["scheduler"], "time_factor": TIME_FACTOR}
    out["total_updates"] = max(int(out.get("total_updates") or 0), steps)
    return out


def script(steps: int) -> list[dict[str, Any]]:
    """The scripted answers: the proposal of the declared initial recipe for `steps` updates, the Critic's review of it,
    and a stop (`ScriptedStrategist` fills the stop's assignment with the one candidate)."""
    candidate = {"parent": "initial", "start": "from_initial", "hypothesis": "no-search control: the declared initial "
                 "recipe", "steps": steps, "lr": None, "kl": None, "group_size": None, "enabled_oracles": None,
                 "aggregator": None, "oracle_utilities": None, "penalties": None, "gates": None}
    base = {"rationale": "no-search control", "candidates": [], "stop": None, "component": None}
    return [{**base, "kind": "propose", "candidates": [candidate]},
            {"concerns": [], "requested_evidence": [], "rationale": "no-search control"},
            {**base, "kind": "stop", "stop": {"outcome": "deliver", "summary": "no-search control", "assignment": []}}]


def prepare_cell(config: Mapping[str, Any], *, steps: int, run_dir: Path, repo_root: Path, scheduler: Scheduler,
                 device: str = "cuda", may_submit: Callable[[], bool] = lambda: True) -> Prepared:
    """A fixed-recipe cell ready to run, with the scripted Strategist and no Coder."""
    cell_config = fixed_recipe_config(config, steps)
    conditions = list(cell_config["control"]["selection"]["plan"]["count"])
    strategist = ScriptedStrategist(script(steps), conditions)
    return prepare(cell_config, run_dir=run_dir, repo_root=repo_root, scheduler=scheduler, transport=strategist,
                   device=device, components=None, may_submit=may_submit)


