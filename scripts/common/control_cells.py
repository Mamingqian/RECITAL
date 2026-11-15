"""Read control-layer cells for the figure builders: a cell is a directory with events/log.jsonl.

Everything here goes through `recital.control.export.load_run`, which replays the hash-chained
log, takes no lock and writes nothing, so a cell a controller is still driving can be read.
A cell is *audited* only when `terminal_audit/summary.json` exists and reports `complete`; an
unfinished cell is never given an audited value.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
for _path in (_REPO / "src", _REPO / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from recital.control.export import RunView, candidates_table, load_run, summary_of  # noqa: E402

SCRATCH_ROOT = Path(os.environ.get("RECITAL_RUNS_ROOT", "runs"))
LOG = Path("events") / "log.jsonl"

# The configuration the paper calls RECITAL is the code's `no_self_validation` arm.
MAIN_ARM = "no_self_validation"

# The campaign roots (directories under SCRATCH_ROOT) and seeds the paper reports. Every figure, table and
# number reads its cells from here. The earlier run of Loop-seq and CM is the same framework run once before
# the Coder was offered (Loop-seq) or before the Coder and the 0.60 delivery distance ceiling (CM).
MAIN_ROOTS = {
    "cre": ("cre_control_round3_20261006",),
    "loopseq": ("loopseq_control_20261008",),
    "cm": ("cm_control_20261008",),
}
MAIN_SEEDS = {"cre": (0, 1, 2), "loopseq": (0, 1, 2), "cm": (2401, 2402, 2403)}
EARLIER_ROOTS = {"loopseq": ("loopseq_control_20261007",), "cm": ("cm_control_20261007",)}
# The no-search CM control (experiments/cm_nosearch_control): the declared initial recipe trained once per cell, for each of
# these update counts, per seed of MAIN_SEEDS["cm"].
NOSEARCH_ROOT = "cm_nosearch_20261023"
NOSEARCH_STEPS = (4000, 8192)


def cell_dirs(root: Path) -> list[Path]:
    """The cells directly under `root`, or `root` itself when it is a cell."""
    if (root / LOG).is_file():
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / LOG).is_file())


@dataclass
class Cell:
    path: Path
    view: RunView
    campaign: str
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def arm(self) -> str:
        return self.view.header.arm

    @property
    def seed(self) -> int:
        return int(self.view.header.seed)

    @property
    def conditions(self) -> list[str]:
        return self.view.conditions

    @property
    def status(self) -> str:
        return self.view.state.finished_status or "running"

    @property
    def audit(self) -> dict[str, Any] | None:
        """`terminal_audit/summary.json` when the audit finished completely, else None."""
        if "audit" not in self._cache:
            path = self.path / "terminal_audit" / "summary.json"
            data = None
            if path.is_file():
                try:
                    data = json.loads(path.read_text())
                except (OSError, ValueError):
                    data = None
            if data is not None and data.get("status") != "complete":
                data = None
            self._cache["audit"] = data
        return self._cache["audit"]

    @property
    def audited(self) -> bool:
        return self.audit is not None

    def audit_values(self, source: str) -> dict[str, Any] | None:
        """`terminal_audit/<source>.json` (per-row values of the frozen library) or None."""
        path = self.path / "terminal_audit" / f"{source}.json"
        if not self.audited or not path.is_file():
            return None
        return json.loads(path.read_text())

    def audit_manifest(self) -> dict[str, Any] | None:
        path = self.path / "terminal_audit" / "manifest.json"
        return json.loads(path.read_text()) if self.audited and path.is_file() else None

    @property
    def summary(self) -> dict[str, Any]:
        if "summary" not in self._cache:
            self._cache["summary"] = summary_of(self.view)
        return self._cache["summary"]

    def capability(self) -> dict[str, Any]:
        """What CLAUDE.md asks to see before a cell is cited."""
        control = (self.view.config().get("control") or {})
        steps = (control.get("budget") or {}).get("steps")
        synthesis = control.get("synthesis") or {}
        summary = self.summary
        return {
            "cell": self.name, "campaign": self.campaign, "arm": self.arm, "seed": self.seed,
            "status": self.status, "audited": self.audited,
            "budget_steps_min_default_max": [steps.get(k) for k in ("min", "default", "max")] if steps else None,
            "synthesis_enabled": synthesis.get("enabled"), "synthesis_roles": synthesis.get("roles"),
            "synthesis_max_requests": synthesis.get("max_requests"),
            "candidates_allocated": summary["candidates"]["allocated"],
            "candidates_trained": summary["candidates_trained"],
            "rounds_completed": summary["rounds"]["completed"],
            "resumes": summary["resumes"], "resolved_models": summary["resolved_models"],
        }

    # ----- candidates ------------------------------------------------------------------------

    def candidates(self) -> pd.DataFrame:
        """One row per candidate: the control export's flat table plus the recipe state."""
        if "candidates" in self._cache:
            return self._cache["candidates"]
        _, rows = candidates_table(self.view)
        frame = pd.DataFrame(rows)
        recipes = {}
        for candidate in self.view.state.ordered_candidates():
            if not candidate.recipe_sha:
                continue
            recipes[candidate.id] = self.view.spec(candidate.id)["recipe"]
        anchor = next((c.id for c in self.view.state.ordered_candidates() if c.anchor and c.id in recipes), None)
        base = recipes.get(anchor, {})
        base_hyper = base.get("hyperparameters") or {}
        base_steps = next((c.steps for c in self.view.state.ordered_candidates() if c.id == anchor), None)
        extra = []
        for _, row in frame.iterrows():
            recipe = recipes.get(row["candidate"], {})
            hyper = recipe.get("hyperparameters") or {}
            gates = [g.get("name") for g in recipe.get("gates") or []]
            enabled = list(recipe.get("enabled_oracles") or [])
            extra.append({
                "enabled": enabled,
                "n_gates": len(gates), "gate_names": gates,
                "penalty_changed": (json.dumps(recipe.get("penalties"), sort_keys=True)
                                    != json.dumps(base.get("penalties"), sort_keys=True)),
                "utility_changed": (json.dumps(recipe.get("oracle_utilities"), sort_keys=True)
                                    != json.dumps(base.get("oracle_utilities"), sort_keys=True)),
                "weight_changed": any(json.dumps((recipe.get("oracle_utilities") or {}).get(name), sort_keys=True)
                                      != json.dumps(utility, sort_keys=True)
                                      for name, utility in (base.get("oracle_utilities") or {}).items()),
                "aggregator_changed": (json.dumps(recipe.get("aggregator"), sort_keys=True)
                                       != json.dumps(base.get("aggregator"), sort_keys=True)),
                "oracles_changed": sorted(enabled) != sorted(base.get("enabled_oracles") or []),
                "kl_changed": bool(hyper.get("kl") != base_hyper.get("kl")),
                "lr_changed": bool(hyper.get("lr") != base_hyper.get("lr")),
                "steps_changed": bool(row["steps"] != base_steps),
                "kl_dir": float(np.sign((hyper.get("kl") or 0) - (base_hyper.get("kl") or 0))),
                "lr_dir": float(np.sign((hyper.get("lr") or 0) - (base_hyper.get("lr") or 0))),
                "steps_dir": float(np.sign((row["steps"] or 0) - (base_steps or 0))),
                "weights": json.dumps((recipe.get("aggregator") or {}).get("weights"), sort_keys=True),
            })
        frame = pd.concat([frame, pd.DataFrame(extra, index=frame.index)], axis=1)
        frame["cell"] = self.name
        frame["campaign"] = self.campaign
        frame["arm"] = self.arm
        frame["seed"] = self.seed
        self._cache["candidates"] = frame
        return frame

    def baseline_measurements(self) -> dict[str, Any] | None:
        """Per-sequence measurements of the initial policy's pool (`baseline/public_measurements.json`)."""
        path = self.path / "baseline" / "public_measurements.json"
        if "baseline_meas" not in self._cache:
            self._cache["baseline_meas"] = (json.loads(path.read_text())["measurements"]
                                            if path.is_file() else None)
        return self._cache["baseline_meas"]

    def baseline_pool(self) -> dict[str, Any] | None:
        path = self.path / "baseline" / "pool.json"
        return json.loads(path.read_text()) if path.is_file() else None

    def candidate_measurements(self, candidate: str) -> dict[str, Any] | None:
        path = self.path / "candidates" / candidate / "public_measurements.json"
        return json.loads(path.read_text())["measurements"] if path.is_file() else None

    def candidate_pool(self, candidate: str) -> dict[str, Any] | None:
        path = self.path / "candidates" / candidate / "pool.json"
        return json.loads(path.read_text()) if path.is_file() else None


def load_cell(path: Path, campaign: str | None = None) -> Cell:
    path = Path(path)
    return Cell(path=path, view=load_run(path), campaign=campaign or path.parent.name)


def load_campaign(root: Path) -> list[Cell]:
    return [load_cell(p, root.name) for p in cell_dirs(root)]


def band(values: np.ndarray, conditions: np.ndarray, condition: str) -> tuple[float, float]:
    """The 5th-95th percentile of a measurement on one condition's rows (as the control layer defines the band)."""
    chosen = np.asarray(values, dtype=float)[np.asarray(conditions) == condition]
    chosen = chosen[np.isfinite(chosen)]
    return float(np.quantile(chosen, 0.05)), float(np.quantile(chosen, 0.95))
