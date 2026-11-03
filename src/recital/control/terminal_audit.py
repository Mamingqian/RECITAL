"""The terminal blind evaluation: score a frozen library with the oracles no arm ever saw.

This is the one place blind oracles are built. `build_runtime(..., audit_oracle_factories=None)`
keeps the task's own lazy blind factories; the controller and every worker replace them
with `{}`. The evaluation reads the library the run froze (`final_sequences.parquet`,
`delivery.json`, `delivery/`), scores exactly those rows, and writes under
`<run_dir>/terminal_audit/`: `manifest.json` (what was scored), one file per blind oracle,
and `summary.json`. It takes no controller lock and imports no agent or LLM code.

A repeat call scores nothing already scored and rewrites identical files; a library or
evaluation definition that differs from the one in `manifest.json` is refused.

Scoring a table of rows is separate from the frozen run: `score_library` takes the table
`{"row_ids", "sequences", "conditions"}` (lists of equal length) and a directory, and writes
the same three kinds of file there. The terminal evaluation calls it for the frozen library; a script that
scores other rows (a candidate's would-be library, an anchor's, a delivered library on another
device) calls the same function, so every table of rows meets the same oracles, the same
`audit_measure` and the same file layout. The frozen run adds only the checks of
`frozen_library` before it and the `blind_audit` status in `delivery.json` around it.
"""
from __future__ import annotations

import gc
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from recital.control.errors import ControlError
from recital.control.export import load_run
from recital.control.jobs import file_sha256, write_json
from recital.control.runtime_build import build_runtime
from recital.dna_session import json_safe

AUDIT_DIR = "terminal_audit"


class TerminalAuditError(ControlError):
    """The run has no frozen library to evaluate, or the library or its evaluation record changed."""


def _read(path: Path):
    return json.loads(path.read_text())


def audit_rows(frame) -> dict[str, list]:
    """The table the blind oracles score, from a frame with `row_id`, `sequence` and `condition` columns."""
    return {"row_ids": frame.row_id.tolist(), "sequences": frame.sequence.tolist(),
            "conditions": frame.condition.tolist()}


def frozen_library(run_dir: Path, view):
    """The delivered rows, after checking that the files on disk are the library the log froze."""
    import pandas as pd

    state = view.state
    if state.finished_status != "frozen" or state.library is None or state.library.status != "frozen":
        raise TerminalAuditError(
            f"terminal audit requires a run that finished frozen; this one is "
            f"{state.finished_status or 'unfinished'}"
            + (f" with library {state.library.status}" if state.library is not None else ""))
    path = run_dir / "final_sequences.parquet"
    if not path.is_file() or file_sha256(path) != state.library.library_sha:
        raise TerminalAuditError("final_sequences.parquet is not the library the run froze")
    delivery_path, artifact_path = run_dir / "delivery.json", run_dir / "delivery" / "delivery_artifact.json"
    if not delivery_path.is_file() or not artifact_path.is_file():
        raise TerminalAuditError("the frozen run lacks delivery.json or delivery/delivery_artifact.json")
    delivery, artifact = _read(delivery_path), _read(artifact_path)
    if not delivery.get("frozen") or delivery.get("outcome") != "qualified_delivery":
        raise TerminalAuditError("terminal audit requires a qualified frozen delivery")
    if delivery.get("id") != artifact["id"] or delivery.get("selected_row_ids") != artifact["selected_row_ids"]:
        raise TerminalAuditError("delivery.json differs from delivery/delivery_artifact.json")
    frame = pd.read_parquet(path)
    if frame.row_id.tolist() != artifact["selected_row_ids"] or len(set(frame.row_id)) != len(frame):
        raise TerminalAuditError("terminal rows do not match the frozen delivery identities")
    pool = _read(run_dir / "delivery" / "pool.json")
    indices = artifact["selected_indices"]
    if (frame.row_id.tolist() != [pool["row_ids"][i] for i in indices]
            or frame.sequence.tolist() != [pool["sequences"][i] for i in indices]
            or frame.condition.tolist() != [pool["conditions"][i] for i in indices]):
        raise TerminalAuditError("terminal sequence values differ from the frozen artifact")
    return delivery, artifact, frame


class HeldOracles:
    """The blind oracles of one runtime, each built when first asked for and kept until `release`.

    `score_library` builds and drops an oracle per call when it is given none; a caller that scores
    many tables in turn passes one of these, so a model loads once rather than once per table.
    """

    def __init__(self, runtime):
        self.factories, self.built = runtime["audit_oracle_factories"], {}

    def __call__(self, name):
        if name not in self.built:
            self.built[name] = self.factories[name]()
        return self.built[name]

    def release(self):
        self.built.clear()
        gc.collect()


def _audit_identity(runtime, rows, artifact_id) -> dict:
    """What `manifest.json` records: the rows, the audit definition and the version of every blind source."""
    contract = runtime["contract"]
    return {
        "artifact_id": artifact_id, "row_ids": list(rows["row_ids"]),
        "sequences": list(rows["sequences"]), "conditions": list(rows["conditions"]),
        "report_definition": deepcopy(dict(contract.blind_report)),
        "sources": list(contract.terminal_blind_sources),
        "source_versions": {name: runtime.get("oracle_versions", {}).get(name)
                            for name in contract.terminal_blind_sources},
    }


def _open_audit(runtime, rows, artifact_id, directory) -> Path:
    """Create `directory` and its `manifest.json`, or check the manifest already there describes these rows."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    identity = _audit_identity(runtime, rows, artifact_id)
    manifest = directory / "manifest.json"
    if manifest.exists():
        if _read(manifest) != json_safe(identity):
            raise TerminalAuditError("terminal audit identity/definition changed during recovery")
    else:
        write_json(manifest, identity)
    return directory


def _score_sources(runtime, rows, directory, *, oracles=None, on_failure=None):
    """Score `rows` with every blind source of the task; returns `{source: values}` and writes one file each.

    A source whose file is complete is read, not scored; a file with some finite values keeps them and only
    the rows without one are scored. A file that belongs to other rows is refused. When an oracle raises, the
    source's file records the failure and the values it held, `on_failure(source)` is called, and the exception
    propagates. `oracles` is a callable from source name to oracle (`HeldOracles`); without it each oracle is
    built for its source and dropped afterwards.
    """
    directory, sequences = Path(directory), rows["sequences"]
    n = len(sequences)
    scored = {}
    for name in runtime["contract"].terminal_blind_sources:
        path = directory / f"{name}.json"
        previous = _read(path) if path.exists() else None
        cached = None
        if previous is not None and "values" in previous:
            cached = np.asarray(previous["values"], dtype=float)
            if (previous.get("row_ids") != list(rows["row_ids"]) or previous.get("source") != name
                    or cached.ndim not in {1, 2} or cached.shape[0] != n):
                raise TerminalAuditError("cached terminal scores belong to different rows")
        if previous is not None and previous.get("status") == "complete":
            if cached is None or not np.isfinite(cached).all():
                raise TerminalAuditError("complete terminal scores contain missing values")
            scored[name] = cached
            continue
        unscored = (np.ones(n, dtype=bool) if cached is None else
                    ~np.isfinite(cached).all(axis=1) if cached.ndim == 2 else ~np.isfinite(cached))
        missing = np.flatnonzero(unscored)
        oracle = None
        try:
            values = cached.copy() if cached is not None else None
            if len(missing):
                oracle = oracles(name) if oracles is not None else runtime["audit_oracle_factories"][name]()
                updated = np.asarray(oracle([sequences[i] for i in missing]), dtype=float)
                if (updated.ndim not in {1, 2} or updated.shape[0] != len(missing)
                        or (cached is not None and updated.shape[1:] != cached.shape[1:])):
                    raise ValueError("terminal scorer output does not align with frozen rows")
                if cached is None:
                    values = np.full((n, *updated.shape[1:]), np.nan)
                    values[missing] = updated
                else:
                    values[missing] = np.where(np.isfinite(cached[missing]), cached[missing], updated)
            if values is None:
                values = np.full((n, *runtime.get("audit_output_shapes", {}).get(name, ())), np.nan)
            finite = bool(n and np.isfinite(values).all())
            write_json(path, {"source": name, "row_ids": list(rows["row_ids"]), "values": json_safe(values),
                              "status": "complete" if finite else "incomplete", "count": n,
                              "ineligible_row_ids": []})
            scored[name] = values
        except Exception as exc:
            failed = {"source": name, "row_ids": list(rows["row_ids"]), "status": "failed",
                      "error_type": type(exc).__name__, "error": str(exc)}
            if cached is not None:
                failed["values"] = json_safe(cached)
            write_json(path, failed)
            if on_failure is not None:
                on_failure(name)
            raise
        finally:
            del oracle
            gc.collect()
    return scored


def _finish_audit(runtime, rows, scored, artifact_id, directory) -> dict:
    """`audit_measure` of the scored rows, written as `summary.json` with the audit's status and row identities."""
    directory = Path(directory)
    report = runtime["audit_measure"](rows["sequences"], rows["conditions"], scored)
    complete = all(_read(directory / f"{name}.json")["status"] == "complete"
                   for name in runtime["contract"].terminal_blind_sources)
    report = {**report, "status": "complete" if complete else "incomplete", "artifact_id": artifact_id,
              "row_ids": list(rows["row_ids"]), "count": len(rows["sequences"])}
    write_json(directory / "summary.json", report)
    return report


def score_library(runtime, rows, artifact_id, directory, *, oracles=None, on_open=None, on_failure=None) -> dict:
    """The terminal blind evaluation of any table of rows: manifest, one file per blind source, `summary.json`.

    Returns the summary. `on_open()` is called once the manifest is written or checked and before any scoring;
    `on_failure(source)` when a blind oracle raises, before the exception propagates.
    """
    _open_audit(runtime, rows, artifact_id, directory)
    if on_open is not None:
        on_open()
    scored = _score_sources(runtime, rows, directory, oracles=oracles, on_failure=on_failure)
    return _finish_audit(runtime, rows, scored, artifact_id, directory)


def run_terminal_audit(config, run_dir, *, repo_root, device):
    """Score the run's frozen library with the task's blind oracles and write the evaluation.

    Returns the summary written to `terminal_audit/summary.json`. Raises TerminalAuditError
    when the run did not finish frozen or its library or evaluation record differs from what the
    log and the manifest describe; an oracle failure is recorded in that oracle's file and
    in `delivery.json`'s `blind_audit` before it propagates.
    """
    run_dir = Path(run_dir).resolve()
    view = load_run(run_dir)
    delivery, artifact, frame = frozen_library(run_dir, view)
    runtime = build_runtime(config, repo_root=repo_root, output=run_dir, device=device,
                            audit_oracle_factories=None)
    contract = runtime["contract"]
    if list(contract.conditions) != view.conditions:
        raise TerminalAuditError(
            f"the config's conditions {list(contract.conditions)} are not the run's {view.conditions}")
    rows, directory = audit_rows(frame), run_dir / AUDIT_DIR
    delivery_path = run_dir / "delivery.json"

    def record_running():
        delivery["blind_audit"] = {"status": "running", "directory": str(directory)}
        write_json(delivery_path, delivery)

    def record_failure(source):
        delivery["blind_audit"] = {"status": "failed", "source": source, "directory": str(directory)}
        write_json(delivery_path, delivery)

    report = score_library(runtime, rows, artifact["id"], directory, on_open=record_running, on_failure=record_failure)
    delivery["blind_audit"] = {"status": report["status"], "summary_path": str(directory / "summary.json")}
    write_json(delivery_path, delivery)
    return report
