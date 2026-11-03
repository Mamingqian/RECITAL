"""A read-only view of a control run and the tables derived from it.

`load_run` replays `events/log.jsonl` and reads blobs and per-candidate result files on
demand. It checks the hash chain, ignores an unfinished final line, takes no lock and writes
nothing, so it can read a run a controller is still driving. `export` turns a view into
`summary.json`, `candidates.parquet`, `selfval.parquet` and `decisions.parquet` beside a copy
of the delivery artifact. Every number comes from the log, its blobs and the candidates' own
result files; nothing here calls a model, the network or a GPU.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from recital.control import events as ev
from recital.control.contract import endpoint_host
from recital.control.errors import ControlError
from recital.control.events import from_record
from recital.control.jobs import write_json
from recital.control.jsonio import atomic_write
from recital.control.ledger import medians
from recital.control.selection import condition_values
from recital.control.state import RunState, fold
from recital.control import store
from recital.control.store import BlobStore, Event
from recital.success import SuccessSpec

LOG_PATH = Path("events") / "log.jsonl"
EXPORT_DIR = "export"


def read_events(run_dir: str | Path) -> list[Event]:
    """The run's events, read without opening the log for writing; a missing log is not a control run."""
    path = Path(run_dir) / LOG_PATH
    if not path.exists():
        raise ControlError(f"{path} does not exist; {run_dir} is not a control run")
    return store.read_events(path)


@dataclass
class RunView:
    """A run as its log and files say it is. Treat as read-only.

    `state` is the fold of every event. Blobs and files are read the first time they are
    asked for. The `*_result` and artifact accessors return None while the file is absent.
    """

    run_dir: Path
    events: tuple[Event, ...]
    models: tuple[ev._Event, ...]
    state: RunState
    blobs: BlobStore
    _memo: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def header(self) -> ev.RunStarted:
        return self.state.header

    @property
    def conditions(self) -> list[str]:
        return list(self.header.conditions)

    def of_type(self, model: type[ev._Event]) -> list[tuple[Event, Any]]:
        """Stored events and their typed payloads for one event type, in log order."""
        return [(stored, typed) for stored, typed in zip(self.events, self.models) if isinstance(typed, model)]

    def _file(self, key: str, path: Path, loader) -> Any:
        if key not in self._memo:
            self._memo[key] = loader(path) if path.is_file() else None
        return self._memo[key]

    def _json_file(self, key: str, *parts: str) -> Any:
        return self._file(key, self.run_dir.joinpath(*parts), lambda path: json.loads(path.read_text()))

    def _blob(self, sha: str | None) -> Any:
        if sha is None:
            return None
        key = f"blob:{sha}"
        if key not in self._memo:
            self._memo[key] = self.blobs.get_json(sha)
        return self._memo[key]

    # ----- per-candidate -----------------------------------------------------------------------

    def candidate_result(self, candidate: str) -> dict[str, Any] | None:
        """`candidate_result.json`, written by the worker of a candidate that finished ok."""
        return self._json_file(f"result:{candidate}", "candidates", candidate, "candidate_result.json")

    def spec(self, candidate: str) -> dict[str, Any] | None:
        """The allocated recipe, training parameters, hypothesis and the agent's proposal."""
        return self._blob(self.state.candidates[candidate].recipe_sha)

    def ledger_row(self, candidate: str) -> dict[str, Any] | None:
        return self._blob(self.state.candidates[candidate].ledger_row_sha)

    def selfval(self, candidate: str) -> dict[str, Any] | None:
        return self._blob(self.state.candidates[candidate].selfval_sha)

    def decision_response(self, decision_id: str) -> dict[str, Any]:
        return self._blob(self.state.decisions[decision_id].response_sha)

    def tool_transcript(self, sha: str | None) -> list[dict[str, Any]] | None:
        return self._blob(sha)

    # ----- the run's files ---------------------------------------------------------------------

    def baseline_result(self) -> dict[str, Any] | None:
        return self._json_file("baseline", "baseline", "baseline_result.json")

    def delivery_artifact(self) -> dict[str, Any] | None:
        return self._json_file("artifact", "delivery", "delivery_artifact.json")

    def terminal_audit(self) -> dict[str, Any] | None:
        return self._json_file("audit", "terminal_audit", "summary.json")

    def needs_attention(self) -> dict[str, Any] | None:
        return self._json_file("attention", "needs_attention.json")

    def declaration(self) -> dict[str, Any] | None:
        return self._json_file("declaration", "declaration.json")

    def success_spec(self) -> SuccessSpec | None:
        """The declared qualification and ranking, from the run's preflight declaration."""
        if "success" not in self._memo:
            success = ((((self.declaration() or {}).get("contract") or {}).get("delivery_requirements") or {})
                       .get("success"))
            self._memo["success"] = SuccessSpec.model_validate(success) if success else None
        return self._memo["success"]

    def config(self) -> dict[str, Any]:
        """The newest config snapshot: `config.yaml`, or the last `config.resume_NNN.yaml`."""
        if "config" not in self._memo:
            saved = sorted(self.run_dir.glob("config.resume_*.yaml")) or [self.run_dir / "config.yaml"]
            path = saved[-1]
            self._memo["config"] = (yaml.safe_load(path.read_text()) or {}) if path.is_file() else {}
        return self._memo["config"]


def load_run(run_dir: str | Path) -> RunView:
    """Fold a run's log into a view; raises when the log is absent, empty or inconsistent."""
    run_dir = Path(run_dir)
    events = read_events(run_dir)
    if not events:
        raise ControlError(f"{run_dir / LOG_PATH} holds no events")
    models = tuple(from_record(event) for event in events)
    return RunView(run_dir=run_dir, events=tuple(events), models=models, state=fold(models),
                   blobs=BlobStore(run_dir))


# ----- tables ----------------------------------------------------------------------------------
# A table is `(columns, rows)`: `columns` is a list of `(name, type)` with type one of
# str, int, float, bool, strs; `rows` is a list of dicts that may omit a column.

Table = tuple[list[tuple[str, str]], list[dict[str, Any]]]

CANDIDATE_COLUMNS = [
    ("candidate", "str"), ("ordinal", "int"), ("round", "int"), ("slot", "int"), ("parent", "str"),
    ("start", "str"), ("steps", "int"), ("status", "str"), ("repeat_of", "str"), ("anchor", "bool"),
    ("fingerprint", "str"), ("attempt", "int"), ("infra_failures", "int"), ("delivered_for", "strs"),
    ("hypothesis", "str"), ("edits", "strs"), ("lr", "float"), ("kl", "float"), ("group_size", "int"),
    ("enabled_oracles", "str"), ("held_out_oracles", "str"),
    ("steps_done", "int"), ("lineage_updates", "int"), ("wall_s", "float"), ("reward_first", "float"),
    ("reward_last", "float"), ("reward_std", "float"), ("frac_zero_std", "float"), ("kl_ref", "float"),
    ("actual_updates", "int"), ("cost_status", "str"), ("failure_type", "str"), ("qualified", "bool"),
]
SELFVAL_COLUMNS = [
    ("candidate", "str"), ("ordinal", "int"), ("round", "int"), ("condition", "str"), ("kind", "str"),
    ("scalar", "str"), ("reward_source", "str"), ("sentinel", "str"), ("rho", "float"),
    ("rho_initial", "float"), ("n", "int"), ("n_initial", "int"), ("source", "str"), ("median", "float"),
    ("initial_median", "float"), ("shift_raw", "float"), ("shift_pctl", "float"),
    ("status", "str"), ("effect", "float"), ("lower", "float"), ("upper", "float"), ("reward_lower", "float"),
]
DECISION_COLUMNS = [
    ("decision_id", "str"), ("round", "int"), ("kind", "str"), ("attempt", "int"), ("status", "str"),
    ("n_valid", "int"), ("problem_codes", "str"), ("tokens_prompt", "int"), ("tokens_cached", "int"),
    ("tokens_completion", "int"), ("tokens_reasoning", "int"), ("latency_s", "float"),
    ("tool_calls", "int"), ("model", "str"), ("resolved_model", "str"), ("finish_reason", "str"),
    ("ts", "str"),
]


def _float(value: Any) -> float | None:
    return None if value is None else float(value)


def candidates_table(view: RunView) -> Table:
    """One row per candidate slot, forfeited ones included.

    Besides the fixed columns, per condition `<c>`: `qualified__<c>`, `failing__<c>` (failing
    requirement ids), `selected__<c>` (rows delivered), `ranking__<c>__<metric>` (one per
    declared ranking criterion, when the run's declaration carries the success spec), and per
    measurement `<name>` the candidate's pool median `pool__<c>__<name>`, its library median
    `library__<c>__<name>` and, for the headline scalars, the pool median minus the initial
    policy's `vs_initial__<c>__<name>`. Medians are at full precision from the candidate's
    result file; the names are those of the ledger row the agent saw.
    """
    state, conditions = view.state, view.conditions
    candidates = state.ordered_candidates()
    rows = {c.id: view.ledger_row(c.id) for c in candidates}
    names = sorted({name for row in rows.values() if row and row.get("pool")
                    for cells in row["pool"].values() for name in cells})
    headline = sorted({name for row in rows.values() if row and row.get("vs_initial")
                       for cells in row["vs_initial"].values() for name in cells})
    baseline = view.baseline_result()
    initial = medians(baseline["pool"]["raw"], conditions, headline) if baseline else {}
    success = view.success_spec()
    criteria = [] if success is None else list(success.ranking)
    ranking_names = []
    for index, criterion in enumerate(criteria):
        twin = sum(1 for earlier in criteria[:index] if earlier.metric == criterion.metric)
        ranking_names.append(criterion.metric if not twin else f"{criterion.metric}#{twin + 1}")
    assignment = state.stop.accepted.assignment if state.stop.accepted else {}

    columns = list(CANDIDATE_COLUMNS)
    for c in conditions:
        columns += [(f"qualified__{c}", "bool"), (f"failing__{c}", "strs"), (f"selected__{c}", "int")]
        columns += [(f"ranking__{c}__{name}", "float") for name in ranking_names]
    for kind, kind_names in (("pool", names), ("library", names), ("vs_initial", headline)):
        columns += [(f"{kind}__{c}__{name}", "float") for c in conditions for name in kind_names]

    table = []
    for candidate in candidates:
        row: dict[str, Any] = {
            "candidate": candidate.id, "ordinal": candidate.ordinal, "round": candidate.round,
            "slot": candidate.slot, "parent": candidate.parent, "start": candidate.start,
            "steps": candidate.steps, "status": candidate.status, "repeat_of": candidate.repeat_of,
            "anchor": candidate.anchor, "fingerprint": candidate.fingerprint,
            "attempt": candidate.completed_attempt, "infra_failures": len(candidate.infra_failures),
            "delivered_for": [c for c in conditions if assignment.get(c) == candidate.id],
        }
        spec = view.spec(candidate.id) if candidate.recipe_sha else None
        if spec is not None:
            recipe = spec["recipe"]
            hyper = recipe.get("hyperparameters") or {}
            row.update(hypothesis=spec.get("hypothesis"), lr=_float(hyper.get("lr")), kl=_float(hyper.get("kl")),
                       group_size=hyper.get("group_size"),
                       enabled_oracles=json.dumps(recipe.get("enabled_oracles"), sort_keys=True),
                       held_out_oracles=json.dumps(recipe.get("held_out_oracles"), sort_keys=True))
        ledger = rows[candidate.id]
        if ledger is not None:
            row["edits"] = list(ledger.get("edits") or [])
        done = candidate.completed
        if done is not None:
            row.update(steps_done=done.steps_done, wall_s=done.wall_s, reward_first=done.reward_first,
                       reward_last=done.reward_last, reward_std=done.reward_std,
                       frac_zero_std=done.frac_zero_std, kl_ref=done.kl_ref, failure_type=done.failure_type)
        result = view.candidate_result(candidate.id) if candidate.status == "ok" else None
        if result is not None:
            end = next(item for item in result["checkpoints"] if item["id"] == result["end_checkpoint_id"])
            row.update(lineage_updates=end["lineage_updates"], actual_updates=result["cost"]["actual_updates"],
                       cost_status=result["cost"]["status"],
                       qualified=all(info["qualified"] for info in result["by_condition"].values()))
            pool = medians(result["pool"]["raw"], conditions, names)
            library = medians(result["delivery"]["selected"], conditions, names)
            qualification = result["delivery"]["qualification"]
            for c in conditions:
                info = result["by_condition"].get(c)
                if info is not None:
                    row[f"qualified__{c}"] = info["qualified"]
                    row[f"failing__{c}"] = list(info["failing_requirements"])
                    row[f"selected__{c}"] = info["selected"]
                if criteria:
                    values = condition_values(success, qualification, c)
                    for name, value in zip(ranking_names, values):
                        row[f"ranking__{c}__{name}"] = _float(value)
                for name in names:
                    row[f"pool__{c}__{name}"] = _float(pool[c][name])
                    row[f"library__{c}__{name}"] = _float(library[c][name])
                for name in headline:
                    now, then = pool[c][name], initial.get(c, {}).get(name)
                    row[f"vs_initial__{c}__{name}"] = None if now is None or then is None else float(now - then)
        table.append(row)
    return columns, table


def selfval_table(view: RunView) -> Table:
    """The self-validation numbers in long form, one row per number.

    `kind` is `rho` (within-pool correlation of a reward source's scalar with a sentinel's:
    `reward_source`, `sentinel`, `rho`, `n`, and the same pair on the initial policy's pool as
    `rho_initial`, `n_initial`) or `shift` (a source's pool median against the initial pool:
    `source`, `median`, `initial_median`, `shift_raw`, `shift_pctl`, `n`) or `flag` (a sentinel's
    verdict on the delivered library: `sentinel`, `reward_source` (the reward oracle that decided it),
    `status`, `effect`, `lower`, `upper`, `reward_lower`, `n`). Candidates whose arm computed no table,
    and failed candidates, have no rows, and a table with no `flag` entry has no `flag` rows.
    """
    table = []
    for candidate in view.state.ordered_candidates():
        if candidate.selfval_sha is None:
            continue
        computed = view.selfval(candidate.id)
        if not computed.get("conditions"):
            continue
        base = {"candidate": candidate.id, "ordinal": candidate.ordinal, "round": candidate.round,
                "scalar": computed["scalar"]}
        for condition in view.conditions:
            cell = computed["conditions"][condition]
            for reward, sentinels in cell["rho"].items():
                for sentinel, found in sentinels.items():
                    initial = cell["rho_initial"][reward][sentinel]
                    table.append({**base, "condition": condition, "kind": "rho", "reward_source": reward,
                                  "sentinel": sentinel, "rho": found["rho"], "n": found["n"],
                                  "rho_initial": initial["rho"], "n_initial": initial["n"]})
            for source, found in cell["shift"].items():
                table.append({**base, "condition": condition, "kind": "shift", "source": source,
                              "median": found["median"], "initial_median": found["initial_median"],
                              "shift_raw": found["shift_raw"], "shift_pctl": found["shift_pctl"], "n": found["n"]})
            # A table with no `flag` entry exports the rows it has.
            for sentinel, found in cell.get("flag", {}).items():
                table.append({**base, "condition": condition, "kind": "flag", "sentinel": sentinel,
                              "reward_source": found["against"], "status": found["status"], "effect": found["effect"],
                              "lower": found["lower"], "upper": found["upper"],
                              "reward_lower": found["reward_lower"], "n": found["n"]})
    return list(SELFVAL_COLUMNS), table


def decisions_table(view: RunView) -> Table:
    """One row per recorded decision: how it was judged and what the call cost."""
    table = []
    for stored, recorded in view.of_type(ev.DecisionRecorded):
        decision = view.state.decisions[recorded.decision_id]
        transcript = view.tool_transcript(recorded.tool_transcript_sha)
        table.append({
            "decision_id": recorded.decision_id, "round": recorded.round, "kind": recorded.kind,
            "attempt": recorded.attempt, "status": decision.status, "n_valid": decision.n_valid,
            "problem_codes": ",".join(problem.code for problem in decision.problems),
            "tokens_prompt": recorded.usage.prompt, "tokens_cached": recorded.usage.cached,
            "tokens_completion": recorded.usage.completion, "tokens_reasoning": recorded.usage.reasoning,
            "latency_s": recorded.latency_s,
            "tool_calls": 0 if transcript is None else sum(
                1 for item in transcript if item.get("type") == "function_call_output"),
            "model": recorded.model, "resolved_model": recorded.resolved_model,
            "finish_reason": recorded.finish_reason, "ts": stored.ts,
        })
    return list(DECISION_COLUMNS), table


def _tokens(usages: list[ev.Usage]) -> dict[str, int]:
    prompt = sum(u.prompt for u in usages)
    completion = sum(u.completion for u in usages)
    return {"calls": len(usages), "prompt": prompt, "cached": sum(u.cached for u in usages),
            "completion": completion, "reasoning": sum(u.reasoning for u in usages),
            "total": prompt + completion}


def _cost_usd(llm: dict[str, Any], usage: ev.Usage) -> float | None:
    from recital.control.transport import Price, Usage

    pricing = llm.get("pricing_usd_per_mtok") or {}
    if any(pricing.get(key) is None for key in ("input", "cached", "output")):
        return None
    price = Price(pricing["input"], pricing["cached"], pricing["output"])
    return price.cost(Usage(input=usage.prompt, cached=usage.cached, output=usage.completion,
                            reasoning=usage.reasoning))


def summary_of(view: RunView) -> dict[str, Any]:
    """The run in one record.

    `status` is the run's final status, or `running` while it has none. `tokens.total` is
    prompt plus completion (completion includes reasoning), the quantity the run's token cap
    counts; `cost_usd` is null unless the declaration priced all three token classes.
    """
    state, header = view.state, view.header
    candidates = state.ordered_candidates()
    by_status = Counter(c.status for c in candidates)
    calls = {"decision": [r.usage for _, r in view.of_type(ev.DecisionRecorded)],
             "critic": [r.usage for _, r in view.of_type(ev.CriticAdvisory)],
             "component": [r.usage for _, r in view.of_type(ev.ComponentAttempt)]}
    accepted = state.stop.accepted
    artifact, success = view.delivery_artifact(), view.success_spec()
    delivered_ranking = None
    if artifact is not None and success is not None and success.ranking:
        delivered_ranking = {
            c: dict(zip((criterion.metric for criterion in success.ranking),
                        (_float(v) for v in condition_values(success, artifact["qualification"], c))))
            for c in view.conditions}
    audit = view.terminal_audit()
    return {
        "arm": header.arm, "seed": header.seed, "task": header.task, "conditions": view.conditions,
        "model": header.model, "endpoint": endpoint_host(header.llm.get("base_url")),
        "resolved_models": list(state.resolved_models),
        "status": state.finished_status or "running",
        "library_status": state.library.status if state.library else None,
        "rounds": {"opened": len(state.rounds),
                   "allocated": sum(1 for r in state.rounds if r.status == "allocated"),
                   "completed": state.rounds_done,
                   "skipped": sum(1 for r in state.rounds if r.status == "skipped"),
                   "minimum": header.budgets.rounds_min, "maximum": header.budgets.rounds_max},
        "candidates": {"budget": header.budgets.candidate_budget, "per_round": header.candidates_per_round,
                       "anchor": header.anchor_candidate,
                       "allocated": len(candidates) - by_status["forfeited"],
                       "by_status": dict(sorted(by_status.items()))},
        "candidates_trained": by_status["ok"],
        "committed_updates": state.committed_updates, "total_updates": header.budgets.total_updates,
        "tokens": {**_tokens([u for group in calls.values() for u in group]),
                   "by_call": {kind: _tokens(group) for kind, group in calls.items()}},
        "cost_usd": _cost_usd(header.llm, state.usage),
        "pricing_usd_per_mtok": header.llm.get("pricing_usd_per_mtok"),
        "resumes": state.resumes,
        "stop": None if accepted is None else {
            "source": accepted.source, "outcome": accepted.outcome, "round": accepted.round,
            "rejections": len(state.stop.rejections)},
        "assignment": None if accepted is None else dict(accepted.assignment),
        "delivery": None if state.delivery is None else {
            "qualified": dict(state.delivery.qualified), "counts": dict(state.delivery.counts)},
        "delivered_ranking": delivered_ranking,
        "terminal_audit": None if audit is None else {k: v for k, v in audit.items() if k != "row_ids"},
        "needs_attention": view.needs_attention(),
        "events": len(view.events), "first_event_ts": view.events[0].ts, "last_event_ts": view.events[-1].ts,
    }


# ----- writing ---------------------------------------------------------------------------------

def _write_table(path: Path, table: Table) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    kinds = {"str": pa.string(), "int": pa.int64(), "float": pa.float64(), "bool": pa.bool_(),
             "strs": pa.list_(pa.string())}
    columns, rows = table
    schema = pa.schema([(name, kinds[kind]) for name, kind in columns])
    data = pa.table({name: [row.get(name) for row in rows] for name, _ in columns}, schema=schema)
    sink = pa.BufferOutputStream()
    pq.write_table(data, sink)
    atomic_write(path, sink.getvalue().to_pybytes())


def export(run_dir: str | Path, out: str | Path | None = None) -> dict[str, Any]:
    """Write the run's tables under `<run_dir>/export/` (or `out`) and return the summary.

    The files are `summary.json`, `candidates.parquet`, `selfval.parquet`, `decisions.parquet`
    and, once the run has delivered, a copy of `delivery_artifact.json`. Re-exporting an
    unchanged run rewrites the same bytes.
    """
    view = load_run(run_dir)
    out = view.run_dir / EXPORT_DIR if out is None else Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary = summary_of(view)
    _write_table(out / "candidates.parquet", candidates_table(view))
    _write_table(out / "selfval.parquet", selfval_table(view))
    _write_table(out / "decisions.parquet", decisions_table(view))
    artifact = view.run_dir / "delivery" / "delivery_artifact.json"
    if artifact.is_file():
        atomic_write(out / "delivery_artifact.json", artifact.read_bytes())
    write_json(out / "summary.json", summary)
    return summary
