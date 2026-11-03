"""What an arm is shown: the static header, the append-only ledger, the volatile tail, and retrieval tools.

The render is a pure function of the run state and the blobs it names. Visibility is a whitelist: a row is
projected through `ledger.visible_row`, a measurement name is shown only when every source behind it is visible
to the arm, and text that leaves the process is scanned by `guard.tripwire` before it is sent.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from recital.success import SuccessSpec

from .agents import ToolRejected
from .contract import ControlContract
from .guard import blind_projection, check_prompt_budget, scrub as scrub_text, tripwire
from .jsonio import canonical_json, sha256_hex
from .ledger import render_advisory_row, render_component_row, render_row, training_curve, visible_row
from .selection import Evidence, leaderboard
from .state import RunState
from .store import BlobStore

TOOL_RESULT_CAP = 12_000
MAX_ROWS = 50


@dataclass(frozen=True)
class ArmView:
    """What one protocol condition may see."""

    hidden_names: frozenset[str]
    blind: bool

    @classmethod
    def from_task(cls, task_contract) -> "ArmView":
        visible = task_contract.visible_sources
        hidden = {tool.name for tool in task_contract.tools if tool.name not in visible}
        hidden |= set(task_contract.terminal_blind_sources)
        return cls(hidden_names=frozenset(hidden), blind=task_contract.protocol_condition != "self_validating")

    def scrub(self, text: str) -> str:
        return scrub_text(text, self.hidden_names)


@dataclass(frozen=True)
class Block:
    text: str            # what is sent
    system_text: str     # the same block without agent- or sandbox-authored words: what the tripwire reads


@dataclass(frozen=True)
class Rendered:
    instructions: str
    input_items: tuple[dict[str, Any], ...]
    manifest: tuple[str, ...]   # blob shas: instructions, each ledger block, tail
    prompt_hash: str
    tokens: int


def _blob(store: BlobStore, sha: str) -> dict[str, Any]:
    return store.get_json(sha)


def ledger_blocks(state: RunState, store: BlobStore, arm: ArmView) -> list[Block]:
    """Rows in the order they became facts: a round's components, its candidates, then its critic advisory."""
    blocks: list[Block] = []
    for rnd in state.rounds:
        for request in sorted((c for c in state.components.values() if c.round == rnd.index),
                              key=lambda c: c.request_id):
            if request.row_sha is not None:
                row = _blob(store, request.row_sha)
                free = {key: row[key] for key in ("description", "signature", "error") if row.get(key)}
                blocks.append(Block(render_component_row(row), render_component_row({**row, **dict.fromkeys(free, "")})))
        for candidate_id in rnd.candidates:
            candidate = state.candidates[candidate_id]
            if candidate.ledger_row_sha is None:
                continue
            shown = visible_row(_blob(store, candidate.ledger_row_sha), scrub=arm.scrub)
            blocks.append(Block(render_row(shown), render_row(shown.model_copy(update={"hypothesis": ""}))))
        if rnd.critic_row_sha is not None:
            blocks.append(Block(render_advisory_row(_blob(store, rnd.critic_row_sha)), ""))
    return blocks


def evidence_of(state: RunState, store: BlobStore) -> list[Evidence]:
    """Selection's view of every candidate that has a frozen row."""
    result = []
    for candidate in state.ordered_candidates():
        if candidate.ledger_row_sha is None:
            continue
        row = _blob(store, candidate.ledger_row_sha)
        result.append(Evidence(candidate.id, candidate.ordinal, candidate.status == "ok", row.get("qualification")))
    return result


def build_tail(state: RunState, store: BlobStore, *, control: ControlContract, success: SuccessSpec,
               conditions: Sequence[str], stop_only: bool, b_effective: int, kept: Sequence[str]) -> str:
    """`b_effective` is the number of candidates still to propose; `kept` describes those already accepted."""
    proposals = control.proposals
    round_index = state.open_round.index if state.open_round else state.next_round
    lines = [
        f"## Round {round_index}",
        f"Completed rounds: {state.rounds_done} of at most {proposals.rounds_max}; a stop is accepted from round "
        f"{proposals.rounds_min + 1}.",
        f"Candidates trained or running: {sum(1 for c in state.candidates.values() if c.status != 'forfeited')} "
        f"of {proposals.candidate_budget}. Updates committed: {state.committed_updates} of "
        f"{state.header.budgets.total_updates} ({state.remaining_updates} remain).",
    ]
    board = leaderboard(success, evidence_of(state, store), conditions, top=5)
    lines.append("Leaderboard (qualified candidates by the declared ranking, best first):\n"
                 + json.dumps(board, sort_keys=True))
    open_round = state.open_round
    if open_round is not None:
        for decision_id in open_round.decision_ids:
            decision = state.decisions[decision_id]
            if decision.status == "invalid" and decision.problems:
                found = [f"{problem.code} {problem.path}: {problem.detail}".strip() for problem in decision.problems]
                what = "stop" if decision.kind == "stop" else decision.kind
                lines.append(f"Your previous answer this round ({what}) was refused:\n- " + "\n- ".join(found))
    if kept:
        lines.append("Already accepted this round and kept; do not propose them again: " + "; ".join(kept) + ".")
    if stop_only:
        lines.append("Instruction: the search is over. Return kind stop with outcome deliver and one completed "
                     "candidate per condition, or outcome no_qualified_candidate if some condition has none qualified.")
    else:
        slots = b_effective
        first = ("Round 1 includes the declared initial recipe as an unchanged reference candidate; propose "
                 f"exactly {slots} more. ") if round_index == 1 and proposals.anchor_candidate else ""
        lines.append(f"Instruction: {first}Return kind propose with exactly {slots} candidates, or kind stop with "
                     "one completed candidate per condition if further rounds will not improve the ranking.")
    return "\n".join(lines)


def render_with_tail(state: RunState, store: BlobStore, *, arm: ArmView, header: str, instructions: str,
                     tail: str, limit: int, schema: Mapping[str, Any], tools: Sequence[Any]) -> Rendered:
    """The full request around a given tail, checked for blinding and for the prompt budget."""
    blocks = ledger_blocks(state, store, arm)
    ledger = "## Ledger\n" + ("\n".join(block.text for block in blocks) if blocks else "(no candidates yet)")
    system_instructions = f"{instructions}\n\n{header}"
    tripwire(system_texts=[system_instructions, tail, *[block.system_text for block in blocks]],
             hidden_names=arm.hidden_names)
    items = ({"role": "user", "content": ledger}, {"role": "user", "content": tail})
    tokens = check_prompt_budget([system_instructions, ledger, tail], limit=limit, schema=schema, tools=tools)
    manifest = tuple(store.put(text.encode("utf-8"))
                     for text in (system_instructions, *[block.text for block in blocks], tail))
    prompt_hash = sha256_hex(canonical_json({"instructions": system_instructions, "input": list(items)}))
    return Rendered(system_instructions, items, manifest, prompt_hash, tokens)


def render(state: RunState, store: BlobStore, *, arm: ArmView, header: str, instructions: str,
           control: ControlContract, success: SuccessSpec, conditions: Sequence[str], stop_only: bool,
           b_effective: int, schema: Mapping[str, Any], tools: Sequence[Any], kept: Sequence[str]) -> Rendered:
    tail = build_tail(state, store, control=control, success=success, conditions=conditions,
                      stop_only=stop_only, b_effective=b_effective, kept=kept)
    return render_with_tail(state, store, arm=arm, header=header, instructions=instructions, tail=tail,
                            limit=control.llm.max_prompt_tokens, schema=schema, tools=tools)


def visible_scalar_names(definitions: Mapping[str, Mapping[str, Any]], visible_sources: Sequence[str]) -> list[str]:
    """Scalar measurements every source behind which is visible: the only names a tool may return."""
    visible = set(visible_sources)
    return sorted(name for name, definition in definitions.items()
                  if definition.get("shape") == [] and set(definition.get("sources", ())) <= visible)


class Retriever:
    """Backs the Strategist's get_candidate and get_rows tools from a run's own artifacts."""

    def __init__(self, *, run_dir: str | Path, state: RunState, store: BlobStore, arm: ArmView,
                 definitions: Mapping[str, Mapping[str, Any]], visible_sources: Sequence[str],
                 conditions: Sequence[str]):
        self.run_dir, self.state, self.store, self.arm = Path(run_dir), state, store, arm
        self.conditions = tuple(conditions)
        self.names = visible_scalar_names(definitions, visible_sources)

    def __call__(self, name: str, arguments: Mapping[str, Any]) -> Any:
        if name == "get_candidate":
            return self.get_candidate(str(arguments.get("id")))
        if name == "get_rows":
            return self.get_rows(str(arguments.get("candidate")), str(arguments.get("condition")),
                                 arguments.get("n"), str(arguments.get("sort")))
        raise ToolRejected(f"unknown tool {name!r}")

    def _completed(self, candidate: str):
        known = self.state.candidates.get(candidate)
        if known is None or known.status != "ok" or known.ledger_row_sha is None:
            completed = sorted(c.id for c in self.state.candidates.values() if c.status == "ok")
            raise ToolRejected(f"{candidate!r} is not a completed candidate; completed: {completed}")
        return known

    def get_candidate(self, candidate: str) -> dict[str, Any]:
        known = self._completed(candidate)
        spec = _blob(self.store, known.recipe_sha)
        result = json.loads((self.run_dir / "candidates" / candidate / "candidate_result.json").read_text())
        history = ((result.get("receipt") or {}).get("training_summary") or {}).get("log_history", [])
        recipe = blind_projection(spec["recipe"]) if self.arm.blind else spec["recipe"]
        # The candidate's ledger row is already in the ledger, so only what the row omits is returned.
        payload = {"recipe": recipe, "train": {
            key: spec["train"][key] for key in ("start_mode", "start_checkpoint", "additional_updates",
                                                "kl_reference", "schedule_policy")},
            "training_curve_tail": training_curve(history)}
        return self._capped(payload, "get_candidate takes no size argument")

    def get_rows(self, candidate: str, condition: str, n: Any, sort: str) -> dict[str, Any]:
        self._completed(candidate)
        if condition not in self.conditions:
            raise ToolRejected(f"unknown condition {condition!r}; conditions: {list(self.conditions)}")
        if not isinstance(n, int) or not 1 <= n <= MAX_ROWS:
            raise ToolRejected(f"n must be an integer from 1 to {MAX_ROWS}")
        if sort != "random" and sort not in self.names:
            raise ToolRejected(f"unknown sort {sort!r}; use 'random' or one of {self.names}")
        directory = self.run_dir / "candidates" / candidate
        pool = json.loads((directory / "pool.json").read_text())
        measured = json.loads((directory / "public_measurements.json").read_text())["measurements"]
        rows = [i for i, label in enumerate(pool["conditions"]) if label == condition]
        values = {name: measured[name]["values"] for name in self.names if name in measured}
        if sort == "random":
            random.Random(f"{candidate}:{condition}:{n}").shuffle(rows)
        else:
            rows.sort(key=lambda i: (values[sort][i] is None, -(values[sort][i] or 0.0), i))
        chosen = rows[:n]
        payload = {"candidate": candidate, "condition": condition, "sort": sort, "rows": [
            {"row_id": pool["row_ids"][i], "sequence": pool["sequences"][i],
             "values": {name: values[name][i] for name in sorted(values)}} for i in chosen]}
        return self._capped(payload, "ask for fewer rows")

    def _capped(self, payload: Mapping[str, Any], advice: str) -> dict[str, Any]:
        text = json.dumps(payload, sort_keys=True, default=str)
        # What a tool returns is outgoing text like any other.
        tripwire(system_texts=[text], hidden_names=self.arm.hidden_names)
        if len(text) > TOOL_RESULT_CAP:
            raise ToolRejected(f"the result is {len(text)} characters, over the {TOOL_RESULT_CAP} cap; {advice}")
        return json.loads(text)
