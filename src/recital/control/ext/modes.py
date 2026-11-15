"""Module M4, modes: who proposes the candidates, and what a person does in the search.

    control.modules.modes: {mode: autopilot}                           the agent alone (the module does nothing)
    control.modules.modes: {mode: copilot, direction: true, review: true, max_vetoes: 1}
    control.modules.modes: {mode: human_only}                          a person writes every round, no agent

`copilot`. The person opens the search with a direction, steers between rounds with notes, and reviews each proposal before
any candidate is submitted: the engine holds (`Extension.holds`) until `approve` names the proposal and the candidates
to veto (at most `max_vetoes` per round; a vetoed slot is asked again like an unresolved one). A direction and the notes
reach the Strategist as ledger text.

Human only. The person answers the Strategist's request through the same response schema; the engine holds until
`propose` posts the round's response, which is recorded as a decision of the model `human` and validated, allocated and
trained like an agent's. The Critic is a no-op (`HumanTransport`).

The person works from files the controller writes (`human/held/`) and posts through files the controller reads
(`human/inbox/`), so the controller stays the only writer of the hash-chained log. Every intervention becomes an event
with its text, its round and the minutes the person says it took; `effort` sums them. Text that goes to the agent is
read by `guard.tripwire` like any text the system writes. A posting the module cannot apply is a `modes.refused` event
with the reason.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import jsonschema

from .. import events as ev
from ..errors import BlindingViolation, ControlError
from ..events import Problem
from ..guard import tripwire
from ..jsonio import atomic_write
from ..projection import Block
from ..state import state_hash
from ..transport import Turn, Usage
from . import Extension, Phase, event, facts, folds

MODES = ("autopilot", "copilot", "human_only")
COPILOT_SETTINGS = {"direction": True, "review": True, "max_vetoes": 1}
MAX_NOTE_CHARS = 1200
POSTING_KINDS = ("direction", "note", "approve", "propose")
HUMAN = "human"


@dataclass(frozen=True)
class Hold:
    kind: Literal["direction", "review", "propose"]
    key: str            # names the file the person works from, `human/held/<key>.md`
    round: int
    reason: str


# ----- events and what they fold to ----------------------------------------------------------------------------------

@event
class HumanNote(ev._Event):
    """A direction (before the first request to the Strategist) or a note, for `round` and the rounds after it."""

    type: Literal["modes.human_note"] = "modes.human_note"
    inbox: str
    round: int
    kind: Literal["direction", "note"]
    text: str
    minutes: float
    posted_at: str


@event
class HumanApproval(ev._Event):
    """The review of a recorded proposal: the candidates (by index in the answer) the person vetoes, none to approve."""

    type: Literal["modes.human_approval"] = "modes.human_approval"
    inbox: str
    round: int
    decision_id: str
    vetoed: list[int]
    text: str
    minutes: float
    posted_at: str


@event
class HumanProposal(ev._Event):
    """The response a person gave in place of the Strategist; `decision_id` is the decision it is recorded as."""

    type: Literal["modes.human_proposal"] = "modes.human_proposal"
    inbox: str
    round: int
    attempt: int
    decision_id: str
    response_sha: str
    minutes: float
    posted_at: str


@event
class Refused(ev._Event):
    """A posting the module could not apply; the minutes still count as effort."""

    type: Literal["modes.refused"] = "modes.refused"
    inbox: str
    round: int
    kind: str
    reason: str
    minutes: float


def _once(state, inbox: str) -> bool:
    """Mark `inbox` as taken; False when an earlier event took it (the first one wins)."""
    taken = facts(state, "modes").setdefault("ingested", [])
    if inbox in taken:
        return False
    taken.append(inbox)
    return True


@folds(HumanNote)
def _fold_note(state, event):
    if _once(state, event.inbox):
        facts(state, "modes").setdefault("notes", []).append(
            {"kind": event.kind, "round": event.round, "text": event.text, "minutes": event.minutes})


@folds(HumanApproval)
def _fold_approval(state, event):
    if _once(state, event.inbox):
        facts(state, "modes").setdefault("approvals", {})[event.decision_id] = {
            "round": event.round, "vetoed": list(event.vetoed), "text": event.text, "minutes": event.minutes}


@folds(HumanProposal)
def _fold_proposal(state, event):
    if _once(state, event.inbox):
        facts(state, "modes").setdefault("proposals", {})[event.decision_id] = {
            "round": event.round, "attempt": event.attempt, "response_sha": event.response_sha,
            "minutes": event.minutes}


@folds(Refused)
def _fold_refused(state, event):
    if _once(state, event.inbox):
        facts(state, "modes").setdefault("refused", []).append(
            {"kind": event.kind, "round": event.round, "reason": event.reason, "minutes": event.minutes})


def effort(state) -> dict[str, Any]:
    """What the person did in this search: interventions by kind, the minutes they report, the characters they wrote."""
    modes = state.ext.get("modes") or {}
    notes, approvals = modes.get("notes", []), modes.get("approvals", {})
    proposals, refused = modes.get("proposals", {}), modes.get("refused", [])
    by_kind = {"direction": sum(1 for n in notes if n["kind"] == "direction"),
               "note": sum(1 for n in notes if n["kind"] == "note"), "approve": len(approvals),
               "veto": sum(len(a["vetoed"]) for a in approvals.values()), "propose": len(proposals),
               "refused": len(refused)}
    minutes = sum(n["minutes"] for n in notes) + sum(a["minutes"] for a in approvals.values()) \
        + sum(p["minutes"] for p in proposals.values()) + sum(r["minutes"] for r in refused)
    characters = sum(len(n["text"]) for n in notes) + sum(len(a["text"]) for a in approvals.values())
    rounds = sorted({n["round"] for n in notes} | {a["round"] for a in approvals.values()}
                    | {p["round"] for p in proposals.values()})
    return {"interventions": sum(by_kind[k] for k in ("direction", "note", "approve", "propose")), "by_kind": by_kind,
            "minutes": minutes, "characters": characters, "rounds": rounds}


def awaiting_approval(state) -> list[str]:
    """Recorded proposals that no `approve` has reviewed yet, oldest first."""
    approved = (state.ext.get("modes") or {}).get("approvals", {})
    return [d for d in state.decisions_awaiting_validation
            if state.decisions[d].kind == "propose" and d not in approved]


# ----- the inbox a person posts to ------------------------------------------------------------------------------

def write_inbox(run_dir: str | Path, kind: str, *, minutes: float, **fields: Any) -> Path:
    """Post one intervention for the controller to take: `human/inbox/<time>-<kind>.json`."""
    if kind not in POSTING_KINDS:
        raise ControlError(f"an intervention is one of {list(POSTING_KINDS)}, got {kind!r}")
    if isinstance(minutes, bool) or not isinstance(minutes, (int, float)) or not math.isfinite(minutes) or minutes < 0:
        raise ControlError(f"minutes is the active time this took, a number of at least 0; got {minutes!r}")
    now = time.time_ns()
    posted_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now // 10**9))
    path = Path(run_dir) / "human" / "inbox" / f"{now:020d}-{kind}.json"
    atomic_write(path, json.dumps({"kind": kind, "minutes": float(minutes), "posted_at": posted_at, **fields},
                                  sort_keys=True, indent=1, allow_nan=False).encode())
    return path


def schema_error(error: jsonschema.ValidationError) -> str:
    """One schema violation as a line a person can act on; the instance is never repeated."""
    where = "/".join(str(part) for part in error.path) or "response"
    if error.validator in ("maxItems", "minItems"):
        return f"{where}: the schema allows {'at most' if error.validator == 'maxItems' else 'at least'} " \
               f"{error.validator_value} items in this round"
    return f"{where}: {error.message[:160]}"


def skeleton(schema: Mapping[str, Any]) -> Any:
    """A response the schema accepts, with every nullable field null: what a person edits to make a proposal."""
    kind = schema.get("type")
    if isinstance(kind, list):
        return None
    if "enum" in schema:
        return schema["enum"][0]
    if kind == "object":
        return {name: skeleton(child) for name, child in schema["properties"].items()}
    if kind == "array":
        return [skeleton(schema["items"]) for _ in range(schema.get("maxItems", 0))]
    return {"string": "", "integer": 0, "number": 0.0, "boolean": False}[kind]


class HumanTransport:
    """The engine's model transport in a `human_only` cell: the Critic is empty and the Strategist is never asked."""

    model = HUMAN

    def respond(self, input_items, *, instructions=None, schema=None, schema_name="response", tools=()):
        if schema_name != "review":
            raise ControlError("a human_only cell asks no model: control.modules.modes holds every request")
        parsed = {"concerns": [], "requested_evidence": [], "rationale": "no critic: a person proposes"}
        return Turn(text=json.dumps(parsed), parsed=parsed, tool_calls=(), items=(), usage=Usage(0, 0, 0, 0),
                    model=self.model, latency_s=0.0)


class Modes(Extension):
    """Module M4: autopilot, copilot and human-only."""

    name = "modes"

    def __init__(self, config: Mapping[str, Any]):
        mode = config.get("mode")
        if mode not in MODES:
            raise ControlError(f"control.modules.modes.mode is one of {list(MODES)}, got {mode!r}")
        own = COPILOT_SETTINGS if mode == "copilot" else {}
        unknown = sorted(set(config) - {"mode"} - set(own))
        if unknown:
            raise ControlError(f"control.modules.modes with mode {mode} has the keys {unknown}; it reads "
                               f"{sorted({'mode'} | set(own))}")
        settings = {**own, **{key: value for key, value in config.items() if key != "mode"}}
        for key in ("direction", "review"):
            if key in settings and not isinstance(settings[key], bool):
                raise ControlError(f"control.modules.modes.{key} is true or false, got {settings[key]!r}")
        if "max_vetoes" in settings and (type(settings["max_vetoes"]) is not int or settings["max_vetoes"] < 0):
            raise ControlError(f"control.modules.modes.max_vetoes is an integer of at least 0, got {settings['max_vetoes']!r}")
        self.mode, self.settings = mode, settings

    @property
    def copilot(self) -> bool:
        return self.mode == "copilot"

    # ----- what the Strategist is told and shown ---------------------------------------------------------------

    def instructions(self) -> str:
        if not self.copilot:
            return ""
        return ("A person supervises this search. Their direction and notes appear in the ledger and the round header; "
                "treat them as instructions from the person who owns the search, and say in the rationale when you do "
                "not follow one. The person reviews each proposal before it trains and may veto a candidate, in which "
                "case you are asked for a replacement.")

    def tail_lines(self, state, store, arm) -> Sequence[str]:
        notes = (state.ext.get("modes") or {}).get("notes", [])
        direction = [n["text"] for n in notes if n["kind"] == "direction"]
        return [f"## Human direction\n{direction[0]}"] if direction else ()

    def round_blocks(self, state, store, arm, rnd) -> Sequence[Block]:
        notes = [n for n in (state.ext.get("modes") or {}).get("notes", [])
                 if n["kind"] == "note" and n["round"] == rnd.index + 1]
        return [Block(text, text) for text in (f"[human note for round {n['round']}] {n['text']}" for n in notes)]

    def validate_candidate(self, engine, state, rnd, index, proposal, resolved) -> Sequence[Problem]:
        pending = state.decisions_awaiting_validation
        approval = (state.ext.get("modes") or {}).get("approvals", {}).get(pending[0]) if pending else None
        if approval is None or index not in approval["vetoed"]:
            return ()
        why = f": {approval['text']}" if approval["text"] else ""
        return (Problem(code="human_veto", path="", detail=f"the person vetoed this candidate{why}"[:300]),)

    # ----- when the engine waits for the person -----------------------------------------------------------------

    @staticmethod
    def _asking(engine, state):
        """The open round when the engine's next move is a request to the Strategist, else None."""
        rnd = state.open_round
        if (rnd is None or state.stop.accepted is not None or state.baseline.status != "ingested"
                or state.decisions_awaiting_validation or state.decisions_awaiting_action
                or rnd.attempts >= engine.s.control.proposals.max_decision_attempts):
            return None
        return rnd

    def _hold(self, engine, state) -> Hold | None:
        if self.mode == "autopilot":
            return None
        if self.copilot:
            if self.settings["review"]:
                for decision_id in awaiting_approval(state):
                    round_ = state.decisions[decision_id].round
                    return Hold("review", decision_id, round_,
                                f"waiting for the person to approve {decision_id} (round {round_})")
            posted = any(n["kind"] == "direction" for n in (state.ext.get("modes") or {}).get("notes", []))
            if self.settings["direction"] and not posted and not state.decisions:
                rnd = self._asking(engine, state)
                if rnd is not None:
                    return Hold("direction", "direction", rnd.index, "waiting for the person's direction")
            return None
        rnd = self._asking(engine, state)
        if rnd is None:
            return None
        return Hold("propose", f"propose-r{rnd.index:02d}-a{rnd.attempts + 1}", rnd.index,
                    f"waiting for the person's proposal for round {rnd.index}")

    def holds(self, engine, state) -> str | None:
        hold = self._hold(engine, state)
        return None if hold is None else hold.reason

    # ----- the controller's two phases -------------------------------------------------------------------------

    def phases(self) -> Sequence[Phase]:
        if self.mode == "autopilot":
            return ()
        return (Phase("modes_ingest", "critic", self._ingest), Phase("modes_publish", "critic", self._publish))

    def _round_now(self, state) -> int:
        return state.open_round.index if state.open_round else state.next_round

    def _ingest(self, engine, state) -> bool:
        proposals = (state.ext.get("modes") or {}).get("proposals", {})
        for decision_id, proposal in proposals.items():
            if decision_id not in state.decisions:      # a controller stopped between the two events
                self._record(engine, state, decision_id, proposal["round"], proposal["response_sha"])
                return True
        taken = set((state.ext.get("modes") or {}).get("ingested", ()))
        for path in sorted((engine.s.run_dir / "human" / "inbox").glob("*.json")):
            if path.stem not in taken:
                self._take(engine, state, path)
                return True
        return False

    def _refuse(self, engine, state, path: Path, kind: str, reason: str, minutes: float = 0.0) -> None:
        engine.emit(Refused(inbox=path.stem, round=self._round_now(state), kind=kind, reason=reason[:300],
                            minutes=minutes))

    def _take(self, engine, state, path: Path) -> None:
        try:
            body = json.loads(path.read_text())
            kind, minutes = body["kind"], float(body["minutes"])
            if kind not in POSTING_KINDS or not math.isfinite(minutes) or minutes < 0:
                raise ValueError(f"kind {kind!r}, minutes {minutes!r}")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return self._refuse(engine, state, path, "unreadable", f"the posting could not be read: {exc}")
        posted_at = str(body.get("posted_at", ""))[:40]
        try:
            if kind in ("direction", "note"):
                self._take_note(engine, state, path, body, kind, minutes, posted_at)
            elif kind == "approve":
                self._take_approval(engine, state, path, body, minutes, posted_at)
            else:
                self._take_proposal(engine, state, path, body, minutes, posted_at)
        except _Refusal as refusal:
            self._refuse(engine, state, path, kind, str(refusal), minutes)

    @staticmethod
    def _clean(engine, text: Any, *, required: bool) -> str:
        """Text bound for the agent: a string of at most MAX_NOTE_CHARS that names no hidden source."""
        if not isinstance(text, str) or (required and not text.strip()):
            raise _Refusal("the text is empty" if required else "the text is not a string")
        if len(text) > MAX_NOTE_CHARS:
            raise _Refusal(f"the text has {len(text)} characters; at most {MAX_NOTE_CHARS}")
        try:
            tripwire(system_texts=[text], hidden_names=engine.arm.hidden_names)
        except BlindingViolation as exc:
            raise _Refusal(f"the text is refused: {exc}") from exc
        return text.strip()

    def _take_note(self, engine, state, path, body, kind, minutes, posted_at) -> None:
        if not self.copilot:
            raise _Refusal(f"mode {self.mode} has no agent to address")
        text = self._clean(engine, body.get("text"), required=True)
        round_ = self._round_now(state)
        if kind == "direction" and (state.decisions or any(
                n["kind"] == "direction" for n in (state.ext.get("modes") or {}).get("notes", []))):
            raise _Refusal("a direction opens the search, once, before the Strategist is asked; post a note")
        if kind == "note" and round_ == 1:
            raise _Refusal("no round has finished; post the direction")
        engine.emit(HumanNote(inbox=path.stem, round=round_, kind=kind, text=text, minutes=minutes,
                              posted_at=posted_at))

    def _take_approval(self, engine, state, path, body, minutes, posted_at) -> None:
        if not (self.copilot and self.settings["review"]):
            raise _Refusal(f"mode {self.mode} with review {self.settings.get('review', False)} reviews no proposal")
        decision_id = body.get("decision_id")
        if decision_id not in awaiting_approval(state):
            raise _Refusal(f"{decision_id!r} is not a proposal waiting for approval; waiting: {awaiting_approval(state)}")
        decision = state.decisions[decision_id]
        answered = engine.blobs.get_json(decision.response_sha)["candidates"]
        vetoed = body.get("vetoed", [])
        if (not isinstance(vetoed, list) or any(type(i) is not int for i in vetoed) or len(set(vetoed)) != len(vetoed)
                or not all(0 <= i < len(answered) for i in vetoed)):
            raise _Refusal(f"vetoed names candidates by index 0 to {len(answered) - 1}, each once; got {vetoed!r}")
        used = sum(len(a["vetoed"]) for a in (state.ext.get("modes") or {}).get("approvals", {}).values()
                   if a["round"] == decision.round)
        if used + len(vetoed) > self.settings["max_vetoes"]:
            raise _Refusal(f"round {decision.round} allows {self.settings['max_vetoes']} vetoes and has used {used}")
        text = self._clean(engine, body.get("text", ""), required=False)
        engine.emit(HumanApproval(inbox=path.stem, round=decision.round, decision_id=decision_id, vetoed=vetoed,
                                  text=text, minutes=minutes, posted_at=posted_at))

    def _take_proposal(self, engine, state, path, body, minutes, posted_at) -> None:
        if self.mode != "human_only":
            raise _Refusal(f"mode {self.mode} asks the Strategist; a proposal is for human_only")
        rnd = self._asking(engine, state)
        if rnd is None:
            raise _Refusal("no proposal is awaited")
        _, schema, _ = engine.ask_prompt(state, rnd)
        response = body.get("response")
        problems = sorted(jsonschema.Draft202012Validator(schema).iter_errors(response), key=lambda e: list(e.path))
        if problems:
            raise _Refusal(schema_error(problems[0]))
        if response["kind"] not in ("propose", "stop"):
            raise _Refusal("a person gives kind propose or stop")
        decision_id = state.next_decision_id
        response_sha = engine.blobs.put_json(response)
        engine.emit(HumanProposal(inbox=path.stem, round=rnd.index, attempt=rnd.attempts + 1, decision_id=decision_id,
                                  response_sha=response_sha, minutes=minutes, posted_at=posted_at))
        self._record(engine, state, decision_id, rnd.index, response_sha)

    def _record(self, engine, state, decision_id: str, round_: int, response_sha: str) -> None:
        """The decision a person's response becomes: the prompt hash is the request the person was shown."""
        rnd = state.rounds[round_ - 1]
        rendered, schema, _ = engine.ask_prompt(state, rnd)
        response = engine.blobs.get_json(response_sha)
        engine.emit(ev.DecisionRecorded(
            decision_id=decision_id, round=round_, kind=response["kind"], attempt=rnd.attempts + 1,
            state_hash=state_hash(state), prompt_hash=rendered.prompt_hash, prompt_manifest=list(rendered.manifest),
            schema_sha=engine.blobs.put_json(schema), response_sha=response_sha, tool_transcript_sha=None,
            usage=ev.Usage(prompt=0, completion=0, cached=0, reasoning=0), model=HUMAN, resolved_model=HUMAN,
            latency_s=0.0))

    # ----- what the person reads ------------------------------------------------------------------------------

    def _publish(self, engine, state) -> bool:
        """Write the file a person works from when the engine starts to wait; never a step of the search."""
        hold = self._hold(engine, state)
        if hold is None:
            return False
        path = engine.s.run_dir / "human" / "held" / f"{hold.key}.md"
        if path.exists():
            return False
        rnd = state.rounds[hold.round - 1]
        rendered, schema, _ = engine.ask_prompt(state, rnd)
        run = engine.s.run_dir
        post = "python -m recital.control.cli"
        if hold.kind == "direction":
            lines = [f"# Direction for round {hold.round}",
                     "The Strategist has not been asked. Say what to try first: the reward to start from, or a direction.",
                     f"    {post} note --run {run} --kind direction --minutes M --text '...'"]
        elif hold.kind == "review":
            answered = engine.blobs.get_json(state.decisions[hold.key].response_sha)
            used = sum(len(a["vetoed"]) for a in (state.ext.get("modes") or {}).get("approvals", {}).values()
                       if a["round"] == hold.round)
            lines = [f"# Review of {hold.key} (round {hold.round})",
                     f"The Strategist proposed {len(answered['candidates'])} candidates. Vetoes used this round: "
                     f"{used} of {self.settings['max_vetoes']}.",
                     f"    {post} approve --run {run} --decision {hold.key} --minutes M [--veto INDEX ...] [--text WHY]",
                     "", "rationale: " + answered["rationale"], ""]
            for index, item in enumerate(answered["candidates"]):
                edits = {key: value for key, value in item.items()
                         if value is not None and key not in ("parent", "start", "hypothesis")}
                lines.append(f"candidates[{index}]: parent {item['parent']}, {item['start']}; {item['hypothesis']}\n"
                             f"    edits: {json.dumps(edits, sort_keys=True)}")
        else:
            lines = [f"# Proposal for round {hold.round} (attempt {rnd.attempts + 1})",
                     "Write the response the Strategist would write, in this schema, and post it:",
                     f"    {post} propose --run {run} --minutes M --file response.json", "",
                     "A skeleton to edit:", "```json", json.dumps(skeleton(schema), indent=1), "```", "",
                     "The schema:", "```json", json.dumps(schema, indent=1), "```"]
        lines += ["", "## What the Strategist is shown", "", "### Instructions", rendered.instructions, ""]
        lines += [item["content"] for item in rendered.input_items]
        atomic_write(path, "\n".join(lines).encode())
        return False


class _Refusal(Exception):
    """Why a posting cannot be applied; becomes a `modes.refused` event."""


def pending_holds(run_dir: str | Path, state) -> list[Path]:
    """The files in `human/held/` that still wait for the person, oldest first (a read for the `show` command)."""
    modes = state.ext.get("modes") or {}
    notes, approvals, proposals = modes.get("notes", []), modes.get("approvals", {}), modes.get("proposals", {})
    answered = {f"propose-r{p['round']:02d}-a{p['attempt']}" for p in proposals.values()}
    pending = []
    for path in sorted((Path(run_dir) / "human" / "held").glob("*.md")):
        key = path.stem
        done = (any(n["kind"] == "direction" for n in notes) if key == "direction"
                else key in approvals if key.startswith("dec_") else key in answered)
        if not done:
            pending.append(path)
    return pending


def build(config: Mapping[str, Any]) -> Modes:
    return Modes(config)
