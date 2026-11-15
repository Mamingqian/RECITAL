"""The MAC-AMP port: an engine-bound transport that runs MAC-AMP's loop in place of the Strategist.

MAC-AMP (arXiv 2602.14926, github.com/CLMFAP/MAC-AMP_v1) trains a peptide generator in three stages. Before each stage a reward
designer writes candidate reward functions, a critic that sees only the reviewers' text approves or rejects each, the survivors are
trained briefly in a sandbox from the stage's starting weights, and a judge picks one; the stage's real training then uses the
judged reward from the same starting weights, and the sandbox weights are discarded. This transport answers the engine's
`decision` and `review` requests with that loop, one candidate per round; every other request (the reviewers panel's) goes to
`inner` unchanged. The engine, `recipes.resolve`, selection, delivery and the terminal audit are the core's.

    Round r (1-based), block = ITERATIONS x TRIALS + 1 = 7 rounds per stage:
      position 0 .. 5  sandbox trial (iteration position // 3 + 1, trial position % 3): SANDBOX_STEPS updates, parent = the stage's
                       starting candidate (the initial policy in stage 1), reward = the designer's, approved by the critic
      position 6       the stage's real candidate: REAL_STEPS updates, the same parent, reward = the judge's pick among the last
                       iteration's three trials; it is the starting candidate of the next stage
      after round 21   stop; the delivered candidate is the best real candidate by the declared ranking

The designer's action space is the reward family and its weights over the enabled signals (`macamp_prompts.DESIGNER`); no training
setting other than `steps` is ever written. Every agent call is appended to `run_dir/macamp_trace.jsonl` with the hash and length of
its prompt, the markers of what it contained and its answer, which `scripts/analysis/macamp_trace_check.py` reads.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import macamp_prompts as prompts
from . import recipes as rc
from .arms import Arm, ArmTransport, call
from .errors import ControlError
from .projection import evidence_of
from .schemas import assert_strict_compatible, object_schema
from .selection import best_candidate
from .transport import Turn, Usage

STAGES, ITERATIONS, TRIALS = 3, 2, 3
BLOCK = ITERATIONS * TRIALS + 1
ROUNDS = STAGES * BLOCK
MAX_DIALOGUE = 4
DEFAULTS = {"sandbox_steps": 128, "real_steps": 384, "signals": {"Sa": "activity_ecoli", "Sb": "amp_likelihood", "St": "nontoxic"},
            "aggregators": ["weighted_sum", "trimmed_mean", "weighted_geometric_mean", "weighted_harmonic_mean", "soft_minimum"]}
REWARD_FIELDS = ("enabled_oracles", "aggregator")

CRITIC_SCHEMA = object_schema({
    "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
    "comments": {"type": "string", "description": "what to change, at the level of family, weights and parameters; at most 80 words"}})
JUDGE_SCHEMA = object_schema({
    "pick": {"type": "integer", "enum": [1, 2, 3]},
    "reason": {"type": "string", "description": "one sentence"}})
NO_CRITIC = {"concerns": [], "requested_evidence": [], "rationale": "the port's critic is the designer dialogue"}


@dataclass(frozen=True)
class Slot:
    round: int
    stage: int
    kind: str          # "sandbox" or "real"
    iteration: int     # 1 .. ITERATIONS for a sandbox trial, 0 for the real candidate
    trial: int         # 0 .. TRIALS - 1, 0 for the real candidate


def slot_of(round_index: int) -> Slot:
    if not 1 <= round_index <= ROUNDS:
        raise ControlError(f"the MAC-AMP port has rounds 1 to {ROUNDS}, got {round_index}")
    stage, position = divmod(round_index - 1, BLOCK)
    if position == BLOCK - 1:
        return Slot(round_index, stage + 1, "real", 0, 0)
    return Slot(round_index, stage + 1, "sandbox", position // TRIALS + 1, position % TRIALS)


def round_of(stage: int, iteration: int, trial: int) -> int:
    return (stage - 1) * BLOCK + (iteration - 1) * TRIALS + trial + 1


def real_round(stage: int) -> int:
    return stage * BLOCK


@dataclass(frozen=True)
class Settings:
    sandbox_steps: int
    real_steps: int
    signals: Mapping[str, str]
    aggregators: Sequence[str]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Settings":
        given = {**DEFAULTS, **dict(config.get("macamp") or {})}
        unknown = set(given) - set(DEFAULTS)
        if unknown:
            raise ControlError(f"config['macamp'] has the keys {sorted(unknown)}; it reads {sorted(DEFAULTS)}")
        return cls(int(given["sandbox_steps"]), int(given["real_steps"]), dict(given["signals"]), list(given["aggregators"]))


class _NoArm(Arm):
    name = "macamp"

    def candidate(self, ctx):  # never asked: the port builds its own proposals
        raise ControlError("the MAC-AMP port does not draw from an arm")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class MacAmpTransport:
    """Answers `decision` and `review` with the MAC-AMP loop; forwards every other request to `inner`."""

    def __init__(self, *, inner, config: Mapping[str, Any], run_dir):
        self.inner = inner
        self.settings = Settings.from_config(config)
        self.run_dir = Path(run_dir)
        self.model = inner.model
        self.engine = None
        self._helper = ArmTransport(arm=_NoArm(0), model="macamp-helper")
        self.trace_path = self.run_dir / "macamp_trace.jsonl"
        self.designs_path = self.run_dir / "macamp_designs.json"

    def bind(self, engine) -> "MacAmpTransport":
        self.engine = engine
        self._helper.bind(engine)
        return self

    # ----- the transport the engine calls ----------------------------------------------------------------------------

    def respond(self, input_items, *, instructions=None, schema=None, schema_name="response", tools=()):
        if schema_name == "review":
            return Turn(text=json.dumps(NO_CRITIC), parsed=dict(NO_CRITIC), tool_calls=(), items=(), usage=Usage(),
                        model=self.model, latency_s=0.0)
        if schema_name == "decision":
            if self.engine is None:
                raise ControlError("the MAC-AMP transport is not bound to an engine")
            return self._decide(schema)
        return self.inner.respond(input_items, instructions=instructions, schema=schema, schema_name=schema_name, tools=tools)

    # ----- the loop --------------------------------------------------------------------------------------------------

    def _decide(self, schema: Mapping[str, Any]) -> Turn:
        started = time.monotonic()
        state = self.engine.state()
        rnd = state.open_round
        if rnd.index > ROUNDS:
            parsed, usage = self._stop(state), Usage()
        else:
            slot = slot_of(rnd.index)
            retried = len(rnd.decision_ids) > 1
            if slot.kind == "sandbox":
                design, usage = self._sandbox_design(slot, state, rnd, schema, retried)
                steps = self.settings.sandbox_steps
                note = f"sandbox trial {slot.trial + 1} of iteration {slot.iteration}, stage {slot.stage}"
            else:
                design, usage = self._judged_design(slot, state)
                steps = self.settings.real_steps
                note = f"real training of stage {slot.stage} with the judged reward"
            parent = self._stage_start(state, slot.stage)
            candidate = {"parent": parent, "start": "from_initial" if parent == "initial" else "from_parent",
                         "hypothesis": f"MAC-AMP port: {note}", "steps": steps, "lr": None, "kl": None, "group_size": None,
                         "enabled_oracles": design["enabled_oracles"], "aggregator": design["aggregator"],
                         "oracle_utilities": None, "penalties": None, "gates": None}
            parsed = {"kind": "propose", "rationale": f"MAC-AMP port, round {rnd.index}: {note}", "candidates": [candidate],
                      "stop": None, "component": None}
            self._trace(role="decision", round=rnd.index, stage=slot.stage, kind=slot.kind, iteration=slot.iteration,
                        trial=slot.trial, parent=parent, steps=steps, retried=retried)
        return Turn(text=json.dumps(parsed), parsed=parsed, tool_calls=(), items=(), usage=usage, model=self.model,
                    latency_s=round(time.monotonic() - started, 3))

    def _stage_start(self, state, stage: int) -> str:
        if stage == 1:
            return "initial"
        return state.rounds[real_round(stage - 1) - 1].candidates[0]

    def _candidate_of(self, state, round_index: int) -> str | None:
        return state.rounds[round_index - 1].candidates[0] if round_index <= len(state.rounds) and state.rounds[round_index - 1].candidates else None

    # ----- designs ---------------------------------------------------------------------------------------------------

    def _load_designs(self) -> dict[str, Any]:
        return json.loads(self.designs_path.read_text()) if self.designs_path.exists() else {}

    def _store_designs(self, designs: Mapping[str, Any]) -> None:
        self.designs_path.parent.mkdir(parents=True, exist_ok=True)
        self.designs_path.write_text(json.dumps(designs, sort_keys=True))

    def _sandbox_design(self, slot: Slot, state, rnd, schema, retried: bool) -> tuple[dict[str, Any], Usage]:
        """The reward of this sandbox trial. The three rewards of an iteration are designed together, at its first trial, from the
        log as it stood before the iteration (MAC-AMP designs its three candidates from the same information); a retry of a trial
        the engine refused designs that trial again."""
        designs = self._load_designs()
        key = f"s{slot.stage}i{slot.iteration}"
        entries = designs.setdefault(key, {})
        usage = Usage()
        if retried:
            feedback = "the system refused the previous proposal of this slot; propose a different valid reward"
            entries[str(slot.trial)], spent = self._dialogue(slot, state, rnd, schema, feedback)
            usage += spent
        elif str(slot.trial) not in entries:
            for trial in range(slot.trial, TRIALS):
                shown = Slot(round_of(slot.stage, slot.iteration, 0), slot.stage, "sandbox", slot.iteration, trial)
                entries[str(trial)], spent = self._dialogue(shown, state, rnd, schema, "")
                usage += spent
        self._store_designs(designs)
        return entries[str(slot.trial)], usage

    def _judged_design(self, slot: Slot, state) -> tuple[dict[str, Any], Usage]:
        trials = [self._candidate_of(state, round_of(slot.stage, ITERATIONS, t)) for t in range(TRIALS)]
        if any(name is None for name in trials):
            raise ControlError(f"stage {slot.stage}: the sandbox trials of the last iteration are not all allocated")
        lines = []
        for index, name in enumerate(trials, 1):
            lines.append(f"{prompts.SIGNALS} trial {index}: {self._describe(state, name)}")
        prompt = prompts.judge_prompt("\n".join(lines))
        parsed, usage = self._ask(prompts.JUDGE, prompt, JUDGE_SCHEMA, "macamp_judge", role="judge", slot=slot)
        pick = int(parsed["pick"])
        design = self._load_designs()[f"s{slot.stage}i{ITERATIONS}"][str(pick - 1)]
        self._trace(role="judge_pick", round=slot.round, stage=slot.stage, pick=pick, reason=parsed.get("reason", ""))
        return design, usage

    def _dialogue(self, slot: Slot, state, rnd, schema, feedback: str) -> tuple[dict[str, Any], Usage]:
        """Designer and critic for one reward function: up to MAX_DIALOGUE turns, the critic seeing only the meta-review."""
        usage, dialogue, last_valid, last_text = Usage(), [], None, ""
        meta = self._meta_review(state, slot)
        for turn in range(1, MAX_DIALOGUE + 1):
            log = self._signals_log(state, slot)
            text = prompts.designer_prompt(slot.stage, slot.iteration, log, feedback)
            if dialogue:
                text += "\nDialogue so far:\n" + "\n".join(dialogue)
            parsed, spent = self._ask(prompts.DESIGNER, text, schema, "macamp_designer", role="designer", slot=slot, turn=turn)
            usage += spent
            proposed = (parsed.get("candidates") or [{}])[0]
            design = {key: proposed.get(key) for key in REWARD_FIELDS}
            summary = self._summary(design)
            problems = self._problems(design, slot, state, rnd, schema)
            if problems:
                feedback = "; ".join(problems)
                dialogue.append(f"[Proposal {turn}] {summary}\n[Validator] refused: {feedback}")
                self._trace(role="validator", round=slot.round, turn=turn, accepted=False, problems=problems)
                continue
            self._trace(role="validator", round=slot.round, turn=turn, accepted=True, problems=[])
            last_valid, last_text = design, summary
            if not meta:
                # No reviewer text exists before the first batch is evaluated; the critic reads only that text, so it is not asked.
                self._trace(role="critic_skipped", round=slot.round, turn=turn, reason="no meta-review yet")
                return design, usage
            verdict, spent = self._critique(meta, dialogue + [f"[Proposal {turn}] {summary}"], slot, turn)
            usage += spent
            if verdict["verdict"] == "PASS":
                return design, usage
            dialogue += [f"[Proposal {turn}] {summary}", f"[Critique {turn}] {verdict['comments']}"]
            feedback = ""
        if last_valid is None:
            raise ControlError(f"round {slot.round}: the designer produced no valid reward in {MAX_DIALOGUE} turns")
        self._trace(role="dialogue_end", round=slot.round, kept_unapproved=last_text)
        return last_valid, usage

    def _critique(self, meta: str, dialogue: Sequence[str], slot: Slot, turn: int) -> tuple[dict[str, Any], Usage]:
        return self._ask(prompts.CRITIC, prompts.critic_prompt(meta, "\n".join(dialogue)), CRITIC_SCHEMA, "macamp_critic",
                         role="critic", slot=slot, turn=turn)

    def _problems(self, design: Mapping[str, Any], slot: Slot, state, rnd, schema) -> list[str]:
        """The rule validator: the whitelist, then the engine's own `recipes.resolve` on the proposal as it will be asked."""
        out = []
        oracles = design.get("enabled_oracles") or []
        allowed = set(self.settings.signals.values())
        if not oracles or set(oracles) - allowed:
            out.append(f"enabled_oracles must be a non-empty subset of {sorted(allowed)}")
        if self.settings.signals["Sa"] not in oracles:
            out.append(f"enabled_oracles must contain {self.settings.signals['Sa']}")
        aggregator = design.get("aggregator") or {}
        if aggregator.get("name") not in self.settings.aggregators:
            out.append(f"aggregator must be one of {list(self.settings.aggregators)}")
        if out:
            return out
        parent = self._stage_start(state, slot.stage)
        candidate = {"parent": parent, "start": "from_initial" if parent == "initial" else "from_parent", "hypothesis": "probe",
                     "steps": self.settings.sandbox_steps, "lr": None, "kl": None, "group_size": None, **{k: design[k] for k in REWARD_FIELDS},
                     "oracle_utilities": None, "penalties": None, "gates": None}
        arm_ctx, parents, history, siblings = self._helper._context(schema, state, rnd)
        resolved, problems = rc.resolve(candidate, self.engine.s.recipe_ctx, parents, committed_updates=state.committed_updates,
                                        siblings=siblings, history=history)
        return [f"{p.code}: {p.detail}" for p in problems] if resolved is None else []

    # ----- what each agent is shown ----------------------------------------------------------------------------------

    def _rows(self, state, name: str) -> Mapping[str, Any]:
        return self.engine.blobs.get_json(state.candidates[name].ledger_row_sha)

    def _describe(self, state, name: str) -> str:
        row = self._rows(state, name)
        condition = list(self.engine.s.task.conditions)[0]
        pool = (row.get("pool") or {}).get(condition) or {}
        spec = self.engine.blobs.get_json(state.candidates[name].recipe_sha)["recipe"]
        sa, sb, st = (pool.get(self.settings.signals[key]) for key in ("Sa", "Sb", "St"))
        panel = self._panel_score(state, name)
        return (f"{name} reward {self._summary(spec)}: Sa={self._num(sa)} Sb={self._num(sb)} St={self._num(st)} "
                f"Sc={self._num(panel)}")

    @staticmethod
    def _num(value) -> str:
        return "n/a" if value is None else f"{float(value):.4f}"

    @staticmethod
    def _summary(design: Mapping[str, Any]) -> str:
        aggregator = design.get("aggregator") or {}
        if isinstance(aggregator, Mapping) and "params_json" in aggregator:
            params = aggregator["params_json"]
        else:
            params = json.dumps({k: v for k, v in aggregator.items() if k != "name"}, sort_keys=True)
        return f"{aggregator.get('name')} {params} over {sorted(design.get('enabled_oracles') or [])}"

    def _signals_log(self, state, slot: Slot) -> str:
        lines = []
        start = self._stage_start(state, slot.stage)
        if start != "initial":
            lines.append(f"{prompts.SIGNALS} stage start: {self._describe(state, start)}")
        first = round_of(slot.stage, 1, 0)
        for r in range(first, slot.round):
            name = self._candidate_of(state, r)
            if name is not None and state.candidates[name].ledger_row_sha:
                lines.append(f"{prompts.SIGNALS} {self._describe(state, name)}")
        return "\n".join(lines) if lines else f"{prompts.SIGNALS} (no earlier trial in this stage)"

    def _panel_score(self, state, name: str):
        record = ((state.ext.get("reviewers") or {}).get("panels") or {}).get(name)
        if not record:
            return None
        from .ext.reviewers import summarize
        found = summarize(record)
        return None if found is None else found["overall"]

    def _meta_review(self, state, slot: Slot) -> str:
        """The reviewers' text for the latest evaluated candidate: the chair's four lines and the meta scores; nothing else."""
        for r in range(slot.round - 1, 0, -1):
            name = self._candidate_of(state, r)
            record = ((state.ext.get("reviewers") or {}).get("panels") or {}).get(name or "")
            if record and record.get("chair"):
                from .ext.reviewers import DIMENSIONS, summarize
                found = summarize(record)
                lines = [f"{d}: {record['chair'][d]}" for d in DIMENSIONS]
                if found is not None:
                    lines.append("meta scores: " + ", ".join(f"{d} {found['dimensions'][d]['metacore']:+.3f}" for d in DIMENSIONS))
                return "\n".join(lines)
        return ""

    # ----- calls and trace -------------------------------------------------------------------------------------------

    def _ask(self, instructions: str, text: str, schema: Mapping[str, Any], name: str, *, role: str, slot: Slot,
             turn: int = 0) -> tuple[dict[str, Any], Usage]:
        if schema is CRITIC_SCHEMA or schema is JUDGE_SCHEMA:
            assert_strict_compatible(schema)
        answer = self.inner.respond([{"role": "user", "content": text}], instructions=instructions, schema=schema,
                                    schema_name=name, tools=())
        if answer.parsed is None:
            raise ControlError(f"the {role} turn ended without a structured answer")
        self._trace(role=role, round=slot.round, stage=slot.stage, iteration=slot.iteration, trial=slot.trial, turn=turn,
                    prompt_sha256=_sha(text), prompt_chars=len(text), instructions_sha256=_sha(instructions),
                    has_signals=prompts.SIGNALS in text, has_meta_review="Meta-review of the latest batch" in text,
                    answer=answer.parsed, usage={"input": answer.usage.input, "output": answer.usage.output})
        return answer.parsed, answer.usage

    def _trace(self, **record: Any) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    # ----- the stop --------------------------------------------------------------------------------------------------

    def _stop(self, state) -> dict[str, Any]:
        engine = self.engine
        conditions = list(engine.s.task.conditions)
        reals = {self._candidate_of(state, real_round(s)) for s in range(1, STAGES + 1)} - {None}
        evidence = [item for item in evidence_of(state, engine.blobs) if item.id in reals]
        pick = best_candidate(engine.s.success, evidence, conditions)
        base = {"kind": "stop", "rationale": "MAC-AMP port: three stages done", "candidates": [], "component": None}
        if pick is None:
            return {**base, "stop": {"outcome": "no_qualified_candidate", "assignment": [],
                                     "summary": "no real candidate qualifies in every condition"}}
        return {**base, "stop": {"outcome": "deliver", "summary": "best real candidate by the declared ranking",
                                 "assignment": [{"condition": c, "candidate": pick} for c in conditions]}}


def parse_aggregator(design: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(design["aggregator"]["params_json"])
