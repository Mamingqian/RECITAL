"""A reviewer panel for a candidate's pool: three reviewers and a chair, scored as MAC-AMP's Area Chair scores a batch.

MAC-AMP (arXiv 2602.14926; github.com/CLMFAP/MAC-AMP_v1, MIT) has three LLM reviewers read each batch of peptides and an Area
Chair write a meta-review; the scalar that comes out is arithmetic on the reviewers' score tags
(`Generation_Module/utils/area_chair_calculate_helper.py:141-256`). That scalar is one number per batch, and the PPO advantage is
standardized over the batch (`ppo_trainer.py:119-122`), so it cancels as a reward term; the reviews act through the reward the
scientist writes next. This module therefore does not touch the reward. It makes the panel's verdict, with that arithmetic, part of
the ledger the Strategist reads.

This file holds the arithmetic (the part that must equal MAC-AMP's), the table the panel reads and the module; the lexicon, the
schemas and the prompts are in `reviewer_panel.py`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Literal, Mapping, Sequence

from .. import events as ev
from ..agents import run_critic
from ..errors import BlindingViolation, ControlError, LedgerCorruption, PromptBudgetExceeded, RunTokenBudgetExceeded
from ..projection import Block, render_standalone
from ..state import spend
from . import Extension, Phase, event, facts, folds
from . import reviewer_panel as panel
from .reviewer_panel import DIMENSIONS, peptide_properties

GAMMA_FLOOR = 0.6
PENALTY_SLOPE = 0.6


def clip_unit(value: float) -> float:
    return max(-1.0, min(1.0, value))


def disagreement_penalty(groups: Sequence[Sequence[int]]) -> float:
    """Mean over the chair's disagreement groups of 1 - |mean of the group's votes|; 0 with no group."""
    values = [1.0 - abs(sum(group) / len(group)) for group in groups if len(group)]
    return sum(values) / len(values) if values else 0.0


def dimension_metacore(scores: Sequence[float], groups: Sequence[Sequence[int]]) -> dict[str, float]:
    """One dimension's score: the mean of the responders' scores times gamma.

    With fewer than three responders there is no penalty (gamma 1); with three or more, gamma is
    `clip(1 - 0.6 x penalty, 0.6, 1)` from the chair's groups, as `compute_metacore_dimension` does.
    """
    responders = len(scores)
    meta = mean(scores) if responders else 0.0
    penalty = disagreement_penalty(groups) if responders >= 3 else 0.0
    gamma = max(GAMMA_FLOOR, min(1.0, 1.0 - PENALTY_SLOPE * penalty)) if responders >= 3 else 1.0
    return {"meta_score": meta, "penalty": penalty, "gamma": gamma, "metacore": gamma * meta}


def meta_score(scores: dict[str, Sequence[float]], groups: dict[str, Sequence[Sequence[int]]]) -> dict[str, object]:
    """The four dimensions' scores and their overall mean clipped to [-1, 1] (`compute_meta_score`)."""
    dimensions = {d: dimension_metacore(scores.get(d, ()), groups.get(d, ())) for d in DIMENSIONS}
    overall = clip_unit(sum(item["metacore"] for item in dimensions.values()) / len(DIMENSIONS))
    return {"overall": overall, "dimensions": dimensions}


# --- the table the panel reads ------------------------------------------------------------------------------------------

def longest_run(sequence: str) -> int:
    best = run = 0
    previous = ""
    for residue in sequence:
        run = run + 1 if residue == previous else 1
        previous = residue
        best = max(best, run)
    return best


def kmer_reuse(sequences: Sequence[str], k: int = 3) -> float:
    """Share of the library's k-mer occurrences that repeat a k-mer already seen in an earlier peptide (0 for no repeats)."""
    seen: set[str] = set()
    total = repeats = 0
    for sequence in sequences:
        grams = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]
        total += len(grams)
        repeats += sum(1 for gram in grams if gram in seen)
        seen.update(grams)
    return repeats / total if total else 0.0


def mean_pairwise_distance(sequences: Sequence[str]) -> float:
    """Mean normalized Levenshtein distance over all pairs (0 for fewer than two sequences)."""
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz.process import cdist

    if len(sequences) < 2:
        return 0.0
    matrix = cdist(sequences, sequences, scorer=Levenshtein.normalized_distance, workers=1)
    n = len(sequences)
    return float((matrix.sum()) / (n * (n - 1)))


def nearest_distances(sequences: Sequence[str], reference: Sequence[str]) -> list[float]:
    """For each sequence the smallest normalized Levenshtein distance to any reference sequence (1.0 without a reference)."""
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz.process import cdist

    if not reference or not sequences:
        return [1.0] * len(sequences)
    matrix = cdist(list(sequences), list(reference), scorer=Levenshtein.normalized_distance, workers=1)
    return [float(value) for value in matrix.min(axis=1)]


def length_band(length: int, bands: Sequence[int]) -> str:
    """`bands` are ascending upper edges; the band of a length is `<=edge` of the first edge it does not exceed."""
    for edge in bands:
        if length <= edge:
            return f"<={edge}"
    return f">{bands[-1]}"


def panel_table(sequences: Sequence[str], measurements: dict[str, Sequence[float]], selected: Sequence[int], *,
                reference: Sequence[str] = (), pool_sample: int = 200, length_bands: Sequence[int] = (10, 20, 32)) -> dict[str, object]:
    """The per-peptide rows of the delivered library and the library and pool statistics the reviewers are given.

    `sequences` and each list of `measurements` are in pool order; `selected` are the delivered rows. Every number is computed
    here from those inputs, so the reviewers are shown arithmetic and not asked to produce it. The pool's pairwise distance is
    taken over its first `pool_sample` rows.
    """
    chosen = [sequences[i] for i in selected]
    nearest = nearest_distances(chosen, reference)
    rows = []
    for position, index in enumerate(selected):
        sequence = sequences[index]
        row = {"sequence": sequence, "length": len(sequence), "length_band": length_band(len(sequence), length_bands),
               "longest_run": longest_run(sequence), "distance_to_nearest_known": round(nearest[position], 3)}
        row.update({name: round(float(values[index]), 4) for name, values in measurements.items() if values[index] is not None})
        row.update(peptide_properties(sequence))
        rows.append(row)
    library = {name: round(float(mean(kept)), 4) for name, values in measurements.items()
               if (kept := [values[i] for i in selected if values[i] is not None])}
    library.update({"mean_pairwise_distance": round(mean_pairwise_distance(chosen), 3),
                    "kmer3_reuse": round(kmer_reuse(chosen), 3), "mean_distance_to_nearest_known": round(mean(nearest), 3),
                    "mean_longest_run": round(mean(row["longest_run"] for row in rows), 2)})
    pool = {"peptides": len(sequences), "mean_pairwise_distance": round(mean_pairwise_distance(list(sequences[:pool_sample])), 3)}
    for name, values in measurements.items():
        finite = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
        if finite:
            pool[f"median_{name}"] = round(finite[len(finite) // 2], 4)
    return {"rows": rows, "library": library, "pool": pool}


# ----- the module ------------------------------------------------------------------------------------------------------------

PANEL_SIZE = 3
ATTEMPTS = 2
MEASUREMENTS = {"activity": "activity_ecoli", "amp_likelihood": "amp_likelihood", "nontoxic": "nontoxic",
                "nonhemolytic": "nonhemolytic", "plddt": "plddt_esmfold"}
SETTINGS = {"known_peptides": None, "sequence_column": "sequence", "pool_sample": 200, "curriculum": True,
            "measurements": MEASUREMENTS, "judge": {"primary": "amp_selection_score", "secondary": "amp_likelihood_mean"}}
INSTRUCTIONS = (
    "Reviewer panel. After each candidate's pool is scored, three reviewers and a chair read its delivered library with computed "
    "columns and tag it in four dimensions: EFF (efficacy), Safe (safety), DevStruct (developability and structure confidence) "
    "and Orig (originality). Code turns the tags into a panel score in [-1, 1]: per dimension the mean of the reviewers' scores "
    "times a factor between 0.6 and 1 that falls as they disagree, overall the mean of the four dimensions. The ledger shows that "
    "score, its four dimensions and the chair's comment beside each row. The panel score is not a measurement of the pool and is "
    "not in the reward, and it does not change which library is delivered, which follows the task's ranking. As advice for "
    "choosing the parent of a round, you may order candidates by {primary}; among those within 0.01 of the best, by {secondary}; "
    "then by the panel score.")
CURRICULUM = (
    "Curriculum for this round: raise the AMP likelihood of the pool first, before the activity of the delivered library; adjust "
    "if the ledger shows AMP likelihood already saturated or activity lagging.",
    "Curriculum for this round: balance the AMP likelihood of the pool against the activity of the delivered library.",
    "Curriculum for this round: focus on the activity of the delivered library while the qualification measurements hold.")


@event
class Reviewed(ev._Event, ev.LlmCall):
    """One reviewer's answer for one candidate: the tags code kept and dropped, and the score code computed from them."""

    type: Literal["reviewers.reviewed"] = "reviewers.reviewed"
    candidate: str
    reviewer: int
    dimensions: dict[str, Any]


@event
class Chaired(ev._Event, ev.LlmCall):
    """The chair's four meta-comments for one candidate."""

    type: Literal["reviewers.chaired"] = "reviewers.chaired"
    candidate: str
    comments: dict[str, str]


@event
class Unavailable(ev._Event):
    """A panel call that failed twice, or a candidate with no delivered library; the search goes on without it."""

    type: Literal["reviewers.unavailable"] = "reviewers.unavailable"
    candidate: str
    role: str
    cause: str


def _panel(state, candidate: str) -> dict[str, Any]:
    return facts(state, "reviewers").setdefault("panels", {}).setdefault(candidate, {"reviews": {}, "chair": None, "failed": []})


@folds(Reviewed)
def _fold_reviewed(state, event):
    spend(state, event)
    _panel(state, event.candidate)["reviews"][str(event.reviewer)] = event.dimensions


@folds(Chaired)
def _fold_chaired(state, event):
    spend(state, event)
    _panel(state, event.candidate)["chair"] = event.comments


@folds(Unavailable)
def _fold_unavailable(state, event):
    _panel(state, event.candidate)["failed"].append(event.role)


def summarize(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """The panel score of a candidate from the reviews on record (`None` with none): groups come from the kept tags."""
    reviews = [record["reviews"][key] for key in sorted(record["reviews"])]
    if not reviews:
        return None
    scores = {d: [review[d]["score"] for review in reviews] for d in DIMENSIONS}
    groups = {d: panel.disagreement_groups(d, [review[d]["kept"] for review in reviews]) for d in DIMENSIONS}
    return {**meta_score(scores, groups), "reviewers": len(reviews), "groups": groups}


def known_peptides(path: Path, column: str) -> list[str]:
    import csv

    with open(path, newline="") as handle:
        return sorted({row[column].upper() for row in csv.DictReader(handle) if row[column].isalpha()})


def _signed(value: float) -> str:
    return f"{value:+.2f}"


class Reviewers(Extension):
    """The closer MAC-AMP arm's panel: tags in four dimensions per candidate, scored by code, shown in the ledger.

    Every hook reads `state.ext` without creating anything in it, so a run that has not reached the module's first event has the
    state, and the state hash, it would have without the module.
    """

    name = "reviewers"

    def __init__(self, config: Mapping[str, Any]):
        unknown = sorted(set(config) - set(SETTINGS))
        if unknown:
            raise ControlError(f"control.modules.reviewers has the keys {unknown}; it reads {sorted(SETTINGS)}")
        self.settings = {**SETTINGS, **config}
        if not self.settings["known_peptides"]:
            raise ControlError("control.modules.reviewers needs `known_peptides`, the table of peptides the reviewers measure novelty against")
        self._known: list[str] | None = None
        self._tables: dict[str, Any] = {}

    # ----- what the Strategist is told ----------------------------------------------------------------------------------

    def instructions(self) -> str:
        return INSTRUCTIONS.format(**self.settings["judge"])

    def tail_lines(self, state, store, arm):
        if not self.settings["curriculum"]:
            return ()
        upcoming = sum(1 for r in state.rounds if r.status == "allocated") + 1
        return (CURRICULUM[min(upcoming, len(CURRICULUM)) - 1],)

    def round_blocks(self, state, store, arm, rnd):
        panels = (state.ext.get("reviewers") or {}).get("panels") or {}
        scrub = getattr(arm, "scrub", lambda text: text)
        shown, system = [], []
        for name in rnd.candidates:
            record = panels.get(name)
            if record is None:
                continue
            found = summarize(record)
            if found is None:
                head = f"[{name}] panel unavailable"
                shown.append(head)
                system.append(head)
                continue
            head = (f"[{name}] panel {_signed(found['overall'])} (" + ", ".join(
                f"{d} {_signed(found['dimensions'][d]['metacore'])}" for d in DIMENSIONS) + f"; {found['reviewers']} reviewers)")
            system.append(head)
            chair = record.get("chair")
            shown.append(head + ("" if not chair else " | " + " | ".join(f"{d}: {scrub(chair[d])}" for d in DIMENSIONS)))
        return [Block("\n".join(shown), "\n".join(system))] if shown else []

    # ----- the phase ------------------------------------------------------------------------------------------------------

    def phases(self) -> Sequence[Phase]:
        return [Phase("reviewers_panel", "rows", self._run)]

    def _pending(self, record: Mapping[str, Any]) -> str | None:
        failed = set(record.get("failed", ()))
        if "panel" in failed:
            return None
        for index in range(PANEL_SIZE):
            if str(index) not in record["reviews"] and f"reviewer{index}" not in failed:
                return f"reviewer{index}"
        if record["reviews"] and record["chair"] is None and "chair" not in failed:
            return "chair"
        return None

    def _run(self, engine, state) -> bool:
        panels = (state.ext.get("reviewers") or {}).get("panels") or {}
        for candidate in state.ordered_candidates():
            if candidate.status != "ok" or candidate.ledger_row_sha is None:
                continue
            record = panels.get(candidate.id) or {"reviews": {}, "chair": None, "failed": []}
            role = self._pending(record)
            if role is None:
                continue
            table = self._table(engine, candidate.id)
            if table is None:
                engine.emit(Unavailable(candidate=candidate.id, role="panel", cause="the candidate delivered no peptides"))
                return True
            return self._call(engine, candidate.id, role, table, record)
        return False

    def _reference(self, engine) -> list[str]:
        if self._known is None:
            path = Path(self.settings["known_peptides"])
            self._known = known_peptides(path if path.is_absolute() else Path(engine.s.repo_root) / path,
                                         self.settings["sequence_column"])
        return self._known

    def _table(self, engine, candidate: str) -> dict[str, Any] | None:
        if candidate not in self._tables:
            directory = engine.candidate_dir(candidate)
            pool = json.loads((directory / "pool.json").read_text())
            measured = json.loads((directory / "public_measurements.json").read_text())["measurements"]
            selected = (engine.result(candidate).model_dump().get("delivery") or {}).get("selected_indices") or []
            columns = {logical: [None if v is None else float(v) for v in measured[name]["values"]]
                       for logical, name in self.settings["measurements"].items() if name in measured}
            self._tables[candidate] = None if not selected else panel_table(
                pool["sequences"], columns, selected, reference=self._reference(engine),
                pool_sample=self.settings["pool_sample"])
        return self._tables[candidate]

    def _review(self, answer: Mapping[str, Any]) -> dict[str, Any]:
        found = {}
        for dimension in DIMENSIONS:
            item = answer[dimension]
            score, kept, dropped = panel.tag_score(dimension, item["tags"])
            found[dimension] = {"comment": panel.one_line(item["comment"], panel.REVIEW_CHARS), "kept": kept, "dropped": dropped,
                                "score": round(score, 4)}
        return found

    def _call(self, engine, candidate: str, role: str, table: Mapping[str, Any], record: Mapping[str, Any]) -> bool:
        from ..engine import llm_fields

        if role == "chair":
            reviews = [record["reviews"][key] for key in sorted(record["reviews"])]
            dimensions = {d: {"reviewer_comments": [engine.arm.scrub(review[d]["comment"]) for review in reviews],
                              "tags": panel.aggregated_tags(d, [review[d]["kept"] for review in reviews]),
                              "polarity_groups": panel.disagreement_groups(d, [review[d]["kept"] for review in reviews])}
                          for d in DIMENSIONS}
            instructions, payload, schema, name = panel.CHAIR_INSTRUCTIONS, panel.chair_payload(dimensions), panel.chair_schema(), "chair"
            system = ()
        else:
            index = int(role.removeprefix("reviewer"))
            instructions = panel.reviewer_instructions(index, len(table["rows"]))
            payload, schema, name = panel.reviewer_payload(table), panel.reviewer_schema(), "reviewer"
            system = (payload,)
        failure: Exception | None = None
        for _ in range(ATTEMPTS):
            try:
                rendered = render_standalone(engine.blobs, arm=engine.arm, instructions=instructions, payload=payload, schema=schema,
                                             limit=engine.s.control.llm.max_prompt_tokens, system_texts=system)
                turn = run_critic(engine.transport, instructions=rendered.instructions, input_items=rendered.input_items,
                                  schema=schema, schema_name=name)
                if role == "chair":
                    comments = {d: panel.one_line(turn.parsed[d], panel.CHAIR_CHARS) for d in DIMENSIONS}
                else:
                    dimensions_found = self._review(turn.parsed)
            except Exception as exc:  # an advisory call must not stop the search
                partial = getattr(exc, "partial_usage", None)
                if partial is not None:
                    engine.meter.add(partial)  # tokens a failed call still cost count against the run's cap
                if isinstance(exc, (BlindingViolation, PromptBudgetExceeded, RunTokenBudgetExceeded, LedgerCorruption)):
                    raise
                failure = exc
                continue
            fields = llm_fields(turn, rendered, engine.blobs.put_json(schema), engine.blobs.put_json(turn.parsed), None)
            if role == "chair":
                engine.emit(Chaired(candidate=candidate, comments=comments, **fields))
            else:
                engine.emit(Reviewed(candidate=candidate, reviewer=index, dimensions=dimensions_found, **fields))
            engine.count(turn)
            return True
        engine.emit(Unavailable(candidate=candidate, role=role, cause=f"{type(failure).__name__}: {failure}"[:200]))
        return True


def build(config: Mapping[str, Any]) -> Reviewers:
    return Reviewers(config)
