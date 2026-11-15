"""M1's Analyst turn: at most three lessons a round, written by the model and kept only where the rows reproduce them.

The Analyst reads the ledger the Strategist reads. Its answer is a typed list of contrastive lessons, each naming an
edit class, the context the edit started from, a measurement, a condition and a direction, and citing the candidates that
show it. `claims.validate_lesson` recomputes every citation from the frozen rows and the module keeps only the lessons
it reproduces, so the ledger never carries a sentence the rows do not support. This file holds the turn's schema, its
instructions and the call; the phase and the event are in `claims.py`.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..agents import AgentTurn, run_critic
from ..schemas import name_schema, object_schema
from ..transport import ResponsesTransport

MAX_LESSONS = 3
MAX_CITED = 8
ANALYST_INSTRUCTIONS = (
    "You are the Analyst of a search over reward and training recipes. You read the ledger and write at most "
    f"{MAX_LESSONS} lessons, each a contrastive statement: edit class E, from a row in context C, moved measurement M "
    "in condition K in direction D, with the candidates that show it. Code recomputes every cited difference from the "
    "rows and keeps only the lessons it reproduces, so cite only candidates whose rows you have checked: each must carry "
    "the edit class, start from a row in that context and move the measurement by two standard errors or more. Prefer a "
    "lesson that separates edit classes the candidates did not share, or that corrects an assumption the claims made. "
    "The note says in a few words what to try or avoid. You advise and do not decide.")


def analyst_schema(*, edit_classes: Sequence[str], contexts: Sequence[str], measurements: Sequence[str],
                   conditions: Sequence[str], candidates: Sequence[str], note_words: int) -> dict[str, Any]:
    """The Analyst's answer in the strict structured-output subset; every name it may use is an enum."""
    lesson = object_schema({
        "edit_class": name_schema(edit_classes, "an edit class of a candidate in the ledger, lever:direction"),
        "context": name_schema(contexts, "the state of the row the edit started from"),
        "measurement": name_schema(measurements, "a measurement of the pool line, or qualified"),
        "condition": name_schema(conditions, "the condition the effect is in"),
        "direction": {"type": "string", "enum": ["up", "down"], "description": "how the measurement moved"},
        "cited": {"type": "array", "maxItems": MAX_CITED, "items": name_schema(candidates, "a candidate that shows it"),
                  "description": "at least two candidates"},
        "note": {"type": "string", "description": f"at most {note_words} words"},
    })
    return object_schema({"lessons": {"type": "array", "maxItems": MAX_LESSONS, "items": lesson,
                                "description": f"at most {MAX_LESSONS} lessons; empty when the ledger supports none"}})


def run_analyst(transport: ResponsesTransport, *, instructions: str, input_items: Sequence[Mapping[str, Any]],
                schema: Mapping[str, Any]) -> AgentTurn:
    return run_critic(transport, instructions=instructions, input_items=input_items, schema=schema,
                      schema_name="analyst")
