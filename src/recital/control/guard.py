"""Checks that run on text before it leaves the process: token budget and arm blinding."""
from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping, Sequence

from .errors import BlindingViolation, PromptBudgetExceeded

ENCODING = "o200k_base"
# The estimator over-counts a little so the cap trips before the endpoint's own limit does.
ESTIMATE_MARGIN = 1.10
HIDDEN_PLACEHOLDER = "<hidden-source>"

_encoder = None


def _encoding():
    global _encoder
    if _encoder is None:
        import tiktoken

        _encoder = tiktoken.get_encoding(ENCODING)
    return _encoder


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def estimate_tokens(texts: Iterable[str], *, schema: Mapping[str, Any] | None = None,
                    tools: Sequence[Any] = ()) -> int:
    """Upper-biased token count of everything one request sends."""
    total = sum(count_tokens(text) for text in texts)
    if schema is not None:
        total += count_tokens(json.dumps(schema, sort_keys=True))
    for tool in tools:
        total += count_tokens(json.dumps(tool.as_request(), sort_keys=True))
    return int(total * ESTIMATE_MARGIN) + 1


def check_prompt_budget(texts: Iterable[str], *, limit: int, schema: Mapping[str, Any] | None = None,
                        tools: Sequence[Any] = ()) -> int:
    estimate = estimate_tokens(texts, schema=schema, tools=tools)
    if estimate > limit:
        raise PromptBudgetExceeded(f"prompt estimated at {estimate} tokens exceeds the {limit}-token cap")
    return estimate


def _name_pattern(name: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", re.IGNORECASE)


# Sentinel-concept words are concepts, not identifiers: `spendable_sentinel` and `held_out_oracles` contain them, so only
# letters bound a match here, while a source name such as `enformer_mpra` is bounded by letters, digits and `_`.
# `flag`, `flagged` and `uncalibrated` are the words of a sentinel's flag line; no header, instruction
# or ledger row of the three tasks contains the word `flag`, and `flags` is not matched.
BLIND_PATTERN = re.compile(
    r"(?<![a-z])(?:held[_\- ]?out|self[_\- ]validat\w*|sentinel|no[_\- ]self[_\- ]validat\w*"
    r"|flag(?:ged)?|uncalibrated)(?![a-z])", re.IGNORECASE)


def mentions_blind_word(text: str) -> bool:
    return BLIND_PATTERN.search(text) is not None


def blind_projection(value: Any) -> Any:
    """A copy of `value` with every key and sentence that names a sentinel concept removed.

    Contract text that describes a sentinel measurement ("Initially held out and eligible for reward promotion"), a
    `held_out_oracles` field, a role id such as `spendable_sentinel`: this arm measures no sentinel, so none of it
    is shown to the agent.
    """
    if isinstance(value, Mapping):
        return {key: blind_projection(item) for key, item in value.items()
                if not (isinstance(key, str) and mentions_blind_word(key))}
    if isinstance(value, (list, tuple)):
        return [blind_projection(item) for item in value]
    if isinstance(value, str) and mentions_blind_word(value):
        sentences = re.split(r"(?<=[.;])\s+", value)
        return " ".join(sentence for sentence in sentences if not mentions_blind_word(sentence)).strip()
    return value


def scrub(text: str, hidden_names: Iterable[str]) -> str:
    """Replace every hidden source name in worker-originated text."""
    for name in sorted(set(hidden_names), key=len, reverse=True):
        text = _name_pattern(name).sub(HIDDEN_PLACEHOLDER, text)
    return text


def tripwire(*, system_texts: Iterable[str], hidden_names: Iterable[str]) -> None:
    """Raise BlindingViolation when text the system wrote names a hidden source.

    A hidden source is a terminal evaluator or a source the arm does not measure; naming one would leak the score
    the delivered libraries are judged by. Words such as "held-out" or "sentinel" in a source note are not a leak and
    do not stop a run: `blind_projection` leaves out the sentences that describe measurements this arm does not make.
    Agent-authored text is not scanned: an agent that writes a public model's name in its own hypothesis has leaked
    nothing, and a frozen row cannot be edited to remove it. Callers pass for `system_texts` only text the system
    produced, with each row's hypothesis taken out.
    """
    hidden = [(name, _name_pattern(name)) for name in sorted(set(hidden_names))]
    for text in system_texts:
        for name, pattern in hidden:
            if pattern.search(text):
                raise BlindingViolation(f"outgoing text names hidden source {name!r}")
