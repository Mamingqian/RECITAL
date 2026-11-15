"""Arm A of the Coder true test: the Strategist's instructions name a gap in the primitive catalog, as a fact.

    control.modules.gap_header: {text: "<paragraph>"}

The module appends `text` to the Strategist's instructions (`Extension.instructions`) and does nothing else: no schema field,
no event, no ledger block. The paragraph is written in the declaration of the arm; this module only checks that it is
non-empty, so that a declaration cannot name the module and forget the text. `guard.tripwire` reads the instructions with
the rest of the prompt.
"""
from __future__ import annotations

from typing import Any, Mapping

from ..errors import ControlError
from . import Extension


class GapHeader(Extension):
    name = "gap_header"

    def __init__(self, text: str):
        self.text = text.strip()

    def instructions(self) -> str:
        return self.text


def build(config: Mapping[str, Any]) -> GapHeader:
    unknown = sorted(set(config) - {"text"})
    text = config.get("text")
    if unknown or not isinstance(text, str) or not text.strip():
        raise ControlError("control.modules.gap_header takes exactly `text`, a non-empty paragraph")
    return GapHeader(text)
