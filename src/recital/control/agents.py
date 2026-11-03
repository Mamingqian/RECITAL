"""The Strategist and Critic turns over the Responses transport.

A Strategist turn may call retrieval tools a bounded number of times and always ends in one strict
structured answer; a Critic turn is a single structured call. Neither validates the answer's meaning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .errors import ControlError
from .transport import ResponsesTransport, ToolSpec, Usage, tool_result_item


class ToolRejected(Exception):
    """A retrieval request the executor refuses; its message goes back to the model as the tool result."""


ToolExecutor = Callable[[str, Mapping[str, Any]], Any]


@dataclass(frozen=True)
class AgentTurn:
    """Everything one decision call left behind."""

    parsed: dict[str, Any]
    transcript: tuple[dict[str, Any], ...]  # model output items and tool results, in order
    usage: Usage
    model: str
    resolved_model: str
    latency_s: float


def run_strategist(
    transport: ResponsesTransport,
    *,
    instructions: str,
    input_items: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
    tools: Sequence[ToolSpec],
    execute_tool: ToolExecutor,
    max_tool_calls: int,
) -> AgentTurn:
    """Let the model retrieve up to `max_tool_calls` times, then require the strict final answer."""
    items = [dict(item) for item in input_items]
    transcript: list[dict[str, Any]] = []
    usage, latency, used = Usage(), 0.0, 0
    resolved = transport.model
    while True:
        offer = tools if used < max_tool_calls else ()
        try:
            turn = transport.respond(items, instructions=instructions, schema=schema,
                                     schema_name="decision", tools=offer)
        except ControlError as exc:
            exc.partial_usage = usage + getattr(exc, "usage", Usage())
            raise
        usage, latency, resolved = usage + turn.usage, latency + turn.latency_s, turn.model
        if turn.tool_calls and not offer:
            raise ControlError("the model called a tool after tools were withdrawn")
        if not turn.tool_calls:
            if turn.parsed is None:
                raise ControlError("the decision turn ended without a structured answer")
            return AgentTurn(turn.parsed, tuple(transcript), usage, transport.model, resolved, round(latency, 3))
        items.extend(turn.items)
        transcript.extend(turn.items)
        for call in turn.tool_calls:
            if used >= max_tool_calls:
                # One model turn may request several calls at once; the budget is per decision, not per turn.
                result = {"error": f"tool-call budget of {max_tool_calls} for this decision is spent"}
            else:
                used += 1
                try:
                    result = execute_tool(call.name, call.arguments)
                except ToolRejected as exc:
                    result = {"error": str(exc)}
            output = tool_result_item(call, result)
            items.append(output)
            transcript.append(output)


def run_critic(
    transport: ResponsesTransport,
    *,
    instructions: str,
    input_items: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
) -> AgentTurn:
    turn = transport.respond([dict(item) for item in input_items], instructions=instructions,
                             schema=schema, schema_name="review")
    if turn.parsed is None:
        raise ControlError("the critic turn ended without a structured answer")
    return AgentTurn(turn.parsed, (), turn.usage, transport.model, turn.model, turn.latency_s)
