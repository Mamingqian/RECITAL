"""Responses transport: strict structured output, native function tools, token accounting.

The endpoint is the OpenAI API unless the config declares another provider's OpenAI-compatible Responses endpoint
(`llm.base_url`) and the file that holds its key (`llm.api_key_file`). The strict schemas are built with every nullable
field written as a type list (`["array", "null"]`); `llm.schema_dialect` names the form an endpoint's strict compiler
accepts, and the transport writes the schemas of one request in that form just before it sends them.

One request carries the static instructions, the input items, optional function tools and an
optional strict json_schema output format. A turn that ends in function calls returns them with
the raw output items so the caller can send those items back together with the tool results.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .errors import ControlError, RunTokenBudgetExceeded


class ResponseIncomplete(ControlError):
    """The model stopped before finishing (output budget exhausted, content filter)."""


class ResponseRefused(ControlError):
    """The model refused instead of answering."""


class ResponseUnparsable(ControlError):
    """A strict-format response did not parse as JSON."""


class TransportUnavailable(ControlError):
    """The endpoint stayed unreachable after the retry budget."""


@dataclass(frozen=True)
class Usage:
    input: int = 0
    cached: int = 0
    output: int = 0
    reasoning: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.input + other.input, self.cached + other.cached,
                     self.output + other.output, self.reasoning + other.reasoning)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Mapping[str, Any]

    def as_request(self) -> dict[str, Any]:
        return {"type": "function", "name": self.name, "description": self.description,
                "strict": True, "parameters": dict(self.parameters)}


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Turn:
    text: str
    parsed: dict[str, Any] | None
    tool_calls: tuple[ToolCall, ...]
    items: tuple[dict[str, Any], ...]
    usage: Usage
    model: str
    latency_s: float


@dataclass(frozen=True)
class Price:
    """USD per million tokens. Cached input is billed at `cached`, the rest of the input at `input`."""
    input: float
    cached: float
    output: float

    def cost(self, usage: Usage) -> float:
        fresh = max(usage.input - usage.cached, 0)
        return (fresh * self.input + usage.cached * self.cached + usage.output * self.output) / 1e6


def load_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    raise ControlError("OPENAI_API_KEY is not set in the environment")


def read_key_file(path: str | Path) -> str:
    """The API key in the file `llm.api_key_file` names: one key, surrounding whitespace removed.

    The errors name the file and never what it holds.
    """
    file = Path(path).expanduser()
    try:
        text = file.read_text()
    except (OSError, UnicodeError) as exc:
        raise ControlError(f"cannot read the API key file {file}: {type(exc).__name__}") from None
    key = text.strip()
    if not key or len(key.split()) != 1:
        raise ControlError(f"the API key file {file} must hold exactly one key")
    return key


def anyof_nullable(schema: Mapping[str, Any]) -> dict[str, Any]:
    """`schema` with every nullable array or object written as `anyOf: [that schema, {"type": "null"}]`.

    The description stays on the `anyOf`; the other keys stay on the non-null branch. Nullable scalars keep their type
    list, and nothing else changes. A schema without a nullable array or object comes back equal.
    """
    out = dict(schema)
    if isinstance(schema.get("properties"), Mapping):
        out["properties"] = {name: anyof_nullable(child) for name, child in schema["properties"].items()}
    if isinstance(schema.get("items"), Mapping):
        out["items"] = anyof_nullable(schema["items"])
    kind = schema.get("type")
    if isinstance(kind, list) and "null" in kind:
        base = [name for name in kind if name != "null"]
        if len(base) == 1 and base[0] in ("array", "object"):
            branch = {key: value for key, value in out.items() if key not in ("type", "description")}
            branch["type"] = base[0]
            wrapped: dict[str, Any] = {"anyOf": [branch, {"type": "null"}]}
            if "description" in out:
                wrapped["description"] = out["description"]
            return wrapped
    return out


# How a request's strict schemas are written for each `llm.schema_dialect`; the first is the form schemas.py builds.
SCHEMA_DIALECTS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "type_list_nullable": dict,
    "anyof_nullable": anyof_nullable,
}


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, ControlError):
        return False
    try:
        import openai
    except ImportError:  # pragma: no cover - the client is a hard dependency of real runs
        return False
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError,
                        openai.RateLimitError, openai.InternalServerError)):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and (status in (408, 429) or status >= 500)


class UsageMeter:
    """Run-wide token total with a hard cap; the cap is operational and never shown to an agent."""

    def __init__(self, max_total_tokens: int | None = None):
        self.max_total_tokens = max_total_tokens
        self.total = Usage()

    def add(self, usage: Usage) -> None:
        self.total = self.total + usage
        if self.max_total_tokens is not None and self.total.input + self.total.output > self.max_total_tokens:
            raise RunTokenBudgetExceeded(
                f"run used {self.total.input + self.total.output} tokens, cap {self.max_total_tokens}")


@dataclass
class ResponsesTransport:
    model: str
    max_output_tokens: int = 16000
    base_url: str | None = None
    api_key_file: str | None = None
    schema_dialect: str = "type_list_nullable"
    timeout_s: float = 300.0
    max_attempts: int = 4
    backoff_s: float = 2.0
    client: Any = None
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False)

    def __post_init__(self) -> None:
        if self.schema_dialect not in SCHEMA_DIALECTS:
            raise ControlError(f"unknown schema dialect {self.schema_dialect!r}; known: {sorted(SCHEMA_DIALECTS)}")
        if self.client is None:
            import openai
            key = load_api_key() if self.api_key_file is None else read_key_file(self.api_key_file)
            self.client = openai.OpenAI(api_key=key, base_url=self.base_url, timeout=self.timeout_s, max_retries=0)

    def respond(
        self,
        input_items: Sequence[Mapping[str, Any]] | str,
        *,
        instructions: str | None = None,
        schema: Mapping[str, Any] | None = None,
        schema_name: str = "response",
        tools: Sequence[ToolSpec] = (),
    ) -> Turn:
        request: dict[str, Any] = {
            "model": self.model,
            "input": input_items if isinstance(input_items, str) else [dict(item) for item in input_items],
            "max_output_tokens": self.max_output_tokens,
        }
        if instructions:
            request["instructions"] = instructions
        written = SCHEMA_DIALECTS[self.schema_dialect]
        if schema is not None:
            request["text"] = {"format": {"type": "json_schema", "name": schema_name,
                                          "schema": written(schema), "strict": True}}
        if tools:
            request["tools"] = [{**tool.as_request(), "parameters": written(tool.parameters)} for tool in tools]
        started = time.monotonic()
        response = self._create_with_retry(request)
        latency = time.monotonic() - started
        return self._turn(response, schema is not None, latency)

    def _create_with_retry(self, request: dict[str, Any]) -> Any:
        last: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.client.responses.create(**request)
            except BaseException as exc:  # classified below; non-retryable errors propagate unchanged
                if not _retryable(exc):
                    raise
                last = exc
                if attempt < self.max_attempts:
                    self.sleep(self.backoff_s * 2 ** (attempt - 1))
        raise TransportUnavailable(
            f"{self.model} unreachable after {self.max_attempts} attempts: {type(last).__name__}: {last}")

    def _turn(self, response: Any, structured: bool, latency: float) -> Turn:
        status = getattr(response, "status", "completed")
        if status == "incomplete":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
            error = ResponseIncomplete(
                f"response incomplete ({reason}); output budget {self.max_output_tokens} "
                f"includes reasoning tokens")
            error.usage = _usage(response.usage)
            raise error
        if status == "failed":
            raise ControlError(f"response failed: {getattr(response, 'error', None)}")
        items = tuple(_dump(item) for item in response.output)
        for item in items:
            if item.get("type") == "message":
                for part in item.get("content") or ():
                    if part.get("type") == "refusal":
                        raise ResponseRefused(str(part.get("refusal"))[:300])
        calls = tuple(
            ToolCall(item["call_id"], item["name"], _arguments(item))
            for item in items if item.get("type") == "function_call")
        text = response.output_text or ""
        parsed = None
        if structured and not calls:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ResponseUnparsable(f"strict-format response is not JSON: {exc}; head={text[:200]!r}") from exc
            if not isinstance(parsed, dict):
                raise ResponseUnparsable(f"strict-format response is {type(parsed).__name__}, not an object")
        return Turn(text=text, parsed=parsed, tool_calls=calls, items=items,
                    usage=_usage(response.usage), model=response.model, latency_s=round(latency, 3))


def _dump(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    return item.model_dump(exclude_none=True)


def _arguments(item: Mapping[str, Any]) -> dict[str, Any]:
    raw = item.get("arguments") or "{}"
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ResponseUnparsable(f"tool call {item.get('name')} has non-JSON arguments: {exc}") from exc
    if not isinstance(value, dict):
        raise ResponseUnparsable(f"tool call {item.get('name')} arguments are not an object")
    return value


def _usage(raw: Any) -> Usage:
    if raw is None:
        return Usage()
    input_details = getattr(raw, "input_tokens_details", None)
    output_details = getattr(raw, "output_tokens_details", None)
    return Usage(
        input=int(getattr(raw, "input_tokens", 0) or 0),
        cached=int(getattr(input_details, "cached_tokens", 0) or 0),
        output=int(getattr(raw, "output_tokens", 0) or 0),
        reasoning=int(getattr(output_details, "reasoning_tokens", 0) or 0),
    )


def tool_result_item(call: ToolCall, result: Any) -> dict[str, Any]:
    """The input item that returns a tool result to the model."""
    return {"type": "function_call_output", "call_id": call.call_id,
            "output": result if isinstance(result, str) else json.dumps(result, sort_keys=True)}
