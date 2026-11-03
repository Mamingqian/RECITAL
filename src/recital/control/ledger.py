"""Ledger rows: the immutable facts about one finished candidate, and the compact text the agent reads.

`build_row` assembles the facts from a worker's `candidate_result.json`; `visible_row` projects them onto the
whitelisted fields an agent may see; `render_row` writes that projection as a line-oriented block. A row is
frozen once written: nothing in it depends on candidates that finish later, so a rendered ledger only grows.
"""
from __future__ import annotations

import math
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from .guard import count_tokens
from .health import REQUIREMENT_ID, UNIQUE_KMER_MEASUREMENT

# The longest row a task can show (eight edits of 200 characters, a 280-character hypothesis, digit-heavy medians, four
# failing requirements and an 80-token library_health reason per condition, every enabled/held-out split of the
# sources) is under 2,000 tokens. The cap is above it with 60% headroom, so no row loses a line to it.
ROW_TOKEN_CAP = 3000
SIGNIFICANT = 4
# What `pairwise_ratio` and `unique_kmer_ratio` in a row's `pool` and `library` lines mean; every arm is shown it.
LIBRARY_HEALTH_LEGEND = """Library health columns. `pairwise_ratio` is a condition's mean pairwise normalized edit \
distance divided by the same quantity on the initial policy's pool (1.0 is as spread out as the initial policy; `pool` \
is the candidate's whole pool, `library` the delivered selection). `unique_kmer_ratio` is the median distinct k-mer \
count per sequence over the initial pool's, and is null when the task measures no k-mer count. A library whose \
`library` pairwise_ratio is below the Search contract's min_pairwise_ratio fails requirement library_health and is \
unqualified, and `qualified` then carries the `reason`. Selection also skips any row closer than \
min_distance_fraction x the initial pool's median nearest-neighbour distance to a row already chosen, and never lowers \
that floor: a condition whose pool cannot supply the required count of rows that far apart delivers fewer (`selected` \
below the required count, `library_health` failing) and is unqualified."""
COMPOSITION_LEGEND = """Composition line. Each row prints, per condition, every source-free measurement (a property of the \
sequences themselves, not a score; the distinct k-mer count is reported by `unique_kmer_ratio` instead) whose pool \
median lies outside the central 90% of the initial policy's own pool for that condition: \
`<name> above|below [<5th percentile>, <95th percentile>] (<the candidate's pool median>)`. `within` means none \
does. It compares the candidate's pool with the initial policy's, and uses no oracle."""
RECIPE_KEYS = ("enabled_oracles", "aggregator", "oracle_utilities", "penalties", "gates", "reward_input")
# The trainer logs about twenty-five numeric fields per step; these are the ones `get_candidate` returns.
CURVE_FIELDS = ("step", "reward", "reward_std", "frac_reward_zero_std", "kl", "loss", "grad_norm", "learning_rate")


def _round(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(f"{value:.{SIGNIFICANT}g}")
    if isinstance(value, Mapping):
        return {key: _round(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(item) for item in value]
    return value


def medians(summary: Mapping[str, Any], conditions: Sequence[str], names: Iterable[str]) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for condition in conditions:
        row = {}
        for name in names:
            cell = (summary.get(name) or {}).get("by_condition", {}).get(condition) or {}
            row[name] = cell.get("median")
        result[condition] = row
    return result


def composition_bands(summary: Mapping[str, Any], conditions: Sequence[str]) -> dict[str, dict[str, tuple[float, float]]]:
    """The 5th-95th percentile band of each source-free scalar measurement on the initial policy's pool.

    A measurement with no spread on the initial pool (a constant such as `legal`) has no band: nothing can lie outside it.
    The distinct k-mer count is left out: `unique_kmer_ratio` already reports it.
    """
    bands: dict[str, dict[str, tuple[float, float]]] = {condition: {} for condition in conditions}
    for name, entry in sorted(summary.items()):
        if entry.get("sources") or name == UNIQUE_KMER_MEASUREMENT:
            continue
        for condition in conditions:
            cell = (entry.get("by_condition") or {}).get(condition) or {}
            low, high = cell.get("q05"), cell.get("q95")
            if low is not None and high is not None and low < high:
                bands[condition][name] = (low, high)
    return bands


def composition_outside(pool: Mapping[str, Mapping[str, float | None]],
                        bands: Mapping[str, Mapping[str, tuple[float, float]]]) -> dict[str, dict[str, Any]]:
    """Per condition, the source-free measurements whose pool median is outside the initial policy's band."""
    outside: dict[str, dict[str, Any]] = {}
    for condition, values in pool.items():
        found = {}
        for name, (low, high) in (bands.get(condition) or {}).items():
            value = values.get(name)
            if value is not None and not low <= value <= high:
                found[name] = {"side": "above" if value > high else "below", "band": [low, high], "value": value}
        outside[condition] = found
    return outside


def shown_measurements(summary: Mapping[str, Any], headline: Sequence[str]) -> list[str]:
    """The headline scalar family of every visible source, plus every source-free scalar measurement."""
    names = []
    for name, entry in sorted(summary.items()):
        cells = (entry.get("by_condition") or {}).values()
        if not cells or not all("median" in cell for cell in cells):
            continue
        if name in headline or not entry.get("sources"):
            names.append(name)
    return names


def recipe_diff(parent: Mapping[str, Any] | None, recipe: Mapping[str, Any]) -> list[str]:
    """What this recipe changes against its parent's, as short strings."""
    base = parent or {}
    changes = []
    for key in RECIPE_KEYS:
        if base.get(key) != recipe.get(key):
            changes.append(f"{key}: {_compact(base.get(key))} -> {_compact(recipe.get(key))}")
    before, after = base.get("hyperparameters") or {}, recipe.get("hyperparameters") or {}
    for key in sorted(set(before) | set(after)):
        if before.get(key) != after.get(key):
            changes.append(f"{key}: {before.get(key)} -> {after.get(key)}")
    return changes


def _compact(value: Any) -> str:
    import json
    return json.dumps(_round(value), sort_keys=True, separators=(",", ":"))


def build_row(
    *,
    candidate: Mapping[str, Any],
    spec: Mapping[str, Any],
    parent_recipe: Mapping[str, Any] | None,
    parent_row: Mapping[str, Any] | None,
    baseline_pool: Mapping[str, Any],
    baseline_bands: Mapping[str, Mapping[str, tuple[float, float]]],
    result: Mapping[str, Any] | None,
    failure: Mapping[str, Any] | None,
    conditions: Sequence[str],
    headline: Sequence[str],
) -> dict[str, Any]:
    """The full row. `candidate` carries id, ordinal, round, slot, parent, start, repeat_of, anchor, status."""
    row: dict[str, Any] = {
        "id": candidate["id"], "round": candidate["round"], "slot": candidate["slot"],
        "parent": candidate["parent"], "start": candidate["start"], "repeat_of": candidate["repeat_of"],
        "anchor": candidate["anchor"], "status": candidate["status"],
        "hypothesis": spec.get("hypothesis", ""), "edits": recipe_diff(parent_recipe, spec["recipe"]),
        "steps": spec["train"]["additional_updates"],
    }
    if result is None:
        row["failure"] = dict(failure or {"type": candidate["status"], "message": ""})
        return _round(row)
    summary = result["summary"]
    end = next(item for item in result["checkpoints"] if item["id"] == result["end_checkpoint_id"])
    row["training"] = {
        "steps": summary["steps_done"], "lineage_updates": end["lineage_updates"],
        "wall_h": None if summary["wall_s"] is None else summary["wall_s"] / 3600.0,
        "reward_first": summary["reward_first"], "reward_last": summary["reward_last"],
        "reward_std": summary["reward_std"], "frac_zero_std": summary["frac_zero_std"],
        "kl_ref": summary["kl_ref"], "cost_status": result["cost"]["status"]}
    pool_summary = result["pool"]["raw"]
    names = shown_measurements(pool_summary, headline)
    row["pool"] = medians(pool_summary, conditions, names)
    row["library"] = medians(result["delivery"]["selected"], conditions, names)
    parent_pool = (parent_row or {}).get("pool") or {}
    row["vs_parent"] = _delta(row["pool"], parent_pool, headline) if parent_pool else None
    row["vs_initial"] = _delta(row["pool"], baseline_pool, headline)
    row["composition"] = composition_outside(row["pool"], baseline_bands)
    row["qualified"] = {condition: {"qualified": info["qualified"], "failing": list(info["failing_requirements"]),
                                    "selected": info["selected"]}
                        for condition, info in result["by_condition"].items()}
    for requirement in result["delivery"]["qualification"].get("requirements", ()):
        if (requirement["requirement_id"] == REQUIREMENT_ID and requirement["status"] != "pass"
                and requirement["stratum"] in row["qualified"]):
            row["qualified"][requirement["stratum"]]["reason"] = requirement["reason"]
    row = _round(row)
    # Not shown to an agent and not rounded: selection ranks candidates, and breaks ties inside a tolerance, from
    # the measured values themselves.
    row["qualification"] = result["delivery"]["qualification"]
    return row


def _delta(current: Mapping[str, Mapping[str, float | None]], reference: Mapping[str, Mapping[str, float | None]],
           headline: Sequence[str]) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for condition, values in current.items():
        out[condition] = {}
        for name in headline:
            now, then = values.get(name), (reference.get(condition) or {}).get(name)
            out[condition][name] = None if now is None or then is None else now - then
    return out


class VisibleRow(BaseModel):
    """The only shape of a candidate row an agent is shown; built field by field from the full row."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    round: int
    slot: int
    parent: str | None
    start: str | None
    repeat_of: str | None
    anchor: bool
    status: str
    hypothesis: str
    edits: list[str]
    steps: int
    training: dict[str, Any] | None = None
    pool: dict[str, dict[str, float | None]] | None = None
    library: dict[str, dict[str, float | None]] | None = None
    vs_parent: dict[str, dict[str, float | None]] | None = None
    vs_initial: dict[str, dict[str, float | None]] | None = None
    composition: dict[str, dict[str, Any]] | None = None
    qualified: dict[str, dict[str, Any]] | None = None
    failure: dict[str, Any] | None = None


# A failure message can carry a run path (`/<output root>/<run name>/...`); an absolute path says nothing the agent
# can act on.
_ABSOLUTE_PATH = re.compile(r"/[^\s'\"]+")


def visible_row(full: Mapping[str, Any], *, scrub: Callable[[str], str] = lambda text: text) -> VisibleRow:
    failure = full.get("failure")
    return VisibleRow(
        id=full["id"], round=full["round"], slot=full["slot"], parent=full["parent"], start=full["start"],
        repeat_of=full["repeat_of"], anchor=full["anchor"], status=full["status"],
        hypothesis=full["hypothesis"][:280], edits=[change[:200] for change in full["edits"]][:8],
        steps=full["steps"], training=full.get("training"), pool=full.get("pool"),
        library=full.get("library"), vs_parent=full.get("vs_parent"), vs_initial=full.get("vs_initial"),
        composition=full.get("composition"),
        qualified=full.get("qualified"),
        failure=None if failure is None else {"type": failure.get("type"),
                                              "message": _ABSOLUTE_PATH.sub("<path>", scrub(str(failure.get("message", ""))))[:300]})


def training_curve(log_history: Sequence[Mapping[str, Any]], *, last: int = 12) -> list[dict[str, Any]]:
    """The last `last` trainer log entries, each cut to the fields that say whether learning was still moving."""
    return [_round({key: entry[key] for key in CURVE_FIELDS if key in entry}) for entry in log_history[-last:]]


def _line(label: str, value: Any) -> str:
    return f" {label}: {_compact(value)}"


def render_row(row: VisibleRow, *, cap: int = ROW_TOKEN_CAP) -> str:
    """Line-oriented text of one row; over `cap` tokens it drops what the ledger already implies, in a fixed order.

    The cap never removes a measurement the instructions describe: the pool and library columns (composition
    included) stay, so a row over the cap is sent whole.
    """
    droppable = ["vs_parent", "text.short"]
    dropped: set[str] = set()
    while True:
        text = _render(row, dropped)
        if count_tokens(text) <= cap or not droppable:
            return text
        dropped.add(droppable.pop(0))


def _render(row: VisibleRow, dropped: set[str]) -> str:
    head = (f"[{row.id}] round {row.round} slot {row.slot} parent={row.parent} start={row.start} "
            f"status={row.status}" + (f" repeat_of={row.repeat_of}" if row.repeat_of else "")
            + (" anchor" if row.anchor else ""))
    short = "text.short" in dropped
    hypothesis = row.hypothesis[:140] if short else row.hypothesis
    edits = [change[:100] for change in row.edits[:3]] if short else row.edits
    lines = [head, f" hypothesis: {hypothesis}", f" edits: {'; '.join(edits) or 'none'}"]
    if row.failure is not None:
        lines.append(_line("failure", row.failure))
        return "\n".join(lines)
    lines.append(_line("training", row.training))
    lines.append(_line("pool", row.pool))
    lines.append(_line("library", row.library))
    lines.append(_line("vs_initial", row.vs_initial))
    if row.composition is not None:
        lines.append(_composition_line(row.composition))
    if row.vs_parent is not None and "vs_parent" not in dropped:
        lines.append(_line("vs_parent", row.vs_parent))
    lines.append(_line("qualified", row.qualified))
    return "\n".join(lines)


def _composition_line(composition: Mapping[str, Mapping[str, Any]]) -> str:
    parts = []
    for condition, found in composition.items():
        parts.append(f"{condition}: " + ("; ".join(
            f"{name} {item['side']} {_compact(item['band'])} ({_compact(item['value'])})"
            for name, item in found.items()) or "within"))
    return " composition " + " | ".join(parts)


def render_component_row(row: Mapping[str, Any]) -> str:
    return (f"[component {row['request_id']}] round {row['round']} role={row['role']} name={row['name']} "
            f"status={row['status']}" + (f" reason={row['reason_code']}" if row.get("reason_code") else "")
            + f"\n description: {str(row.get('description', ''))[:240]}"
            + (f"\n signature: {row['signature']}" if row.get("signature") else "")
            + (f"\n error: {str(row['error'])[:300]}" if row.get("error") else ""))


def render_advisory_row(row: Mapping[str, Any]) -> str:
    concerns = "; ".join(str(item)[:200] for item in row.get("concerns", ())[:6]) or "none"
    evidence = "; ".join(str(item)[:160] for item in row.get("requested_evidence", ())[:3]) or "none"
    return (f"[critic round {row['round']}] concerns: {concerns}\n requested_evidence: {evidence}\n"
            f" rationale: {str(row.get('rationale', ''))[:300]}")
