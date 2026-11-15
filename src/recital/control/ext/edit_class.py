"""The edit classes of a recipe: what changed against the base it was built from, as lever, direction and detail.

The base is the parent's recipe, or the initial recipe for a candidate that starts from the initial policy. Every
count of edits in the modules and in the endpoint tables goes through `edit_classes`, so a repeat rate or an
attribution share means the same thing in each.

An oracle that enters or leaves `enabled_oracles` is one edit (`oracle_set`); its `oracle_utilities` entry and its
aggregator weight move with it and are not counted again. `held_out_oracles` follows `enabled_oracles` and is never
an agent edit, so it is left out.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Iterable, Mapping

# Recipe keys that have their own lever; every other key that differs is `other`.
_HANDLED = ("enabled_oracles", "oracle_utilities", "aggregator", "penalties", "gates", "hyperparameters",
            "held_out_oracles")
SCALAR_HYPERPARAMETERS = ("lr", "kl", "group_size", "steps")
# Every `EditClass.lever` that `edit_classes` produces. (`recipes.LEVERS` is something else: the two lever families,
# reward and training, that `control.levers` opens.)
LEVER_CLASSES = ("oracle_set", "weights", "aggregator", "utility", "penalty", "gate", *SCALAR_HYPERPARAMETERS, "other")


@dataclass(frozen=True, order=True)
class EditClass:
    lever: str
    direction: str
    detail: str = ""

    @property
    def key(self) -> str:
        return ":".join((self.lever, self.direction, self.detail))

    @property
    def coarse(self) -> tuple[str, str]:
        return self.lever, self.direction


def _direction(old: Any, new: Any) -> str:
    numbers = all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (old, new))
    if numbers and old != new:
        return "up" if new > old else "down"
    return "changed"


def _named(items: Any) -> dict[str, Mapping[str, Any]]:
    return {item["name"]: item for item in items or ()}


def _primitive_edits(lever: str, old: Any, new: Any) -> list[EditClass]:
    """Penalties and gates are lists of named entries: one class per entry added, removed or changed."""
    before, after = _named(old), _named(new)
    classes = [EditClass(lever, "added", name) for name in sorted(set(after) - set(before))]
    classes += [EditClass(lever, "removed", name) for name in sorted(set(before) - set(after))]
    for name in sorted(set(before) & set(after)):
        for field in sorted(set(before[name]) | set(after[name])):
            if before[name].get(field) != after[name].get(field):
                classes.append(EditClass(lever, _direction(before[name].get(field), after[name].get(field)),
                                         f"{name}.{field}"))
    return classes


def edit_classes(base: Mapping[str, Any] | None, recipe: Mapping[str, Any]) -> list[EditClass]:
    """The classes of `recipe` against `base`, sorted; empty when the recipes are the same."""
    base = base or {}
    classes: list[EditClass] = []
    entered = set(recipe.get("enabled_oracles") or ()) - set(base.get("enabled_oracles") or ())
    left = set(base.get("enabled_oracles") or ()) - set(recipe.get("enabled_oracles") or ())
    classes += [EditClass("oracle_set", "added", name) for name in sorted(entered)]
    classes += [EditClass("oracle_set", "removed", name) for name in sorted(left)]
    moved = entered | left

    before_aggregator, after_aggregator = base.get("aggregator") or {}, recipe.get("aggregator") or {}
    if before_aggregator.get("name") != after_aggregator.get("name"):
        classes.append(EditClass("aggregator", "changed", str(after_aggregator.get("name"))))
    before_weights, after_weights = before_aggregator.get("weights") or {}, after_aggregator.get("weights") or {}
    for name in sorted((set(before_weights) | set(after_weights)) - moved):
        if before_weights.get(name) != after_weights.get(name):
            classes.append(EditClass("weights", _direction(before_weights.get(name), after_weights.get(name)), name))
    if before_aggregator.get("name") == after_aggregator.get("name"):
        for field in sorted((set(before_aggregator) | set(after_aggregator)) - {"name", "weights"}):
            if before_aggregator.get(field) != after_aggregator.get(field):
                classes.append(EditClass("aggregator", _direction(before_aggregator.get(field),
                                                                  after_aggregator.get(field)), field))

    before_utilities, after_utilities = base.get("oracle_utilities") or {}, recipe.get("oracle_utilities") or {}
    for name in sorted((set(before_utilities) | set(after_utilities)) - moved):
        if before_utilities.get(name) != after_utilities.get(name):
            classes.append(EditClass("utility", "changed", name))

    classes += _primitive_edits("penalty", base.get("penalties"), recipe.get("penalties"))
    classes += _primitive_edits("gate", base.get("gates"), recipe.get("gates"))

    before_hyper, after_hyper = base.get("hyperparameters") or {}, recipe.get("hyperparameters") or {}
    for field in sorted(set(before_hyper) | set(after_hyper)):
        if before_hyper.get(field) == after_hyper.get(field):
            continue
        if field in SCALAR_HYPERPARAMETERS:
            classes.append(EditClass(field, _direction(before_hyper.get(field), after_hyper.get(field))))
        else:
            classes.append(EditClass("other", "changed", f"hyperparameters.{field}"))
    for key in sorted((set(base) | set(recipe)) - set(_HANDLED)):
        if base.get(key) != recipe.get(key):
            classes.append(EditClass("other", "changed", key))
    return sorted(classes)


def repeat_counts(proposed: Iterable[Iterable[str]], seen: Collection[str]) -> tuple[int, int]:
    """(occurrences of a class that is in `seen`, occurrences) over the class keys of the proposed candidates.

    This is the repeat rate of the endpoint table (`scripts/analysis/module_endpoints.py`). A class counts once per
    candidate that carries it, keys are `EditClass.key` (lever, direction and detail), `seen` holds the keys carried by
    the candidates of the rounds before, and the rate is the ratio of the two sums over a search, or over the decisions
    of an arm.
    """
    occurrences = [key for keys in proposed for key in keys]
    return sum(1 for key in occurrences if key in seen), len(occurrences)
