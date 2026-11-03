"""Turn an agent's candidate proposal into the exact training parameters a worker executes.

The proposal names a parent, a start mode and the edits it wants; everything else (optimizer, schedule,
reward-state and randomness policy, KL reference, held-out sources, fixed task targets) is derived here,
once, so the agent surface stays small and every rule has one home.
Functions are pure: a refusal is a list of typed `Problem`s, never an exception.
"""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import jsonschema

from .events import Problem
from .jsonio import canonical_json, sha256_hex

START_TO_MODE = {"from_initial": "initial_restart", "from_parent": "weights_new_stage",
                 "continue_parent": "full_state_continuation"}
EDITED_HYPERPARAMETERS = ("lr", "kl", "group_size")
# A job killed mid-training resumes from its newest full-state checkpoint, so one is kept every this many updates.
RECOVERY_INTERVAL = 500
SENTINEL_ROLES = ("required_sentinel", "sentinel", "spendable_sentinel")


@dataclass(frozen=True)
class RecipeContext:
    """What the task and the control contract fix, as plain values."""

    initial_recipe: Mapping[str, Any]
    initial_checkpoint: str
    initial_schedule_horizon: int
    hyperparameter_bounds: Mapping[str, Any]
    fixed_target_objectives: Mapping[str, Any]
    reward_eligible: frozenset[str]
    visible_sources: frozenset[str]
    sentinel_sources: frozenset[str]
    steps_min: int
    steps_default: int
    steps_max: int
    total_updates: int
    validate_recipe: Callable[[dict[str, Any]], list[str]] = field(compare=False)


def context_from(task_contract, control, validate_recipe: Callable[[dict[str, Any]], list[str]]) -> RecipeContext:
    """Build from the task's `ExperimentContract` and the run's `ControlContract`."""
    visible = task_contract.visible_sources
    tools = {tool.name: tool for tool in task_contract.tools}
    # The declared recipe's `steps` is the length of a first segment this design does not force; a candidate that
    # starts from the initial recipe trains the declared default, so that is the recipe everything else reads.
    initial = deepcopy(dict(task_contract.initial_recipe))
    initial["hyperparameters"] = {**initial.get("hyperparameters", {}), "steps": control.budget.steps.default}
    return RecipeContext(
        initial_recipe=initial,
        initial_checkpoint=task_contract.initial_checkpoint,
        initial_schedule_horizon=task_contract.initial_schedule_horizon,
        hyperparameter_bounds=task_contract.hyperparameter_bounds,
        fixed_target_objectives=task_contract.task.get("reward_target_objectives", {}),
        reward_eligible=frozenset(name for name in visible if "reward_active" in tools[name].uses),
        visible_sources=frozenset(visible),
        sentinel_sources=frozenset(name for name in visible if tools[name].declared_role in SENTINEL_ROLES),
        steps_min=control.budget.steps.min, steps_default=control.budget.steps.default,
        steps_max=control.budget.steps.max, total_updates=task_contract.total_updates,
        validate_recipe=validate_recipe)


@dataclass(frozen=True)
class Resolved:
    parent: str
    start: str
    steps: int
    recipe: dict[str, Any]
    train: dict[str, Any]
    fingerprint: str
    repeat_of: str | None = None
    anchor: bool = False


def initial_parent_record(ctx: RecipeContext) -> dict[str, Any]:
    return {"id": "initial", "path": ctx.initial_checkpoint, "parent": None, "action_id": None, "step": 0,
            "lineage_updates": 0, "full_state_available": False, "weights_available": True,
            "recipe": deepcopy(dict(ctx.initial_recipe)), "schedule": None, "kl_reference": "initial"}


def visible_recipe(recipe: Mapping[str, Any], visible: Iterable[str]) -> dict[str, Any]:
    """The recipe as one arm may read it: held-out sources the arm cannot see are not named."""
    shown, names = deepcopy(dict(recipe)), set(visible)
    shown["held_out_oracles"] = [name for name in recipe.get("held_out_oracles", ()) if name in names]
    shown["enabled_oracles"] = [name for name in recipe.get("enabled_oracles", ()) if name in names]
    return shown


def anchor_proposal() -> dict[str, Any]:
    """The declared initial recipe, unchanged, as a proposal; steps take the declared default."""
    return {"parent": "initial", "start": "from_initial", "hypothesis": "declared initial recipe, unchanged",
            "steps": None, "lr": None, "kl": None, "group_size": None, "enabled_oracles": None,
            "aggregator": None, "oracle_utilities": None, "penalties": None, "gates": None}


def _problem(code: str, path: str = "", detail: str = "") -> Problem:
    return Problem(code=code, path=path, detail=detail[:300])


def _params(text: str, path: str) -> tuple[dict[str, Any] | None, Problem | None]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, _problem("bad_params_json", path, str(exc))
    if not isinstance(value, dict):
        return None, _problem("bad_params_json", path, "params_json must be a JSON object")
    return value, None


def _call(entry: Mapping[str, Any], path: str, problems: list[Problem]) -> dict[str, Any] | None:
    params, problem = _params(entry["params_json"], path)
    if problem is not None:
        problems.append(problem)
        return None
    if "name" in params:
        problems.append(_problem("name_in_params", path, "the primitive name belongs in `name`, not params_json"))
        return None
    return {"name": entry["name"], **params}


def _apply_edits(proposal: Mapping[str, Any], base: Mapping[str, Any], ctx: RecipeContext,
                 problems: list[Problem]) -> dict[str, Any]:
    recipe = deepcopy(dict(base))
    if proposal.get("enabled_oracles") is not None:
        enabled = list(proposal["enabled_oracles"])
        if not enabled or len(set(enabled)) != len(enabled):
            problems.append(_problem("bad_enabled_oracles", "enabled_oracles", "needs distinct, nonempty sources"))
        for name in enabled:
            if name not in ctx.reward_eligible:
                problems.append(_problem("not_reward_eligible", "enabled_oracles",
                                         f"{name!r} may not enter the reward; eligible: {sorted(ctx.reward_eligible)}"))
        recipe["enabled_oracles"] = enabled
        # A utility for a source that left the reward has nothing to act on.
        recipe["oracle_utilities"] = {name: ref for name, ref in (recipe.get("oracle_utilities") or {}).items()
                                      if name in enabled}
    if proposal.get("aggregator") is not None:
        aggregator = _call(proposal["aggregator"], "aggregator", problems)
        if aggregator is not None:
            recipe["aggregator"] = aggregator
    if proposal.get("oracle_utilities") is not None:
        utilities = {}
        for index, entry in enumerate(proposal["oracle_utilities"]):
            path = f"oracle_utilities[{index}]"
            reference = _call({"name": entry["name"], "params_json": entry["params_json"]}, path, problems)
            if reference is not None:
                utilities[entry["oracle"]] = reference
        recipe["oracle_utilities"] = utilities
    for key in ("penalties", "gates"):
        if proposal.get(key) is not None:
            calls = [_call(entry, f"{key}[{index}]", problems) for index, entry in enumerate(proposal[key])]
            recipe[key] = [call for call in calls if call is not None]
    hyperparameters = dict(recipe.get("hyperparameters") or {})
    for key in EDITED_HYPERPARAMETERS:
        if proposal.get(key) is not None:
            hyperparameters[key] = proposal[key]
    recipe["hyperparameters"] = hyperparameters
    return recipe


def _check_bounds(hyperparameters: Mapping[str, Any], ctx: RecipeContext, problems: list[Problem]) -> None:
    for key, bound in ctx.hyperparameter_bounds.items():
        if key == "steps" or key not in hyperparameters:
            continue
        try:
            jsonschema.Draft202012Validator(bound).validate(hyperparameters[key])
        except jsonschema.ValidationError as exc:
            problems.append(_problem("hyperparameter_out_of_bounds", key, f"{key}={hyperparameters[key]}: {exc.message}"))


def _derive_held_out(recipe: dict[str, Any], parent: Mapping[str, Any], ctx: RecipeContext) -> None:
    """Held-out evidence is every visible sentinel the reward uses neither here nor anywhere up the recipe's lineage.

    A sentinel the reward used anywhere up the lineage stays out of the held-out set of a descendant that stops using
    it; the agent cannot drop a held-out source either.
    """
    inherited = ctx.sentinel_sources if parent["id"] == "initial" else (
        ctx.sentinel_sources & set(parent["recipe"].get("held_out_oracles", ())))
    recipe["held_out_oracles"] = sorted(inherited - set(recipe["enabled_oracles"]))


def _fix_targets(recipe: dict[str, Any], ctx: RecipeContext, problems: list[Problem]) -> None:
    if not ctx.fixed_target_objectives:
        return
    missing = sorted(set(recipe["enabled_oracles"]) - set(ctx.fixed_target_objectives))
    if missing:
        problems.append(_problem("no_target_coordinates", "enabled_oracles",
                                 f"{missing} have no declared task target coordinates"))
        return
    recipe["target_objectives"] = {name: deepcopy(ctx.fixed_target_objectives[name])
                                   for name in recipe["enabled_oracles"]}


def _training(start: str, parent: Mapping[str, Any], steps: int, recipe: Mapping[str, Any],
              ctx: RecipeContext, problems: list[Problem]) -> dict[str, Any]:
    mode = START_TO_MODE[start]
    train: dict[str, Any] = {"start_mode": mode, "additional_updates": steps}
    if start == "continue_parent":
        if not parent.get("full_state_available"):
            problems.append(_problem("continue_needs_full_state", "start", "the parent kept no complete training state"))
        prior = parent.get("schedule")
        if not prior:
            problems.append(_problem("continue_needs_schedule", "start", "the parent has no declared schedule to continue"))
        else:
            parent_lr = parent["recipe"].get("hyperparameters", {}).get("lr")
            if recipe["hyperparameters"].get("lr") != parent_lr:
                problems.append(_problem("continue_changes_lr", "lr",
                                         f"continue_parent keeps the parent's lr {parent_lr}; changing it requires "
                                         "start from_parent or from_initial"))
            left = prior["start_step"] + prior["horizon"] - parent["step"]
            if steps > left:
                problems.append(_problem("schedule_exhausted", "steps",
                                         f"the parent's schedule has {left} updates left, {steps} requested; "
                                         "start a new stage with from_parent"))
        train.update(start_checkpoint=parent["id"], optimizer_policy="retain",
                     schedule_policy={"kind": "continue_segment"}, reward_state_policy="inherit_compatible",
                     randomness_policy="inherit_stream", kl_reference=parent.get("kl_reference", "initial"))
    else:
        schedule = {"kind": "new_segment", "horizon": max(steps, ctx.initial_schedule_horizon),
                    "start_lr": recipe["hyperparameters"].get("lr"), "type": "linear", "warmup_updates": 0}
        train.update(start_checkpoint="initial" if start == "from_initial" else parent["id"],
                     optimizer_policy="reset", schedule_policy=schedule, reward_state_policy="reset_all",
                     randomness_policy="new_stream", kl_reference="initial")
    train["checkpoint_steps"] = sorted({*range(RECOVERY_INTERVAL, steps, RECOVERY_INTERVAL), steps})
    return train


def _fingerprint(train: Mapping[str, Any], recipe: Mapping[str, Any]) -> str:
    identity = {"start": train["start_mode"], "from": train["start_checkpoint"], "kl": train["kl_reference"],
                "steps": train["additional_updates"], "schedule": train["schedule_policy"], "recipe": recipe}
    return sha256_hex(canonical_json(identity))


def resolve(
    proposal: Mapping[str, Any],
    ctx: RecipeContext,
    parents: Mapping[str, Mapping[str, Any]],
    *,
    committed_updates: int,
    siblings: Sequence[Resolved] = (),
    history: Mapping[str, str] | None = None,
    anchor: bool = False,
) -> tuple[Resolved | None, list[Problem]]:
    """Resolve one proposal against completed parents.

    `parents` maps 'initial' and every completed candidate id to its end-checkpoint record. `siblings` are
    the already accepted candidates of this round (their steps count against the budget and an identical
    fingerprint among them is refused). `history` maps earlier fingerprints to the first candidate that ran
    them, so a repeat across rounds is allowed and recorded.
    """
    problems: list[Problem] = []
    parent_id = proposal["parent"]
    parent = parents.get(parent_id)
    if parent is None:
        return None, [_problem("unknown_parent", "parent",
                               f"{parent_id!r} is not 'initial' or a completed candidate; available: {sorted(parents)}")]
    start = proposal["start"]
    if start not in START_TO_MODE:
        return None, [_problem("bad_start", "start", f"{start!r} is not one of {sorted(START_TO_MODE)}")]
    if start != "from_initial" and parent_id == "initial":
        problems.append(_problem("bad_start", "start", "the initial policy can only be restarted: use from_initial"))
    steps = proposal.get("steps")
    steps = ctx.steps_default if steps is None else steps
    if not ctx.steps_min <= steps <= ctx.steps_max:
        problems.append(_problem("steps_out_of_bounds", "steps",
                                 f"steps {steps} outside the declared [{ctx.steps_min}, {ctx.steps_max}]"))
    spent = committed_updates + sum(sibling.steps for sibling in siblings)
    if steps > ctx.total_updates - spent:
        problems.append(_problem("over_budget", "steps",
                                 f"{steps} updates requested, {max(ctx.total_updates - spent, 0)} remain unallocated"))
    recipe = _apply_edits(proposal, parent["recipe"], ctx, problems)
    recipe["hyperparameters"]["steps"] = steps
    _check_bounds(recipe["hyperparameters"], ctx, problems)
    if not recipe.get("enabled_oracles"):
        problems.append(_problem("bad_enabled_oracles", "enabled_oracles", "the recipe has no reward source"))
    if problems:
        return None, problems
    _derive_held_out(recipe, parent, ctx)
    _fix_targets(recipe, ctx, problems)
    if problems:
        return None, problems
    detail = ctx.validate_recipe(recipe)
    if detail:
        return None, [_problem("invalid_recipe", "recipe", "; ".join(detail))]
    train = _training(start, parent, steps, recipe, ctx, problems)
    if problems:
        return None, problems
    train["recipe"] = recipe
    fingerprint = _fingerprint(train, recipe)
    if any(sibling.fingerprint == fingerprint for sibling in siblings):
        return None, [_problem("duplicate_in_round", "", "an identical candidate is already proposed in this round")]
    return Resolved(parent=parent_id, start=start, steps=steps, recipe=recipe, train=train,
                    fingerprint=fingerprint, repeat_of=(history or {}).get(fingerprint), anchor=anchor), []
