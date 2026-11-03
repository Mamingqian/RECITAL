"""The task contract the task runtimes build and the control layer reads.

`ExperimentContract` is what a task declares: its conditions, its measurement sources with the role of each,
the initial recipe, the declared hyperparameter windows and the delivery requirements. `ToolCapability` is one
source. `generation_plan` resolves the pool size and sampling of one generation against the contract.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import jsonschema


PROTOCOL_CONDITIONS = ("self_validating", "no_self_validation")
USES = ("observed", "reward_active", "selection_used")
# What each declared role permits. A sentinel is measured and may rank a selection but never rewards; a
# spendable one is the sentinel the task declares promotable into the reward. A source with no declared role
# keeps all three uses: absence of a declaration is not a restriction to enforce.
ROLE_USES = {"reward": USES, "spendable_sentinel": USES, "audit": (),
             "required_sentinel": ("observed", "selection_used"),
             "sentinel": ("observed", "selection_used")}


def _object(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required),
            "additionalProperties": False}


GENERATION_SCHEMA = _object({
    "count": {"type": "integer", "minimum": 1},
    "count_scope": {"enum": ["per_condition", "total"]},
    "temperature": {"type": "number", "exclusiveMinimum": 0},
    "top_p": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
    "batch_size": {"type": "integer", "minimum": 1},
    "condition_counts": {"type": "object", "additionalProperties": {
        "type": "integer", "minimum": 0}},
})


@dataclass(frozen=True)
class ToolCapability:
    name: str
    description: str
    units: str = "declared predictor coordinates"
    direction: str = "task_dependent"
    dependencies: tuple[str, ...] = ()
    available: bool = True
    availability_reason: str | None = None
    # The task's own designation for this predictor: "reward", "sentinel",
    # "spendable_sentinel" (held out to start, declared promotable into the reward) or
    # "audit". It decides `uses`: a plain sentinel is what the contract reserves as
    # held out, so it carries no reward permission, while the spendable one is exactly
    # the source the task declares promotable.
    declared_role: str | None = None
    uses: tuple[str, ...] = field(init=False, default=USES)

    def __post_init__(self):
        object.__setattr__(self, "uses", ROLE_USES.get(self.declared_role, USES))

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ExperimentContract:
    task: Mapping[str, Any]
    conditions: tuple[str, ...]
    initial_checkpoint: str
    initial_recipe: Mapping[str, Any]
    tools: tuple[ToolCapability, ...]
    total_updates: int
    evaluation_plan: Mapping[str, Any]
    delivery_requirements: Mapping[str, Any]
    initial_schedule_horizon: int
    protocol_condition: str = "self_validating"
    feedback_sources: tuple[str, ...] = ()
    terminal_blind_sources: tuple[str, ...] = ()
    blind_report: Mapping[str, Any] = field(default_factory=dict)
    hyperparameter_bounds: Mapping[str, Any] = field(default_factory=dict)
    initial_training_updates: int | None = None
    seed: int = 0

    def __post_init__(self):
        if self.protocol_condition not in PROTOCOL_CONDITIONS:
            raise ValueError("unknown protocol condition")
        if not self.conditions or len(set(self.conditions)) != len(self.conditions):
            raise ValueError("conditions must be nonempty and unique")
        if type(self.total_updates) is not int or self.total_updates < 0:
            raise ValueError("total_updates must be a nonnegative integer")
        names = {tool.name for tool in self.tools}
        if len(names) != len(self.tools) or not names:
            raise ValueError("tool names must be nonempty and unique")
        if names & set(self.terminal_blind_sources):
            raise ValueError("terminal blind tools cannot be experiment capabilities")
        if not set(self.feedback_sources) <= names:
            raise ValueError("feedback sources must be declared tools")
        if self.initial_schedule_horizon < 1:
            raise ValueError("schedule horizon must be positive")
        if self.initial_training_updates is not None and not (
            1 <= self.initial_training_updates <= self.total_updates
        ):
            raise ValueError("initial training segment exceeds training allowance")
        self.validate_sources(self.initial_recipe.get("enabled_oracles", []), "reward_active")
        dropped = sorted(self.required_sentinels - set(self.initial_recipe.get("held_out_oracles", [])))
        if dropped:
            raise ValueError(f"the declared first segment drops required sentinel(s) {dropped}")

    @property
    def visible_sources(self) -> frozenset[str]:
        hidden = set(self.feedback_sources) if self.protocol_condition != "self_validating" else set()
        return frozenset(tool.name for tool in self.tools if tool.name not in hidden)

    @property
    def required_sentinels(self) -> frozenset[str]:
        """Declared sentinels this run can still hold out.

        Only the sources the contract lists as required: CM declares `cm_max` in the
        sentinel pool but not under `required_sentinels`, so it may be dropped from
        `held_out_oracles` while `function_2026_t5` and `cm_poet` may not. All three
        are still refused a reward, which `uses` decides.

        Scoped to the visible sources because `no_self_validation` hides the feedback
        sources outright -- a run that cannot see a sentinel cannot be asked to declare
        it, and demanding it there would refuse every recipe.
        """
        visible = self.visible_sources
        return frozenset(tool.name for tool in self.tools
                         if tool.declared_role == "required_sentinel" and tool.name in visible)

    def validate_sources(self, names, use="observed"):
        if not isinstance(names, (list, tuple, set, frozenset)):
            raise ValueError("tool references must be a collection of source names")
        capabilities = {tool.name: tool for tool in self.tools}
        for name in names:
            if name not in self.visible_sources:
                # Name the request and the alternatives: a rejection the agent cannot
                # locate is one it cannot correct on the next turn.
                raise ValueError(f"source {name!r} is not available to this run; "
                                 f"available sources are {sorted(self.visible_sources)}")
            tool = capabilities[name]
            if use not in tool.uses:
                raise ValueError(f"tool {name!r} cannot execute use {use!r}")
            if not tool.available:
                raise ValueError(f"tool {name!r} is unavailable: {tool.availability_reason}")

    def public_description(self):
        """What a prompt may say about the task: its description, its delivery requirements, one card per source
        the run can see and the declared hyperparameter windows."""
        task = deepcopy(dict(self.task))
        if "reward_target_objectives" in task:
            task["reward_target_objectives"] = {
                name: target for name, target in task["reward_target_objectives"].items()
                if name in self.visible_sources
            }
        tools = []
        for tool in self.tools:
            if tool.name in self.visible_sources:
                record = tool.to_dict()
                record["dependencies"] = [name for name in tool.dependencies if name in self.visible_sources]
                tools.append(record)
        return {
            "task": task,
            "delivery_requirements": deepcopy(dict(self.delivery_requirements)),
            "capabilities": {"tools": tools, "hyperparameters": deepcopy(dict(self.hyperparameter_bounds))},
        }


def generation_plan(value, contract: ExperimentContract):
    defaults = dict(contract.evaluation_plan.get("generation_plan", {}))
    defaults.update(value or {})
    defaults.setdefault("count", 512)
    defaults.setdefault("count_scope", "per_condition" if len(contract.conditions) > 1 else "total")
    defaults.setdefault("temperature", 1.0)
    defaults.setdefault("top_p", 1.0)
    defaults.setdefault("batch_size", 32)
    jsonschema.Draft202012Validator(GENERATION_SCHEMA).validate(defaults)
    counts = defaults.get("condition_counts")
    if counts is None:
        n = defaults["count"]
        if defaults["count_scope"] == "total":
            if n % len(contract.conditions):
                raise ValueError("unequal total allocation requires explicit condition_counts")
            n //= len(contract.conditions)
        counts = {condition: n for condition in contract.conditions}
    if set(counts) != set(contract.conditions):
        raise ValueError("generation allocation must name exactly the task conditions")
    if defaults["count_scope"] == "total":
        if sum(counts.values()) != defaults["count"]:
            raise ValueError("condition counts differ from the declared total generation count")
    elif any(n != defaults["count"] for n in counts.values()):
        raise ValueError("per-condition count conflicts with condition_counts; use total allocation")
    defaults["condition_counts"] = counts
    return defaults
