"""ControlContract: the control layer's declaration, read from the run config.

The researcher fixes these values before launch. The agent proposes recipes and
decides where to stop and what to deliver; round count, batch width, budgets, the
within-candidate row rule and the stop rules are declared here. The ranking is the
task contract's `success.ranking`, not a field of this one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Collection, Literal, Mapping
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from recital.control.errors import ControlError
from recital.control.events import Role
from recital.control.jsonio import canonical_json, sha256_hex
from recital.library_selection import SelectionPlan

Positive = Annotated[int, Field(ge=1)]
Count = Annotated[int, Field(ge=0)]
Name = Annotated[str, StringConstraints(min_length=1)]
Fraction = Annotated[float, Field(gt=0, lt=1)]
# The plan fields `health` derives from the baseline pool; a plan may not declare them.
DIVERSITY_FIELDS = ("minimum_distance", "distance_metric", "diversity_scope")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Proposals(_Model):
    candidates_per_round: Positive
    candidate_budget: Positive
    rounds_min: Positive
    rounds_max: Positive
    anchor_candidate: bool
    max_decision_attempts: Positive
    max_skipped_rounds: Count

    @model_validator(mode="after")
    def _consistent(self) -> "Proposals":
        if self.rounds_min > self.rounds_max:
            raise ValueError("rounds_min exceeds rounds_max")
        # With anchor_candidate the anchor takes slot 1 of round 1, so the total stays
        # candidates_per_round * rounds_max and the agent proposes one fewer in round 1.
        if self.candidate_budget != self.candidates_per_round * self.rounds_max:
            raise ValueError("candidate_budget must equal candidates_per_round * rounds_max")
        if self.anchor_candidate and self.candidates_per_round < 2:
            raise ValueError("an anchor candidate leaves no slot for the agent when B is 1")
        return self


class StepBudget(_Model):
    min: Positive
    default: Positive
    max: Positive

    @model_validator(mode="after")
    def _ordered(self) -> "StepBudget":
        if not self.min <= self.default <= self.max:
            raise ValueError("steps need min <= default <= max")
        return self


class Budget(_Model):
    steps: StepBudget


class Selection(_Model):
    visible_only: bool
    plan: dict[str, Any]

    @model_validator(mode="after")
    def _executable(self) -> "Selection":
        owned = [name for name in DIVERSITY_FIELDS if name in self.plan]
        if owned:
            raise ValueError(f"selection.plan declares {owned}; the diversity floor belongs to control.health, "
                             "which derives it from the baseline pool")
        try:
            SelectionPlan.from_dict(self.plan)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"selection.plan is not an executable plan: {exc}") from exc
        return self


class Health(_Model):
    """Library health, as two fractions of what the initial policy's baseline pool measures.

    Both are relative to that pool, so they carry no alphabet or sequence length. The selection plan's minimum
    distance is `min_distance_fraction` of the baseline pool's median nearest-neighbour normalized edit distance;
    a selected library qualifies only if its mean pairwise normalized edit distance is at least
    `min_pairwise_ratio` of the baseline pool's. Every arm is held to the same two numbers.
    """

    min_distance_fraction: Fraction
    min_pairwise_ratio: Fraction


class StopRules(_Model):
    require_assignment_qualified: bool
    min_updates_fraction: Annotated[float, Field(ge=0, le=1)]


class SelfvalSpec(_Model):
    """`scalar` names the per-row measurement each source exposes (`selfval.scalar_name`)."""

    scalar: Name


class SynthesisSpec(_Model):
    enabled: bool
    roles: list[Role]
    max_requests: Count
    attempts_per_request: Positive

    @model_validator(mode="after")
    def _usable(self) -> "SynthesisSpec":
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("synthesis.roles repeats a role")
        if self.enabled and (not self.roles or self.max_requests < 1):
            raise ValueError("enabled synthesis needs roles and max_requests >= 1")
        return self


class Pricing(_Model):
    input: Annotated[float, Field(ge=0)] | None
    cached: Annotated[float, Field(ge=0)] | None
    output: Annotated[float, Field(ge=0)] | None


OPENAI_HOST = "api.openai.com"


def endpoint_host(base_url: str | None) -> str:
    """The host a cell's requests go to: the declared `llm.base_url`, else the OpenAI API."""
    return OPENAI_HOST if base_url is None else urlsplit(base_url).hostname


class LlmSpec(_Model):
    """The model and the endpoint it is asked through.

    With `base_url` and `api_key_file` both null the requests go to the OpenAI API with OPENAI_API_KEY. A declared
    `base_url` is another provider's OpenAI-compatible Responses endpoint; it needs the `api_key_file` that holds the
    key alone (the key itself is never declared), so no key is ever sent to a host it was not issued for.
    A refused URL is not echoed back in the error, because the refusal may be for credentials in it.
    `schema_dialect` is how the transport writes the strict response schemas, which `schemas.py` builds with each
    nullable field as a type list: `type_list_nullable` sends them as built (OpenAI), `anyof_nullable` writes a nullable
    array or object as `anyOf` with null, the form DeepSeek's strict compiler accepts.
    """

    model_config = ConfigDict(hide_input_in_errors=True)

    model: Name
    max_prompt_tokens: Positive
    max_output_tokens: Positive
    tool_calls_per_decision: Count
    max_total_tokens: Positive | None
    pricing_usd_per_mtok: Pricing
    base_url: Name | None = None
    api_key_file: Name | None = None
    schema_dialect: Literal["type_list_nullable", "anyof_nullable"] = "type_list_nullable"

    @model_validator(mode="after")
    def _endpoint(self) -> "LlmSpec":
        if self.base_url is not None:
            parts = urlsplit(self.base_url)
            if (parts.scheme != "https" or not parts.hostname or parts.username is not None
                    or parts.password is not None or parts.query or parts.fragment):
                raise ValueError("llm.base_url must be an https URL with a host and no credentials, query or fragment")
            if self.api_key_file is None:
                raise ValueError("llm.base_url needs llm.api_key_file: the OpenAI key is not sent to another host")
        if self.api_key_file is not None and not Path(self.api_key_file).expanduser().is_absolute():
            raise ValueError("llm.api_key_file must be an absolute path or start with ~/")
        return self


class SchedulerSpec(_Model):
    """Operational parameters: they change how jobs run, not what is measured."""

    max_concurrent_jobs: Positive
    partition: Name
    gres: Name
    cpus: Positive
    mem: Annotated[str, StringConstraints(pattern=r"^[0-9]+[KMGT]$")]
    time_factor: Annotated[float, Field(ge=1)]
    max_infra_retries: Count
    poll_seconds: Positive


def measurement_source(name: str) -> str | None:
    """The source a derived measurement `<kind>/<source>` depends on, else None.

    A bare name is a code-computed quality measurement or a source's own matrix.
    The runtime measurement catalog only emits sources visible to the arm, so a bare
    name for a hidden source cannot resolve at selection time.
    """
    return name.rsplit("/", 1)[1] if "/" in name else None


class ControlContract(_Model):
    proposals: Proposals
    budget: Budget
    selection: Selection
    health: Health
    stop: StopRules
    selfval: SelfvalSpec
    synthesis: SynthesisSpec
    llm: LlmSpec
    scheduler: SchedulerSpec

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ControlContract":
        """Build from `config["control"]`, `config["llm"]` and `config["scheduler"]`."""
        for key in ("control", "llm", "scheduler"):
            if key not in config:
                raise ControlError(f"the config declares no {key!r} block")
        control = config["control"]
        if not isinstance(control, Mapping):
            raise ControlError("config['control'] must be a mapping")
        shadowed = sorted(set(control) & {"llm", "scheduler"})
        if shadowed:
            raise ControlError(f"config['control'] may not contain {shadowed}")
        return cls.model_validate(
            {**control, "llm": config["llm"], "scheduler": config["scheduler"]})

    def contract_hash(self) -> str:
        """Digest of the control blocks plus the model name; the scheduler is left out.

        Recorded in `run_started` as provenance. Nothing compares it.
        """
        payload = {
            "control": self.model_dump(mode="json", exclude={"llm", "scheduler"}),
            "llm_model": self.llm.model,
        }
        return sha256_hex(canonical_json(payload))

    def validate_against(self, visible_sources: Collection[str], conditions: list[str],
                         delivery_count: int) -> None:
        """Check the selection plan against one arm's task: sources, conditions, counts."""
        plan = SelectionPlan.from_dict(self.selection.plan)
        conditions = list(conditions)
        if not conditions or len(set(conditions)) != len(conditions):
            raise ControlError("conditions must be nonempty and unique")
        if type(delivery_count) is not int or delivery_count < 1:
            raise ControlError("delivery_count must be a positive integer")
        counts = (dict(plan.count) if isinstance(plan.count, Mapping)
                  else dict.fromkeys(plan.strata or conditions, plan.count))
        if set(counts) != set(conditions):
            raise ControlError(
                f"the selection plan counts {sorted(counts)}, "
                f"the task's conditions are {sorted(conditions)}")
        wrong = {condition: n for condition, n in counts.items() if n != delivery_count}
        if wrong:
            raise ControlError(
                f"the selection plan counts {wrong}, "
                f"the task delivers {delivery_count} per condition")
        if self.selection.visible_only:
            visible = set(visible_sources)
            hidden = []
            for name in plan.measurement_names:
                source = measurement_source(name)
                if source is not None and source not in visible:
                    hidden.append(name)
            if hidden:
                raise ControlError(
                    f"the selection plan names measurements {hidden} "
                    "whose source is not visible to this arm")
