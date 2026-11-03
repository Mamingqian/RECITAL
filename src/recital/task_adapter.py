"""Causal-LM sequence task adapter interfaces.

A task adapter owns task semantics and model/oracle wiring.  The shared outer
loop only consumes these typed descriptors and never branches on task names.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from recital.failure_modes import FailureProfile, shared_failure_profile
from recital.spec.composer import SourceRolePolicy
from recital.sft import SFTRecipe
from recital.task_provider import ObjectiveDescriptor
from recital.token_sequence import RepresentationSpec
from recital.success import SuccessSpec
from recital.capabilities import RecipeCapabilities


@dataclass
class TaskProgram:
    """The natural-language task program supplied by the user contract."""

    name: str
    description: str
    biological_constraints: list[str] = field(default_factory=list)
    target_objective: str = ""


@dataclass(frozen=True)
class SequenceDataSpec:
    """Semantic columns and split policy for a sequence table."""

    source: str | None = None
    sequence_column: str = "sequence"
    condition_column: str | None = None
    label_columns: tuple[str, ...] = ()
    split_column: str | None = "split"
    train_split: str = "train"


@dataclass(frozen=True)
class CausalLMPolicySpec:
    """Causal-LM policy settings needed by the shared generation trainer."""

    model: str | None = None
    conditioning_prompts: tuple[str, ...] = ()
    token_alphabet: str = "ACGT"
    max_completion_length: int = 0
    min_new_tokens: int | None = None
    sft_status: str = "unknown"


@dataclass(frozen=True)
class CausalLMTaskSpec:
    """Task metadata shared by planning, training, and diagnostics."""

    name: str
    modality: str
    sequence_length: int
    alphabet: str
    data: SequenceDataSpec
    policy: CausalLMPolicySpec
    program: TaskProgram
    objective: ObjectiveDescriptor
    initial_reward_spec: dict[str, Any] = field(default_factory=dict)
    source_role_policy: SourceRolePolicy | None = None
    failure_profile: FailureProfile = field(default_factory=shared_failure_profile)
    primitive_allowlist: tuple[str, ...] | None = None
    # Policy-stage metadata is optional: an adapter that declares no SFT recipe
    # stays valid.  The shared RL loop never trains SFT implicitly; runners may use
    # an adapter's provider when an explicit SFT stage is requested.
    sft_recipe: SFTRecipe | None = None
    # Optional neutral representation contract.  DNA adapters leave this unset
    # and use the plain character alphabet; non-DNA sequence adapters should
    # declare their tokenizer/length/EOS semantics here.
    representation: RepresentationSpec | None = None
    success_spec: SuccessSpec | None = None
    recipe_capabilities: RecipeCapabilities | None = None


def validate_task_spec(spec: CausalLMTaskSpec) -> list[str]:
    """Return structural issues in an adapter descriptor."""
    problems: list[str] = []
    if not spec.name.strip():
        problems.append("task name must be non-empty")
    modality = spec.modality.lower().strip()
    if modality not in {"dna", "protein"}:
        problems.append(
            "causal-LM sequence adapters support DNA and protein modalities"
        )
    if spec.sequence_length < 1:
        problems.append("sequence_length must be positive")
    if not spec.alphabet:
        problems.append("alphabet must be non-empty")
    allowed_lengths = {0, spec.sequence_length}
    representation = spec.representation
    if representation is not None:
        if representation.modality != modality:
            problems.append("representation modality must match task modality")
        if representation.alphabet != spec.alphabet:
            problems.append("representation alphabet must match task alphabet")
        if representation.max_length is not None:
            allowed_lengths.add(representation.max_length)
            if representation.variable_length and representation.eos_token is not None:
                # An explicit terminator consumes a token but is not a residue.
                allowed_lengths.add(representation.max_length + 1)
        if representation.min_length > spec.sequence_length:
            problems.append("representation min_length exceeds sequence_length")
    elif modality == "protein":
        problems.append("protein task specs must declare a representation")
    if spec.policy.max_completion_length not in allowed_lengths:
        problems.append(
            "policy.max_completion_length must be zero or match the task length"
        )
    if (
        modality == "dna"
        or representation is None
        or representation.tokenizer == "residue"
    ) and spec.policy.token_alphabet.upper() != spec.alphabet.upper():
        problems.append("policy token_alphabet must match task alphabet")
    if spec.failure_profile is None:
        problems.append("failure_profile must be provided")
    return problems


def source_role_policy_from_contract(
    contract: dict[str, Any],
    *,
    fold: str = "primary",
) -> SourceRolePolicy:
    """Translate either shipped contract role layout into one shared policy."""
    candidates = {
        str(entry.get("id"))
        for entry in (contract.get("oracles", {}).get("candidate_pool") or [])
        if entry.get("id")
    }
    fold_names = (fold, f"{fold}_fold")
    roles = next(
        (
            role_map.get(name)
            for role_map in (
                contract.get("source_roles") or {},
                contract.get("oracles", {}).get("source_roles") or {},
                contract.get("self_validation") or {},
            )
            for name in fold_names
            if role_map.get(name) is not None
        ),
        None,
    )
    if roles is None or not isinstance(roles, dict):
        raise ValueError(f"contract has no source-role fold {fold!r}")

    def names(*keys: str) -> frozenset[str]:
        for key in keys:
            value = roles.get(key)
            if value is not None:
                return frozenset(str(item) for item in value)
        return frozenset()

    reward = names("reward_pool", "reward_oracles")
    sentinel = names("sentinel_pool", "held_out_oracles", "default_held_out_oracles")
    required_held = names("required_sentinels", "required_held_out_oracles")
    required_enabled = names("required_enabled_oracles")
    audit = names("audit_oracles")
    unknown = sorted((reward | sentinel | required_held | required_enabled | audit) - candidates)
    if unknown:
        raise ValueError(f"contract source roles name undeclared oracles: {unknown}")
    return SourceRolePolicy(
        known_oracles=frozenset(candidates),
        audit_oracles=audit,
        require_held_out=bool(sentinel),
        required_held_out_oracles=required_held,
        required_enabled_oracles=required_enabled,
        reward_oracles=reward,
        sentinel_oracles=sentinel,
    )


def task_spec_from_contract(
    contract: dict[str, Any],
    *,
    objective: ObjectiveDescriptor,
    sequence_length: int | None = None,
    sequence_column: str = "sequence",
    condition_column: str | None = None,
    initial_reward_spec: dict[str, Any] | None = None,
    source_role_policy: SourceRolePolicy | None = None,
    failure_profile: FailureProfile | None = None,
    sft_recipe: SFTRecipe | None = None,
    representation: RepresentationSpec | None = None,
    primitive_allowlist: tuple[str, ...] | None = None,
    recipe_capabilities: RecipeCapabilities | None = None,
) -> CausalLMTaskSpec:
    """Build a common task descriptor from an existing YAML contract."""
    task = contract.get("task") or {}
    data = contract.get("data") or {}
    representation_payload = (
        contract.get("representation")
        or data.get("representation")
        or {}
    )
    base_models = contract.get("base_models") or []
    base_model = base_models[0] if base_models else {}
    prompts = tuple(str(prompt) for prompt in data.get("conditioning_prompts") or [])
    modality = str(contract.get("modality") or "dna").strip().lower()
    alphabet_value = representation_payload.get("alphabet") or data.get(
        "token_alphabet"
    ) or "ACGT"
    # The shared policy descriptor keeps a character alphabet for consumers that
    # read one.  Token boundaries, when present, remain in RepresentationSpec.
    alphabet = (
        "".join(str(token) for token in alphabet_value)
        if not isinstance(alphabet_value, str)
        else str(alphabet_value)
    )
    length_raw = data.get("max_completion_length") or sequence_length
    if length_raw is None:
        raise ValueError(
            "task adapter must provide sequence_length when the contract does not"
        )
    length = int(length_raw)
    if representation is None and modality == "protein":
        representation_max = representation_payload.get("max_length", length)
        representation = RepresentationSpec(
            modality=modality,
            alphabet=alphabet,
            tokenizer=str(representation_payload.get("tokenizer", "residue")),
            min_length=int(
                representation_payload.get(
                    "min_length", data.get("min_new_tokens") or 1
                )
            ),
            max_length=(
                None
                if representation_max is None
                else int(representation_max)
            ),
            variable_length=bool(representation_payload.get("variable_length", False)),
            pad_token=representation_payload.get("pad_token", "<pad>"),
            eos_token=representation_payload.get("eos_token", "<eos>"),
            unk_token=representation_payload.get("unk_token", "<unk>"),
            prefix=str(representation_payload.get("prefix", "")),
            supports_edit_mask=bool(
                representation_payload.get("supports_edit_mask", False)
            ),
            token_vocabulary=tuple(
                str(token)
                for token in (representation_payload.get("token_vocabulary") or ())
            ),
        )
    program = TaskProgram(
        name=str(task.get("name") or ""),
        description=str(task.get("description") or ""),
        biological_constraints=list(task.get("biological_constraints") or []),
        target_objective=str(task.get("target_objective") or ""),
    )
    spec = CausalLMTaskSpec(
        name=program.name,
        modality=modality,
        sequence_length=length,
        alphabet=alphabet,
        data=SequenceDataSpec(
            source=str(data.get("unified_parquet") or data.get("source") or "") or None,
            sequence_column=sequence_column,
            condition_column=condition_column,
            split_column=str(data.get("split_column") or "split"),
        ),
        policy=CausalLMPolicySpec(
            model=str(base_model.get("model") or "") or None,
            conditioning_prompts=prompts,
            token_alphabet=alphabet,
            max_completion_length=length,
            min_new_tokens=(
                int(data["min_new_tokens"])
                if modality == "protein" and data.get("min_new_tokens") is not None
                else None
            ),
            sft_status=(
                sft_recipe.status
                if sft_recipe is not None
                else "reuse_existing"
                if base_model
                else "unknown"
            ),
        ),
        program=program,
        objective=objective,
        initial_reward_spec=dict(initial_reward_spec or {}),
        source_role_policy=source_role_policy,
        failure_profile=failure_profile or shared_failure_profile(),
        primitive_allowlist=primitive_allowlist,
        recipe_capabilities=(
            recipe_capabilities
            if recipe_capabilities is not None
            else RecipeCapabilities.from_dict(contract["recipe_capabilities"])
            if contract.get("recipe_capabilities") is not None
            else None
        ),
        sft_recipe=sft_recipe,
        representation=representation,
        success_spec=(
            SuccessSpec.model_validate(contract["success"])
            if contract.get("success") is not None
            else None
        ),
    )
    problems = validate_task_spec(spec)
    if problems:
        raise ValueError("invalid task spec: " + "; ".join(problems))
    return spec
