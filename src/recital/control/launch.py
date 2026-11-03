"""Build one runnable cell from a run config: runtime, contract, header, backend, engine.

The controller process loads no model: its runtime carries no blind scorers, its oracle factories are lazy, and
the composition screen and delivery assembly read scores the workers already cached.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Collection, Mapping

import yaml

from recital.actions import generation_plan
from recital.capabilities import primitive_catalog, validate_recipe_capabilities
from recital.spec.composer import SourceRolePolicy, validate_spec
from recital.success import SuccessSpec

from . import events as ev
from . import recipes as rc
from .backend import SequenceExperimentBackend
from .contract import ControlContract
from .engine import ComponentRunner, Engine, EngineSettings
from .errors import ControlError
from .jobs import write_json
from .jsonio import canonical_json, sha256_hex
from .guard import blind_projection
from .projection import ArmView
from .prompts import static_header, strategist_instructions
from .runtime_build import (build_runtime, experiment_recipe_capabilities, mount_synthesis, persist_oracle_versions,
                            preflight)
from .scheduler import Scheduler
from .store import BlobStore, EventLog
from .transport import ResponsesTransport

CONFIG_SNAPSHOT = "config.yaml"
ARM = "no_self_validation"


def config_hash(config: Mapping[str, Any]) -> str:
    return sha256_hex(canonical_json(dict(config)))


def require_arm(task) -> None:
    """The control layer runs the one arm; a run config that names another protocol condition is refused."""
    if task.protocol_condition != ARM:
        raise ControlError(f"protocol_condition must be {ARM!r}, got {task.protocol_condition!r}")


def source_commit(repo_root: Path) -> str | None:
    """The commit of the source this run executes, recorded for provenance and never compared.

    A frozen copy of the source has no .git and carries its commit in SOURCE_COMMIT.
    """
    recorded = repo_root / "SOURCE_COMMIT"
    if recorded.is_file():
        return recorded.read_text().strip() or None
    try:
        done = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() or None if done.returncode == 0 else None


def snapshot_config(run_dir: Path, config: Mapping[str, Any]) -> Path:
    """`config.yaml` is the config the run started with; a changed config on resume is saved beside it.

    Workers read the newest snapshot, so a fix to a run's config takes effect on resume and the files still
    record what ran when.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(dict(config), sort_keys=True)
    base = run_dir / CONFIG_SNAPSHOT
    if not base.exists():
        base.write_text(text)
        return base
    saved = sorted(run_dir.glob("config.resume_*.yaml"))
    newest = saved[-1] if saved else base
    if newest.read_text() == text:
        return newest
    path = run_dir / f"config.resume_{len(saved) + 1:03d}.yaml"
    path.write_text(text)
    return path


def selection_rule_text(plan: Mapping[str, Any]) -> str:
    return ("The system, not you, chooses each condition's library from a candidate's evaluated pool with this rule: "
            + yaml.safe_dump(dict(plan), sort_keys=True, default_flow_style=True).strip())


def ranking_text(success: SuccessSpec) -> str:
    if not success.ranking:
        return "No ranking is declared: among qualified candidates the earliest wins."
    parts = []
    for criterion in success.ranking:
        tie = ("a value is tied with the best when its gap to the best is within the standard error of that gap"
               if criterion.tolerance == "auto" else f"a value within {criterion.tolerance} of the best is tied with it")
        parts.append(f"{criterion.direction} {criterion.metric} ({tie})")
    return ("Qualified candidates are compared per condition, each on that condition's own value of: "
            + "; then ".join(parts) + ". Tied candidates fall to the next criterion, then to the earlier candidate.")


def primitive_names(capabilities) -> dict[str, list[str]]:
    names: dict[str, list[str]] = {}
    for role, name in sorted(capabilities.allowed_primitives):
        names.setdefault(role, []).append(name)
    return names


def reward_role_policy(runtime: Mapping[str, Any]) -> SourceRolePolicy:
    """The part of the task's source-role policy that an agent's recipe can break.

    `held_out_oracles` is derived (`recipes._derive_held_out`) and this arm cannot see the sentinels the policy
    requires held out, so only the reward side is checked: every enabled source comes from the reward pool and every
    required reward source is enabled. The recipe validator enforces this policy and the header marks its
    `required_enabled_oracles` on their cards. A required source the arm cannot see is refused here: the arm's
    recipes could not enable it, and the refusal text that says so would name it.
    """
    policy = runtime["adapter"].task_spec().source_role_policy
    contract = runtime["contract"]
    hidden = sorted(set(policy.required_enabled_oracles) - set(contract.visible_sources))
    if hidden:
        raise ControlError(f"the {contract.protocol_condition} arm cannot see {hidden}, which the task's source-role "
                           "policy requires every recipe to enable")
    return SourceRolePolicy(known_oracles=policy.known_oracles, audit_oracles=policy.audit_oracles,
                            reward_oracles=policy.reward_oracles,
                            required_enabled_oracles=policy.required_enabled_oracles)


def recipe_validator(runtime: Mapping[str, Any], capabilities_of: Callable[[], Any],
                     resolver_of: Callable[[], Any]) -> Callable[[dict[str, Any]], list[str]]:
    contract = runtime["contract"]
    visible = sorted(contract.visible_sources)
    initial = dict(contract.initial_recipe)
    policy = reward_role_policy(runtime)

    def validate(recipe: dict[str, Any]) -> list[str]:
        problems = validate_spec(recipe, available_oracles=visible, source_role_policy=policy,
                                 primitive_resolver=resolver_of())
        problems += validate_recipe_capabilities(recipe, capabilities_of(), current_spec=initial)
        return problems

    return validate


AGENT_ROLES = ("per_oracle_utility", "cross_oracle_aggregation", "per_sequence_penalty", "hard_biological_gate")


def primitive_section(catalog: list[dict[str, Any]]) -> dict[str, Any]:
    """The catalog plus what an absent role means: a field with no available primitive is left null."""
    offered = {entry["role"] for entry in catalog}
    return {"catalog": catalog,
            "roles_with_no_primitive": [role for role in AGENT_ROLES if role not in offered],
            "null_aggregator": "the enabled oracles' per-oracle utilities are averaged"}


def header_for(task, control: ControlContract, ctx: rc.RecipeContext, success: SuccessSpec,
               catalog: list[dict[str, Any]], *, required: Collection[str]) -> str:
    """The static header one arm's every decision starts with.

    The header reads the contract with every sentinel concept taken out: held-out fields, sentinel role ids,
    and the sentences that explain them. The role it sees for a spendable source is "reward". `required` names the
    sources every recipe must enable (`reward_role_policy`); the card of each one the arm can see says so with
    `required: true`, and no other card has the key.
    """
    public = task.public_description()
    arm = ArmView.from_task(task)
    shown = blind_projection if arm.blind else (lambda value: value)
    cards = [{**card, "declared_role": ("reward" if "reward_active" in card["uses"] else "observed")}
             if arm.blind else card for card in public["capabilities"]["tools"]]
    cards = [{**card, "required": True} if card["name"] in required else card for card in cards]
    return static_header(
        task=shown(public["task"]), conditions=list(task.conditions),
        delivery_requirements=shown(public["delivery_requirements"]), sources=shown(cards),
        hyperparameter_bounds=shown(public["capabilities"]["hyperparameters"]),
        primitives=shown(primitive_section(catalog)),
        contract_summary={
            "candidates_per_round": control.proposals.candidates_per_round,
            "anchor": control.proposals.anchor_candidate,
            "rounds": {"minimum_before_a_stop": control.proposals.rounds_min, "maximum": control.proposals.rounds_max},
            "candidate_budget": control.proposals.candidate_budget,
            "steps": control.budget.steps.model_dump(), "total_updates": task.total_updates,
            "library_health": control.health.model_dump(),
            "components_available": control.synthesis.enabled,
            "stop": "needs every condition assigned a qualified completed candidate"},
        selection_rule=selection_rule_text(control.selection.plan), ranking=ranking_text(success),
        initial_recipe=shown(rc.visible_recipe(ctx.initial_recipe, task.visible_sources)))


@dataclass
class Prepared:
    engine: Engine
    settings: EngineSettings
    runtime: dict[str, Any]
    backend: SequenceExperimentBackend
    synthesis: Any
    log: EventLog
    declaration: dict[str, Any]

    def close(self) -> None:
        self.log.close()
        if self.synthesis is not None:
            self.synthesis.catalog.close()


def prepare(
    config: Mapping[str, Any],
    *,
    run_dir: str | Path,
    repo_root: str | Path,
    scheduler: Scheduler,
    transport: ResponsesTransport,
    device: str = "cuda",
    components: Callable[[Any], ComponentRunner] | None = None,
    may_submit: Callable[[], bool] = lambda: True,
) -> Prepared:
    """Everything a cell needs to run. `components(prepared_parts)` builds the Coder runner when synthesis is on."""
    run_dir, repo_root = Path(run_dir).expanduser().resolve(), Path(repo_root).resolve()
    control = ControlContract.from_config(config)
    runtime = build_runtime(config, repo_root=repo_root, output=run_dir, device="cpu", audit_oracle_factories={})
    declaration = preflight(runtime, device=device)
    task = runtime["contract"]
    require_arm(task)
    delivery_count = int(config["delivery"]["count"])
    control.validate_against(task.visible_sources, list(task.conditions), delivery_count)
    required = reward_role_policy(runtime).required_enabled_oracles
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = snapshot_config(run_dir, config)
    persist_oracle_versions(runtime, run_dir)
    write_json(run_dir / "declaration.json", declaration)

    synthesis = mount_synthesis(runtime, run_dir) if control.synthesis.enabled else None
    static = experiment_recipe_capabilities(runtime)
    capabilities_of = (lambda: synthesis.capabilities) if synthesis is not None else (lambda: static)
    resolver_of = (lambda: synthesis.catalog.resolve) if synthesis is not None else (lambda: None)
    ctx = rc.context_from(task, control, recipe_validator(runtime, capabilities_of, resolver_of))
    success = SuccessSpec.model_validate(task.delivery_requirements["success"])
    header_text = header_for(task, control, ctx, success, primitive_catalog(static), required=required)
    header = ev.RunStarted(
        schema_version=ev.SCHEMA_VERSION, task=str(config.get("task_family")), arm=task.protocol_condition,
        seed=task.seed, conditions=list(task.conditions), candidates_per_round=control.proposals.candidates_per_round,
        anchor_candidate=control.proposals.anchor_candidate,
        budgets=ev.Budgets(candidate_budget=control.proposals.candidate_budget, rounds_min=control.proposals.rounds_min,
                           rounds_max=control.proposals.rounds_max, total_updates=task.total_updates,
                           max_total_tokens=control.llm.max_total_tokens),
        contract_hash=control.contract_hash(), config_hash=config_hash(config),
        source_commit=source_commit(repo_root),
        oracle_versions_hash=sha256_hex(canonical_json(runtime.get("oracle_versions", {}))),
        model=control.llm.model, llm=control.llm.model_dump(mode="json"))
    settings = EngineSettings(
        run_dir=run_dir, config_path=config_path, repo_root=repo_root, device=device, task=task, control=control,
        recipe_ctx=ctx, success=success, definitions=runtime["measurement_definitions"],
        generation_plan=generation_plan(None, task), header_text=header_text,
        instructions=strategist_instructions(synthesis=control.synthesis.enabled), run_header=header,
        primitives=lambda: primitive_names(capabilities_of()))
    log = EventLog(run_dir / "events" / "log.jsonl")
    blobs = BlobStore(run_dir)
    backend = SequenceExperimentBackend(runtime, output=run_dir, device="cpu", synthesis=synthesis)
    parts = {"settings": settings, "runtime": runtime, "backend": backend, "synthesis": synthesis,
             "transport": transport, "blobs": blobs, "capabilities_of": capabilities_of}
    runner = components(parts) if control.synthesis.enabled and components is not None else None
    engine = Engine(settings, log=log, blobs=blobs, scheduler=scheduler, transport=transport,
                    assemble=backend.assemble_delivery, components=runner, may_submit=may_submit)
    state = engine.state()
    if state.run is not None and (not state.terminal or state.finished_status == "needs_attention"):
        engine.emit(ev.Resumed(torn_bytes=log.repaired_bytes, source_commit=header.source_commit,
                               contract_hash=header.contract_hash, config_hash=header.config_hash,
                               oracle_versions_hash=header.oracle_versions_hash))
        (run_dir / "needs_attention.json").unlink(missing_ok=True)
    return Prepared(engine=engine, settings=settings, runtime=runtime, backend=backend, synthesis=synthesis,
                    log=log, declaration=declaration)
