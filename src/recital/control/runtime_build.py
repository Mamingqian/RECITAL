"""Build a task runtime for the controller and its workers.

Blind oracles exist only in the terminal blind evaluation, so a runtime built here carries no
blind factories. `terminal_audit.py` builds its own.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, replace
from importlib import import_module
import json
from pathlib import Path

from recital.capabilities import RecipeCapabilities
from recital.control.jobs import write_json
from recital.dna_session import json_safe
from recital.library_selection import selection_capabilities
from recital.search_semantics import EVIDENCE_GUIDED
from recital.success import SuccessSpec
from recital.training.policy_precision import apply_training_precision, training_precision

TASK_MODULES = {"loopseq": "tasks.loopseq.experiment", "cre": "tasks.promoter.experiment",
                "cm": "tasks.protein.cm.experiment"}


def build_runtime(config, *, repo_root, output, device, audit_oracle_factories):
    """The task's runtime with its blind-oracle factories replaced.

    `audit_oracle_factories` is what the returned runtime carries under that key. The
    controller and every worker pass `{}`; the task builds its blind factories lazily,
    so nothing is loaded and the replacement drops them. `None` keeps the task's own,
    for the terminal blind evaluation alone.
    """
    family = config.get("task_family")
    if family not in TASK_MODULES:
        raise ValueError(f"task_family must be one of {sorted(TASK_MODULES)}")
    runtime = import_module(TASK_MODULES[family]).build_runtime(config, Path(repo_root), Path(output), device)
    if audit_oracle_factories is not None:
        runtime["audit_oracle_factories"] = dict(audit_oracle_factories)
    # The run declares its own policy-training precision, so a campaign that spans
    # hardware generations records which one each run trained in.
    runtime["training_precision"] = training_precision(config)
    apply_training_precision(runtime, config)
    return runtime


def _experiment_requirements(base, contract):
    """Read-only prompt context: adapter-declared requirements plus the contract's own.

    The adapter owns task requirements the contract does not restate -- loopseq's
    ``success`` and ``success_measurement``, CRE's ``score_semantics`` and
    ``training_reward_semantics``, CM's distance bounds, ``reference_libraries``
    bound ids and gate-cost caution. The contract's ``task`` and
    ``delivery_requirements`` are authoritative for their own keys, and the two
    ``delivery`` views are merged because they carry different fields: the adapter
    declares protocol, deduplication and exclusions, the contract declares required
    counts, identity and generation requirements.

    The keys of ``SuccessSpec`` describe a ranked search that stops on its own
    ranking, which this method does not do, so those keys alone are dropped.
    """
    legacy_search_keys = set(SuccessSpec.model_fields)
    requirements = {name: deepcopy(value) for name, value in dict(base.frozen_requirements).items()
                    if name not in legacy_search_keys}
    declared_delivery = requirements.get("delivery")
    delivery = deepcopy(dict(contract.delivery_requirements))
    if isinstance(declared_delivery, Mapping):
        delivery = {**deepcopy(dict(declared_delivery)), **delivery}
    elif declared_delivery is not None:
        delivery["declared"] = deepcopy(declared_delivery)
    requirements["task"] = deepcopy(dict(contract.task))
    requirements["delivery"] = delivery
    return requirements


def experiment_recipe_capabilities(runtime):
    """The task's recipe capabilities with its actual primitives, no frozen hyperparameters and primitive requests allowed."""
    adapter, contract = runtime["adapter"], runtime["contract"]
    if hasattr(adapter, "resolved_recipe_capabilities"):
        base = adapter.resolved_recipe_capabilities()
    elif hasattr(adapter, "capabilities"):
        base = adapter.capabilities()
    else:
        base = adapter.task_spec().recipe_capabilities
    if not isinstance(base, RecipeCapabilities):
        raise ValueError("task must declare its actual reward-component capabilities")
    allowed = set(base.allowed_primitives)
    initial = contract.initial_recipe
    fields = {"transforms": "per_oracle_normalization", "aggregator": "cross_oracle_aggregation",
              "shape": "per_sequence_shaping", "penalties": "per_sequence_penalty", "gates": "hard_biological_gate"}
    for field, role in fields.items():
        values = initial.get(field)
        for value in values if isinstance(values, list) else [values]:
            if isinstance(value, dict) and value.get("name"):
                allowed.add((role, value["name"]))
    editable = deepcopy(dict(contract.hyperparameter_bounds))
    editable["steps"] = {"type": "integer", "minimum": 1, "maximum": max(1, contract.total_updates)}
    # Library selection has its own catalog route and does not masquerade as a
    # reward pipeline slot. Existing tensor signatures determine reward roles.
    return replace(base,
        allowed_reward_fields=tuple(sorted(set(base.allowed_reward_fields) | {
            "enabled_oracles", "held_out_oracles", "target_objectives", "oracle_utilities",
            "aggregator", "penalties", "gates",
        })),
        editable_hyperparameters=editable, frozen_hyperparameters={},
        allowed_primitives=tuple(sorted(allowed)),
        frozen_requirements=_experiment_requirements(base, contract),
        frozen_target_objectives=deepcopy(contract.task.get("reward_target_objectives", {})),
        allow_primitive_requests=True, admitted_primitives={}, approved_references={},
        search_semantics=EVIDENCE_GUIDED,
    )


def _active_sources(contract, *, blind):
    active = set(contract.visible_sources)
    if blind:
        active |= set(contract.terminal_blind_sources)
    return active


def _missing_assets(runtime, *, blind):
    contract = runtime["contract"]
    active = _active_sources(contract, blind=blind)
    all_sources = {tool.name for tool in contract.tools} | set(contract.terminal_blind_sources)
    return [record["path"] for source, records in runtime.get("asset_manifest", {}).items()
            if source in active or source not in all_sources for record in records if not record["exists"]]


def _check_scorers(runtime):
    contract = runtime["contract"]
    if not Path(contract.initial_checkpoint).is_dir():
        raise FileNotFoundError(f"initial policy is absent: {contract.initial_checkpoint}")
    if set(runtime["oracle_factories"]) != {tool.name for tool in contract.tools}:
        raise ValueError("runtime scorer inventory differs from experiment capabilities")


def check_worker_runtime(runtime):
    """What a worker needs before it loads a model: its policy, its scorers' assets.

    Blind sources are not checked: no worker scores them.
    """
    _check_scorers(runtime)
    missing = _missing_assets(runtime, blind=False)
    if missing:
        raise FileNotFoundError("required experiment assets are absent: " + "; ".join(missing))
    contract = runtime["contract"]
    if not _active_sources(contract, blind=False) <= runtime.get("oracle_versions", {}).keys():
        raise ValueError("active experiment scorers lack model version declarations")


def preflight(runtime, *, device):
    """The controller's read-only check and declaration: no model, no API call."""
    contract = runtime["contract"]
    capabilities = experiment_recipe_capabilities(runtime)
    _check_scorers(runtime)
    missing = _missing_assets(runtime, blind=True)
    if missing:
        raise FileNotFoundError("required experiment assets are absent: " + "; ".join(missing))
    if runtime.get("audit_oracle_factories"):
        raise ValueError("a controller runtime carries no blind scorers")
    if not _active_sources(contract, blind=True) <= runtime.get("oracle_versions", {}).keys():
        raise ValueError("active experiment scorers lack model version declarations")
    return json_safe({
        "status": "preflight_passed", "device": device,
        "training_precision": runtime.get("training_precision", "task_default"),
        "bf16_requested": bool(runtime.get("bf16", str(device) != "cpu")),
        "contract": asdict(contract),
        "recipe_capabilities": asdict(capabilities),
        "selection_capabilities": selection_capabilities(),
        "training_reference_rows": len(runtime["training_sequences"]),
        "representation": runtime["representation"], "assets": runtime.get("asset_manifest", {}),
        "measurement_definitions": runtime.get("measurement_definitions", {}),
        "models_loaded": False, "api_calls": 0, "training_started": False,
    })


def persist_oracle_versions(runtime, output):
    """Bind cached measurements to the predictor files declared at run creation."""
    contract = runtime["contract"]
    active = _active_sources(contract, blind=True)
    versions = json_safe({name: runtime["oracle_versions"][name] for name in sorted(active)})
    all_sources = {tool.name for tool in contract.tools} | set(contract.terminal_blind_sources)
    versions["_references"] = json_safe({name: records for name, records in runtime.get("asset_manifest", {}).items()
                                         if name not in all_sources})
    path = Path(output) / "oracle_versions.json"
    if path.exists() and json.loads(path.read_text()) != versions:
        raise ValueError("experiment scorer assets changed; cached measurements cannot be reused")
    if not path.exists():
        write_json(path, versions)


def mount_synthesis(runtime, output):
    """The run's extension catalog, with every published extension loaded.

    No Coder is mounted: a worker resolves primitives that already exist, and the
    controller owns synthesis. The caller closes `synthesis.catalog`.
    """
    from recital.synthesis.bindings import build_recipe_synthesis

    references_bytes = sum(len(sequence) + 4 for sequence in runtime["training_sequences"])
    synthesis, _ = build_recipe_synthesis(
        output_dir=Path(output), capabilities=experiment_recipe_capabilities(runtime),
        training_sequences=runtime["training_sequences"], allowed_oracles=set(runtime["contract"].visible_sources),
        representation={key: runtime["representation"][key] for key in ("alphabet", "min_length", "max_length")},
        limits={"max_message_bytes": max(16 * 1024**2, 2 * references_bytes + 1024**2)},
    )
    return synthesis
