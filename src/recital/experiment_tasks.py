"""Small task-factory helpers shared by the three experiment adapters."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from importlib import import_module
from pathlib import Path

import numpy as np
import jsonschema

from recital.actions import ExperimentContract, GENERATION_SCHEMA
from recital.library_selection import SelectionMeasurement
from recital.sequence_complexity import _dinucleotide_entropy, _kmer_counts, _max_homopolymer
from recital.success import SuccessSpec, assess_candidate


def path_from_root(value, repo_root):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(repo_root) / path).resolve()


def oracle_factory(class_path, **kwargs):
    """Do not import or initialize a model while describing a task."""
    def create():
        module, name = class_path.rsplit(".", 1)
        return getattr(import_module(module), name)(**kwargs)
    return create


def asset_manifest(paths):
    """Describe local inputs without loading models or imposing source-code gates."""
    result = {}
    for source, values in paths.items():
        result[source] = []
        for value in values:
            path = Path(value).expanduser().resolve()
            record = {"path": str(path), "exists": path.exists(), "kind": "directory" if path.is_dir() else "file"}
            if path.is_file():
                stat = path.stat()
                record.update(size_bytes=stat.st_size, modified_ns=stat.st_mtime_ns)
            elif path.is_dir() and ((path / "config.json").is_file() or (path / "saved_model.pb").is_file()):
                # Model directories need their actual weights/tokenizers bound,
                # while code repositories and environments are not code gates.
                members = [member for member in path.iterdir() if member.is_file()]
                if (path / "variables").is_dir():
                    members.extend(member for member in (path / "variables").rglob("*") if member.is_file())
                record["files"] = [{"path": str(member.relative_to(path)), "size_bytes": member.stat().st_size,
                                    "modified_ns": member.stat().st_mtime_ns} for member in sorted(members)]
            result[source].append(record)
    return result



DECLARED_SOURCE_ROLES = ("reward", "spendable_sentinel", "required_sentinel", "sentinel", "audit")


def declared_source_role(name, *, reward_pool=(), sentinel_pool=(), required=(), use_for=None):
    """The contract's designation for one predictor, as data rather than prose.

    CRE and CM declare it by pool membership: a source in both ``reward_pool`` and
    ``sentinel_pool`` is the one the task means to be promotable. Loop-seq declares
    the same vocabulary per candidate entry instead, so ``use_for`` is the fallback.

    ``required`` separates the two kinds of sentinel, and the distinction is the
    contract's, not a shade of one role: CM declares ``cm_max`` in the sentinel pool
    and outside the reward pool, but lists only ``function_2026_t5`` and ``cm_poet``
    under ``required_sentinels``. So ``cm_max`` may be dropped from
    ``held_out_oracles`` while those two may not, and none of the three may be
    rewarded.
    """
    reward_pool, sentinel_pool = set(reward_pool), set(sentinel_pool)
    in_reward, in_sentinel = name in reward_pool, name in sentinel_pool
    if in_reward and in_sentinel:
        return "spendable_sentinel"
    if in_reward:
        return "reward"
    if name in set(required):
        return "required_sentinel"
    if in_sentinel:
        return "sentinel"
    if reward_pool or sentinel_pool:
        # The contract declares pools and this source is in neither, which is how these
        # contracts express deliberate exclusion: a contract that leaves `grammar_prior` outside
        # every role cannot reward the accessibility oracle. Falling through to `use_for` would
        # read a `spendable_sentinel` hint on that entry and grant exactly what the pools exclude.
        return "sentinel"
    if use_for is None:
        # The task declares no roles at all. Absence of a declaration is not a
        # restriction to enforce, and `ROLE_USES` keeps all three uses for None.
        return None
    if use_for not in DECLARED_SOURCE_ROLES:
        # A contract may also declare `panel`, `optional_audit` or `ablation`, which this
        # mapping does not cover. Returning None for them would grant `reward_active` -- the
        # permissive answer to "what did you mean?".
        raise ValueError(
            f"source {name!r} declares use_for={use_for!r}, which carries no reward permission "
            f"in this framework; declare one of {list(DECLARED_SOURCE_ROLES)}")
    return use_for

def measurement(values, sources=(), **metadata):
    values = np.asarray(values)
    status = "measured" if np.isfinite(values).all() else "partial"
    return SelectionMeasurement(values=values, sources=tuple(sources), status=status, metadata=metadata)


def dna_quality_definitions(k):
    """Cheap sequence-local diagnostics, without task-specific cutoffs."""
    return {name: {"sources": [], "shape": [], "units": units, "meaning": meaning}
            for name, units, meaning in (
                ("gc_fraction", "fraction", "Fraction of sequence positions occupied by G or C"),
                ("dinucleotide_entropy", "bits", "Shannon entropy of overlapping sequence 2-mers"),
                ("unique_kmer_count", "count", f"Number of distinct overlapping sequence {k}-mers"),
                ("max_kmer_copy", "count", f"Maximum number of overlapping copies of one sequence {k}-mer"),
                ("max_homopolymer_length", "bases", "Longest contiguous run of one sequence symbol"),
            )}


def dna_quality_measurements(sequences, *, k):
    """Per-sequence measurements of the coordinates `dna_quality_definitions` declares, without pairwise/NN work."""
    values = {name: [] for name in dna_quality_definitions(k)}
    for sequence in sequences:
        counts = _kmer_counts(sequence, k)
        values["gc_fraction"].append(sum(base in "GC" for base in sequence) / len(sequence) if sequence else np.nan)
        values["dinucleotide_entropy"].append(_dinucleotide_entropy(sequence))
        values["unique_kmer_count"].append(len(counts))
        values["max_kmer_copy"].append(max(counts.values(), default=0))
        values["max_homopolymer_length"].append(_max_homopolymer(sequence))
    return {name: measurement(values[name], **{key: value for key, value in declaration.items()
                                              if key not in {"sources", "shape"}})
            for name, declaration in dna_quality_definitions(k).items()}


def counts_from_config(config, conditions):
    delivery = config["delivery"]
    count = delivery["count"]
    if type(count) is not int or count < 1:
        raise ValueError("delivery count must be a positive integer")
    if delivery["count_scope"] == "total":
        if count % len(conditions):
            raise ValueError("total delivery count must allocate equally across declared conditions")
        count //= len(conditions)
    elif delivery["count_scope"] != "per_condition":
        raise ValueError("unknown delivery count scope")
    if count < 1:
        raise ValueError("delivery requires a positive count per condition")
    return dict.fromkeys(conditions, count)


def delivery_success(base, counts):
    """Keep the declared bounds and the declared ranking; resize only the library counts.

    The declared ranking is what tells the agent when one qualified delivery beats
    another; over an empty ranking the first qualified library is already optimal,
    so the ranking must be carried through.

    `stop_when_qualified` stays False: qualification is a feasibility result, and the
    framework still states no preference about *when* to stop, only about which
    delivery is better.
    """
    value = deepcopy(base)
    value["stop_when_qualified"] = False
    for requirement in value["requirements"]:
        groups = requirement["strata"]
        n = sum(counts.values()) if groups == ["all"] else min(counts[group] for group in groups)
        if requirement.get("min_count", 1) > 1:
            requirement["min_count"] = n
        if requirement["metric"] in {"generated_count", "cm_delivered"}:
            requirement["minimum"] = n
            requirement["maximum"] = n
        elif requirement["metric"] == "cm_eligible":
            # A count of rows, so its floor is the library size the run actually
            # delivers, not the size the contract file was written for. No maximum:
            # the count is over the candidate pool, which is larger by design.
            requirement["minimum"] = n
    # Ranking metrics need their provenance kept too: `SuccessSpec.validate_information_access`
    # refuses a spec whose metrics lack explicit source provenance.
    used = {item["metric"] for item in (*value["requirements"], *(value.get("ranking") or []))}
    value["metric_sources"] = {key: sources for key, sources in value.get("metric_sources", {}).items()
                               if key in used}
    return SuccessSpec.model_validate(value)


def qualification(success, measured):
    result = assess_candidate(success, measured)
    return {"qualified": result.qualified, "status": result.status,
            "requirements": [asdict(item) for item in result.evidence], "measurements": measured}


def scalar_measurement(value, count, standard_error=None):
    result = {"value": None if value is None else float(value), "count": int(count),
              "status": "unavailable" if value is None else "measured"}
    if standard_error is not None:
        result["standard_error"] = float(standard_error)
    return result


BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0


def median_measurement(values):
    """A library's median over its finite rows with the bootstrap standard error that `tolerance: auto` reads.

    The error is the standard deviation (ddof=1) of the medians of 1000 resamples of the finite rows, each drawn with
    replacement to the library's own size from a generator seeded with 0, so one library always reports one error.
    A library of fewer than two finite rows carries no error.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return scalar_measurement(None, 0)
    if len(finite) < 2:
        return scalar_measurement(np.median(finite), len(finite))
    rows = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(finite), size=(BOOTSTRAP_RESAMPLES, len(finite)))
    error = np.std(np.median(finite[rows], axis=1), ddof=1)
    return scalar_measurement(np.median(finite), len(finite), standard_error=error)


def make_contract(config, *, task, conditions, initial_checkpoint, initial_recipe,
                  tools, feedback_sources, blind_sources, success, bounds, identity=None):
    for key, minimum in (("total_updates", 0), ("initial_schedule_horizon", 1), ("seed", 0)):
        value = config.get(key, 0) if key == "seed" else config[key]
        if type(value) is not int or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    initial_updates = config.get("initial_training_updates")
    if initial_updates is not None and (type(initial_updates) is not int or initial_updates < 1):
        raise ValueError("initial_training_updates must be a positive integer or null")
    jsonschema.Draft202012Validator(GENERATION_SCHEMA).validate(config["online_generation"])
    if not {"count", "count_scope"} <= config["online_generation"].keys():
        raise ValueError("online_generation requires count and count_scope")
    delivery_requirements = {"required_counts": counts_from_config(config, conditions),
                             "success": success.model_dump(mode="json")}
    if identity:
        # The task defines what counts as one delivered sequence; a selection plan
        # inherits it and cannot select a library under a weaker definition.
        delivery_requirements["identity"] = deepcopy(identity)
    # The size of the pool each candidate is evaluated on, stated from `online_generation`, the one declaration
    # that decides it.
    delivery_requirements["generation_requirements"] = {
        "count": config["online_generation"]["count"], "count_scope": config["online_generation"]["count_scope"]}
    condition = config.get("protocol_condition", "self_validating")
    hidden = set(feedback_sources) if condition != "self_validating" else set()
    visible = {tool.name for tool in tools} - hidden
    initial = deepcopy(initial_recipe)
    initial["held_out_oracles"] = [name for name in initial.get("held_out_oracles", []) if name in visible]
    # A run config declares its own window for a lever by name; it replaces the task
    # contract's window for that lever, and nothing else does.
    bounds = {**bounds, **deepcopy(config.get("hyperparameter_bounds", {}))}
    # A run may move its declared first segment off the contract's anchor, which is how
    # one spec declares more than one starting point. The override is still a
    # declaration and the contract's own bounds decide it: an unchecked value produces a
    # run whose declared first segment the agent is then refused for restating.
    anchor = config.get("initial_hyperparameters", {})
    for name, value in anchor.items():
        if name not in bounds:
            raise ValueError(f"initial_hyperparameters names {name!r}, which the contract does not declare editable")
        error = next(jsonschema.Draft202012Validator(bounds[name]).iter_errors(value), None)
        if error is not None:
            raise ValueError(f"initial_hyperparameters.{name} is outside the declared bound: {error.message}")
    initial["hyperparameters"].update(anchor)
    initial["hyperparameters"]["steps"] = initial_updates or config["initial_schedule_horizon"]
    return ExperimentContract(
        task=task, conditions=tuple(conditions), initial_checkpoint=str(initial_checkpoint),
        initial_recipe=initial, tools=tuple(tools), total_updates=config["total_updates"],
        initial_schedule_horizon=config["initial_schedule_horizon"],
        initial_training_updates=initial_updates,
        evaluation_plan={"generation_plan": deepcopy(config["online_generation"]),
                         "scorers": sorted(visible)},
        delivery_requirements=delivery_requirements,
        protocol_condition=condition, feedback_sources=tuple(feedback_sources),
        terminal_blind_sources=tuple(blind_sources),
        blind_report=deepcopy(config.get("blind_report", {})),
        hyperparameter_bounds=deepcopy(bounds), seed=config.get("seed", 0),
    )


AUDIT_STATISTICS = {"mean": np.mean, "median": np.median}


def declared_audit_statistic(blind_report):
    """The within-condition statistic the run declared before it started.

    The implementation fixes the other two fields, so a declaration it does not
    implement is refused rather than quietly reported under the wrong caliber.
    """
    statistic = blind_report.get("within_condition", "median")
    if statistic not in AUDIT_STATISTICS:
        raise ValueError(f"blind_report.within_condition={statistic!r} is not an implemented statistic")
    if blind_report.get("across_conditions", "minimum") != "minimum":
        raise ValueError("blind_report.across_conditions is reported as the worst condition")
    if blind_report.get("incomplete_scores",
                        "report_incomplete_with_original_denominator") != "report_incomplete_with_original_denominator":
        raise ValueError("incomplete blind scores are reported against the original denominator")
    return statistic


def summarize_audit(values, conditions, groups, *, statistic="median"):
    """Incomplete frozen-library scores never acquire a smaller denominator."""
    if statistic not in AUDIT_STATISTICS:
        raise ValueError(f"unknown audit statistic {statistic!r}")
    values = np.asarray(values, dtype=float)
    if values.shape != (len(conditions),):
        raise ValueError("audit values must align with the frozen library")
    per_condition = {}
    for group in groups:
        subset = values[np.asarray(conditions) == group]
        complete = bool(len(subset) and np.isfinite(subset).all())
        fn = AUDIT_STATISTICS[statistic]
        per_condition[group] = {"count": len(subset), "measured_count": int(np.isfinite(subset).sum()),
                                "value": float(fn(subset)) if complete else None,
                                "status": "complete" if complete else "incomplete"}
    complete = all(item["status"] == "complete" for item in per_condition.values())
    return {"statistic": statistic, "per_condition": per_condition,
            "worst_condition": min(item["value"] for item in per_condition.values()) if complete else None,
            "status": "complete" if complete else "incomplete"}
