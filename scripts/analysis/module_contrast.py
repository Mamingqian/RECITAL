#!/usr/bin/env python3
"""The pass rule of `docs/reports/module_pass_rules_2026-11-04.md` applied to module arms, per task.

    module_contrast.py declare --n-arm TASK=N [TASK=N ...] [--n-comparator TASK=N ...] [--single-modules MODULE ...] --out DIR
    module_contrast.py floor   --stored CSV --fresh CSV --out DIR
    module_contrast.py core    --arm CSV --declared-tag TAG --declared-path PATH --floor-tag TAG --floor-path PATH --stored CSV --fresh CSV [--out DIR]
    module_contrast.py matched --arm CSV --control CSV [CSV ...] (the same tags and core files as core) [--out DIR]
    module_contrast.py amp     --arm CSV --comparator CSV --declared-tag TAG --declared-path PATH [--out DIR]
    module_contrast.py stack   --arm CSV --control CSV [CSV ...] (the same tags and core files as core) [--out DIR]

Each CSV is the `module_endpoints.csv` that `module_endpoints.py` writes for a group of cells. Nothing is read from a cell here.

Two files are committed and tagged in git before the arm they judge. `declare` writes `declared.json`, the arm size of each task of the pre-registration
(and the comparator size of the amp task); its tag must be earlier than the first event of every arm cell. `floor` writes `floor.json`: per task the noise
floor, the pooling decision and the scale, with the sha256 of the two core CSVs; it needs the fresh cores finished, so its tag must be earlier than the first
result of every arm cell (the earlier of its `run_finished` event and its terminal audit file, `finished_ts` of the endpoint table), not earlier than the
arm's start. Both tags are earlier by at least 60 s. The commands read the files from the tags (`git show TAG:PATH`), never from the working copy, refuse a core
CSV that no longer has the recorded sha256, and recompute every floor from the core CSVs and require equality with the file. The verdict records the blob
hashes it read. Residual: a back-dated commit, a forged `first_event_ts` or a forged `finished_ts` would defeat the time check, and `finished_ts` marks only the first `run_finished`
event or the audit file (an interim curve can be read before it); each needs a deliberate act of one of our own sessions, and the git history, the tag and the
hash-chained event log are the record (the repository keeps no SHA or source-identity approval gates).

  floor f       |mean(fresh) - mean(stored)| of aubc_norm, both normalized by the mean final value of the stored and fresh cores
                together; the floor of the blind result is the same difference of the task's blind result
  controls      (core) stored and fresh together when the exact two-sided permutation p of stored against fresh is at least 0.2,
                otherwise the fresh core alone; (matched) the control searches given, the best by mean `aubc` when several
  aubc_norm     aubc divided by the mean final value of the controls used (a search with no qualified library has aubc 0 and final 0)
  p             exact one-sided permutation p that the arm's mean aubc_norm exceeds the controls'; every split of the pooled
                values into groups of the two sizes is enumerated
  pass          arm of 4 or more searches: p <= 0.05, mean difference > f, and the arm's mean blind result not below the controls'
                by more than the blind floor. Arm of 3 or fewer: every arm search above the controls' mean aubc_norm, mean
                difference > f, and the same blind condition.

When the cores do not pool, the floor is in the scale of the two cores together and the difference in the fresh core's scale (the
reading the owner confirmed); `floor_in_control_scale` and `passes_if_floor_in_control_scale` show whether the other scale would
change the verdict.

What is refused: a cell whose status is `running` or `needs_attention`; a frozen cell without a complete terminal audit; an arm of a size other
than the one declared in `floor.json`; a core group other than three searches; a cell listed twice or in two groups; groups (arm, cores and matched
controls) whose `total_updates`, `arm` or `endpoint_metric` differ, or whose arm, fresh core and matched controls differ in resolved model (the
stored cores may differ in model: that is what the floor measures); core cells with a module declared; a cell with more than one resolved model.
A search that ends `completed_no_qualified_candidate` is scored: aubc 0, final value 0, and no blind result. The blind condition takes the
conservative reading until the owner decides: an arm with such a search does not meet it (the search then counts twice, as AUBC 0 and as a failed blind condition). `blind_not_worse_frozen_only` and
`passes_if_failures_left_out_of_blind` print the other reading (the search left out of the blind mean). An arm in which every search failed
fails (`passes` False). The blind result of each task is `audit_primary` (Loop-seq: `audit_secondary`, the worst-group median margin).

The amp task has no stored control; its rule is section 6 of `docs/reports/amp_task_design_2026-11-04.md`, applied to core (`amp --arm`)
against the reward-only arm (`--comparator`) with the sizes of `declared.json` (`n_arm` and `n_comparator` of the task `amp`) read from a tag
committed before the arm started (`--declared-tag`, `--declared-path`): `aubc_norm` normalizes by the comparator's mean final value; the floor is the sample standard deviation of the comparator's `aubc_norm`
values; the blind result is `audit_primary` (minus log10 of the median APEX E. coli MIC, so higher is better) and the arm's median across
searches may fall below the comparator's by at most 0.1 (the same conservative reading for a search without a library).

`stack` runs `core` (the full-stack arm against the cores) and `matched` (against the best of the single-module controls given, which must be exactly
the `single_modules` of `declared.json`, one CSV each) on the three tasks cre, loopseq and cm from the same arm CSV, so nothing is read from an unsigned verdict. The stack passes when it passes against core on at least two tasks,
the enhancer (cre) meets its three-search rule (every search above the control mean), and on every task it is not worse than the best single module: its
difference to that module is not below zero and the blind condition against that module is met (no tolerance until the owner confirms one).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

REPO = Path(__file__).resolve().parents[2]

# The terminal blind result a task is judged on (pass rules, section 3): Loop-seq on the worst-group median margin, which
# `module_endpoints.py` writes as `audit_secondary` (its `audit_primary` is the saturated hit rate); the others on `audit_primary`.
BLIND = {"loopseq": "audit_secondary"}
AMP_BLIND_TOLERANCE = 0.1
N_CORE = 3
MIN_TASKS_BETTER = 2
MARGIN_SECONDS = 60
STACK_TASKS = ('cre', 'loopseq', 'cm')
ALPHA = 0.05
POOL_P = 0.2
EPS = 1e-12
TERMINAL = ("frozen", "completed_no_qualified_candidate")
REQUIRED = ("cell", "campaign", "task", "status", "failed", "total_updates", "aubc", "final_best", "audit_primary",
            "audit_secondary", "arm", "modules", "models", "audited", "endpoint_metric", "first_event_ts", "finished_ts")


def permutation_p(a: Sequence[float], b: Sequence[float], *, alternative: str) -> float:
    """Exact permutation p of the difference of means mean(a) - mean(b); `greater` or `two-sided`."""
    if alternative not in ("greater", "two-sided"):
        raise ValueError(f"alternative {alternative!r} is neither greater nor two-sided")
    pooled = [*a, *b]
    n, total = len(a), sum(pooled)
    observed = sum(a) / len(a) - sum(b) / len(b)
    hits = splits = 0
    for chosen in itertools.combinations(range(len(pooled)), n):
        mean_a = sum(pooled[i] for i in chosen) / n
        difference = mean_a - (total - mean_a * n) / (len(pooled) - n)
        splits += 1
        hits += (difference >= observed - EPS) if alternative == "greater" else (abs(difference) >= abs(observed) - EPS)
    return hits / splits


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


# ----- what a group of cells must be before it is compared ----------------------------------------------------------

def require_columns(name: str, rows: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED if column not in rows.columns]
    if missing:
        raise SystemExit(f"{name}: the CSV lacks {missing}; write it again with the current module_endpoints.py")


def prepare(name: str, rows: pd.DataFrame, *, size: int | None = None, controls: bool = False) -> pd.DataFrame:
    """The group's rows, refused when they cannot be compared; a search with no library gets aubc 0 and final 0."""
    require_columns(name, rows)
    if rows.empty:
        raise SystemExit(f"{name}: no cells")
    open_ = rows[~rows["status"].isin(TERMINAL)]
    if len(open_):
        raise SystemExit(f"{name}: cells that are not finished cannot be compared: {', '.join(open_['cell'])}")
    if rows.duplicated("cell").any():
        raise SystemExit(f"{name}: a cell is listed twice: {', '.join(sorted(set(rows[rows.duplicated('cell')]['cell'])))}")
    if rows.duplicated("first_event_ts").any():
        raise SystemExit(f"{name}: cells with the same first event time: the same cell under two names "
                         f"({', '.join(sorted(set(rows[rows.duplicated('first_event_ts', keep=False)]['cell'])))})")
    if size is not None and len(rows) != size:
        raise SystemExit(f"{name}: {len(rows)} searches, the declared size is {size}")
    if controls and (rows["modules"] != "none").any():
        raise SystemExit(f"{name}: control cells with a module declared: {', '.join(rows[rows['modules'] != 'none']['cell'])}")
    if rows["models"].astype(str).str.contains(",").any():
        raise SystemExit(f"{name}: a cell recorded more than one resolved model id")
    unaudited = rows[(rows["status"] == "frozen") & ~rows["audited"].astype(bool)]
    if len(unaudited):
        raise SystemExit(f"{name}: frozen cells without a complete terminal audit: {', '.join(unaudited['cell'])}")
    rows = rows.copy()
    empty = rows["status"] == "completed_no_qualified_candidate"
    rows.loc[empty, ["aubc", "final_best"]] = 0.0
    if rows[["aubc", "final_best"]].isna().any().any():
        raise SystemExit(f"{name}: a finished cell has no aubc or final value")
    return rows


def compatible(groups: Mapping[str, pd.DataFrame], same_model: Sequence[str] = ()) -> None:
    """The groups of one comparison share the budget, the arm, the endpoint criterion and no cell; the groups named in `same_model`
    share the resolved model (the stored cores are left out: a different model is part of what their floor measures)."""
    frames = list(groups.values())
    for column, what in (("total_updates", "total updates"), ("arm", "protocol condition"), ("endpoint_metric", "endpoint criterion")):
        values = sorted({str(v) for frame in frames for v in frame[column]})
        if len(values) != 1:
            raise SystemExit(f"the groups differ in {what}: {values}")
    models = sorted({str(v) for name in same_model for v in groups[name]["models"]})
    if len(models) > 1:
        raise SystemExit(f"{', '.join(same_model)} differ in resolved model: {models}")
    seen = pd.concat([frame[["campaign", "cell"]].assign(group=name) for name, frame in groups.items()])
    shared = seen[seen.duplicated(["campaign", "cell"], keep=False)]
    if len(shared):
        raise SystemExit(f"a cell is in two groups: {', '.join(sorted(set(shared['cell'])))}")
    starts = {name: set(frame["first_event_ts"].astype(str)) for name, frame in groups.items()}
    for first, second in itertools.combinations(starts, 2):
        if starts[first] & starts[second]:
            raise SystemExit(f"{first} and {second} hold a cell with the same first event time ({sorted(starts[first] & starts[second])[0]}): "
                             "the same cell under two labels")


def blind_values(rows: pd.DataFrame, blind: str, *, required: bool = True) -> list[float]:
    """The blind results of the cells that have a library; a search with no library has none."""
    found = rows[rows["status"] == "frozen"][blind]
    if found.isna().any():
        raise SystemExit(f"a frozen cell has no {blind}")
    if found.empty and required:
        raise SystemExit("no cell has a blind result")
    return found.tolist()


def read_tagged(tag: str, path: str, repo: Path = REPO) -> tuple[str, str, str]:
    """(content, blob hash, commit time) of `path` as the tag `tag` holds it; the working copy is never read."""
    def git(*command: str) -> str:
        done = subprocess.run(["git", "-C", str(repo), *command], capture_output=True, text=True)
        if done.returncode:
            raise SystemExit(f"git {' '.join(command)}: {done.stderr.strip() or 'failed'}")
        return done.stdout
    content = git("show", f"{tag}:{path}")
    return content, git("rev-parse", f"{tag}:{path}").strip(), git("log", "-1", "--format=%cI", tag).strip()


def predates(commit_time: str, arm: pd.DataFrame, column: str, what: str) -> None:
    """The file was committed at least `MARGIN_SECONDS` before the earliest `column` of the arm's cells."""
    times = pd.to_datetime(arm[column], utc=True)
    if pd.isna(times).any():
        raise SystemExit(f"an arm cell has no {column}")
    if not pd.Timestamp(commit_time).tz_convert("UTC") + pd.Timedelta(seconds=MARGIN_SECONDS) < times.min():
        raise SystemExit(f"the {what} was committed at {commit_time}, not {MARGIN_SECONDS} s before the earliest {column} of the arm ({times.min()})")


def git_blob_hash(path: Path) -> str:
    """What `git hash-object` prints for the file."""
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# ----- the floor, written before any arm is read --------------------------------------------------------------------

def floors(stored: pd.DataFrame, fresh: pd.DataFrame, blind: str = "audit_primary") -> dict[str, object]:
    """The noise floor of one task, the pooling decision and the scale they are in; the two frames hold one task's cores."""
    stored, fresh = prepare("stored", stored, size=N_CORE, controls=True), prepare("fresh", fresh, size=N_CORE, controls=True)
    compatible({"stored": stored, "fresh": fresh})
    scale = pd.concat([stored, fresh])["final_best"].mean()
    if not scale > 0:
        raise SystemExit("the stored and fresh cores have no qualified library: aubc_norm has no scale")
    stored_norm, fresh_norm = (stored["aubc"] / scale).tolist(), (fresh["aubc"] / scale).tolist()
    p = permutation_p(fresh_norm, stored_norm, alternative="two-sided")
    return {"scale": float(scale), "floor": abs(_mean(fresh_norm) - _mean(stored_norm)),
            "blind_floor": abs(_mean(blind_values(fresh, blind)) - _mean(blind_values(stored, blind))),
            "p_stored_vs_fresh": p, "cores_pooled": bool(p >= POOL_P), "blind_column": blind,
            "n_stored": len(stored), "n_fresh": len(fresh)}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_floor(recorded: Mapping[str, object], found: Mapping[str, object]) -> list[str]:
    """The keys of the recomputed floor that differ from the recorded one."""
    different = []
    for key, value in found.items():
        old = recorded.get(key)
        same = (math.isclose(old, value, rel_tol=1e-9, abs_tol=1e-12)
                if isinstance(value, float) and isinstance(old, (int, float)) and not isinstance(old, bool) else old == value)
        if not same:
            different.append(key)
    return different


# ----- the verdict --------------------------------------------------------------------------------------------------

def blind_check(arm: pd.DataFrame, reference: Sequence[float], blind: str, tolerance: float, *, median: bool = False
                ) -> tuple[float, bool, bool]:
    """(difference of the arm's blind result to the reference's, met over the frozen cells only, met on the conservative reading).

    A search without a library has no blind result. The conservative reading is that an arm with such a search does not meet the
    condition; an arm in which every search failed has no blind difference and does not meet it either."""
    values = blind_values(arm, blind, required=False)
    failed = int((arm["status"] == "completed_no_qualified_candidate").sum())
    if not values:
        return math.nan, False, False
    center = pd.Series(values).median() if median else _mean(values)
    base = pd.Series(reference).median() if median else _mean(reference)
    difference = float(center - base)
    frozen_only = difference >= -tolerance - EPS
    return difference, bool(frozen_only), bool(frozen_only and failed == 0)


def arm_id(arm: pd.DataFrame) -> str:
    """A short hash of the arm's cells, so two verdicts of one task can be shown to be about the same searches."""
    names = sorted(f"{campaign}/{cell}" for campaign, cell in zip(arm["campaign"], arm["cell"]))
    return hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]


def verdict(arm: pd.DataFrame, controls: pd.DataFrame, info: Mapping[str, object], *, blind: str, n_arm: int,
            control_name: str = "controls") -> dict[str, object]:
    """One arm against the controls used, with the floors of `info` (prepared frames of one task)."""
    normalizer = controls["final_best"].mean()
    if not normalizer > 0:
        raise SystemExit("the controls have no qualified library: aubc_norm has no scale")
    arm_norm, control_norm = (arm["aubc"] / normalizer).tolist(), (controls["aubc"] / normalizer).tolist()
    floor, scale = float(info["floor"]), float(info["scale"])
    difference = _mean(arm_norm) - _mean(control_norm)
    p = permutation_p(arm_norm, control_norm, alternative="greater")
    p_worse = permutation_p(control_norm, arm_norm, alternative="greater")
    small_arm = n_arm <= 3
    wins = all(value > _mean(control_norm) for value in arm_norm) if small_arm else p <= ALPHA
    blind_difference, blind_frozen, blind_ok = blind_check(arm, blind_values(controls, blind), blind, float(info["blind_floor"]))
    floor_in_control_scale = floor * scale / normalizer
    above = difference > floor + EPS
    return {
        "arm_id": arm_id(arm), "n_arm": len(arm), "n_controls": len(controls), "control_name": control_name, "normalizer": normalizer,
        "floor": floor, "floor_scale": scale, "floor_in_control_scale": floor_in_control_scale,
        "blind_floor": info["blind_floor"], "cores_pooled": info.get("cores_pooled"),
        "p_stored_vs_fresh": info.get("p_stored_vs_fresh"), "arm_mean": _mean(arm_norm), "control_mean": _mean(control_norm),
        "difference": difference, "p_one_sided": p, "p_worse": p_worse,
        "rule": "same direction in every arm search" if small_arm else "p <= 0.05",
        "rule_met": bool(wins), "above_floor": bool(above), "blind_column": blind,
        "blind_difference": blind_difference, "blind_not_worse": blind_ok, "blind_not_worse_frozen_only": blind_frozen,
        "failed_arm_searches": int((arm["status"] == "completed_no_qualified_candidate").sum()),
        "passes": bool(wins and above and blind_ok),
        "passes_if_failures_left_out_of_blind": bool(wins and above and blind_frozen),
        "passes_if_floor_in_control_scale": bool(wins and difference > floor_in_control_scale + EPS and blind_ok),
    }


def contrast(arm: pd.DataFrame, stored: pd.DataFrame, fresh: pd.DataFrame, info: Mapping[str, object] | None = None, *,
             n_arm: int, blind: str = "audit_primary") -> dict[str, object]:
    """Core: the arm against the stored and fresh cores pooled, or the fresh core alone when they do not pool."""
    info = dict(info) if info is not None else floors(stored, fresh, blind)
    stored, fresh = prepare("stored", stored, size=N_CORE, controls=True), prepare("fresh", fresh, size=N_CORE, controls=True)
    arm = prepare("arm", arm, size=n_arm)
    compatible({"arm": arm, "stored": stored, "fresh": fresh}, same_model=("arm", "fresh"))
    controls = pd.concat([stored, fresh]) if info["cores_pooled"] else fresh
    return {**verdict(arm, controls, info, blind=blind, n_arm=n_arm), "mode": "core"}


def matched(arm: pd.DataFrame, controls: Mapping[str, pd.DataFrame], stored: pd.DataFrame, fresh: pd.DataFrame,
            info: Mapping[str, object] | None = None, *, n_arm: int, blind: str = "audit_primary") -> dict[str, object]:
    """The arm against a matched control (or the best of several single modules, by mean aubc), with the cores' floor."""
    info = dict(info) if info is not None else floors(stored, fresh, blind)
    stored, fresh = prepare("stored", stored, size=N_CORE, controls=True), prepare("fresh", fresh, size=N_CORE, controls=True)
    arm = prepare("arm", arm, size=n_arm)
    prepared = {name: prepare(name, rows) for name, rows in controls.items()}
    compatible({"arm": arm, "stored": stored, "fresh": fresh, **prepared}, same_model=("arm", "fresh", *prepared))
    best = max(prepared, key=lambda name: prepared[name]["aubc"].mean())
    result = {**verdict(arm, prepared[best], info, blind=blind, n_arm=n_arm, control_name=best), "mode": "matched"}
    result["control_candidates"] = {name: float(rows["aubc"].mean()) for name, rows in prepared.items()}
    return result


def amp_contrast(arm: pd.DataFrame, comparator: pd.DataFrame, *, n_arm: int, n_comparator: int) -> dict[str, object]:
    """The amp pass rule (design section 6) for one arm against its comparator arm."""
    arm, comparator = prepare("arm", arm, size=n_arm), prepare("comparator", comparator, size=n_comparator)
    compatible({"arm": arm, "comparator": comparator}, same_model=("arm", "comparator"))
    if len(comparator) < 2:
        raise SystemExit("the floor is the spread of the comparator's searches: it needs at least two")
    normalizer = comparator["final_best"].mean()
    if not normalizer > 0:
        raise SystemExit("the comparator has no qualified library: aubc_norm has no scale")
    arm_norm, comparator_norm = (arm["aubc"] / normalizer).tolist(), (comparator["aubc"] / normalizer).tolist()
    floor = float(pd.Series(comparator_norm).std(ddof=1))
    difference = _mean(arm_norm) - _mean(comparator_norm)
    p = permutation_p(arm_norm, comparator_norm, alternative="greater")
    small_arm = n_arm <= 3
    wins = all(value > _mean(comparator_norm) for value in arm_norm) if small_arm else p <= ALPHA
    blind_difference, blind_frozen, blind_ok = blind_check(arm, blind_values(comparator, "audit_primary"), "audit_primary",
                                                           AMP_BLIND_TOLERANCE, median=True)
    above = difference > floor + EPS
    return {
        "n_arm": len(arm), "n_comparator": len(comparator), "normalizer": normalizer, "floor": floor,
        "arm_mean": _mean(arm_norm), "comparator_mean": _mean(comparator_norm), "difference": difference,
        "p_one_sided": p, "rule": "same direction in every arm search" if small_arm else "p <= 0.05", "rule_met": bool(wins),
        "above_floor": bool(above), "blind_difference": blind_difference, "blind_tolerance": AMP_BLIND_TOLERANCE,
        "blind_not_worse": blind_ok, "blind_not_worse_frozen_only": blind_frozen,
        "failed_arm_searches": int((arm["status"] == "completed_no_qualified_candidate").sum()),
        "passes": bool(wins and above and blind_ok), "passes_if_failures_left_out_of_blind": bool(wins and above and blind_frozen),
    }


def full_stack(vs_core: Mapping[str, Mapping[str, object]], vs_best: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """The full-stack row of the pass rules from per-task verdicts against core (`core`) and against the best single module (`matched`)."""
    for name, verdicts, mode in (("vs-core", vs_core, "core"), ("vs-best", vs_best, "matched")):
        if set(verdicts) != set(STACK_TASKS):
            raise SystemExit(f"{name}: the full stack is judged on {sorted(STACK_TASKS)}, the file holds {sorted(verdicts)}")
        wrong = sorted(task for task, result in verdicts.items() if result.get("mode") != mode)
        if wrong:
            raise SystemExit(f"{name}: the verdicts of {wrong} do not come from `{mode}`")
    blobs = {result.get("floor_blob") for verdicts in (vs_core, vs_best) for result in verdicts.values()}
    if len(blobs) != 1 or None in blobs:
        raise SystemExit(f"the verdicts do not share one floor file: {sorted(map(str, blobs))}")
    different = sorted(task for task in STACK_TASKS if vs_core[task].get("arm_id") != vs_best[task].get("arm_id"))
    if different:
        raise SystemExit(f"the two files are not about the same stack searches on {different}")
    better = sorted(task for task, result in vs_core.items() if result["passes"])
    enhancer_ok = bool(vs_core["cre"]["rule_met"])
    not_worse = {task: bool(result["difference"] >= 0 and result["blind_not_worse"]) for task, result in vs_best.items()}
    return {"tasks": sorted(vs_core), "better_than_core_on": better, "enough_tasks": len(better) >= MIN_TASKS_BETTER,
            "enhancer_same_direction_3_of_3": enhancer_ok, "not_worse_than_best_single": not_worse,
            "passes": bool(len(better) >= MIN_TASKS_BETTER and enhancer_ok and all(not_worse.values()))}


# ----- the command line ---------------------------------------------------------------------------------------------

def _by_task(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    return frame[frame["task"] == task]


def _emit(result: object, out: Path | None, name: str) -> None:
    text = json.dumps(result, indent=1, sort_keys=True, default=lambda v: None if isinstance(v, float) and math.isnan(v) else v)
    print(text)
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / name).write_text(text + "\n")


def _read_floor(tag: str, path: str, stored: Path, fresh: Path, repo: Path = REPO
                ) -> tuple[dict[str, dict[str, object]], pd.DataFrame, pd.DataFrame, str, str]:
    """The recorded floors as the tag holds them, after the core CSVs and every recorded floor have been checked."""
    content, blob, committed = read_tagged(tag, path, repo)
    record = json.loads(content)
    for key, csv in (("stored_csv", stored), ("fresh_csv", fresh)):
        if record[key]["sha256"] != sha256_of(csv):
            raise SystemExit(f"{csv} is not the file the floor was written from ({key} sha256 differs)")
    stored_rows, fresh_rows = pd.read_csv(stored), pd.read_csv(fresh)
    for task, recorded in record["tasks"].items():
        found = floors(_by_task(stored_rows, task), _by_task(fresh_rows, task), BLIND.get(task, "audit_primary"))
        different = same_floor(recorded, found)
        if different:
            raise SystemExit(f"{task}: the floor file does not match the core files it names, in {different}")
    return record["tasks"], stored_rows, fresh_rows, blob, committed


def _read_declared(tag: str, path: str, repo: Path = REPO) -> tuple[dict[str, object], str, str]:
    """The declaration as the tag holds it: ({"n_arm": {task: N}, "n_comparator": {task: N}, "single_modules": [names]}, blob hash, commit time)."""
    content, blob, committed = read_tagged(tag, path, repo)
    declared = json.loads(content)
    if not {"n_arm", "n_comparator"} <= set(declared) <= {"n_arm", "n_comparator", "single_modules"}:
        raise SystemExit(f"{path} must hold n_arm and n_comparator (maps from task to size) and may hold single_modules")
    return declared, blob, committed


def _sizes(items: Sequence[str]) -> dict[str, int]:
    sizes = {}
    for item in items:
        task, _, size = item.partition("=")
        if not size.isdigit() or int(size) < 1:
            raise SystemExit(f"--n-arm takes TASK=N with N a positive integer, got {item!r}")
        sizes[task] = int(size)
    return sizes


def run_stack(arm: pd.DataFrame, controls: Mapping[str, pd.DataFrame], stored: pd.DataFrame, fresh: pd.DataFrame,
              floor: Mapping[str, Mapping[str, object]], sizes: Mapping[str, int], blob: str,
              single_modules: Sequence[str] = ()) -> dict[str, object]:
    """The full stack from its searches: `core` and `matched` are run here on the three tasks, then `full_stack`.

    The single-module controls must be exactly the modules the declaration lists, one CSV each, so a stronger module cannot be left out."""
    if set(arm["task"]) != set(STACK_TASKS):
        raise SystemExit(f"stack: the arm CSV holds tasks {sorted(set(arm['task']))}, the full stack is judged on {sorted(STACK_TASKS)}")
    if not single_modules:
        raise SystemExit("stack: the declaration lists no single_modules")
    named: dict[str, str] = {}
    for control_name, frame in controls.items():
        modules = sorted(set(frame["modules"].astype(str)))
        if len(modules) != 1:
            raise SystemExit(f"{control_name}: a single-module control holds one module, not {modules}")
        if modules[0] in named:
            raise SystemExit(f"two control files hold the module {modules[0]}")
        named[modules[0]] = control_name
    if sorted(named) != sorted(single_modules):
        raise SystemExit(f"the controls are the modules {sorted(named)}, the declaration lists {sorted(single_modules)}")
    vs_core, vs_best = {}, {}
    for task in STACK_TASKS:
        blind = BLIND.get(task, "audit_primary")
        if task not in floor or task not in sizes:
            raise SystemExit(f"{task}: the floor file or the declared sizes hold nothing for this task")
        vs_core[task] = {**contrast(_by_task(arm, task), _by_task(stored, task), _by_task(fresh, task), floor[task],
                                    n_arm=sizes[task], blind=blind), "floor_blob": blob}
        singles = {name: _by_task(frame, task) for name, frame in controls.items()}
        vs_best[task] = {**matched(_by_task(arm, task), singles, _by_task(stored, task), _by_task(fresh, task), floor[task],
                                   n_arm=sizes[task], blind=blind), "floor_blob": blob}
    return {"vs_core": vs_core, "vs_best": vs_best, "stack": full_stack(vs_core, vs_best)}


def _add_tags(command: argparse.ArgumentParser, *, floor: bool, declared: bool = True) -> None:
    if declared:
        command.add_argument("--declared-tag", required=True, help="the git tag on the commit that holds declared.json")
        command.add_argument("--declared-path", required=True, help="the repository path of declared.json in that tag")
    if floor:
        command.add_argument("--floor-tag", required=True, help="the git tag on the commit that holds floor.json")
        command.add_argument("--floor-path", required=True, help="the repository path of floor.json in that tag")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("declare", "floor", "core", "matched", "amp", "stack"):
        command = sub.add_parser(name)
        command.add_argument("--out", type=Path)
        command.add_argument("--repo", type=Path, default=REPO, help="the repository whose tags hold the committed files")
        if name == "declare":
            command.add_argument("--n-arm", nargs="+", required=True, metavar="TASK=N", help="the declared arm size of each task")
            command.add_argument("--n-comparator", nargs="*", default=[], metavar="TASK=N", help="the declared comparator size (amp)")
            command.add_argument("--single-modules", nargs="*", default=[], metavar="MODULE",
                                 help="the single modules a full stack is compared with (the `modules` value of each control file)")
        if name in ("floor", "core", "matched", "stack"):
            command.add_argument("--stored", type=Path, required=True)
            command.add_argument("--fresh", type=Path, required=True)
        if name in ("core", "matched", "amp", "stack"):
            command.add_argument("--arm", type=Path, required=True)
        if name in ("core", "matched", "stack"):
            _add_tags(command, floor=True)
        if name == "amp":
            _add_tags(command, floor=False)
            command.add_argument("--comparator", type=Path, required=True)
        if name in ("matched", "stack"):
            command.add_argument("--control", type=Path, nargs="+", required=True)
    args = parser.parse_args(argv)

    if args.command == "declare":
        declared = {"n_arm": _sizes(args.n_arm), "n_comparator": _sizes(args.n_comparator), "single_modules": sorted(args.single_modules)}
        _emit(declared, args.out, "declared.json")
        return 0
    if args.command == "floor":
        stored, fresh = pd.read_csv(args.stored), pd.read_csv(args.fresh)
        require_columns("stored", stored)
        require_columns("fresh", fresh)
        record = {"stored_csv": {"path": str(args.stored), "sha256": sha256_of(args.stored)},
                  "fresh_csv": {"path": str(args.fresh), "sha256": sha256_of(args.fresh)},
                  "tasks": {task: floors(_by_task(stored, task), _by_task(fresh, task), BLIND.get(task, "audit_primary"))
                            for task in sorted(fresh["task"].unique())}}
        _emit(record, args.out, "floor.json")
        return 0

    arm = pd.read_csv(args.arm)
    require_columns("arm", arm)
    declared, declared_blob, declared_time = _read_declared(args.declared_tag, args.declared_path, args.repo)
    predates(declared_time, arm, "first_event_ts", "declared sizes")
    if args.command == "amp":
        comparator = pd.read_csv(args.comparator)
        require_columns("comparator", comparator)
        others = sorted((set(arm["task"]) | set(comparator["task"])) - {"amp"})
        if others:
            raise SystemExit(f"amp is the rule of the amp task; the files also hold {others}")
        if "amp" not in declared["n_arm"] or "amp" not in declared["n_comparator"]:
            raise SystemExit(f"{args.declared_path} declares no amp sizes")
        result = amp_contrast(arm, comparator, n_arm=declared["n_arm"]["amp"], n_comparator=declared["n_comparator"]["amp"])
        _emit({"amp": {**result, "declared_blob": declared_blob}}, args.out, "contrast.json")
        return 0

    recorded, stored, fresh, floor_blob, floor_time = _read_floor(args.floor_tag, args.floor_path, args.stored, args.fresh, args.repo)
    predates(floor_time, arm, "finished_ts", "floor")
    controls = {str(path): pd.read_csv(path) for path in getattr(args, "control", [])}
    for control_name, frame in controls.items():
        require_columns(control_name, frame)
    if args.command == "stack":
        _emit(run_stack(arm, controls, stored, fresh, recorded, declared["n_arm"], floor_blob, declared.get("single_modules", [])),
              args.out, "stack.json")
        return 0
    result = {}
    for task in sorted(arm["task"].unique()):
        blind = BLIND.get(task, "audit_primary")
        if task not in recorded or task not in declared["n_arm"]:
            raise SystemExit(f"{task}: the floor file or the declared sizes hold nothing for this task")
        info, size = recorded[task], int(declared["n_arm"][task])
        if args.command == "core":
            result[task] = contrast(_by_task(arm, task), _by_task(stored, task), _by_task(fresh, task), info, n_arm=size, blind=blind)
        else:
            singles = {name: _by_task(frame, task) for name, frame in controls.items()}
            result[task] = matched(_by_task(arm, task), singles, _by_task(stored, task), _by_task(fresh, task), info, n_arm=size, blind=blind)
        result[task]["floor_blob"], result[task]["declared_blob"] = floor_blob, declared_blob
    _emit(result, args.out, "contrast.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
