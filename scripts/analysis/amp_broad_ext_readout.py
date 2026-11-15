#!/usr/bin/env python3
"""The pooled reading of the AMP broad-spectrum extension: P1 and P2 at twelve against twelve, the visible clauses and the blind clause read apart.

    amp_broad_ext_readout.py --selftest
    amp_broad_ext_readout.py visible --fixed CSV --reward-only CSV --core CSV [--out DIR]
    amp_broad_ext_readout.py blind --fixed ROOT ROOT --reward-only ROOT ROOT --core ROOT ROOT [--out DIR]

An arm is its twelve cells: the six of stage 1 (seeds 3101 to 3106, roots `amp_broad_*_blockT_20261109`) and the six of the extension (seeds 3107 to 3112,
roots `amp_broad_*_blockT_ext_20261112`).

`visible` takes, per arm, the `module_endpoints.py --metric amp=amp_selection_score_min3` table of its twelve cells. The sizes are the 12 and 12 of the extension's
`declared.json` as tag prereg-amp-broad-ext-2026-11-12 holds it (`module_contrast._read_declared`, the committed parser). P2 is `module_contrast.amp_contrast`
(normalized AUBC, core against reward-only). P1 is `amp_final_value_contrast.final_value_contrast` of each LLM arm against the fixed recipe (final value normalized by the
fixed recipe's mean) with the `same_model` check of `module_contrast.compatible` left out, as in stage 1: the fixed recipe's resolved model is `scripted`, the searches'
`gpt-6-luna`. Only the visible keys of those results are printed or written; the E. coli-head fields they also compute (`audit_primary`) are not. The seeds 3101 to 3106
and 3107 to 3112 are also read apart at six against six (descriptive).

`blind` reads the terminal audit of every cell with the stage-1 script's functions (`amp_broad_blind_readout`): B = minus log10 of the median over the delivered 30 of each
peptide's worst-pathogen APEX MIC; the three conditions of the stage-1 declaration (P1 each LLM arm against the fixed recipe at least +0.1, P2 core against reward-only at least
-0.1) on the median B of each arm's twelve cells, over the cells that have a B and on the conservative reading. Each root must hold exactly its six cells.

Both commands refuse unless tag prereg-amp-broad-stage1-2026-11-09 holds the stage-1 declared files unchanged and tag prereg-amp-broad-ext-2026-11-12 exists, is more than 60 s old
and holds the extension's declared files byte for byte. The extension's declaration is `docs/reports/tier1_item9_amp_broad_extension_declaration_2026-11-12.md`.
Run with PYTHONPATH including src and scripts.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import pandas as pd

from analysis import amp_broad_blind_readout as bb
from analysis import amp_final_value_contrast as fv
from analysis import module_contrast as mc

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
STAGE1_TAG = "prereg-amp-broad-stage1-2026-11-09"
EXT_TAG = "prereg-amp-broad-ext-2026-11-12"
CAMPAIGN = REPO / "experiments/recital_amp_broad_ext_20261112"
DECLARED_PATH = "experiments/recital_amp_broad_ext_20261112/declared.json"
DECLARATION = REPO / "docs/reports/tier1_item9_amp_broad_extension_declaration_2026-11-12.md"
DECLARED_FILES = (CAMPAIGN / "campaign.json", CAMPAIGN / "declared.json", DECLARATION, HERE,
                  REPO / "slurm/amp_broad_ext_block_t_holder.sbatch", REPO / "slurm/amp_broad_ext_block_t_submit.sh")
STAGE1_SEEDS = tuple(range(3101, 3107))
EXT_SEEDS = tuple(range(3107, 3113))
VISIBLE = ("n_arm", "n_comparator", "normalizer", "floor", "arm_mean", "comparator_mean", "difference", "p_one_sided", "rule", "rule_met", "above_floor",
           "failed_arm_searches")


def verify_files(tag: str, files: Sequence[Path]) -> None:
    """The tag exists, is more than 60 s old, and holds each file byte for byte."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout.strip()
    try:
        commit = git("rev-parse", f"{tag}^{{commit}}")
    except subprocess.CalledProcessError:
        raise SystemExit(f"tag {tag} does not exist") from None
    created = int(git("for-each-ref", f"refs/tags/{tag}", "--format=%(creatordate:unix)"))
    if time.time() - created < mc.MARGIN_SECONDS:
        raise SystemExit(f"tag {tag} is less than {mc.MARGIN_SECONDS} s old")
    for path in files:
        relative = path.relative_to(REPO).as_posix()
        blob = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{relative}"], capture_output=True).stdout
        if blob != path.read_bytes():
            raise SystemExit(f"{relative} differs from the file at tag {tag}")
    print(f"declaration verified at tag {tag} ({commit[:12]})")


def verify_tags() -> None:
    bb.check_declaration(STAGE1_TAG)
    verify_files(EXT_TAG, DECLARED_FILES)


def declared_sizes() -> tuple[int, int]:
    declared, _, _ = mc._read_declared(EXT_TAG, DECLARED_PATH, REPO)
    return int(declared["n_arm"]["amp"]), int(declared["n_comparator"]["amp"])


@contextlib.contextmanager
def without_same_model() -> Iterator[None]:
    """`module_contrast.compatible` with every check but the resolved model's: the fixed recipe's is `scripted`, the searches'."""
    original = mc.compatible
    mc.compatible = lambda groups, same_model=(): original(groups, ())
    try:
        yield
    finally:
        mc.compatible = original


def pick(result: Mapping[str, object], *extra: str) -> dict[str, object]:
    found = {key: result[key] for key in (*VISIBLE, *extra)}
    found["visible_clauses_met"] = bool(result["rule_met"] and result["above_floor"])
    return found


def require_seeds(name: str, frame: pd.DataFrame, expected: Sequence[int]) -> None:
    seeds = sorted(int(s) for s in frame["seed"])
    if seeds != sorted(expected):
        raise SystemExit(f"{name}: seeds {seeds}, the declared seeds are {sorted(expected)}")


def read_visible(fixed: pd.DataFrame, reward_only: pd.DataFrame, core: pd.DataFrame, *, n_arm: int, n_comparator: int) -> dict[str, object]:
    """P2 (core against reward-only, normalized AUBC) and P1 (each LLM arm against the fixed recipe, final value), visible keys only."""
    arm, comparator = mc.prepare("core", core, size=n_arm), mc.prepare("reward_only", reward_only, size=n_comparator)
    p2 = pick(mc.amp_contrast(core, reward_only, n_arm=n_arm, n_comparator=n_comparator))
    normalizer = comparator["final_best"].mean()
    p2["p_reverse_reward_only_higher"] = mc.permutation_p((comparator["aubc"] / normalizer).tolist(), (arm["aubc"] / normalizer).tolist(), alternative="greater")
    p1 = {}
    with without_same_model():
        for name, frame in (("core", core), ("reward_only", reward_only)):
            result = fv.final_value_contrast(frame, fixed, n_arm=n_arm, n_comparator=n_comparator)
            p1[name] = pick(result, "p_two_sided")
    return {"P1": p1, "P2": p2}


def read_visible_by_stage(fixed: pd.DataFrame, reward_only: pd.DataFrame, core: pd.DataFrame) -> dict[str, dict[str, object]]:
    """The same reading on the six seeds of stage 1 and the six of the extension apart (descriptive)."""
    return {label: read_visible(*(frame[frame["seed"].isin(seeds)] for frame in (fixed, reward_only, core)), n_arm=len(seeds), n_comparator=len(seeds))
            for label, seeds in (("stage1_seeds_3101_3106", STAGE1_SEEDS), ("extension_seeds_3107_3112", EXT_SEEDS))}


def seed_of_cell(cell: Path) -> int:
    found = re.search(r"seed(\d+)$", cell.name)
    if not found:
        raise SystemExit(f"{cell.name}: no seed in the cell name")
    return int(found.group(1))


def read_blind(arm_roots: Mapping[str, Sequence[Path]]) -> dict[str, list[dict[str, object]]]:
    """One row per cell of each arm's two roots (the stage-1 root, then the extension's), each root holding exactly its six cells."""
    rows: dict[str, list[dict[str, object]]] = {}
    for name, roots in arm_roots.items():
        if len(roots) != 2:
            raise SystemExit(f"{name}: two roots are given, the stage-1 root and the extension's")
        rows[name] = []
        for root, seeds in zip(roots, (STAGE1_SEEDS, EXT_SEEDS)):
            found = sorted(seed_of_cell(cell) for cell in bb.cells_of(root))
            if found != list(seeds):
                raise SystemExit(f"{root}: seeds {found}, the declared seeds are {list(seeds)}")
            rows[name] += [{**row, "seed": seed_of_cell(Path(str(row["cell"])))} for row in bb.read_arm(root, len(seeds))]
    return rows


def blind_conditions(rows: Mapping[str, Sequence[Mapping]]) -> dict[str, dict[str, dict[str, object]]]:
    """The three conditions of the stage-1 declaration on the twelve cells of each arm, and apart on the six of stage 1 and the six of the extension."""
    def of(chosen: Sequence[int]) -> dict[str, dict[str, object]]:
        return bb.contrast(*([r for r in rows[name] if r["seed"] in chosen] for name in ("fixed", "reward_only", "core")))
    return {"pooled_12": of(STAGE1_SEEDS + EXT_SEEDS), "stage1_seeds_3101_3106": of(STAGE1_SEEDS), "extension_seeds_3107_3112": of(EXT_SEEDS)}


def selftest() -> None:
    cell = Path("amp_no_self_validation_seed3109")
    assert seed_of_cell(cell) == 3109
    flat = [1.0 + 0.01 * i for i in range(12)]
    rows = {name: [{"B": v, "seed": 3101 + i} for i, v in enumerate(values)]
            for name, values in (("fixed", flat), ("reward_only", flat), ("core", [v + 0.2 for v in flat]))}
    pooled = blind_conditions(rows)
    assert pooled["pooled_12"]["P1_core_vs_fixed"]["met"] and pooled["pooled_12"]["P2_core_vs_reward_only"]["met"]
    assert pooled["stage1_seeds_3101_3106"]["P1_core_vs_fixed"]["difference"] > 0.19
    original = mc.compatible
    with without_same_model():
        assert mc.compatible is not original
    assert mc.compatible is original
    frame = pd.DataFrame({"seed": [3101, 3102]})
    require_seeds("x", frame, [3102, 3101])
    try:
        require_seeds("x", frame, [3101, 3103])
    except SystemExit:
        pass
    else:
        raise AssertionError("seeds that are not the declared ones must be refused")
    print("selftest ok")


def table(fixed: pd.DataFrame, reward_only: pd.DataFrame, core: pd.DataFrame) -> pd.DataFrame:
    wide = pd.DataFrame({"seed": fixed["seed"].astype(int).to_numpy(), "fixed": fixed["final_best"].to_numpy()}).set_index("seed")
    wide["reward_only"] = reward_only.set_index(reward_only["seed"].astype(int))["final_best"]
    wide["core"] = core.set_index(core["seed"].astype(int))["final_best"]
    return wide.sort_index()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--selftest", action="store_true")
    commands = parser.add_subparsers(dest="command")
    visible = commands.add_parser("visible")
    for arm in ("fixed", "reward-only", "core"):
        visible.add_argument(f"--{arm}", type=Path, required=True)
    visible.add_argument("--out", type=Path)
    blind = commands.add_parser("blind")
    for arm in ("fixed", "reward-only", "core"):
        blind.add_argument(f"--{arm}", type=Path, nargs=2, required=True, metavar=("STAGE1_ROOT", "EXT_ROOT"))
    blind.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return 0
    if not args.command:
        parser.print_help()
        return 1
    verify_tags()
    out = args.out
    if out:
        out.mkdir(parents=True, exist_ok=True)
    if args.command == "visible":
        n_arm, n_comparator = declared_sizes()
        frames = {name: pd.read_csv(path) for name, path in (("fixed", args.fixed), ("reward_only", args.reward_only), ("core", args.core))}
        for name, frame in frames.items():
            require_seeds(name, frame, STAGE1_SEEDS + EXT_SEEDS)
        pooled = read_visible(frames["fixed"], frames["reward_only"], frames["core"], n_arm=n_arm, n_comparator=n_comparator)
        stages = read_visible_by_stage(frames["fixed"], frames["reward_only"], frames["core"])
        print(table(frames["fixed"], frames["reward_only"], frames["core"]).to_string(float_format=lambda v: f"{v:.4f}"))
        print(json.dumps({"pooled_12": pooled, **stages}, indent=1, default=float))
        if out:
            (out / "visible_pooled.json").write_text(json.dumps({"pooled_12": pooled, **stages}, indent=1, default=float) + "\n")
        return 0
    rows = read_blind({"fixed": args.fixed, "reward_only": args.reward_only, "core": args.core})
    frame = pd.concat([pd.DataFrame(r).assign(arm=name) for name, r in rows.items()], ignore_index=True)
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    conditions = blind_conditions(rows)
    for name, group in frame.groupby("arm", sort=False):
        print(f"{name}: median B {group.B.median():.4f} over {int(group.B.notna().sum())} of {len(group)} cells")
    for label, found in conditions.items():
        for name, c in found.items():
            print(f"{label} {name}: median difference {c['difference']:+.4f} (margin {c['margin']:+.1f}); met over cells with a B: {c['met_with_b']}; "
                  f"met on the conservative reading: {c['met']} ({c['arm_cells_without_b']} arm cells without a B)")
    if out:
        frame.to_csv(out / "blind_by_cell.csv", index=False)
        (out / "blind_conditions.json").write_text(json.dumps(conditions, indent=1, allow_nan=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
