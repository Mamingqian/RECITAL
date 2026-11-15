#!/usr/bin/env python3
"""Recompute post-RECOMB journal numbers directly from frozen cells on demand.

    RECITAL_RUNS_ROOT=/Users/mamingqian/RECITAL/work_data \
      PYTHONPATH=src:scripts:. python3 scripts/analysis/journal_paper_numbers.py

Reads (read-only, never modifies any cell):
  - AMP conflict objective: main batch (seeds 3101-3106: fixed, macamp, reward_only, core)
    and finished extension cells (seeds 3107-3110: reward_only, core; 10 vs 10 descriptive)
  - AMP broad-spectrum min3: 36 cells (12 vs 12 vs 12: fixed, reward_only, core, seeds 3101-3112)
  - AMP E. coli baseline matrix (block T & block A): S1 random, Eureka-style, AgentHPO-style,
    MAC-AMP port, core, reward_only, fixed_recipe_1280
  - Enhancer matched-depth fixed recipes (E4, 9,000 updates: motif, motif_gate, seeds 0-2)

Writes:
  - manuscript/numbers_journal.tex
  - docs/reports/journal_day1_ledger_2026-11-15/journal_numbers_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
for _entry in (REPO / "src", REPO / "scripts", REPO):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import fisher_exact  # noqa: E402

from analysis import (  # noqa: E402
    amp_broad_ext_readout,
    amp_conflict_extension_readout,
    amp_conflict_readout,
    e4_reading,
    module_contrast,
    module_endpoints,
    scripted_same_model,
)


def fmt(val: float, digits: int = 2) -> str:
    return f"{val:.{digits}f}"


def compute_amp_conflict(scratch: Path) -> dict[str, Any]:
    fixed_root = scratch / "amp_conflict_fixed_recipe_1280_h100_20261114"
    macamp_root = scratch / "amp_conflict_macamp_h100_20261114"
    ro_root = scratch / "amp_conflict_reward_only_h100_20261114"
    core_root = scratch / "amp_conflict_core_h100_20261114"
    ro_ext_root = scratch / "amp_conflict_reward_only_ext_h100_20261114"
    core_ext_root = scratch / "amp_conflict_core_ext_h100_20261114"

    # Declared 6v6v6v6 batch
    batch_cells = pd.concat(
        [
            amp_conflict_readout.read_arm("fixed", fixed_root, 6),
            amp_conflict_readout.read_arm("macamp", macamp_root, 6),
            amp_conflict_readout.read_arm("reward_only", ro_root, 6),
            amp_conflict_readout.read_arm("core", core_root, 6),
        ],
        ignore_index=True,
    )
    batch_summary = amp_conflict_readout.arm_table(batch_cells).set_index("arm")

    # Pooled 10v10 extension (descriptive)
    ext_cells = amp_conflict_extension_readout.read_arms(
        [
            f"fixed={fixed_root}:6",
            f"macamp={macamp_root}:6",
            f"reward_only={ro_root}:6,{ro_ext_root}:4",
            f"core={core_root}:6,{core_ext_root}:4",
        ]
    )
    ext_summary = amp_conflict_readout.arm_table(ext_cells.drop(columns=["seed", "part"])).set_index("arm")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        module_endpoints.main(["--out", str(td_path / "core6"), str(core_root)])
        module_endpoints.main(["--out", str(td_path / "ro6"), str(ro_root)])
        module_endpoints.main(["--out", str(td_path / "core10"), str(core_root), str(core_ext_root)])
        module_endpoints.main(["--out", str(td_path / "ro10"), str(ro_root), str(ro_ext_root)])

        df_core6 = pd.read_csv(td_path / "core6" / "module_endpoints.csv")
        df_ro6 = pd.read_csv(td_path / "ro6" / "module_endpoints.csv")
        df_core10 = pd.read_csv(td_path / "core10" / "module_endpoints.csv")
        df_ro10 = pd.read_csv(td_path / "ro10" / "module_endpoints.csv")
        contrast_10v10 = amp_conflict_extension_readout.descriptive_contrast(
            td_path / "core10" / "module_endpoints.csv",
            td_path / "ro10" / "module_endpoints.csv",
        )

    _, p_batch_1s = fisher_exact([[6, 0], [0, 6]], alternative="greater")
    _, p_batch_2s = fisher_exact([[6, 0], [0, 6]], alternative="two-sided")
    _, p_ext_1s = fisher_exact([[10, 0], [0, 6]], alternative="greater")
    _, p_ext_2s = fisher_exact([[10, 0], [0, 6]], alternative="two-sided")

    return {
        "batch_6v6": {
            "fixed_qual": int(batch_summary.loc["fixed", "cells_with_a_qualified_library"]),
            "macamp_qual": int(batch_summary.loc["macamp", "cells_with_a_qualified_library"]),
            "reward_only_qual": int(batch_summary.loc["reward_only", "cells_with_a_qualified_library"]),
            "core_qual": int(batch_summary.loc["core", "cells_with_a_qualified_library"]),
            "p_one_sided_vs_fixed": float(p_batch_1s),
            "p_two_sided_vs_fixed": float(p_batch_2s),
            "core_activity_mean": float(df_core6["final_best"].mean()),
            "ro_activity_mean": float(df_ro6["final_best"].mean()),
            "core_apex_median_log10": float(df_core6["audit_primary"].median()),
            "ro_apex_median_log10": float(df_ro6["audit_primary"].median()),
            "core_apex_median_mic_uM": float(10 ** (-df_core6["audit_primary"].median())),
            "ro_apex_median_mic_uM": float(10 ** (-df_ro6["audit_primary"].median())),
        },
        "pooled_10v10_descriptive": {
            "reward_only_qual": int(ext_summary.loc["reward_only", "cells_with_a_qualified_library"]),
            "core_qual": int(ext_summary.loc["core", "cells_with_a_qualified_library"]),
            "p_one_sided_vs_fixed": float(p_ext_1s),
            "p_two_sided_vs_fixed": float(p_ext_2s),
            "core_activity_mean": float(df_core10["final_best"].mean()),
            "ro_activity_mean": float(df_ro10["final_best"].mean()),
            "core_aubc_mean": float(df_core10["aubc"].mean()),
            "ro_aubc_mean": float(df_ro10["aubc"].mean()),
            "core_apex_median_log10": float(df_core10["audit_primary"].median()),
            "ro_apex_median_log10": float(df_ro10["audit_primary"].median()),
            "core_apex_median_mic_uM": float(10 ** (-df_core10["audit_primary"].median())),
            "ro_apex_median_mic_uM": float(10 ** (-df_ro10["audit_primary"].median())),
            "core_nonhemolytic_mean": float(ext_summary.loc["core", "mean_nonhemolytic_of_qualified"]),
            "ro_nonhemolytic_mean": float(ext_summary.loc["reward_only", "mean_nonhemolytic_of_qualified"]),
            "core_plddt_mean": float(ext_summary.loc["core", "mean_plddt_esmfold_of_qualified"]),
            "ro_plddt_mean": float(ext_summary.loc["reward_only", "mean_plddt_esmfold_of_qualified"]),
            "contrast": contrast_10v10,
        },
    }


def compute_amp_broad(scratch: Path) -> dict[str, Any]:
    fixed_r = [scratch / "amp_broad_fixed_recipe_1280_blockT_20261109", scratch / "amp_broad_fixed_recipe_1280_blockT_ext_20261112"]
    ro_r = [scratch / "amp_broad_reward_only_blockT_20261109", scratch / "amp_broad_reward_only_blockT_ext_20261112"]
    core_r = [scratch / "amp_broad_core_blockT_20261109", scratch / "amp_broad_core_blockT_ext_20261112"]

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for name, roots in [("fixed", fixed_r), ("ro", ro_r), ("core", core_r)]:
            module_endpoints.main(
                ["--metric", "amp=amp_selection_score_min3", "--out", str(td_path / name), *[str(r) for r in roots]]
            )
        df_fix = pd.read_csv(td_path / "fixed" / "module_endpoints.csv")
        df_ro = pd.read_csv(td_path / "ro" / "module_endpoints.csv")
        df_core = pd.read_csv(td_path / "core" / "module_endpoints.csv")

        rows = amp_broad_ext_readout.read_blind({"fixed": fixed_r, "reward_only": ro_r, "core": core_r})
        frame = pd.concat([pd.DataFrame(r).assign(arm=name) for name, r in rows.items()], ignore_index=True)
        medians = {name: float(group["B"].median()) for name, group in frame.groupby("arm", sort=False)}
        blind = {"medians": medians, "conditions": amp_broad_ext_readout.blind_conditions(rows)}

    norm_base = float(df_fix["final_best"].mean())
    return {
        "n_per_arm": 12,
        "fixed_min3_mean": float(df_fix["final_best"].mean()),
        "ro_min3_mean": float(df_ro["final_best"].mean()),
        "core_min3_mean": float(df_core["final_best"].mean()),
        "core_norm_final": float(df_core["final_best"].mean() / norm_base),
        "ro_norm_final": float(df_ro["final_best"].mean() / norm_base),
        "blind": blind,
    }


def compute_amp_ecoli_baselines(scratch: Path) -> dict[str, Any]:
    roots = {
        "core_blockT": [scratch / "amp_core_blockT_20261107"],
        "ro_blockT": [scratch / "amp_reward_only_blockT_20261107"],
        "s1_random_blockT": [scratch / "amp_scripted_random_blockT_20261109"],
        "eureka_blockT": [scratch / "amp_eureka_style_blockT_20261107"],
        "agenthpo_blockT": [scratch / "amp_agenthpo_style_blockT_20261107"],
        "fixed1280_blockT": [scratch / "amp_fixed_recipe_1280_blockT_20261107"],
        "core_12_blockT": [scratch / "amp_core_blockT_20261107", scratch / "amp_core_blockT_ext_20261108"],
        "ro_12_blockT": [scratch / "amp_reward_only_blockT_20261107", scratch / "amp_reward_only_blockT_ext_20261108"],
        "macamp_blockA": [scratch / "amp_macamp_port_20261107"],
        "core_blockA": [scratch / "amp_core_20261105"],
        "ro_blockA": [scratch / "amp_reward_only_20261105"],
    }
    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        dfs: dict[str, pd.DataFrame] = {}
        for name, rlist in roots.items():
            module_endpoints.main(["--out", str(td_path / name), *[str(r) for r in rlist]])
            df = pd.read_csv(td_path / name / "module_endpoints.csv")
            dfs[name] = df
            out[name] = {
                "n": int(len(df)),
                "final_mean": float(df["final_best"].mean()),
                "aubc_mean": float(df["aubc"].mean()),
                "apex_median_log10": float(df["audit_primary"].median()),
                "apex_median_mic_uM": float(10 ** (-df["audit_primary"].median())),
            }
        # Exact permutation contrasts on normalized AUBC
        scripted_same_model.install()
        out["s1_vs_ro_blockT"] = module_contrast.amp_contrast(dfs["s1_random_blockT"], dfs["ro_blockT"], n_arm=6, n_comparator=6)
        out["s1_vs_core_blockT"] = module_contrast.amp_contrast(dfs["s1_random_blockT"], dfs["core_blockT"], n_arm=6, n_comparator=6)
        out["agenthpo_vs_core_blockT"] = module_contrast.amp_contrast(dfs["agenthpo_blockT"], dfs["core_blockT"], n_arm=6, n_comparator=6)
        out["eureka_vs_ro_blockT"] = module_contrast.amp_contrast(dfs["eureka_blockT"], dfs["ro_blockT"], n_arm=6, n_comparator=6)
    return out


def compute_e4_enhancer(scratch: Path) -> dict[str, Any]:
    e4_root = scratch / "journal_e4_20261104"
    rows = []
    for arm in ("motif", "motif_gate"):
        for seed in (0, 1, 2):
            cell = e4_root / f"cre_fixedrecipe_{arm}_steps9000_seed{seed}"
            summary = json.loads((cell / "terminal_audit" / "summary.json").read_text())
            k562_cal = float(summary["primary"]["per_condition"]["K562"]["value"]) - 2.9068624042
            hepg2_cal = float(summary["primary"]["per_condition"]["HepG2"]["value"]) - (-2.9068624042)
            rows.append({"arm": arm, "seed": seed, "k562": k562_cal, "hepg2": hepg2_cal})
    df = pd.DataFrame(rows)
    return {
        "motif_k562_mean": float(df[df["arm"] == "motif"]["k562"].mean()),
        "motif_gate_k562_mean": float(df[df["arm"] == "motif_gate"]["k562"].mean()),
        "motif_hepg2_mean": float(df[df["arm"] == "motif"]["hepg2"].mean()),
        "motif_gate_hepg2_mean": float(df[df["arm"] == "motif_gate"]["hepg2"].mean()),
        "per_seed": rows,
    }


def write_tex_macros(summary: dict[str, Any], tex_path: Path) -> None:
    c6 = summary["amp_conflict"]["batch_6v6"]
    c10 = summary["amp_conflict"]["pooled_10v10_descriptive"]
    br = summary["amp_broad"]
    ec = summary["amp_ecoli_baselines"]
    e4 = summary["e4_enhancer"]

    b_meds = br["blind"]["medians"]
    fix_b_mic = 10 ** (-b_meds["fixed"])
    ro_b_mic = 10 ** (-b_meds["reward_only"])
    core_b_mic = 10 ** (-b_meds["core"])

    lines = [
        "% Generated by scripts/analysis/journal_paper_numbers.py — do not edit by hand.",
        "% AMP conflict-objective main batch (6 vs 6 vs 6 vs 6)",
        f"\\def\\ampConfBatchN{{6}}",
        f"\\def\\ampConfFixedQual{{{c6['fixed_qual']}}}",
        f"\\def\\ampConfMacampQual{{{c6['macamp_qual']}}}",
        f"\\def\\ampConfRoQual{{{c6['reward_only_qual']}}}",
        f"\\def\\ampConfCoreQual{{{c6['core_qual']}}}",
        f"\\def\\ampConfFisherOneSided{{{c6['p_one_sided_vs_fixed']:.4f}}}",
        f"\\def\\ampConfFisherTwoSided{{{c6['p_two_sided_vs_fixed']:.4f}}}",
        f"\\def\\ampConfCoreActMean{{{c6['core_activity_mean']:.2f}}}",
        f"\\def\\ampConfRoActMean{{{c6['ro_activity_mean']:.2f}}}",
        f"\\def\\ampConfCoreApexMic{{{c6['core_apex_median_mic_uM']:.1f}}}",
        f"\\def\\ampConfRoApexMic{{{c6['ro_apex_median_mic_uM']:.1f}}}",
        "% AMP conflict-objective pooled with extension (10 vs 10 descriptive)",
        f"\\def\\ampConfExtTotalN{{10}}",
        f"\\def\\ampConfExtRoQual{{{c10['reward_only_qual']}}}",
        f"\\def\\ampConfExtCoreQual{{{c10['core_qual']}}}",
        f"\\def\\ampConfExtCoreActMean{{{c10['core_activity_mean']:.2f}}}",
        f"\\def\\ampConfExtRoActMean{{{c10['ro_activity_mean']:.2f}}}",
        f"\\def\\ampConfExtCoreApexMic{{{c10['core_apex_median_mic_uM']:.1f}}}",
        f"\\def\\ampConfExtRoApexMic{{{c10['ro_apex_median_mic_uM']:.1f}}}",
        "% AMP broad-spectrum min3 (12 vs 12 vs 12)",
        f"\\def\\ampBroadN{{12}}",
        f"\\def\\ampBroadFixMinThree{{{br['fixed_min3_mean']:.2f}}}",
        f"\\def\\ampBroadRoMinThree{{{br['ro_min3_mean']:.2f}}}",
        f"\\def\\ampBroadCoreMinThree{{{br['core_min3_mean']:.2f}}}",
        f"\\def\\ampBroadFixWorstMic{{{fix_b_mic:.1f}}}",
        f"\\def\\ampBroadRoWorstMic{{{ro_b_mic:.1f}}}",
        f"\\def\\ampBroadCoreWorstMic{{{core_b_mic:.1f}}}",
        "% AMP E. coli baseline suite (block T, 6 seeds per arm)",
        f"\\def\\ampBaseN{{6}}",
        f"\\def\\ampSOneFinal{{{ec['s1_random_blockT']['final_mean']:.2f}}}",
        f"\\def\\ampSOneApexMic{{{ec['s1_random_blockT']['apex_median_mic_uM']:.1f}}}",
        f"\\def\\ampAgenthpoFinal{{{ec['agenthpo_blockT']['final_mean']:.2f}}}",
        f"\\def\\ampAgenthpoApexMic{{{ec['agenthpo_blockT']['apex_median_mic_uM']:.1f}}}",
        f"\\def\\ampEurekaFinal{{{ec['eureka_blockT']['final_mean']:.2f}}}",
        f"\\def\\ampEurekaApexMic{{{ec['eureka_blockT']['apex_median_mic_uM']:.1f}}}",
        f"\\def\\ampCoreTFinal{{{ec['core_blockT']['final_mean']:.2f}}}",
        f"\\def\\ampCoreTApexMic{{{ec['core_blockT']['apex_median_mic_uM']:.1f}}}",
        f"\\def\\ampRoTFinal{{{ec['ro_blockT']['final_mean']:.2f}}}",
        f"\\def\\ampRoTApexMic{{{ec['ro_blockT']['apex_median_mic_uM']:.1f}}}",
        "% CRE matched-depth 9,000-update fixed baselines (E4, 3 seeds per arm)",
        f"\\def\\creEFourMotifK{{{e4['motif_k562_mean']:.2f}}}",
        f"\\def\\creEFourMotifGateK{{{e4['motif_gate_k562_mean']:.2f}}}",
        f"\\def\\creEFourMotifH{{{e4['motif_hepg2_mean']:.2f}}}",
        f"\\def\\creEFourMotifGateH{{{e4['motif_gate_hepg2_mean']:.2f}}}",
    ]
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scratch",
        type=Path,
        default=Path(os.environ.get("RECITAL_RUNS_ROOT", "runs")),
    )
    parser.add_argument("--tex", type=Path, default=REPO / "manuscript" / "numbers_journal.tex")
    parser.add_argument(
        "--json",
        type=Path,
        default=REPO / "docs" / "reports" / "journal_day1_ledger_2026-11-15" / "journal_numbers_summary.json",
    )
    args = parser.parse_args()

    summary = {
        "amp_conflict": compute_amp_conflict(args.scratch),
        "amp_broad": compute_amp_broad(args.scratch),
        "amp_ecoli_baselines": compute_amp_ecoli_baselines(args.scratch),
        "e4_enhancer": compute_e4_enhancer(args.scratch),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    write_tex_macros(summary, args.tex)
    print(f"Wrote {args.tex} and {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
