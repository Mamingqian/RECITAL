"""CM cross-assay data preparation and frozen-encoder qualification."""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import average_precision_score, mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from .data import ALPHABET, download, write_fasta
from .family import FamilyReference, POTTS_ALPHABET
from .predictors import ESMFeatures, ProtT5Features, evaluation_metrics

CM_2020_URL = "https://zenodo.org/api/records/11176444/files/CM_table.csv/content"
ALPHAS = (10., 100., 1000., 10000.)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def homology_groups(sequences: list[str], *, distance: float = .2) -> np.ndarray:
    """Connected components ensure every close pair remains in the same group."""
    distances = process.cdist(sequences, sequences, scorer=Levenshtein.normalized_distance,
                             dtype=np.float32, workers=4)
    _, groups = connected_components(distances <= distance + 1e-7, directed=False)
    return groups


def grouped_split(table: pd.DataFrame, *, seed: int) -> np.ndarray:
    strata = table.sources.astype(str) + ":" + table.functional.astype(str)
    splitter = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    folds = list(splitter.split(table, strata, table.homology_group))
    result = np.full(len(table), "train", dtype=object)
    result[folds[0][1]], result[folds[1][1]] = "valid", "test"
    return result


def prepare_qualification(root: Path, pilot_root: Path, *, seed: int = 2001) -> dict:
    root, pilot_root = root.resolve(), pilot_root.resolve()
    for name in ("data", "oracles", "assets"):
        (root / name).mkdir(parents=True, exist_ok=True)
    for folder, names in {
        # prot-t5 belongs here: qualify_encoder("prot-t5") reads
        # root/assets/prot-t5.
        "assets": ["esm2-650m", "prot-t5", "bmDCA_params.zip", "meDCA_params.zip",
                   "eaDCA_params.zip", "edDCA_params.zip"],
        "data": ["natural_aligned.fasta", "sft.csv"],
    }.items():
        for name in names:
            path = root / folder / name
            if not path.exists():
                path.symlink_to(pilot_root / folder / name)
    if not (root / "sft").exists():
        (root / "sft").symlink_to(pilot_root / "sft")
    new = pd.read_csv(pilot_root / "data/function.csv")
    new["dataset"] = "2026"
    source = download(CM_2020_URL, root / "assets/CM_table.csv")
    raw = pd.read_csv(source)
    raw["sequence"] = raw.Sequence.str.replace("-", "", regex=False).str.upper()
    raw["target"] = pd.to_numeric(raw["norm r.e."], errors="coerce")
    eligible = raw[np.isfinite(raw.target) & raw.sequence.map(
        lambda s: 80 <= len(s) <= 96 and set(s) <= set(ALPHABET))].copy()
    overlap = eligible.sequence.isin(new.sequence)
    old = eligible.loc[~overlap, ["sequence", "target", "Model", "Activity"]].copy()
    old = old.rename(columns={"Model": "sources", "Activity": "functional"})
    if not old.sequence.is_unique or not new.sequence.is_unique:
        raise ValueError("Aggregate duplicate measurements before assigning splits")
    old["functional"] = old.functional.astype(bool)
    old["artificial"] = old.sources.ne("natural")
    old["dataset"] = "2020"
    old = old.sort_values("sequence").reset_index(drop=True)
    all_sequences = new.sequence.tolist() + old.sequence.tolist()
    groups = homology_groups(all_sequences)
    new["homology_group"], old["homology_group"] = groups[:len(new)], groups[len(new):]
    old["split"] = grouped_split(old, seed=seed)
    table = pd.concat([new, old], ignore_index=True)
    family = FamilyReference(pilot_root)
    descriptions = [family.describe(s) for s in all_sequences]
    table["natural_distance"] = [d["natural_distance"] for d in descriptions]
    table["baseline_natural"] = -table.natural_distance
    # EcCM is the named WT reference, not a function-selected natural sequence.
    wt = raw.loc[raw["ID to EcCM"].eq(1), "sequence"].iloc[0]
    table["baseline_eccm"] = [-Levenshtein.normalized_distance(s, wt) for s in all_sequences]
    codes = {aa: i for i, aa in enumerate(POTTS_ALPHABET)}
    alignment = np.asarray([[codes[a] for a in s] for s in family.aligned])
    counts = np.stack([(alignment == i).sum(0) for i in range(21)], axis=1) + 1.
    logp = np.log(counts / counts.sum(1, keepdims=True))
    table["baseline_profile"] = [float(logp[np.arange(96), [codes[a] for a in d["aligned"]]].mean())
                                 for d in descriptions]
    table.to_csv(root / "data/qualification.csv", index=False)
    for dataset in ("2026", "2020"):
        table[table.dataset.eq(dataset)].to_csv(root / f"data/function_{dataset}.csv", index=False)
    # Delivery exclusions include both regressor training sets plus the common SFT.
    training = set(family.training) | set(table.loc[table.split.eq("train"), "sequence"])
    write_fasta(root / "data/training_sequences.fasta", sorted(training))
    sft = set(pd.read_csv(root / "data/sft.csv").sequence)
    report = {
        "seed": seed, "2020_source": CM_2020_URL, "2020_raw": len(raw),
        "2020_excluded_exact_overlap": int(overlap.sum()), "2020_retained": len(old),
        "homology_rule": "connected components at normalized whole-sequence edit distance <= 0.20",
        "2020_group_disjoint": bool(old.groupby("homology_group").split.nunique().max() == 1),
        "2020_groups": int(old.homology_group.nunique()),
        "2020_largest_group": int(old.homology_group.value_counts().max()),
        "2020_sft_exact_overlap": int(old.sequence.isin(sft).sum()),
        "2020_labels": "published Activity classification; continuous target is norm r.e.",
        "2026_labels": "published Null Mode 3Sigma classification; mean M9c Norm R.E.",
        "2026_splits": "preserved pilot splits; meDCA source test already inspected in pilot",
        "counts": table.groupby(["dataset", "split"]).size().astype(int).to_dict(),
        "source_counts": old.groupby(["split", "sources"]).size().astype(int).to_dict(),
        "cross_dataset_related_groups": len(set(new.homology_group) & set(old.homology_group)),
    }
    for key in ("counts", "source_counts"):
        report[key] = {"/".join(k): int(v) for k, v in report[key].items()}
    write_json(root / "data/qualification_summary.json", report)
    return report


def cached_features(root: Path, model: str, sequences: list[str], *, device: str,
                    pilot_root: Path | None = None) -> np.ndarray:
    """Resume extraction in batches; sequence rows are the cache keys."""
    path = root / "oracles" / f"{model}_features.npz"
    cache = {}
    sources = [path]
    if pilot_root is not None:
        sources.append(pilot_root / "oracles" / f"{model}_features.npz")
    for source in sources:
        if source.exists():
            with np.load(source, allow_pickle=False) as stored:
                cache.update(zip(stored["sequences"].tolist(), stored["features"]))
    missing = list(dict.fromkeys(s for s in sequences if s not in cache))
    if missing:
        cls = ProtT5Features if model == "prot-t5" else ESMFeatures
        encoder = cls(root / "assets" / model, device=device)
        for start in range(0, len(missing), 256):
            chunk = missing[start:start + 256]
            cache.update(zip(chunk, encoder(chunk)))
            keys = [s for s in dict.fromkeys(sequences) if s in cache]
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, sequences=np.asarray(keys),
                                features=np.stack([cache[s] for s in keys]))
            temporary.replace(path)
            print(f"{model} embeddings: {min(start + 256, len(missing))}/{len(missing)} new", flush=True)
        del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.stack([cache[s] for s in sequences])


def fit_head(features: np.ndarray, target: np.ndarray, train: np.ndarray,
             valid: np.ndarray, *, seed: int) -> tuple[dict, np.ndarray]:
    scaler = StandardScaler().fit(features[train])
    x = scaler.transform(features)
    losses = {}
    for alpha in ALPHAS:
        head = Ridge(alpha=alpha).fit(x[train], target[train])
        losses[alpha] = float(mean_squared_error(target[valid], head.predict(x[valid])))
    alpha = min(losses, key=losses.get)
    heads = []
    for member in range(3):
        rng = np.random.default_rng(seed + member)
        indices = rng.choice(train, len(train), replace=True)
        heads.append(Ridge(alpha=alpha).fit(x[indices], target[indices]))
    prediction = np.stack([h.predict(x) for h in heads]).mean(0)
    # Affine, monotone scaling for online selection; never fit using terminal blind-evaluation labels.
    calibration = LinearRegression(positive=True).fit(prediction[valid, None], target[valid])
    return {"scaler": scaler, "heads": heads, "alpha": alpha, "valid_mse": losses,
            "calibration": {"slope": float(calibration.coef_[0]),
                            "intercept": float(calibration.intercept_)}}, prediction


def grouped_cv(features: np.ndarray, table: pd.DataFrame, *, seed: int) -> np.ndarray:
    """Three outer group folds; select alpha on an inner group holdout per fold."""
    groups = table.homology_group.to_numpy()
    target = table.target.to_numpy(dtype=np.float32)
    result = np.full(len(table), np.nan)
    for fold, (development, test) in enumerate(GroupKFold(3).split(features, groups=groups)):
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed + fold).split(
            features[development], groups=groups[development]))
        _, prediction = fit_head(features, target, development[a], development[b], seed=seed + fold)
        result[test] = prediction[test]
    return result


def metrics(table: pd.DataFrame, prediction: np.ndarray) -> dict:
    if not len(table):
        return {"n": 0}
    result = evaluation_metrics(table.target.to_numpy(), prediction,
                                table.functional.to_numpy(dtype=float))
    functional = table.functional.to_numpy(dtype=bool)
    result["auprc"] = float(average_precision_score(functional, prediction)) if functional.any() else None
    k = max(1, int(np.ceil(.1 * len(table))))
    indices = np.argsort(-prediction, kind="stable")[:k]
    result["top_decile_n"] = k
    result["top_decile_functional_fraction"] = float(functional[indices].mean())
    result["homology_groups"] = int(table.homology_group.nunique())
    return result


def cluster_intervals(table: pd.DataFrame, prediction: np.ndarray, *, seed: int,
                      repeats: int = 300) -> dict:
    if len(table) < 10:
        return {}
    groups = table.homology_group.to_numpy()
    blocks = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    target = table.target.to_numpy()
    rng = np.random.default_rng(seed)
    values = {"spearman": []}
    baselines = ["baseline_natural", "baseline_eccm", "baseline_profile"]
    values.update({"delta_spearman_vs_" + b: [] for b in baselines})
    for _ in range(repeats):
        indices = np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])
        corr = spearmanr(target[indices], prediction[indices]).statistic
        if not np.isfinite(corr):
            continue
        values["spearman"].append(float(corr))
        for name in baselines:
            baseline = spearmanr(target[indices], table[name].to_numpy()[indices]).statistic
            if np.isfinite(baseline):
                values["delta_spearman_vs_" + name].append(float(corr - baseline))
    return {key: np.quantile(value, [.025, .975]).tolist() for key, value in values.items() if value}


def evaluate_subsets(table: pd.DataFrame, prediction: np.ndarray, *, seed: int,
                     intervals: bool = True) -> dict:
    masks = {"all": np.ones(len(table), dtype=bool), "artificial": table.artificial.to_numpy()}
    masks.update({"source/" + str(s): table.sources.eq(s).to_numpy() for s in table.sources.unique()})
    for low, high in ((0., .2), (.2, .4), (.4, .6), (.6, 1.01)):
        masks[f"distance/{low:.1f}-{high:.1f}"] = (
            table.natural_distance.ge(low) & table.natural_distance.lt(high)).to_numpy()
    result = {}
    for name, mask in masks.items():
        subset, pred = table.loc[mask], prediction[mask]
        result[name] = metrics(subset, pred)
        if intervals and (name in ("all", "artificial", "source/BmDCA")):
            result[name]["cluster_bootstrap_95ci"] = cluster_intervals(subset, pred, seed=seed)
    return result


def qualify_encoder(root: Path, model: str, *, device: str, seed: int = 2001,
                    pilot_root: Path | None = None) -> dict:
    start = time.monotonic()
    table = pd.read_csv(root / "data/qualification.csv", dtype={"dataset": str})
    features = cached_features(root, model, table.sequence.tolist(), device=device, pilot_root=pilot_root)
    result = {"feature_model": model, "seed": seed, "datasets": {}}
    for dataset in ("2026", "2020"):
        train = np.flatnonzero(table.dataset.eq(dataset) & table.split.eq("train"))
        valid = np.flatnonzero(table.dataset.eq(dataset) & table.split.eq("valid"))
        target = table.target.to_numpy(dtype=np.float32)
        head, prediction = fit_head(features, target, train, valid, seed=seed)
        head.update(feature_model=model, dataset=dataset, target="normalized CM complementation enrichment",
                    training_sequences=table.iloc[train].sequence.tolist())
        name = ("function" if dataset == "2026" else "audit") + "_" + dataset
        name += "_t5" if model == "prot-t5" else "_esm"
        joblib.dump(head, root / "oracles" / f"{name}.joblib")
        table[name] = prediction
        report = {"alpha": head["alpha"], "calibration": head["calibration"], "evaluation": {}}
        for evaluated_dataset in ("2026", "2020"):
            for split in ("valid", "test"):
                mask = (table.dataset.eq(evaluated_dataset) & table.split.eq(split)).to_numpy()
                report["evaluation"][f"{evaluated_dataset}/{split}"] = evaluate_subsets(
                    table.loc[mask], prediction[mask], seed=seed, intervals=split == "test")
        if dataset == "2026":
            development = np.flatnonzero(table.dataset.eq(dataset) & table.split.ne("test"))
            oof = grouped_cv(features[development], table.iloc[development], seed=seed)
            table.loc[development, name + "_group_oof"] = oof
            report["group_cv_2026_development"] = evaluate_subsets(
                table.iloc[development], oof, seed=seed, intervals=False)
        result["datasets"][dataset] = report
        print(f"Fitted {name}: alpha={head['alpha']}, test="
              f"{report['evaluation'][dataset + '/test']['artificial']}", flush=True)
        table.to_csv(root / "oracles" / f"{model}_predictions.csv", index=False)
    result["elapsed_seconds"] = time.monotonic() - start
    result["test_usage"] = "evaluation only; alpha/calibration use within-dataset validation labels"
    write_json(root / "oracles" / f"{model}_qualification.json", result)
    return result


def summarize_qualification(root: Path) -> dict:
    table = pd.read_csv(root / "data/qualification.csv", dtype={"dataset": str})
    result = {"models": {}, "baselines": {}}
    for model in ("esm2-650m", "prot-t5"):
        path = root / "oracles" / f"{model}_qualification.json"
        if path.exists():
            result["models"][model] = json.loads(path.read_text())
    for dataset in ("2020", "2026"):
        subset = table[table.dataset.eq(dataset) & table.split.eq("test")]
        result["baselines"][dataset] = {
            name: evaluate_subsets(subset, subset[name].to_numpy(), seed=2001, intervals=False)
            for name in ("baseline_natural", "baseline_eccm", "baseline_profile")}
    result["complete"] = len(result["models"]) == 2
    reasons = []
    if result["complete"]:
        for model, report in result["models"].items():
            for dataset, subset_name in (("2026", "artificial"), ("2020", "source/BmDCA")):
                scores = report["datasets"][dataset]["evaluation"][dataset + "/test"][subset_name]
                baseline = max((result["baselines"][dataset][name][subset_name]["spearman"] or 0.)
                               for name in result["baselines"][dataset])
                interval = scores.get("cluster_bootstrap_95ci", {}).get("spearman", [0., 0.])
                if (scores.get("spearman") or 0.) <= baseline or interval[0] <= 0:
                    reasons.append(f"{model}/{dataset}: ranking has not exceeded simple baselines with a positive correlation interval")
                if scores["top_decile_functional_fraction"] <= scores["background_functional_fraction"]:
                    reasons.append(f"{model}/{dataset}: no measured-positive enrichment in the top decile")
            if report["datasets"]["2026"]["calibration"]["slope"] <= 0:
                reasons.append(f"{model}/2026: online validation calibration has no positive slope")
    result["decision"] = ("awaiting_models" if not result["complete"] else
                          "needs_model_review" if reasons else "ready_for_prior_development")
    result["decision_reasons"] = reasons
    result["decision_scope"] = "development-stage evidence check; not a significance claim"
    write_json(root / "oracles/qualification_summary.json", result)
    return result
