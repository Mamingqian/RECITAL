"""Common candidate filtering, visible selection and terminal-only CM scoring."""
from __future__ import annotations


from collections.abc import Callable
from numbers import Real

import numpy as np
import pandas as pd
import torch
from rapidfuzz.distance import Levenshtein

from tasks.protein.progen import ProGenSequenceCodec
from .data import ALPHABET


def combine_function_scores(first, second, ranking: str = "mean") -> np.ndarray:
    """Combine calibrated predictions without changing their individual scales.

    "primary" ranks on the first score alone. The second (function_2026_t5) is a
    PURE sentinel: it shares ProtT5 with audit_2020_t5, the primary endpoint, so a
    two-model rule buys its margin from the endpoint's own backbone, and under the
    blinded condition it puts sentinel evidence into the selection the agent's policy is
    judged by. It is still measured and recorded; it just stops picking the library.
    """
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Function score arrays must have identical shapes")
    if ranking == "mean":
        return (a + b) * .5
    if ranking == "min":
        return np.minimum(a, b)
    if ranking == "primary":
        return a
    raise ValueError(f"Unknown CM selection ranking: {ranking}")


def sample_policy(model, tokenizer, *, count: int, seed: int, batch_size: int = 32,
                  progress_callback=None) -> list[str]:
    torch.manual_seed(seed)
    codec = ProGenSequenceCodec(tokenizer)
    legal = set(codec.legal_token_ids(include_eos=True))
    suppressed = [i for i in range(model.config.vocab_size) if i not in legal]
    sequences = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            n = min(batch_size, count - start)
            encoded = tokenizer(["<bos>1"] * n, add_special_tokens=False,
                                return_tensors="pt", padding=True)
            encoded = {key: value.to(model.device) for key, value in encoded.items()
                       if key in ("input_ids", "attention_mask")}
            tokens = model.generate(**encoded, max_new_tokens=97, min_new_tokens=80,
                                    do_sample=True, temperature=1., top_p=1., top_k=0,
                                    suppress_tokens=suppressed, forced_eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id)
            sequences.extend(tokenizer.batch_decode(tokens[:, encoded["input_ids"].shape[1]:],
                                                    skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False))
            if progress_callback is not None:
                progress_callback(len(sequences), count)
    return sequences


def select_library(sequences: list[str], family, function, cnn, *, count: int = 128,
                   min_distance: float = 0.2,
                   score_names: tuple[str, str] = ("cm_function", "cm_cnn"),
                   calibrators: tuple | None = None,
                   ranking: str = "mean", family_requirement: dict | None = None,
                   family_score: Callable | None = None) -> tuple[pd.DataFrame, dict]:
    combine_function_scores([], [], ranking)  # Validate even for an empty pool.
    if family_requirement is not None:
        if (not isinstance(family_requirement, dict)
                or set(family_requirement) != {"oracle", "minimum"}
                or not isinstance(family_requirement["oracle"], str)
                or not family_requirement["oracle"].strip()
                or isinstance(family_requirement["minimum"], bool)
                or not isinstance(family_requirement["minimum"], Real)
                or not np.isfinite(family_requirement["minimum"])):
            raise ValueError("Family requirement needs an oracle name and finite numeric minimum")
        if not callable(family_score):
            raise ValueError("Family requirement needs its scoring callable")
        family_requirement = {**family_requirement, "minimum": float(family_requirement["minimum"])}
    rows = []
    seen = set()
    for index, sequence in enumerate(sequences):
        valid = 80 <= len(sequence) <= 96 and set(sequence) <= set(ALPHABET)
        duplicate = sequence in seen
        seen.add(sequence)
        row = {"candidate_index": index, "sequence": sequence, "valid": valid,
               "duplicate": duplicate, "eligible": False, "selected": False,
               "score_valid": False, "training_copy": False}
        if valid:
            description = family.describe(sequence)
            row.update({key: value for key, value in description.items() if key != "aligned"})
            row["eligible"] = not duplicate and not description["training_copy"] and description["natural_distance"] >= min_distance
        rows.append(row)
    table = pd.DataFrame(rows, columns=list(dict.fromkeys(
        ["candidate_index", "sequence", "valid", "duplicate", "eligible", "selected", "score_valid", "training_copy"]
        + [key for row in rows for key in row]
    )))
    for column in ("valid", "duplicate", "eligible", "selected", "score_valid", "training_copy"):
        table[column] = table[column].astype(bool)
    for column in (*score_names, "selection_score", "natural_distance", "training_distance"):
        if column not in table:
            table[column] = np.nan
    valid_indices = table.index[table.valid].tolist()
    first, second = score_names
    # Which of score_names the ranking actually consumes. Used for eligibility and
    # for the selection_rule string written into every delivered packet.
    consumed = (0,) if ranking == "primary" else (0, 1)
    if valid_indices:
        valid_sequences = table.loc[valid_indices, "sequence"].tolist()
        for name, oracle in zip(score_names, (function, cnn)):
            values = np.asarray(oracle(valid_sequences), dtype=float)
            if values.shape != (len(valid_sequences),):
                raise ValueError(f"{name} must return one score per input sequence")
            table.loc[valid_indices, name] = values
        scores = [table[name].to_numpy() for name in score_names]
        if calibrators is not None:
            if len(calibrators) != 2:
                raise ValueError("CM selection requires one calibrator per function oracle")
            scores = [np.asarray(calibrate(values), dtype=float)
                      for calibrate, values in zip(calibrators, scores)]
        if any(values.shape != (len(table),) for values in scores):
            raise ValueError("Calibration must preserve one score per candidate")
        table["selection_score"] = combine_function_scores(*scores, ranking=ranking)
        # Eligibility turns on the scores the RULE reads. Requiring a finite t5 under
        # "primary" would keep the sentinel as a delivery gate after taking it out of
        # the ranking -- a quieter version of the same channel.
        score_valid = table.valid & np.isfinite(table.selection_score)
        for position in consumed:
            score_valid &= (np.isfinite(table[score_names[position]])
                            & np.isfinite(scores[position]))
        table["score_valid"] = score_valid
        table["eligible"] &= table.score_valid
    family_unknown_count = family_rejected_count = 0
    if family_requirement is not None:
        table["family_score"] = np.nan
        if valid_indices:
            values = np.asarray(family_score(table.loc[valid_indices, "sequence"].tolist()), dtype=float)
            if values.shape != (len(valid_indices),):
                raise ValueError("Family scorer must return one score per input sequence")
            table.loc[valid_indices, "family_score"] = values
        table["family_score_valid"] = table.valid & np.isfinite(table.family_score)
        family_unknown_count = int((table.eligible & ~table.family_score_valid).sum())
        family_rejected_count = int((table.eligible & table.family_score_valid
                                     & (table.family_score < family_requirement["minimum"])).sum())
        table["family_eligible"] = (table.eligible & table.family_score_valid
                                    & (table.family_score >= family_requirement["minimum"]))
        table["eligible"] &= table.family_eligible
    from recital.library_selection import stable_top_indices
    selected_indices = table.index[stable_top_indices(
        table.selection_score.to_numpy(), table.eligible.to_numpy(), count)]
    table.loc[selected_indices, "selected"] = True
    selected = table[table.selected]
    def median(column):
        values = selected[column].dropna()
        return float(values.median()) if len(values) else None
    diversity = [Levenshtein.normalized_distance(a, b)
                 for i, a in enumerate(selected.sequence) for b in selected.sequence.iloc[i + 1:]]
    summary = {"candidate_count": len(table), "valid_count": int(table.valid.sum()),
               "invalid_score_count": int((table.valid & ~table.score_valid).sum()),
               "eligible_count": int(table.eligible.sum()), "selected_count": len(selected),
               "required_count": count, "complete_delivery": len(selected) == count,
               "visible_function_median": median(first),
               "score_medians": {name: median(name) for name in score_names},
               "selection_score_median": median("selection_score"),
               "natural_distance_median": median("natural_distance"),
               "training_distance_median": median("training_distance"),
               "mean_pairwise_normalized_edit_distance": float(np.mean(diversity)) if diversity else None,
               "selection_aggregation": ranking,
               "selection_rule": "unique novel candidates, ranked by " + ranking + " of "
                                 + " and ".join(score_names[position] for position in consumed)
                                 + (" after validation calibration" if calibrators else ""),
               "distance_definition": "full-sequence Levenshtein / max(lengths)"}
    if family_requirement is not None:
        summary.update(family_requirement=family_requirement,
            family_eligible_count=int(table.family_eligible.sum()),
            family_unknown_count=family_unknown_count, family_rejected_count=family_rejected_count,
            family_score_median=median("family_score"))
        summary["selection_rule"] += (f"; finite {family_requirement['oracle']} "
                                      f">= {family_requirement['minimum']:g} before ranking")
    if second == "cm_cnn":
        summary["cnn_median"] = median(second)
    return table, summary


def audit_library(table: pd.DataFrame, audit) -> dict:
    selected = table[table.selected]
    if not len(selected):
        return {"n": 0, "median": None, "mean": None, "status": "empty_delivery"}
    scores = audit(selected.sequence.tolist())
    table.loc[selected.index, "cm_audit"] = scores
    return {"n": len(scores), "median": float(np.median(scores)), "mean": float(np.mean(scores)),
            "status": "computational_function_prediction", "oracle": "ESM2-650M regression ensemble"}
