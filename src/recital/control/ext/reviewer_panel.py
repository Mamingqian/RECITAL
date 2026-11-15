"""The reviewer panel's vocabulary and arithmetic: lexicon, sign map, tag scores, aggregation, peptide properties, schemas, prompts.

Adapted from MAC-AMP (arXiv 2602.14926; github.com/CLMFAP/MAC-AMP_v1, MIT): `Generation_Module/utils/prompt/
INDEPENDENT_REVIEWER_AGENT_en.txt` (the weighted lexicon of four dimensions, the 1.00/0.85/0.60/0.40 confidence levels, at most
four tags a dimension, the score `b_a + sum of weight x p`) and `AREA_CHAIR_AGENT_en.txt` (the sign map, the polarity multiset of a
core that two or more reviewers tagged, the meta-comment). Four differences, each deliberate:

* a tag is chosen from an enum of `core=state` pairs of the dimension, so a core or state outside the lexicon cannot be written;
* code computes every score and every disagreement group from the tags (the Area Chair prompt says its groups are "only polarity
  multiset and counting"; its score line is hard logic downstream), and the chair model writes the four comments only;
* the cores that need OmegaFold, Foldseek templates or the Boman table are not offered: `plddt` is ESMFold's, `foldseek_similarity`
  becomes `known_similarity` (distance to the public MIC table), `boman_index` and `template_motif_reuse` are dropped;
* MAC-AMP's three reviewers are three vendors' models on one prompt (`discussion_agents.py`, the `openai`, `genai` and `perplexity`
  wrappers of `INDEPENDENT_REVIEWER_AGENT`); here one model answers three times, each time with a one-line emphasis.

No function here reads a file or calls a model.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from ..schemas import object_schema

DIMENSIONS = ("EFF", "Safe", "DevStruct", "Orig")
P_LEVELS = (1.0, 0.85, 0.6, 0.4)
MAX_TAGS = 4
REVIEW_CHARS, CHAIR_CHARS = 200, 300

# core -> state -> (weight, polarity). Weights are MAC-AMP's; polarity is its sign map (-1, 0, +1: a position, not a judgement).
LEXICON: dict[str, dict[str, dict[str, tuple[float, int]]]] = {
    "EFF": {
        "mic_band": {"low": (0.50, -1), "medium": (0.20, 0), "high": (-0.56, 1)},
        "amp_likelihood": {"low": (-0.20, -1), "medium": (0.10, 0), "high": (0.20, 1)},
        "net_charge": {"low": (-0.28, -1), "optimal": (0.32, 0), "high": (-0.30, 1)},
        "hydrophobicity": {"low": (-0.22, -1), "balanced": (0.20, 0), "high": (-0.26, 1)},
        "amphipathicity": {"balanced": (0.28, 0), "imbalanced": (-0.26, 1)},
        "hydrophobic_moment": {"in_band": (0.28, 0), "off_band": (-0.26, 1)},
        "charge_hydrophobic_ratio": {"balanced": (0.12, 0), "imbalanced": (-0.16, 1)},
        "cation_pi_motif": {"present": (0.10, 1), "absent": (0.00, -1)},
    },
    "Safe": {
        "toxinpred": {"low": (0.60, -1), "medium": (0.00, 0), "high": (-0.80, 1)},
        "hydrophobicity_safety": {"low": (0.10, -1), "balanced": (0.20, 0), "high": (-0.20, 1)},
        "over_cationic_level": {"low": (0.20, -1), "moderate": (0.00, 0), "high": (-0.30, 1)},
        "aromatic_enrichment": {"low": (0.15, -1), "moderate": (0.00, 0), "high": (-0.20, 1)},
        "rk_run_length": {"low": (0.15, -1), "medium": (-0.05, 0), "high": (-0.20, 1)},
    },
    "DevStruct": {
        "plddt": {"low": (-0.45, -1), "medium": (0.10, 0), "high": (0.40, 1)},
        "instability_index": {"high": (-0.35, 1), "medium": (0.10, 0), "low": (0.30, -1)},
        "solubility_proxy": {"poor": (-0.35, -1), "moderate": (0.10, 0), "good": (0.30, 1)},
        "aliphatic_index": {"low": (-0.10, -1), "moderate": (0.10, 0), "high": (0.15, 1)},
        "length_band": {"short": (-0.20, -1), "optimal": (0.25, 0), "long": (-0.20, 1)},
        "cysteine_complexity": {"high": (-0.15, 1), "medium": (0.00, 0), "low": (0.10, -1)},
        "aggregation_proxy": {"high": (-0.25, 1), "medium": (0.00, 0), "low": (0.20, -1)},
    },
    "Orig": {
        "known_similarity": {"low": (0.40, -1), "medium": (0.20, 0), "high": (-0.40, 1)},
        "kmer_reuse": {"low": (0.25, -1), "medium": (0.10, 0), "high": (-0.30, 1)},
        "batch_diversity": {"low": (-0.30, -1), "medium": (0.10, 0), "high": (0.30, 1)},
        "low_complexity_content": {"high": (-0.25, 1), "medium": (0.00, 0), "low": (0.10, -1)},
    },
}


def tag_names(dimension: str) -> list[str]:
    """Every `core=state` of a dimension, the enum a reviewer picks from."""
    return [f"{core}={state}" for core, states in LEXICON[dimension].items() for state in states]


# ----- one reviewer's tags to a score ---------------------------------------------------------------------------------------

def tag_score(dimension: str, tags: Sequence[Mapping[str, Any]]) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]]]:
    """`(score, kept, dropped)` of one dimension: `sum(weight x p)` over at most four tags, one state a core, the first kept.

    A tag outside the dimension's lexicon, a second tag of a core, or a fifth tag is dropped with its reason; no tag left scores 0.
    """
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tag in tags:
        core, _, state = str(tag["tag"]).partition("=")
        if state not in LEXICON[dimension].get(core, {}):
            dropped.append({**tag, "reason": "not in the lexicon"})
        elif core in seen:
            dropped.append({**tag, "reason": "core already tagged"})
        elif len(kept) >= MAX_TAGS:
            dropped.append({**tag, "reason": "more than four tags"})
        else:
            seen.add(core)
            kept.append({"tag": f"{core}={state}", "p": float(tag["p"])})
    score = sum(LEXICON[dimension][item["tag"].partition("=")[0]][item["tag"].partition("=")[2]][0] * item["p"] for item in kept)
    return score, kept, dropped


def polarity(dimension: str, tag: str) -> int:
    core, _, state = tag.partition("=")
    return LEXICON[dimension][core][state][1]


def disagreement_groups(dimension: str, reviews: Sequence[Sequence[Mapping[str, Any]]]) -> list[list[int]]:
    """The chair's groups: for each core that two or more reviewers tagged, the polarities of their states, sorted ascending.

    `reviews` holds each reviewer's kept tags of the dimension. Cores come in the order a reviewer first tagged them.
    """
    by_core: dict[str, list[int]] = defaultdict(list)
    for tags in reviews:
        for item in tags:
            by_core[item["tag"].partition("=")[0]].append(polarity(dimension, item["tag"]))
    return [sorted(values) for values in by_core.values() if len(values) >= 2]


def aggregated_tags(dimension: str, reviews: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Each tagged core with the states the reviewers gave it, e.g. `{"core": "net_charge", "states": ["optimal", "high"]}`."""
    states: dict[str, list[str]] = defaultdict(list)
    for tags in reviews:
        for item in tags:
            core, _, state = item["tag"].partition("=")
            states[core].append(state)
    return [{"core": core, "states": values} for core, values in states.items()]


# ----- the peptide columns the reviewers read ------------------------------------------------------------------------------

HYDROPHOBIC = frozenset("AILMFVW")
AROMATIC = frozenset("FWY")
CATIONIC = frozenset("KR")
HELIX_DEGREES = 100.0


def longest_run_of(sequence: str, members: frozenset[str]) -> int:
    best = run = 0
    for residue in sequence:
        run = run + 1 if residue in members else 0
        best = max(best, run)
    return best


def cation_pi_pairs(sequence: str, window: int = 2) -> int:
    """Pairs of a K or R and an F, W or Y at most `window` positions apart."""
    return sum(1 for i, a in enumerate(sequence) if a in CATIONIC
               for j in range(max(0, i - window), min(len(sequence), i + window + 1)) if sequence[j] in AROMATIC)


def hydrophobic_moment(sequence: str, scale: Mapping[str, float], degrees: float = HELIX_DEGREES) -> float:
    """Eisenberg's moment at `degrees` per residue, on `scale`, divided by the length."""
    angle = math.radians(degrees)
    x = sum(scale[a] * math.cos(angle * i) for i, a in enumerate(sequence))
    y = sum(scale[a] * math.sin(angle * i) for i, a in enumerate(sequence))
    return math.hypot(x, y) / len(sequence)


def peptide_properties(sequence: str) -> dict[str, float | int]:
    """ProtParam-style columns for one peptide of standard residues (Biopython; Kyte-Doolittle for GRAVY and the moment)."""
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    from Bio.SeqUtils.ProtParamData import kd

    analysis = ProteinAnalysis(sequence)
    n = len(sequence)
    mole_percent = lambda letters: 100.0 * sum(sequence.count(x) for x in letters) / n  # noqa: E731
    return {
        "net_charge_ph7.4": round(float(analysis.charge_at_pH(7.4)), 2),
        "gravy": round(float(analysis.gravy()), 3),
        "hydrophobic_moment": round(hydrophobic_moment(sequence, kd), 3),
        "aromaticity": round(float(analysis.aromaticity()), 3),
        "instability_index": round(float(analysis.instability_index()), 1),
        "aliphatic_index": round(mole_percent("A") + 2.9 * mole_percent("V") + 3.9 * (mole_percent("I") + mole_percent("L")), 1),
        "cysteines": sequence.count("C"),
        "cation_pi_pairs": cation_pi_pairs(sequence),
        "longest_hydrophobic_stretch": longest_run_of(sequence, HYDROPHOBIC),
        "longest_kr_run": longest_run_of(sequence, CATIONIC),
    }


# ----- the schemas ----------------------------------------------------------------------------------------------------------

def reviewer_schema() -> dict[str, Any]:
    """One reviewer's answer: for each dimension a comment and at most four tags of that dimension's lexicon."""
    def dimension(name: str) -> dict[str, Any]:
        tag = object_schema({
            "tag": {"type": "string", "enum": tag_names(name), "description": "core=state of this dimension's lexicon"},
            "p": {"type": "number", "enum": list(P_LEVELS),
                  "description": "1.0 certain, 0.85 confident, 0.6 possible, 0.4 speculative"}})
        return object_schema({
            "comment": {"type": "string", "description": f"at most {REVIEW_CHARS} characters, one line"},
            "tags": {"type": "array", "maxItems": MAX_TAGS, "items": tag,
                     "description": "at most four, one state a core; empty when the evidence does not support one"}})
    return object_schema({name: dimension(name) for name in DIMENSIONS})


def chair_schema() -> dict[str, Any]:
    return object_schema({name: {"type": "string", "description": f"the meta-comment, at most {CHAIR_CHARS} characters, one line"}
                          for name in DIMENSIONS})


def one_line(text: str, limit: int) -> str:
    return " ".join(str(text).split())[:limit]


# ----- the prompts ----------------------------------------------------------------------------------------------------------

EMPHASES = (
    "Weigh the efficacy evidence (predicted MIC, AMP likelihood, charge, hydrophobicity, amphipathicity) most heavily when you "
    "decide which tags are certain and which are speculative.",
    "Weigh the safety evidence (toxicity and hemolysis scores, cationicity, hydrophobicity, aromatic content, K/R runs) most "
    "heavily when you decide which tags are certain and which are speculative.",
    "Weigh the developability and originality evidence (structure confidence, instability, solubility, length, similarity to known "
    "peptides, repeats and diversity) most heavily when you decide which tags are certain and which are speculative.",
)


def lexicon_text() -> str:
    lines = []
    for dimension, cores in LEXICON.items():
        lines.append(f"{dimension}:")
        lines.extend(f"  {core}: " + ", ".join(f"{state} ({weight:+.2f})" for state, (weight, _) in states.items())
                     for core, states in cores.items())
    return "\n".join(lines)


REVIEWER_INSTRUCTIONS = (
    "You are an independent reviewer of a library of antimicrobial peptides designed by a generative model. You read one table "
    "of {count} peptides with computed columns and review it as a whole in four dimensions: EFF (antimicrobial efficacy), Safe "
    "(safety), DevStruct (developability and structural reliability) and Orig (originality). Judge only from the table; the "
    "columns are model predictions and sequence statistics.\n"
    "Columns: seq; amp_score, the AMP probability of a sequence classifier (higher is more AMP-like); mic_um, the predicted MIC in "
    "micromolar (lower is stronger); toxicity_score and hemolysis_score, predicted probabilities (higher is worse); plddt_score, "
    "structure confidence in [0, 1] or null; known_distance, the normalized edit distance to the nearest peptide of a public MIC "
    "table (0 is a copy, 1 is unrelated); prop, a one-line summary of the sequence statistics. `batch` holds statistics of the "
    "whole library and `pool` of the model's 1,000 samples that the library was drawn from.\n"
    "For each dimension write a comment (at most 200 characters, one line, you may cite ids) and choose at most four tags from "
    "that dimension's lexicon below. A tag is core=state; give a core at most one state. Decide a state from the peptide's "
    "relative position in this library together with the other columns, and give each tag a confidence p of 1.0 (certain), 0.85 "
    "(confident), 0.6 (possible) or 0.4 (speculative): the more signals agree, the higher p; if the evidence conflicts or is thin, "
    "choose the more conservative state and a lower p. If no tag is supported, return none for that dimension. Code computes the "
    "scores, so do not.\n"
    "Lexicon (the weight beside a state is added to the dimension's score, times p):\n{lexicon}\n{emphasis}")

CHAIR_INSTRUCTIONS = (
    "You are the Area Chair of a panel of three reviewers of a library of antimicrobial peptides. For each dimension you receive "
    "the reviewers' comments and the tags they gave, aggregated by core, with the polarity groups code computed (a group lists "
    "the polarities, -1 low, 0 middle, +1 high, of the states that two or more reviewers gave one core; (1, 1) is agreement and "
    "(-1, 1) is opposition). Write one meta-comment per dimension, at most 300 characters on one line: first the consensus, then "
    "where reviewers diverge, naming the cores. You do not compute or change scores and you add no evidence of your own.")


def reviewer_instructions(index: int, count: int) -> str:
    return REVIEWER_INSTRUCTIONS.format(count=count, lexicon=lexicon_text(), emphasis=EMPHASES[index % len(EMPHASES)])


def reviewer_payload(table: Mapping[str, Any]) -> str:
    """The table as MAC-AMP's reviewers read it: arrays of one length, one index a peptide, plus the library and pool statistics."""
    rows = table["rows"]
    activity = [row.get("activity") for row in rows]
    summary = lambda row: "; ".join(f"{name} {row[name]}" for name in (  # noqa: E731
        "length", "net_charge_ph7.4", "gravy", "hydrophobic_moment", "aromaticity", "instability_index", "aliphatic_index",
        "cysteines", "cation_pi_pairs", "longest_hydrophobic_stretch", "longest_kr_run", "longest_run") if name in row)
    payload = {
        "id": [f"X{i + 1}" for i in range(len(rows))], "seq": [row["sequence"] for row in rows],
        "amp_score": [row.get("amp_likelihood") for row in rows],
        "mic_um": [None if a is None or a <= 0 else round(1.0 / a - 1.0, 3) for a in activity],
        "toxicity_score": [None if row.get("nontoxic") is None else round(1.0 - row["nontoxic"], 4) for row in rows],
        "hemolysis_score": [None if row.get("nonhemolytic") is None else round(1.0 - row["nonhemolytic"], 4) for row in rows],
        "plddt_score": [row.get("plddt") for row in rows],
        "known_distance": [row.get("distance_to_nearest_known") for row in rows],
        "prop": [summary(row) for row in rows], "batch": table["library"], "pool": table["pool"]}
    return json.dumps(payload, separators=(",", ":"))


def chair_payload(dimensions: Mapping[str, Mapping[str, Any]]) -> str:
    return json.dumps(dimensions, separators=(",", ":"))
