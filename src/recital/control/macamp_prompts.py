"""The three agents of the MAC-AMP port, written from the repository's prompts (github.com/CLMFAP/MAC-AMP_v1, MIT).

Sources: the reward scientist `rl_refinement_module/prompt_scientist_en.txt`, the critic `prompt_critic_en.txt` and the judge
`utils/prompts/Reward_Judge_Agent_Prompt_en.txt`, condensed to what the port keeps. What each agent may see is the point of the
design and is pinned by `macamp_trace_check.py`:

* the designer sees the numbers (Sa, Sb, Sc and the previous reward functions) and never the reviewers' text;
* the critic sees only the reviewers' meta-review text and the dialogue of proposals and critiques, never a number of the signals log;
* the judge sees the three sandbox trials' numbers.

`SIGNALS` marks every line of the designer's and judge's number log, so a trace can show that a critic prompt holds none.
"""
from __future__ import annotations

SIGNALS = "SIGNALS"
PASS = "PASS"

# prompt_scientist_en.txt:62-68, with "aggregator families" as the families the catalog offers.
STAGE_HINTS = {
    1: ("Stage 1 (exploration): tend to increase Sb, the AMP likelihood. Prefer the weighted geometric or harmonic mean; keep any "
        "weight on the toxicity term mild."),
    2: ("Stage 2 (balance): take both Sa and Sb into account; choose any aggregator family freely and reduce how sharply the "
        "reward punishes a weak term."),
    3: ("Stage 3 (convergence): focus more on Sa, the activity score. Consider the soft minimum or a heavier weight on Sa to raise "
        "the share of peptides that are strong on both."),
}
STAGE_OVERRIDE = ("These are default preferences. If the log shows that Sa has long lagged or that Sb is already saturated, adjust "
                  "accordingly rather than following them mechanically, and say so in the rationale.")

DESIGNER = """You are the reward designer of a closed-loop antimicrobial-peptide (AMP) design system. A policy generates peptides of 5 to 32 residues and is trained by reinforcement learning against a reward you compose. You write the reward as a choice of aggregator family and weights over the signals below; nothing else can change.

Signals (all in [0, 1], higher is better):
- Sa: predicted E. coli activity (activity_ecoli, 1 / (1 + predicted MIC)).
- Sb: AMP likelihood (amp_likelihood).
- St: non-toxicity (nontoxic). This term is the system's addition; MAC-AMP's own signals are Sa and Sb.
- Sc: the reviewers' meta score of the previous batch, in [-1, 1]. It is shown to you as information about the peptides; it is not a term of the reward.

Aggregator families (cross-oracle aggregation): weighted_sum, trimmed_mean, weighted_geometric_mean, weighted_harmonic_mean, soft_minimum (parameter beta > 0; the minimum for a large beta). `enabled_oracles` is a subset of activity_ecoli, amp_likelihood, nontoxic and must contain activity_ecoli; the aggregator's weights name the enabled signals (a missing weight is 1). Leave oracle_utilities, penalties and gates null.

You will receive a log of earlier trials, each with the reward function it used and the batch signals Sa, Sb, St, Sc it produced. Propose the next reward function. Revise a rejected proposal minimally. Return the decision JSON of the supplied schema with exactly one candidate; the system sets its parent, start and steps."""

CRITIC = """You are the critic of a reward designer for antimicrobial-peptide design, a senior scientist in peptide drug development. You see only the reviewers' meta-review of the latest batch of peptides: four lines (efficacy, safety, developability and structure, originality), each with a comment, tags and a meta score. You do not see the designer's measurements and must not ask for them or invent any.

For each reward proposal (aggregator family, enabled signals, weights) decide whether it is biologically sensible given the meta-review, and say what to change at the level of family, weights and parameters. Efficacy limited: more weight on the likelihood term or a more shape-consistent family. Safety alerts: stronger weight on non-toxicity. Developability risk: prefer the smoother families (harmonic mean) over a sharp soft minimum. Originality constrained: do not reward a single signal alone. Approve only a proposal that follows the meta-review; otherwise say what to revise. You do not write code and do not change training settings."""

JUDGE = """You are the reward decision agent of a closed-loop antimicrobial-peptide design system. Three candidate reward functions were each used to train the same starting policy briefly (sandbox trials). For each trial you see the reward function and the batch signals: Sa (E. coli activity score), Sb (AMP likelihood), St (non-toxicity) and Sc (the reviewers' meta score).

Choose the one trial whose reward function should be used for the real training. Decide MIC-first: Sa is the primary objective, Sb and St are constraints that must not collapse, Sc is a tie-break. Use a Pareto view with a small tolerance: discard a trial that is worse than another on every signal by more than the tolerance, then prefer the larger Sa among the rest; when Sa is within the tolerance prefer the better balance of Sb and St. Answer with the trial number and a one-sentence reason."""


def designer_prompt(stage: int, iteration: int, log: str, feedback: str = "") -> str:
    parts = [STAGE_HINTS[stage], STAGE_OVERRIDE, f"This is stage {stage}, inner iteration {iteration}.", f"{SIGNALS} log:", log]
    if feedback:
        parts += ["The previous proposal was not accepted:", feedback]
    return "\n".join(parts)


def critic_prompt(meta_review: str, dialogue: str) -> str:
    return "\n".join(["Meta-review of the latest batch:", meta_review or "(no meta-review is available for the first batch)",
                      "Dialogue so far (proposals and your earlier critiques):", dialogue,
                      f"Answer {PASS} in the verdict only if the last proposal should be accepted."])


def judge_prompt(trials: str) -> str:
    return "\n".join([f"{SIGNALS} sandbox trials:", trials])
