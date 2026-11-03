# How a cell runs

What a cell executes, read from the code. The entry point is `python -m recital.control.cli` on a CPU node; training
runs as separate jobs (`recital.control.worker`), submitted to Slurm or run as local subprocesses. CRE
(`configs/cre_experiment.yaml`), Loop-seq (`configs/loopseq_experiment.yaml`) and CM (`configs/cm_experiment.yaml`)
declare a `control:` block and offer the Coder (CM only the roles its adapter lets it write). The framework is one
arm, protocol condition `no_self_validation`: the Strategist plus the oracle-free checks (library health, the
composition line, qualification, the primitive catalog text). CM delivers within a maximum natural distance of 0.60,
applied by the selection plan.

## 1. A cell is an event log

`events/log.jsonl` is append-only and hash-chained; large payloads (prompts, responses, ledger
rows) are content-addressed blobs. `state.fold(events)` is a pure function of the log, and
`Engine.step()` (`control/engine.py`) advances the first phase that has work: baseline job,
completed jobs, self-validation, ledger rows, finish, submissions, the Critic, the next decision.
Every outward action is logged as an intent before it is taken and as a result after. A crashed
controller resumes by building a new `Engine` on the same files and logging a `Resumed` event;
hashes of the contract, config and source are recorded, never compared. An undecided question is
asked again, so a resumed cell can diverge from an uninterrupted one; the resume count is part of
the record.

## 2. One round: B candidates, then measurements

Each round the Strategist answers one schema (`control/schemas.py`) with `kind` = `propose`,
`stop` or `develop_component`. A `propose` carries up to B=3 candidates, each a parent (the initial
policy or an earlier candidate), a start (`from_initial`, `from_parent`, `continue_parent`), steps,
lr, kl, group size, reward oracles, aggregator, per-oracle utilities, penalties and gates; a null
field inherits from the parent. `recipes.resolve` is the only place a proposal becomes an executable
recipe, and it returns typed problems instead of repairing anything. When an answer is refused
(`max_decision_attempts` per round) its valid candidates are kept: the retry is asked only for the
slots still open, and a kept candidate counts as a sibling for duplicates and the update budget; on
the last attempt, or once the attempts run out on refused stops, whatever is kept is allocated and
the remaining slots are forfeited; a round whose answers kept nothing is skipped. A valid
candidate gets the next
ordinal `(round-1)*B + slot` and the id `cand_%06d`, whatever order the jobs finish in. The first
candidate of round 1 is the declared initial recipe (the anchor). The update budget counts every
allocated candidate, including siblings still running. When rounds reach `rounds_max` or the budget
is spent, only a stop is offered. After repeated invalid stops in such a round the engine takes the
declared-ranking best candidate per condition and records the stop source as `contract_default`.

The agent does not choose the task, the conditions, the delivery requirements, the row rule, the
ranking, B, or whether a stop is accepted. `selection.validate_assignment` accepts a stop only after
`rounds_min` rounds, when the assignment names every condition once, every named candidate completed,
and its library for that condition meets the delivery requirements.

Qualified candidates are ranked per condition on that condition's own value of the declared ranking metric
(`selection.condition_values` restricts the criterion's strata to the condition, so a `reduction` over
conditions is never applied). A candidate whose value is within the criterion's `tolerance` of the best is tied
with it and the earlier candidate wins. A number is an absolute band; `auto` ties a candidate when its gap to the
best is no more than the standard error of that gap, `hypot(se_candidate, se_best)`, from the `standard_error`
each library measurement carries (`experiment_tasks.median_measurement`: 1,000 bootstrap resamples of the
library's finite rows, seed 0). A measurement without one under `auto` is refused, not read as zero.

## 3. What the Strategist sees

A static header (task, conditions, delivery requirements, visible sources, bounds, primitive catalog
and the roles that have none, initial recipe, search contract, row rule, ranking), then one ledger
row per candidate in the order they became facts, then a short volatile tail (round, remaining
budget, per-condition leaderboard, the last typed validation error). The header and the ledger
prefix are byte-stable, so the endpoint's prefix cache applies. A row is rendered whole
(pool and library medians with the composition columns and deltas); over 3,000 tokens
(`ledger.ROW_TOKEN_CAP`) `control/ledger.py` drops only `vs_parent` and then shortens the text. `get_candidate` returns the effective recipe, start and training-curve tail
(the row is already in the ledger), `get_rows` up to 50 pool rows; at most four calls per decision
across parallel calls. `guard.check_prompt_budget` refuses a prompt over
`llm.max_prompt_tokens` before it is sent. The transport is the Responses API
(`control/transport.py`): function tools and a strict `json_schema` in one request, reasoning on,
model from config `llm.model`, token counts and the resolved model id logged per decision. The endpoint
is the OpenAI API with `OPENAI_API_KEY` unless the config declares `llm.base_url` (another provider's
OpenAI-compatible Responses endpoint, https only) and `llm.api_key_file` (the file that holds its key;
the key is read when the transport is built and is in no snapshot, event or blob). `cli status` and the
export summary print the endpoint host beside the resolved model ids. `schemas.py` builds each nullable field of the
strict schemas as a type list (`["array", "null"]`); `llm.schema_dialect: anyof_nullable` makes the transport send a
nullable array or object as `anyOf` with null instead, which is the form some providers' strict compilers accept.

Every row also carries a `composition` line: for each source-free scalar measurement (a property of the
sequences, not a score; the distinct k-mer count is left to `unique_kmer_ratio`), the condition's pool median when
it lies outside the 5th-95th percentile band of the initial policy's own pool, with its side (`above` or `below`),
or `within` when none does. It reads no oracle (`ledger.composition_bands`, `composition_outside`; the baseline
pool's summary stores `q05` and `q95`). The primitive catalog gives each primitive its first docstring paragraph.

## 4. Reward during training sees only the enabled oracles

The reward is evaluated per TRL reward-function call over GRPO completions, over the recipe's
`enabled_oracles` only (`trainer.py`, "Only the spec's enabled oracles are scored online"). Sources
that are not enabled are not scored online. CRE's reward is the
scalar `cell_gap` (`spec/cell_gap.py`) reduced per oracle before any utility, then `gap_ema_zscore`;
`reward_input` is not an agent field.

## 5. Every source is scored on the same evaluation pool

A worker trains a candidate, generates one pool, and scores every source the arm can see on the same rows
(`control/backend.py`, bound through `prediction_pool_binding`). The baseline job scores the initial policy's pool first
(`baseline/`, `initial_policy_sample.json`); the composition line and library health are measured against it.

## 6. One protocol condition

`ExperimentContract.visible_sources` (`actions.py`) lists the sources the arm may see; `projection.ArmView.from_task`
turns that into what a prompt may contain: the hidden source names (the terminal blind sources and any source the arm
does not measure) and whether sentinel concepts are projected out. `guard.tripwire` raises `BlindingViolation` before a
request is sent if system-written text names a hidden source; vocabulary alone does not stop a run.
`guard.blind_projection` drops the keys and sentences that describe measurements this arm does not make (the
`held_out_oracles` field, "scored every iteration" notes) from the header, the tool results and the Coder's context.
Agent- and sandbox-written text is not scanned.

## 7. What "self-validation" is in the framework

The oracle-free checks, each in code: the composition line of a ledger row (the pool's median of every
source-free scalar outside the 5th-95th percentile band of the initial policy's pool, `ledger.composition_bands` and
`composition_outside`; `unique_kmer_count` is excluded), library health (`control/health.py`: the pairwise-distance
ratio against the baseline pool, 0.20, and the minimum-distance fraction, 0.10), qualification (the contract's
requirements, e.g. a complete 128-row delivery within the selection plan's filters) and the primitive catalog with the
rationale of each gate. No rank-correlation table or flag is computed for this arm: its `selfval` record reads
`not_computed`. `control/selfval.py` keeps only the per-row scalar names (`scalar_name`).

## 8. What the code spends, and what only the agent can

`recipes.resolve` rejects an enabled source whose role is not reward-eligible (`not_reward_eligible`), so the
terminal blind sources cannot be trained on. Every reward-pool source may be enabled, spendable ones included (CRE `grammar_prior`,
Loop-seq `mixbend`, CM `cm_bm` and `cm_max`); `_derive_held_out` derives `held_out_oracles`, which the arm does not see.

The recipe validator also refuses a recipe that leaves out a reward source the contract's role policy
requires enabled (`required_enabled_oracles`: CRE `malinois`, Loop-seq `cnn`, CM `function_2026_esm` and
`cm_novelty`). The header reads the same policy (`launch.reward_role_policy`) and gives the card of each such
source `required: true` (the key is absent on every other card), and the Strategist
instructions say "A source marked required must be enabled in every recipe."

## 9. Components the agent asks for

A `develop_component` request names a role (per-oracle utility, cross-oracle aggregation,
per-sequence penalty, hard gate), a new name, what it computes and why, and a reference candidate.
`control/coder.py` has the Coder write it (at most three attempts per request), the sandbox runs it
(`synthesis/`), the composition screen tests it on the baseline pool, and an admitted component
becomes usable by name. A request that breaks a binding or permission rule is returned without
costing an attempt. Code that exceeds the sandbox's CPU, time or output limit is the code's fault
and is repaired; an isolation failure halts the cell. With `control.synthesis.enabled: false` the
schema has no component kind and no Coder is built.

## 10. Delivery and the terminal blind evaluation

At an accepted stop `selection.best_assignment`/`backend.assemble_delivery` compose, for each
condition, the library of the candidate the assignment names, apply the contract's row rule
(`cre_single_mingap_malinois_diverse_v1`: legal rows, exact deduplication, top 128 by
`mingap/malinois` among rows at least the diversity floor, in normalized edit distance, from every
row already chosen for the condition; a pool that cannot supply 128 such rows delivers fewer and is
unqualified; CM's `cm_single_function_esm_v1` takes the top 128 by calibrated `function_2026_esm` among legal, distinct,
non-training-copy rows at least 0.2 from the natural family), check qualification, and freeze `final_sequences.parquet` and `delivery.json`. A
delivery can therefore combine candidates from different recipes. `control/terminal_audit.py` runs
afterwards, outside the controller, and is the only place the blind oracles are built; it scores
exactly the frozen rows and refuses if the library or the evaluation definition differs from
`manifest.json`. The scoring of a table of rows is its `score_library`: the terminal evaluator calls it for the frozen
library, and any other table of rows scored through the same function meets the same oracles, `audit_measure` and file
layout.

Library health (`control.health`, `control/health.py`) is a validity requirement, defined
relative to the baseline pool so it carries no alphabet or length: distance is whole-sequence
Levenshtein over the longer length, which compares sequences of unequal length. When the baseline
is ingested the engine measures, per condition and over at most 512 legal baseline rows, the
median nearest-neighbour distance and the mean pairwise distance, and refuses a baseline with fewer
than 30 usable rows or no spread. The selection plan declares no diversity rule; its minimum
distance is `min_distance_fraction` of the smallest condition's median nearest-neighbour distance
(0.10 in the CRE declaration), the same number for every condition. Each selected library then gets
one `library_health` requirement per condition: its mean pairwise distance must be at least
`min_pairwise_ratio` of the baseline's (0.20), and a condition that selection could not fill at the
floor fails it with the count it reached. The floor is never lowered to fill the count. The baseline
pool is selected from by no rule. Every ledger row carries `pairwise_ratio` for the pool and for
the selected library, and `unique_kmer_ratio` (median distinct k-mer count over the baseline's,
null when the task emits no `unique_kmer_count`), in the `pool` and `library` lines.

Loop-seq's row rule (`loopseq_single_margin_cnn_quality_v1`) takes, per condition, the 128 rows of greatest
`margin/cnn` among legal rows that are not in the training set, are not reverse-complement duplicates, and meet the
condition's floors on dinucleotide entropy and distinct 3-mers (the conditional quality reference's own). Its
terminal blind evaluation reports DNAcycP2's in-band fraction (primary) and the median signed distance inside the band
(secondary), each per condition and as the minimum over conditions.

## 11. Training precision is a run declaration

`training_precision` is `task_default` or `fp32`, validated once in
`recital/training/policy_precision.py` and bound in `build_runtime`; the run's
`declaration.json` records it. There is no automatic value. A device whose bf16 is emulated rather
than native (compute capability below 8.0) is refused at the trainer call site unless the run
declared `fp32`, because on such a device `torch.cuda.is_bf16_supported()` is True and autocast
runs (measured with torch 2.6.0+cu124 on a GPU of compute capability 7.5).

## 12. Files of a cell

A cell is one directory. Everything the agent decided is in the event log;
everything measured is under a candidate.

| Path | Holds |
|---|---|
| `events/log.jsonl`, `blobs/` | the hash-chained event log; prompts, responses and ledger rows by content hash |
| `config.yaml`, `config.resume_NNN.yaml`, `declaration.json`, `oracle_versions.json` | the first config the cell ran, any later config a resume was given, the run's declaration |
| `baseline/` | the initial policy's pool and scores (`pool.json`, `scores/`, `public_measurements.json`) |
| `candidates/<cand_NNNNNN>/` | `training_receipt.json`, `training_statistics.json`, `pool.json`, `scores/`, `evaluation.json`, `selection_measurements.json`, `candidate_result.json`, `host_aN.json` (node and attempt) |
| `jobs/<id>_aN/` | the `JobSpec` of each Slurm attempt, its `sbatch.sh` and its log |
| `delivery/`, `delivery.json`, `final_sequences.parquet` | the frozen library, with `assignment.json` naming the candidate delivered for each condition |
| `terminal_audit/` | `summary.json` and one file per blind oracle, written only after the cell froze |
| `export/` | `candidates.parquet`, `selfval.parquet`, `decisions.parquet`, `summary.json` from `cli export` |
| `needs_attention.json` | present only while the last event is a halt; a resume removes it |
