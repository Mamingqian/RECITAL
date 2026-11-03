# RECITAL

RECITAL searches for the reward-and-training recipe of a biological sequence generator. Each round a
language-model Strategist proposes candidate recipes, deterministic validators check them, each accepted candidate
trains as its own GRPO job, and the Strategist reads a ledger of the measured libraries to propose the next round. A
Coder agent can write new reward components that run in a sandbox. Every prompt, decision, measurement and delivered
library is recorded in a hash-chained event log. Three task families share the engine: Loop-seq DNA cyclizability,
cell-type-specific enhancers (CRE) and chorismate mutase (CM).

Paper: *Agentic Adaptation of Rewards and Training for Biological Sequence Design* (manuscript to be submitted).
Code: <https://github.com/Mamingqian/RECITAL>, released under the Apache License 2.0 (`LICENSE`).

This repository holds the framework, the prompts of the Strategist, Critic and Coder, the recipe schema, the reward
components and the task declarations. The analysis and baseline scripts, the third-party model code and the search
records of the paper's runs are not part of it; the external models are installed separately (see below).

## Where things are

| Item | Path |
|---|---|
| Framework | `src/recital/control/` (entry point `cli.py`, the round loop in `engine.py`) |
| Agent prompts | Strategist and Critic: `src/recital/control/prompts.py` (`STRATEGIST_SYSTEM`, `CRITIC_SYSTEM`, the Coder-available note and the header of every decision). Coder: `system_prompt` in `src/recital/control/coder.py`. The health and composition legends the Strategist reads are in `ledger.py` |
| Recipe schema | `src/recital/control/schemas.py` (`round_response_schema`; the checks a recipe passes are in `recipes.py`) |
| Reward components (primitive registry) | `src/recital/spec/registry.py` (the components are in `src/recital/spec/primitives/`, `src/tasks/dna/primitives.py` and `src/tasks/protein/primitives.py`) |
| Task declarations | `configs/`: per task a run config (`<task>_experiment.yaml`), its contract (`<task>_contract.yaml`, named by `contract_path`) and, for CRE and Loop-seq, the initial recipe (`<task>_initial_spec.json`) |

The rest of `src/recital/` is the science layer: the GRPO trainer, reward composition (`spec/`), the library
selectors and the sandbox that runs Coder-written components (`synthesis/`). `src/tasks/` holds the three task
packages (adapter, oracle wrappers, contract) and the alphabet-scoped primitives.

## Install

Python 3.11, PyTorch 2.6.x and Transformers 4.57.3. `environment.yml` pins the packages the code imports:

```bash
conda env create -f environment.yml
conda activate recital
pip install -e . --no-deps
```

Training needs a CUDA GPU; the control layer runs on CPU.

The Strategist, Critic and Coder call an OpenAI-compatible Responses API (function tools and a strict `json_schema`
in one request, reasoning on). The model is the `llm.model` of the run's config. The endpoint is the OpenAI API with
the key in `OPENAI_API_KEY`, unless the config declares `llm.base_url` and `llm.api_key_file` for another provider.
Prompt budgets are counted with tiktoken's `o200k_base` vocabulary, downloaded at first use (`TIKTOKEN_CACHE_DIR`).

## Run a task

One cell is one search: one task, one seed. The controller runs on a CPU node, trains nothing and submits the training
of each candidate as its own job. Run directories, checkpoints and model caches are large: put them on a volume with
space.

```bash
RUNS=/path/with/space                  # where the run directories go

# CRE, Loop-seq and CM, one seed each. preflight checks the assets and the contract: it loads no model, calls no API,
# trains nothing
python -m recital.control.cli preflight --config configs/cre_experiment.yaml     --output "$RUNS/cre_seed0"     --seed 0
python -m recital.control.cli run       --config configs/cre_experiment.yaml     --output "$RUNS/cre_seed0"     --seed 0

python -m recital.control.cli preflight --config configs/loopseq_experiment.yaml --output "$RUNS/loopseq_seed0" --seed 0
python -m recital.control.cli run       --config configs/loopseq_experiment.yaml --output "$RUNS/loopseq_seed0" --seed 0

python -m recital.control.cli preflight --config configs/cm_experiment.yaml      --output "$RUNS/cm_seed0"      --seed 0
python -m recital.control.cli run       --config configs/cm_experiment.yaml      --output "$RUNS/cm_seed0"      --seed 0

# a campaign: one cell per seed
python -m recital.control.cli campaign --config configs/loopseq_experiment.yaml --output-root "$RUNS/loopseq" --seeds 0 1 2

# state of cells: status, rounds, candidates, tokens, resume count, resolved model ids, endpoint host
python -m recital.control.cli status "$RUNS"/loopseq/*

# a cell's candidates and decisions as tables
python -m recital.control.cli export "$RUNS/cre_seed0"

# the Coder with a scripted Strategist: real Coder and real training, for checking the sandbox
python -m recital.control.cli drill-coder --config configs/cre_experiment.yaml --output "$RUNS/drill_cre" \
    --drill-oracles malinois grammar_prior
```

`--scheduler slurm` (the default) submits each worker with `sbatch`; `--scheduler local` runs them as subprocesses on
the machine that runs the controller. For Slurm, set `scheduler.partition` and `scheduler.gres` in the config to a GPU
partition of your cluster. A worker runs the source tree under `--repo-root` (default: this repository) with the
interpreter that runs the controller, so start the controller from a copy of the repository if queued jobs must not
follow later edits. The controller runs for days; it stops cleanly on SIGTERM and `run` on the same `--output` resumes
it. A resumed cell keeps its first `config.yaml`; a changed config is saved beside it as `config.resume_NNN.yaml`.

## Declare a task

- **Contract** (`contract_path` of a run config, for example `configs/cre_contract.yaml`): the task text the
  agent reads, the reward sources with their roles, units and direction, the conditions, the delivery requirements,
  the ranking of qualified candidates, and the bounds on the training levers.
- **Run config** (`configs/<task>_experiment.yaml`): the contract, the initial recipe (`initial_spec_path`), the
  update budget, the `control:` block (candidates per round, rounds, step bounds, the selection plan that takes a
  library from a pool, library health, the stop rule, whether the Coder is offered and which roles it may write), the
  `llm:` block and the `scheduler:` block.
- **Recipe**: what the Strategist returns each round (`schemas.py`). Each candidate names its parent, how it starts
  (`from_initial`, `from_parent`, `continue_parent`), `steps`, `lr`, `kl` and `group_size` inside the declared bounds,
  the reward oracles, the aggregator, per-oracle utilities, penalties and gates. A null field inherits from the parent.
  `recipes.resolve` turns a proposal into an executable recipe or returns typed problems.
- **Primitives**: `@register_primitive(role, name)` adds a component to the registry. The roles are per-oracle
  normalization, per-oracle utility, cross-oracle aggregation, per-sequence shaping, per-sequence penalty and hard
  biological gate. A primitive whose thresholds depend on the alphabet registers from its task package
  (`tasks/dna/primitives.py`, `tasks/protein/primitives.py`).
- **Task package** (`src/tasks/<task>/`): an adapter that binds the contract to the oracle wrappers, the policy loader
  and the terminal blind evaluator.

## What a run writes

A cell is one directory whose truth is `events/log.jsonl`, an append-only hash-chained log; prompts, responses and
ledger rows are content-addressed blobs. It also holds `config.yaml`, `declaration.json`, `baseline/`,
`candidates/<cand_NNNNNN>/` (training receipt, pool, scores, evaluation), `jobs/`, `delivery/` with
`final_sequences.parquet`, and `terminal_audit/`, written after the cell froze by the terminal blind evaluator.
`docs/METHOD.md` describes what each phase of the controller does and lists each file of a cell.

## External models and data

The repository contains no code or weights of the models below. The oracle and policy wrappers import them lazily, so
the package imports without them; running a task needs the ones it uses.

| Task | You provide |
|---|---|
| All DNA tasks | HyenaDNA policy checkpoints, loaded with `trust_remote_code` from the checkpoint directory (`src/recital/policy_loading.py`) |
| CRE | the `promoter_rl` scorer package (path in `src/tasks/promoter/assets.py`) with Malinois (boda2), the Agarwal lentiMPRA model and ChromBPNet weights; `enformer-pytorch` and a fine-tuned Enformer checkpoint; a JASPAR MEME motif file for the grammar prior; the Gosai et al. MPRA dataset |
| Loop-seq | DNAcycP2 (its own Python environment, called as a subprocess); MIXBend (DNABERT-6) and BendNet, each through a `scorer: package.module:factory` entry that wraps the upstream code; the trained CNN ensemble checkpoints and the Loop-seq dataset (`src/tasks/loopseq/assets.py`) |
| CM | ProGen3-112M, ESM2-35M and ESM2-650M, the ProtT5 encoder (Hugging Face); PoET-2 in its own environment (non-commercial model license); the natural family and DCA parameters of Russ et al. 2020 and Netti et al. 2026 |

The CM prerequisites (data, fitted function heads, the SFT policy) are built by `scripts/run_cm.py` (`--stage prepare`,
`oracles`, `sft`), `scripts/qualify_cm.py` and `scripts/develop_cm.py`. Set `LOOPSEQ_ASSET_ROOT` and `PROMOTER_ASSET_ROOT` to move the
asset roots (default `backends/loopseq_rl` and `backends/promoter_rl`).

## License

RECITAL is released under the Apache License, Version 2.0; see `LICENSE`. The external models it calls keep their own licenses.
