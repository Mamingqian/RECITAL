"""Sequence-only PoET-2 CM generation and scoring using the official inference API.

The fixed generation prompt uses natural, functional 2026 training sequences also
present in SFT training. Raw draws are retained, including length and EOS failures.

The ``cm_poet`` sentinel scorer lives here too so that model loading and prompt
selection exist once. Its prompt is a different pool on purpose; see
``select_scoring_context``. Everything in this module has to be importable inside
the separate PoET-2 interpreter, which is where the official package can
actually be imported.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from .data import ALPHABET, read_fasta, save_json, save_table


def _journal(output: Path, event: str, **values) -> None:
    with (output / "generation_calls.jsonl").open("a") as handle:
        handle.write(json.dumps({"event": event, "unix_time": time.time(), **values},
                                allow_nan=False) + "\n")


CONTEXT_COUNT = 64


def select_context(root: Path, *, count: int = CONTEXT_COUNT, seed: int = 2001) -> pd.DataFrame:
    """One prespecified GENERATION prompt, using training labels only, favoring distinct clusters."""
    if count < 1:
        raise ValueError("PoET-2 context count must be positive")
    root = Path(root)
    sft = pd.read_csv(root / "data/sft.csv")
    training = set(sft.loc[sft.split == "train", "sequence"])
    natural = {sequence.replace("-", "") for _, sequence in read_fasta(
        (root / "data/natural_aligned.fasta").read_text())}
    measured = pd.read_csv(root / "data/function_2026.csv")
    eligible = measured.loc[
        (measured.split == "train") & measured.functional.eq(True)
        & measured.sequence.isin(training & natural),
        ["sequence", "homology_group"],
    ].drop_duplicates("sequence").sort_values("sequence", kind="stable")
    eligible = eligible.loc[eligible.sequence.map(
        lambda sequence: 80 <= len(sequence) <= 96 and set(sequence) <= set(ALPHABET))]
    if len(eligible) < count:
        raise ValueError(f"Only {len(eligible)} eligible natural training positives for {count} prompts")
    shuffled = eligible.iloc[np.random.default_rng(seed).permutation(len(eligible))]
    diverse = shuffled.drop_duplicates("homology_group", keep="first")
    remainder = shuffled.loc[~shuffled.sequence.isin(diverse.sequence)]
    chosen = pd.concat([diverse, remainder], ignore_index=True).iloc[:count].copy()
    chosen.insert(0, "context_index", range(len(chosen)))
    return chosen.reset_index(drop=True)


def select_scoring_context(root: Path, *, count: int = CONTEXT_COUNT,
                           seed: int = 2001) -> list[str]:
    """The fixed UNMEASURED natural prompt behind the cm_poet sentinel.

    Deliberately not ``select_context``. That prompt belongs to the generation
    baseline and is drawn from MEASURED 2026 training positives, which a
    generator allowed to use training labels may do. cm_poet is a sentinel: a
    measured sequence inside its prompt would make it non-blind, so this pool
    excludes every row of data/qualification.csv. The two differ in the pool,
    not in the size -- both take CONTEXT_COUNT sequences under the same seed.
    """
    if count < 1:
        raise ValueError("PoET-2 scoring context count must be positive")
    root = Path(root)
    measured = set(pd.read_csv(root / "data/qualification.csv").sequence)
    natural = sorted({sequence.replace("-", "") for _, sequence in read_fasta(
        (root / "data/natural_aligned.fasta").read_text())})
    eligible = [s for s in natural if s not in measured and set(s) <= set(ALPHABET)]
    if len(eligible) < count:
        raise ValueError(
            f"Only {len(eligible)} unmeasured natural sequences for {count} PoET-2 prompts")
    order = np.random.default_rng(seed).permutation(len(eligible))[:count]
    return [eligible[index] for index in order]


def decode_draw(tokens, alphabet) -> dict:
    """Decode to the first emitted EOS; preserve the entire returned token row.

    Upstream keeps generating already-finished rows until all rows finish.
    Characters beyond their first EOS are raw decoder material, not residues
    in the delivered sequence. Missing EOS never becomes a complete protein.
    """
    ids = [int(token) for token in tokens]
    has_bos = bool(ids and ids[0] == alphabet.start_token)
    start = int(has_bos)
    eos = next((i for i in range(start, len(ids)) if ids[i] == alphabet.stop_token), None)
    end = len(ids) if eos is None else eos
    sequence = alphabet.decode(np.asarray(ids[start:end], dtype=np.int64)).decode("ascii")
    raw = alphabet.decode(np.asarray(ids, dtype=np.int64)).decode("ascii")
    standard = bool(sequence) and set(sequence) <= set(ALPHABET)
    valid_length = 80 <= len(sequence) <= 96
    reasons = []
    if not has_bos:
        reasons.append("missing_bos")
    if eos is None:
        reasons.append("missing_eos")
    if not standard:
        reasons.append("nonstandard_or_empty_sequence")
    if not valid_length:
        reasons.append("length_outside_80_96")
    return {"sequence": sequence, "raw_sequence": raw,
            "raw_token_ids": json.dumps(ids, separators=(",", ":")),
            "raw_token_count": len(ids), "sequence_length": len(sequence),
            "has_bos": has_bos, "has_eos": eos is not None,
            "first_eos_index": eos, "post_eos_token_count": 0 if eos is None else len(ids) - eos - 1,
            "standard_amino_acids": standard, "valid_length": valid_length,
            "generation_valid": not reasons, "invalid_reason": ";".join(reasons)}


def _decode_runtime(device_type: str, rotary_backend) -> dict:
    """Respect the official decoder's optional cached-rotary capability."""
    rotary_available = bool(getattr(rotary_backend, "FLASH_ROTARY_AVAILABLE", False))
    cache_available = callable(getattr(rotary_backend, "flash_attn_with_kvcache", None))
    cached = device_type == "cuda" and rotary_available and cache_available
    return {"decode_mode": "cached" if cached else "full_prefix",
            "upstream_flash_rotary_available": rotary_available,
            "upstream_kv_cache_available": cache_available}


class OfficialPoET2:
    """The official package, one loaded model, and one cached prompt memory.

    Generation and sentinel scoring share this so there is a single place where
    the PoET-2 source checkout is located, the hardware requirement is enforced
    and the checkpoint is loaded.
    """

    def __init__(self, source: Path, checkpoint: Path, context: list[str], *, device="cuda"):
        import torch

        source = Path(source).resolve()
        if not (source / "src/poet_2/models/poet_2_helpers.py").is_file():
            raise FileNotFoundError(f"Missing official PoET-2 source: {source}")
        sys.path.insert(0, str(source / "src"))
        from poet_2.alphabet.sparse_uniref_cluster2 import Alphabet
        from poet_2.models import poet_2_helpers as helpers
        from poet_2.models.modules import transformer_rotary

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("PoET-2 requires an allocated GPU")
        self.runtime_info = {"torch": torch.__version__, "device": str(self.device)}
        if self.device.type == "cuda":
            capability = torch.cuda.get_device_capability(self.device)
            if capability[0] < 8:
                raise RuntimeError("Official PoET-2 FlashAttention requires an Ampere or newer GPU")
            self.runtime_info.update(gpu=torch.cuda.get_device_name(self.device),
                                     compute_capability=list(capability))
        self.rotary_backend = transformer_rotary
        self.helpers = helpers
        self.alphabet = Alphabet()
        self.context = list(context)
        self.model = helpers.load_model(path=checkpoint, device=self.device)
        self.model.eval().requires_grad_(False)
        with torch.inference_mode():
            self.memory, _ = helpers.compute_memory(self.model, prompts=[self.context])


class OfficialPoET2Scorer(OfficialPoET2):
    """Conditional log likelihood per predicted token under the fixed prompt."""

    def score(self, sequences: list[str], *, batch_size: int = 16) -> np.ndarray:
        import torch

        if not sequences:
            return np.empty(0, dtype=float)
        # A prompt copy is already in the conditioning context; scoring one would
        # report the prompt back, not a novel design.
        if set(sequences) & set(self.context):
            raise ValueError("PoET-2 scoring must exclude exact prompt copies")
        if any(not s or not set(s) <= set(ALPHABET) for s in sequences):
            raise ValueError("PoET-2 CM inputs must contain standard amino acids")
        with torch.inference_mode():
            total = self.helpers.score_sequences(
                model=self.model, memory=self.memory, sequences=list(sequences),
                alphabet=self.alphabet, batch_size=batch_size)
        # tokenize_seqs appends start/stop, and the scorer predicts every token
        # after the start: the residues plus the stop. Dividing by len + 1 keeps
        # magnitudes comparable across the declared 80-96 length band.
        predicted = np.asarray([len(sequence) + 1 for sequence in sequences], dtype=float)
        values = total.detach().float().cpu().numpy() / predicted
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite PoET-2 scores")
        return values


class OfficialPoET2Generator(OfficialPoET2):
    """One official model and one cached prompt memory, with no structure inputs."""

    def __init__(self, source: Path, checkpoint: Path, context: list[str], *, device="cuda"):
        super().__init__(source, checkpoint, context, device=device)
        # The released backend may disable flash rotary even when FlashAttention
        # is installed. Its supported full-prefix decoder keeps the same model,
        # prompt and sampling settings without forcing that disabled kernel on.
        self.runtime_info.update(_decode_runtime(self.device.type, self.rotary_backend))
        self.fast_decode = self.runtime_info["decode_mode"] == "cached"

    def sample(self, count: int, *, seed: int, max_length: int,
               top_p: float, temperature: float) -> list[dict]:
        import torch

        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        inputs = torch.full((count, 1), self.alphabet.start_token,
                            device=self.device, dtype=torch.long)
        with torch.inference_mode():
            logits, sequences, refs, logps = self.model.generate(
                input_ids=inputs, seqid=None, memory=self.memory, self_prompt=None,
                plddt=None, s3di=None, atomx=None, atomb=None, ys_ref_values=None,
                max_length=max_length, top_k=0, top_p=top_p,
                temperature=temperature, force_consistent=False,
                fast=self.fast_decode, show_pbar=False,
            )
        token_rows = sequences.detach().cpu().tolist()
        raw_logps = logps.detach().float().cpu().tolist()
        if len(token_rows) != count or len(raw_logps) != count:
            raise ValueError("PoET-2 returned a different number of draws than requested")
        rows = []
        for tokens, values in zip(token_rows, raw_logps):
            row = decode_draw(tokens, self.alphabet)
            row["raw_decoder_logps"] = json.dumps(values, allow_nan=False)
            rows.append(row)
        del logits, sequences, refs, logps
        return rows


def score_request(request_path: Path, response_path: Path, *, source: Path, checkpoint: Path,
                  device: str = "cuda", batch_size: int = 16, scorer_factory=None) -> dict:
    """Score one batch of sequences against one prompt and write the answer out.

    This is the whole in-process side of ``cm_poet``. The training environment
    cannot import the official package at all, so the host writes a request,
    runs this under the PoET-2 interpreter, and reads the response back.
    """
    request = json.loads(Path(request_path).read_text())
    if set(request) != {"context", "sequences"}:
        raise ValueError("PoET-2 scoring request needs exactly context and sequences")
    factory = scorer_factory or OfficialPoET2Scorer
    scorer = factory(source, checkpoint, request["context"], device=device)
    started = time.monotonic()
    values = scorer.score(request["sequences"], batch_size=batch_size)
    response = {"log_likelihood_per_token": [float(value) for value in values],
                "count": len(request["sequences"]), "context_count": len(request["context"]),
                "elapsed_seconds": time.monotonic() - started,
                "runtime": getattr(scorer, "runtime_info", {}),
                "units": "mean conditional log likelihood per predicted token, "
                         "including the terminal stop token"}
    save_json(Path(response_path), response)
    return response


def generate(root: Path, output: Path, *, source: Path, checkpoint: Path,
             seed: int, count: int = 40960, batch_size: int = 32,
             context_count: int = 64, context_seed: int = 2001,
             max_length: int = 128, top_p: float = .9, temperature: float = 1.,
             device: str = "cuda", generator_factory=None) -> dict:
    """Commit each deterministic batch separately; resumed draws keep their streams."""
    if type(count) is not int or count < 1:
        raise ValueError("Generation count must be a positive integer of raw draws")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Batch size must be positive")
    if max_length < 2 or not 0 < top_p <= 1 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid generation length, top-p, or temperature")
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    call_started = time.monotonic()
    settings = {"method": "PoET-2", "source": str(Path(source).resolve()),
                "checkpoint": str(Path(checkpoint).resolve()), "seed": seed,
                "count": count, "batch_size": batch_size, "context_count": context_count,
                "context_seed": context_seed, "max_length_including_bos": max_length,
                "top_p": top_p, "temperature": temperature, "top_k": 0,
                "prompt_recipe": "natural_sft_train_and_2026_functional_train_cluster_diverse",
                "structure_conditioning": False, "force_eos": False,
                "terminal_scores_used_for_prompt_or_generation": False}
    config_path = output / "generation_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != settings:
        raise ValueError("Saved PoET-2 generation settings differ; use a new output directory")
    save_json(config_path, settings)
    _journal(output, "invocation_started", seed=seed, count=count)
    context_path = output / "context.csv"
    if context_path.exists():
        context = pd.read_csv(context_path, keep_default_na=False)
        if len(context) != context_count or context.sequence.nunique() != context_count:
            raise ValueError("Incomplete saved PoET-2 prompt")
    else:
        context = select_context(root, count=context_count, seed=context_seed)
        save_table(context_path, context)
    save_json(output / "context_summary.json", {
        "source_tables": [str((root / "data" / name).resolve()) for name in
                          ("sft.csv", "natural_aligned.fasta", "function_2026.csv")],
        "count": len(context), "homology_clusters": int(context.homology_group.nunique()),
        "prompt_tokens_including_bos_eos": int(context.sequence.str.len().sum() + 2 * len(context)),
        "selection_seed": context_seed, "recipe": settings["prompt_recipe"],
        "scope": "Prespecified training-positive context, not a tuned or proven optimal prompt"})

    factory = generator_factory or OfficialPoET2Generator
    generator = None
    model_context_seconds = 0.
    generated_batches = 0
    tables = []
    for batch_index, start in enumerate(range(0, count, batch_size)):
        wanted = min(batch_size, count - start)
        batch_seed = int(np.random.SeedSequence([seed, batch_index]).generate_state(1)[0])
        path = output / "batches" / f"batch_{batch_index:05d}.csv"
        if path.exists():
            table = pd.read_csv(path, keep_default_na=False)
            if (len(table) != wanted or table.candidate_index.tolist() != list(range(start, start + wanted))
                    or not table.batch_seed.eq(batch_seed).all()):
                raise ValueError(f"Inconsistent saved PoET-2 batch: {path}")
        else:
            if generator is None:
                started = time.monotonic()
                _journal(output, "model_context_started")
                try:
                    generator = factory(source, checkpoint, context.sequence.tolist(), device=device)
                except Exception as exc:
                    _journal(output, "model_context_finished", status="failed",
                             elapsed_seconds=time.monotonic() - started,
                             failure_type=type(exc).__name__)
                    raise
                model_context_seconds = time.monotonic() - started
                if getattr(generator, "runtime_info", None):
                    save_json(output / "runtime.json", generator.runtime_info)
                _journal(output, "model_context_finished", status="complete",
                         elapsed_seconds=model_context_seconds,
                         runtime=getattr(generator, "runtime_info", {}))
            started = time.monotonic()
            _journal(output, "batch_started", batch_index=batch_index, seed=batch_seed, count=wanted)
            try:
                rows = generator.sample(wanted, seed=batch_seed, max_length=max_length,
                                        top_p=top_p, temperature=temperature)
            except Exception as exc:
                _journal(output, "batch_finished", batch_index=batch_index, status="failed",
                         elapsed_seconds=time.monotonic() - started,
                         failure_type=type(exc).__name__)
                raise
            if len(rows) != wanted:
                raise ValueError(f"PoET-2 returned {len(rows)} draws; expected {wanted}")
            table = pd.DataFrame(rows)
            table.insert(0, "candidate_index", range(start, start + wanted))
            table.insert(1, "batch_index", batch_index)
            table.insert(2, "batch_seed", batch_seed)
            table["batch_elapsed_seconds"] = time.monotonic() - started
            save_table(path, table)
            generated_batches += 1
            _journal(output, "batch_finished", batch_index=batch_index, status="committed",
                     count=wanted, elapsed_seconds=time.monotonic() - started)
        tables.append(table)
        done = start + wanted
        save_json(output / "progress.json", {"status": "running", "completed_draws": done,
                                               "raw_draw_budget": count})
        print(f"PoET-2 generated {done}/{count}", flush=True)
    table = pd.concat(tables, ignore_index=True)
    save_table(output / "sequences.csv", table)
    summary = {"status": "complete", **settings, "unique": int(table.sequence.nunique()),
               "generation_valid": int(table.generation_valid.sum()),
               "missing_eos": int((~table.has_eos).sum()),
               "raw_draws": len(table), "median_sequence_length": float(table.sequence_length.median()),
               "measured_completed_batch_seconds": float(table.groupby("batch_index").batch_elapsed_seconds.first().sum()),
               "this_call_model_context_seconds": model_context_seconds,
               "this_call_elapsed_seconds": time.monotonic() - call_started,
               "this_call_new_batches": generated_batches,
               "runtime": (json.loads((output / "runtime.json").read_text())
                           if (output / "runtime.json").exists() else {}),
               "scope": "All raw draws counted; BOS/first EOS delimit sequences; no residue truncation or length repair. Interrupted uncommitted batches may add physical work. Raw decoder logps include post-EOS decoder material and are not selection scores."}
    save_json(output / "generation_summary.json", summary)
    save_json(output / "progress.json", {"status": "complete", "completed_draws": count,
                                           "raw_draw_budget": count})
    _journal(output, "invocation_finished", status="complete", generated_batches=generated_batches,
             elapsed_seconds=time.monotonic() - call_started,
             model_context_seconds=model_context_seconds)
    return summary
