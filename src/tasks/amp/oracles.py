"""The scorers of the amp task: MIC regressors in the training process, Macrel, ToxinPred3 and APEX in worker processes.

Every scorer is a callable from a list of upper-case peptides of 5 to 32 residues (the engine passes legal rows only) to a
float array of one value per peptide. Sources the search sees are oriented so that higher is better.

* `activity_<pathogen>`: `1 / (1 + MIC)`, MIC in micromolar from the pathogen's AMP-Designer MIC regressor, the activity MAC-AMP
  prints. The regressor right-pads a batch with [PAD] and reads the last position without an attention mask, so a score depends
  on the longest peptide in its batch; every row is padded to 34 tokens here, which makes a score independent of its batch.
* `amp_likelihood`, `nonhemolytic`: Macrel 1.6.1 AMP probability, and one minus its hemolysis probability.
* `nontoxic`: one minus the ToxinPred3 1.4 `Hybrid Score`.
* `plddt_esmfold` (offered by the contract of the reviewed reward-design arm only): the mean pLDDT of ESMFold v1 in [0, 1].
* `apex_<pathogen>` (terminal audit only): the APEX MIC in micromolar, lower is better.

Macrel, ToxinPred3 and APEX run in their own interpreters as persistent workers (`workers/`), one JSON line per request, so a
scoring call costs milliseconds rather than the start-up of a package.
"""
from __future__ import annotations

import atexit
import json
import subprocess
from pathlib import Path

import numpy as np

from .policy import MAX_LENGTH, MIN_LENGTH, PAD_ID, RESIDUES, residue_ids

PATHOGENS = {"ecoli": ("Ecoli_regress.pth", "E. coli"), "saureus": ("Saureus_regress.pth", "S. aureus"),
             "paeruginosa": ("Paeruginosa_regress.pth", "P. aeruginosa")}
REGRESSOR_WIDTH = 34
WORKER_DIR = Path(__file__).parent / "workers"
_WORKERS: dict[tuple, "Worker"] = {}


def check_peptides(sequences) -> list[str]:
    rows = [str(sequence) for sequence in sequences]
    for row in rows:
        if not MIN_LENGTH <= len(row) <= MAX_LENGTH or not set(row) <= set(RESIDUES):
            raise ValueError(f"a scorer received a peptide that is not 5 to 32 standard residues: {row!r}")
    return rows


class Worker:
    """One persistent scorer process: a JSON request per line in, a JSON reply per line out."""

    def __init__(self, python, script, *arguments):
        self.command = [str(python), "-I", str(script), *map(str, arguments)]
        self.process = None

    def request(self, sequences) -> dict:
        if self.process is None or self.process.poll() is not None:
            self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.process.stdin.write(json.dumps({"sequences": list(sequences)}) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"scorer worker exited: {' '.join(self.command)}")
        answer = json.loads(line)
        if "error" in answer:
            raise RuntimeError(f"scorer worker failed: {answer['error']}")
        return answer

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None


def worker(python, script, *arguments) -> Worker:
    key = (str(python), str(script), tuple(map(str, arguments)))
    if key not in _WORKERS:
        _WORKERS[key] = Worker(python, script, *arguments)
    return _WORKERS[key]


@atexit.register
def close_workers():
    for each in _WORKERS.values():
        each.close()
    _WORKERS.clear()


class WorkerScore:
    """One output column of a worker reply, transformed to the orientation of its source."""

    def __init__(self, process: Worker, field: str, transform=None, column: int | None = None):
        self.process, self.field, self.transform, self.column = process, field, transform, column

    def __call__(self, sequences) -> np.ndarray:
        rows = check_peptides(sequences)
        values = np.asarray(self.process.request(rows)[self.field], dtype=float) if rows else np.zeros(0)
        if self.column is not None:
            values = values.reshape(len(rows), -1)[:, self.column]
        return values if self.transform is None else self.transform(values)


def macrel(source: str, python) -> WorkerScore:
    process = worker(python, WORKER_DIR / "macrel_worker.py")
    if source == "amp_likelihood":
        return WorkerScore(process, "amp")
    if source == "nonhemolytic":
        return WorkerScore(process, "hemolytic", transform=lambda values: 1.0 - values)
    raise KeyError(source)


def toxinpred3(python, scratch) -> WorkerScore:
    return WorkerScore(worker(python, WORKER_DIR / "toxinpred3_worker.py", scratch), "hybrid", transform=lambda values: 1.0 - values)


def apex(pathogen: str, python, apex_dir) -> WorkerScore:
    column = list(PATHOGENS).index(pathogen)
    return WorkerScore(worker(python, WORKER_DIR / "apex_worker.py", apex_dir), "mic", column=column)


class Plddt:
    """Mean pLDDT of ESMFold v1 (`facebook/esmfold_v1` through Hugging Face transformers), in [0, 1], over the atoms it places.

    Peptides of one length are folded together, so no row is padded and a score does not depend on its batch.
    """

    def __init__(self, model, tokenizer, device: str = "cpu", batch_size: int = 16):
        import torch

        self.torch, self.tokenizer, self.device, self.batch_size = torch, tokenizer, device, batch_size
        self.model = model.eval().to(device)

    @classmethod
    def from_pretrained(cls, weights: Path, device: str = "cpu", batch_size: int = 16):
        from transformers import AutoTokenizer, EsmForProteinFolding

        return cls(EsmForProteinFolding.from_pretrained(str(weights)), AutoTokenizer.from_pretrained(str(weights)),
                   device=device, batch_size=batch_size)

    def __call__(self, sequences) -> np.ndarray:
        rows = check_peptides(sequences)
        by_length: dict[int, list[int]] = {}
        for index, row in enumerate(rows):
            by_length.setdefault(len(row), []).append(index)
        out = np.zeros(len(rows))
        for indices in by_length.values():
            for first in range(0, len(indices), self.batch_size):
                chunk = indices[first:first + self.batch_size]
                tokens = self.tokenizer([rows[i] for i in chunk], return_tensors="pt", add_special_tokens=False).to(self.device)
                with self.torch.no_grad():
                    folded = self.model(**tokens)
                atoms = folded.atom37_atom_exists
                out[chunk] = ((folded.plddt * atoms).sum(dim=(1, 2)) / atoms.sum(dim=(1, 2))).float().cpu().numpy()
        if rows and not (np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0 + 1e-3):
            raise ValueError(f"ESMFold pLDDT left [0, 1]: {out.min()} to {out.max()}")
        return out


class Activity:
    """`1 / (1 + MIC)` of one AMP-Designer MIC regressor, evaluated in this process."""

    def __init__(self, weights: Path, architecture: Path, device: str = "cpu", batch_size: int = 64):
        import torch
        from torch import nn
        from transformers import GPT2Config, GPT2LMHeadModel

        class Regressor(nn.Module):
            def __init__(self):
                super().__init__()
                self.submodel1 = GPT2LMHeadModel(GPT2Config.from_json_file(str(architecture)))
                self.submodel2 = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 1))

            def forward(self, ids):
                hidden = self.submodel1(ids, output_hidden_states=True).hidden_states[-1]
                return self.submodel2(hidden[:, -1, :]).squeeze(-1)

        self.torch, self.device, self.batch_size = torch, device, batch_size
        self.model = Regressor()
        state = torch.load(weights, map_location="cpu", weights_only=True)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if [key for key in missing if not key.endswith(".attn.bias")] or [key for key in unexpected if "attn" not in key]:
            raise ValueError(f"MIC regressor weights do not fit the architecture: {missing[:3]} {unexpected[:3]}")
        self.model = self.model.eval().to(device)

    def log10_mic(self, sequences) -> np.ndarray:
        rows = check_peptides(sequences)
        out = np.zeros(len(rows))
        for first in range(0, len(rows), self.batch_size):
            ids = [residue_ids(row) for row in rows[first:first + self.batch_size]]
            width = max(REGRESSOR_WIDTH, max(len(each) for each in ids))
            batch = self.torch.tensor([each + [PAD_ID] * (width - len(each)) for each in ids], device=self.device)
            with self.torch.no_grad():
                out[first:first + len(ids)] = self.model(batch).float().cpu().numpy()
        return out

    def __call__(self, sequences) -> np.ndarray:
        return 1.0 / (1.0 + 10.0 ** self.log10_mic(sequences))


def load_oracle(name: str, *, zenodo: Path, device: str, python: dict, apex_dir: Path, scratch: Path, esmfold: Path | None = None):
    """The scorer called `name`; imports nothing heavy until it is built."""
    if name.startswith("activity_"):
        weights, _ = PATHOGENS[name.removeprefix("activity_")]
        return Activity(Path(zenodo) / weights, Path(zenodo) / "config.json", device=device)
    if name in ("amp_likelihood", "nonhemolytic"):
        return macrel(name, python["macrel"])
    if name == "nontoxic":
        return toxinpred3(python["toxinpred3"], scratch)
    if name == "plddt_esmfold":
        if esmfold is None:
            raise ValueError("plddt_esmfold needs `esmfold_dir`, the directory of the facebook/esmfold_v1 snapshot")
        return Plddt.from_pretrained(esmfold, device=device)
    if name.startswith("apex_"):
        return apex(name.removeprefix("apex_"), python["apex"], apex_dir)
    raise KeyError(name)
