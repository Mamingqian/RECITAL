"""The cm_poet sentinel: PoET-2 log likelihood on a fixed unmeasured CM context.

PoET-2 cannot be imported in the training environment. Its pinned
FlashAttention wheel is built for Python 3.10 / torch 2.6+cu118 and needs
GLIBC_2.32, and it requires an Ampere or newer GPU. Every call therefore runs
the official package, a checkout the user provides, in its own interpreter and
exchanges plain JSON. The host never imports ``poet_2``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from .data import ALPHABET
from .poet2 import CONTEXT_COUNT, select_scoring_context

REPOSITORY = Path(__file__).resolve().parents[4]
WORKER = REPOSITORY / "scripts/cm_poet2.py"
CONTRACT = REPOSITORY / "configs/cm_contract.yaml"


def poet_settings(contract: Path = CONTRACT) -> dict:
    """The declared cm_poet block, so diagnostics use the contract's own paths."""
    import yaml

    return yaml.safe_load(Path(contract).read_text())["oracles"]["poet"]


def run_worker(command: list[str]) -> None:
    """Run the official PoET-2 interpreter; its stderr is the failure report."""
    environment = {**os.environ,
                   "PYTHONPATH": f"{REPOSITORY / 'src'}:{REPOSITORY / 'scripts'}",
                   "PYTHONNOUSERSITE": "1"}
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            f"PoET-2 scoring worker failed ({result.returncode}): {result.stderr[-2000:]}")


class PoETOracle:
    """Per-token conditional log likelihood under one fixed unmeasured prompt.

    ``context_count`` defaults to the PoET-2 generation baseline's prompt size
    (``CONTEXT_COUNT``), so the sentinel and the generation baseline read the
    same model under the same prompt size. The pool differs; see
    ``poet2.select_scoring_context``.
    """

    def __init__(self, root: Path, *, python: str | Path, source: str | Path,
                 checkpoint: str | Path, device: str = "cuda",
                 context_count: int = CONTEXT_COUNT, batch_size: int = 16,
                 seed: int = 2001, runner=run_worker):
        self.root = Path(root)
        self.python, self.source = Path(python), Path(source)
        self.checkpoint = Path(checkpoint)
        self.device, self.batch_size = device, batch_size
        self.context = select_scoring_context(self.root, count=context_count, seed=seed)
        self.runner = runner

    def command(self, request: Path, response: Path) -> list[str]:
        return [str(self.python), str(WORKER), "--phase", "loglik",
                "--source", str(self.source), "--checkpoint", str(self.checkpoint),
                "--request", str(request), "--response", str(response),
                "--device", str(self.device), "--batch-size", str(self.batch_size)]

    def __call__(self, sequences: list[str]) -> np.ndarray:
        if not sequences:
            return np.empty(0)
        if set(sequences) & set(self.context):
            raise ValueError("PoET scoring must exclude exact prompt copies")
        if any(not s or not set(s) <= set(ALPHABET) for s in sequences):
            raise ValueError("PoET CM inputs must contain standard amino acids")
        with tempfile.TemporaryDirectory(prefix="cm-poet2-") as directory:
            request = Path(directory) / "request.json"
            response = Path(directory) / "response.json"
            request.write_text(json.dumps(
                {"context": self.context, "sequences": list(sequences)}, allow_nan=False))
            self.runner(self.command(request, response))
            payload = json.loads(response.read_text())
        values = np.asarray(payload["log_likelihood_per_token"], dtype=float)
        if values.shape != (len(sequences),):
            raise ValueError("PoET-2 returned a different number of scores than requested")
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite PoET scores")
        return values
