"""DNAcycP2 cyclizability predictor.

Runs the upstream ``dnacycp2`` package in its own Python environment through a subprocess. Startup and TensorFlow
loading make it too slow for inner-loop reward calls, so the primary protocol uses it only as a computational
predictor in the terminal blind evaluation.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tasks.loopseq.assets import DNACYCP2_REPO

DEFAULT_VENV_PYTHON = DNACYCP2_REPO / ".venv311" / "bin" / "python"

_RUNNER = """
from pathlib import Path
import os
import sys

repo_path = Path(sys.argv[1]).resolve()
input_path = Path(sys.argv[2]).resolve()
basename = Path(sys.argv[3]).resolve()
smooth = sys.argv[4] == "1"

os.chdir(repo_path)
from dnacycp2 import cycle_txt
cycle_txt(str(input_path), str(basename), smooth=smooth)
"""


def _trim_to_50(sequence: str) -> str:
    sequence = str(sequence).upper()
    if len(sequence) == 50:
        return sequence
    if len(sequence) == 100:
        return sequence[25:75]
    raise ValueError(f"DNAcycP2 expects 50 bp or 100 bp inputs, got length={len(sequence)} for {sequence!r}")


@dataclass
class DNAcycP2Oracle:
    repo_path: str | Path = DNACYCP2_REPO
    python_bin: str | Path = DEFAULT_VENV_PYTHON
    smooth: bool = True
    normalized: bool = True

    def __post_init__(self) -> None:
        self.repo_path = Path(self.repo_path).resolve()
        self.python_bin = Path(self.python_bin)

    def _output_suffix(self) -> str:
        if self.smooth:
            return "_C0S_norm.txt" if self.normalized else "_C0S_unnorm.txt"
        return "_C0_norm.txt" if self.normalized else "_C0_unnorm.txt"

    def __call__(self, seqs: list[str]) -> np.ndarray:
        trimmed = [_trim_to_50(sequence) for sequence in seqs]
        with tempfile.TemporaryDirectory(prefix="loopseq_dnacycp2_") as directory:
            input_path = Path(directory) / "sequences.txt"
            basename = Path(directory) / "pred"
            input_path.write_text("\n".join(trimmed) + "\n", encoding="utf-8")
            env = os.environ.copy()
            env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
            subprocess.run(
                [str(self.python_bin), "-c", _RUNNER, str(self.repo_path), str(input_path), str(basename),
                 "1" if self.smooth else "0"],
                check=True, cwd=str(self.repo_path), env=env)
            values = np.atleast_1d(np.loadtxt(f"{basename}{self._output_suffix()}", dtype=np.float32)).reshape(-1)
        if len(values) != len(trimmed):
            raise ValueError(
                f"DNAcycP2 output row count does not match input sequence count: {len(values)} vs {len(trimmed)}")
        return values.astype(float)
