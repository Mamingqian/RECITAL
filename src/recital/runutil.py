"""Small shared helpers for the live task runners (kept identical across tasks)."""
from __future__ import annotations

from importlib import metadata
import os
import platform
import random
import re
import sys
from typing import Any


def runtime_environment_record() -> dict[str, Any]:
    """Return a nonsecret, content-addressed software/hardware snapshot."""
    distributions = (
        "recital",
        "numpy",
        "pandas",
        "pyarrow",
        "torch",
        "transformers",
        "tokenizers",
        "safetensors",
        "trl",
        "datasets",
        "accelerate",
        "openai",
        "jsonschema",
        "pydantic",
        "scipy",
        "scikit-learn",
        "tensorflow",
    )
    packages: dict[str, str | None] = {}
    for distribution in distributions:
        try:
            packages[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            packages[distribution] = None

    torch_record: dict[str, Any]
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        devices = []
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                devices.append(
                    {
                        "index": index,
                        "name": properties.name,
                        "compute_capability": [properties.major, properties.minor],
                        "total_memory_bytes": int(properties.total_memory),
                    }
                )
        torch_record = {
            "available": True,
            "version": str(torch.__version__),
            "cuda_available": cuda_available,
            "cuda_runtime": str(torch.version.cuda),
            "cudnn_version": (
                int(torch.backends.cudnn.version())
                if torch.backends.cudnn.is_available()
                else None
            ),
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "devices": devices,
        }
    except (ImportError, RuntimeError) as exc:
        torch_record = {
            "available": False,
            "error_type": type(exc).__name__,
        }

    record = {
        "schema_version": 1,
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "torch": torch_record,
        "scheduler": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "SLURM_JOB_NODELIST",
            )
            if os.environ.get(key) is not None
        },
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    return record


def seed_everything(seed: int) -> dict[str, Any]:
    """Apply one experiment seed to every available stochastic runtime.

    Returns a JSON-serializable record for the run manifest. Setting
    ``PYTHONHASHSEED`` cannot retroactively change the current interpreter's hash
    seed, but recording it keeps child processes consistent.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Import optional runtimes *before* establishing the final Python/NumPy
    # streams.  A first import of transformers can import TensorFlow/JAX and
    # consume NumPy randomness as part of module initialisation.  Seeding NumPy
    # only before that import would make the first call differ from every
    # subsequent call in the same process.
    import numpy as np

    torch_module = None
    transformers_set_seed = None
    record: dict[str, Any] = {
        "python_random": seed,
        "numpy": seed,
        "python_hash_seed": seed,
        "torch": None,
        "transformers": None,
    }
    try:
        import torch

        torch_module = torch
    except ImportError:
        pass
    try:
        from transformers import set_seed

        transformers_set_seed = set_seed
    except ImportError:
        pass

    # Seed the broad helper first, then finish with explicit Python/NumPy/Torch
    # calls.  The final explicit calls are intentional: optional framework
    # imports and helper internals must not advance the streams callers receive.
    if transformers_set_seed is not None:
        transformers_set_seed(seed)
        record["transformers"] = seed
    random.seed(seed)
    np.random.seed(seed)
    if torch_module is not None:
        torch_module.manual_seed(seed)
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(seed)
        record["torch"] = seed
    return record


def load_training_sample(
    parquet_path: str,
    n: int | None = 2000,
    *,
    sequence_length: int | None = None,
    split_column: str = "split",
    split_value: str = "train",
    condition_column: str | None = None,
    condition_value: str | None = None,
) -> list[str]:
    """Load the exact first ``n`` rows, or the full physical training split with None.

    Filtering uses exact, case-sensitive equality and does not sort or shuffle,
    so the resulting order is the parquet's physical row order after the split
    and optional condition filter. This is framework-owned reward input and therefore
    fails closed on
    missing columns, short inputs, nulls, non-ACGT text, or wrong lengths.
    """
    if not parquet_path:
        raise ValueError("training-reference parquet path is empty")
    if n is not None and (isinstance(n, bool) or int(n) < 1):
        raise ValueError("training-reference selection size must be positive")
    if sequence_length is not None and (
        isinstance(sequence_length, bool) or int(sequence_length) < 1
    ):
        raise ValueError("training-reference sequence length must be positive")

    import pandas as pd

    frame = pd.read_parquet(parquet_path)
    if split_column not in frame.columns:
        raise ValueError(
            f"training-reference parquet has no {split_column!r} split column: "
            f"{parquet_path}"
        )
    if (condition_column is None) != (condition_value is None):
        raise ValueError(
            "condition_column and condition_value must be supplied together"
        )
    if condition_column is not None and condition_column not in frame.columns:
        raise ValueError(
            f"training-reference parquet has no {condition_column!r} condition column: "
            f"{parquet_path}"
        )
    sequence_column = next(
        (
            column
            for column in (
                "sequence",
                "seq",
                "sequences",
                "seq50",
                "full_sequence",
            )
            if column in frame.columns
        ),
        None,
    )
    if sequence_column is None:
        raise ValueError(
            f"training-reference parquet has no sequence column: {parquet_path}"
        )
    selection_mask = frame[split_column].eq(split_value)
    if condition_column is not None:
        selection_mask &= frame[condition_column].eq(condition_value)
    selected = frame.loc[selection_mask, sequence_column]
    if n is not None:
        selected = selected.head(int(n))
    if n is not None and len(selected) != int(n):
        raise ValueError(
            f"training-reference split {split_column} == {split_value!r} has "
            f"{len(selected)} usable rows; expected at least {int(n)}"
        )
    if selected.empty:
        raise ValueError("training-reference split is empty")
    if selected.isna().any():
        raise ValueError("training-reference selection contains null sequences")
    sequences = selected.tolist()
    if sequence_length is None and sequence_column == "seq50":
        sequence_length = 50
    for index, sequence in enumerate(sequences):
        if not isinstance(sequence, str):
            raise ValueError(
                f"training-reference sequence {index} is not text"
            )
        if sequence_length is not None and len(sequence) != int(sequence_length):
            raise ValueError(
                f"training-reference sequence {index} has length {len(sequence)}; "
                f"expected {int(sequence_length)}"
            )
        if not sequence or re.fullmatch(r"[ACGT]+", sequence) is None:
            raise ValueError(
                f"training-reference sequence {index} is not uppercase ACGT DNA"
            )
    return sequences
