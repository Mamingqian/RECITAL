"""Keep measurement side effects out of policy sampling and optimization."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import random
from collections.abc import Iterator
from typing import Any

import numpy as np


@contextmanager
def isolated_rng() -> Iterator[None]:
    """Restore Python, NumPy and available Torch CPU/CUDA global streams.

    This does not seed a scorer or rewind its explicitly owned generators. A
    stochastic oracle must own and record its inference RNG. In particular, a
    deterministic oracle's lazy model initialization or DataLoader must never
    advance the policy's global stream. Restoration also applies on failure.
    TensorFlow/JAX generators, when used, remain the oracle adapter's concern.
    """
    try:
        import torch
    except ImportError:
        torch = None
    py_state = random.getstate()
    np_state = np.random.get_state()
    devices = (
        list(range(torch.cuda.device_count()))
        if torch is not None and torch.cuda.is_available() else []
    )
    torch_context = torch.random.fork_rng(devices=devices) if torch is not None else nullcontext()
    with torch_context:
        try:
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


@contextmanager
def evaluation_mode(model: Any) -> Iterator[None]:
    """Run under ``torch.no_grad`` with ``model`` in eval mode.

    Every submodule's own ``training`` flag is restored afterwards, including a
    mixed train/eval configuration and a failure inside the block.
    """
    import torch

    modes = [(module, module.training) for module in model.modules()] if isinstance(
        model, torch.nn.Module
    ) else []
    try:
        if modes:
            model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training
