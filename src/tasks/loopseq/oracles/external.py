"""Loader for the scorers of external models whose code this repository does not contain."""
from __future__ import annotations

import importlib
from typing import Callable


def scorer_factory(spec: str, oracle: str) -> Callable:
    """Resolve ``package.module:factory`` to the callable that builds the scorer of ``oracle``.

    The factory takes the oracle's declared paths as keyword arguments and returns an object with
    ``predict(sequences, batch_size)``.
    """
    if not spec:
        raise ImportError(
            f"{oracle} runs the upstream model's code, which is not part of this repository; set `scorer` to "
            "'package.module:factory', a callable returning an object with predict(sequences, batch_size)")
    module, attribute = spec.split(":")
    return getattr(importlib.import_module(module), attribute)
