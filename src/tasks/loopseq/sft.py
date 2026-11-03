"""Loop-seq asset-root and asset-path resolution.

The configured asset root supplies data, weights and optional configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def resolve_asset_root(repo_root: str | Path = ".") -> Path:
    """Resolve the Loop-seq backend, allowing a machine-local override."""

    configured = os.environ.get("LOOPSEQ_ASSET_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    root = Path(repo_root).expanduser().resolve()
    backend = root / "backends" / "loopseq_rl"
    return backend.resolve() if backend.exists() else root


_BACKEND_NATIVE_PREFIXES = {
    "data",
    "external",
    "MIXBend",
    "outputs",
    "params",
    "tools",
}


def _relative_suffix(path: Path, root: Path) -> Path | None:
    """Return ``path`` below ``root`` without resolving either first.

    Keeping one lexical check is important here: ``backends/loopseq_rl`` is
    normally a symlink, and resolving it too early loses the checkout-relative
    spelling that tells us which asset root should receive the suffix.
    """

    try:
        return path.relative_to(root)
    except ValueError:
        return None


def _resolve_asset_path(
    value: Any,
    *,
    backend: Path,
    repo_root: Path,
) -> Path:
    """Resolve checkout-relative and backend-native Loop-seq paths.

    Contract paths use ``backends/loopseq_rl/...`` while backend YAMLs use
    paths such as ``data/...``.  The former must be rebased onto an override
    asset root; blindly doing ``backend / value`` would otherwise produce
    ``.../loopseq_rl/backends/loopseq_rl/...``.
    """

    path = Path(str(value)).expanduser()
    checkout_backend = repo_root / "backends" / "loopseq_rl"
    if path.is_absolute():
        suffix = _relative_suffix(path, checkout_backend)
        if suffix is not None:
            return (backend / suffix).resolve()
        # A path that was already resolved follows the default backend
        # symlink, so also recognize the resolved default target and rebase
        # its suffix.
        try:
            suffix = path.resolve().relative_to(checkout_backend.resolve())
        except ValueError:
            suffix = None
        if suffix is not None:
            return (backend / suffix).resolve()
        return path.resolve()

    parts = path.parts
    if parts[:2] == ("backends", "loopseq_rl"):
        return (backend.joinpath(*parts[2:])).resolve()
    if parts and parts[0] == backend.name:
        return (backend.joinpath(*parts[1:])).resolve()
    # These names are unambiguous backend-native paths in trainer YAML; do
    # not let a same-named checkout directory shadow an asset override.
    if parts and parts[0] in _BACKEND_NATIVE_PREFIXES:
        return (backend / path).resolve()
    # A genuine checkout path (for example a custom config under ``configs``
    # outside the backend-native namespace) remains rooted at the checkout.
    candidate = repo_root / path
    if candidate.exists():
        return candidate.resolve()
    if parts and parts[0] == "configs":
        return (backend / path).resolve()
    return (backend / path).resolve()


__all__ = [
    "resolve_asset_root",
]
