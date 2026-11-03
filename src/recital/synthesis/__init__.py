"""Isolated generated Python primitives, admitted to run-scoped catalogs only."""
from .bundle import RuntimeBundle
from .catalog import ConstructorProxy, ExtensionCatalog, PrimitiveProxy, StagedExtension
from .models import (
    IsolationError, PrimitiveRequest, ROLE_CROSS_ORACLE_AGG, ROLE_HARD_GATE,
    ROLE_PER_ORACLE_UTILITY, ROLE_PER_SEQ_PENALTY, ROLE_LIBRARY_SELECTOR, RuntimeLimits, SynthesisError,
    canonical_json, probe_representation,
)
from .references import ReferenceBinder, ReferenceBinding

__all__ = [
    "ConstructorProxy", "ExtensionCatalog", "IsolationError", "PrimitiveProxy",
    "PrimitiveRequest", "ReferenceBinder", "ReferenceBinding", "RuntimeBundle",
    "RuntimeLimits", "StagedExtension", "SynthesisError", "canonical_json",
    "ROLE_CROSS_ORACLE_AGG", "ROLE_HARD_GATE",
    "ROLE_PER_ORACLE_UTILITY", "ROLE_PER_SEQ_PENALTY", "ROLE_LIBRARY_SELECTOR", "probe_representation",
]
