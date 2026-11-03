"""Task-independent immutable training references and run-scoped synthesis setup."""
from __future__ import annotations

from pathlib import Path


def build_training_reference_binder(sequences: list[str], labels: list[str] | None = None,
                                    *, source: str = "training"):
    """Create immutable grants from supplied rows and their verified labels."""
    from recital.synthesis import ReferenceBinder, ReferenceBinding

    if not sequences:
        raise ValueError("synthesis requires a nonempty declared TRAIN reference")
    if labels is not None and (len(labels) != len(sequences)
            or any(type(label) is not str or not label for label in labels)):
        raise ValueError("TRAIN reference needs exactly one nonempty condition label per row")
    values = {"training_reference": (sequences, labels, None)}
    if labels is not None:
        for condition in sorted(set(labels)):
            identifier = f"training_reference_condition_{condition}"
            rows = [sequence for sequence, label in zip(sequences, labels, strict=True) if label == condition]
            values[identifier] = (rows, [condition] * len(rows), condition)
    grants = {
        identifier: ReferenceBinding(
            binding_id=identifier,
            source=source,
            condition=condition,
            schema={"type": "array", "items": {"type": "string", "minLength": 1}},
            count=len(rows),
        )
        for identifier, (rows, row_labels, condition) in values.items()
    }
    binder = ReferenceBinder(grants, allowed_sources={source})
    for identifier, (rows, row_labels, _) in values.items():
        binder.approve(identifier, rows, labels=row_labels)
    return binder


def build_recipe_synthesis(*, output_dir: Path, capabilities, training_sequences: list[str],
                           allowed_oracles: set[str], representation: dict, limits=None,
                           training_conditions: list[str] | None = None,
                           reference_source: str = "training"):
    """Run-scoped controller for synthesis."""
    from recital.synthesis import ExtensionCatalog, RuntimeLimits
    from recital.synthesis.integration import RecipeSynthesis

    binder = build_training_reference_binder(training_sequences, training_conditions, source=reference_source)
    configured = RuntimeLimits(**limits) if limits is not None else RuntimeLimits(max_message_bytes=16 * 1024 ** 2)
    catalog = ExtensionCatalog(output_dir / "extensions", allowed_oracles=allowed_oracles,
                               representation=representation,
                               reference_binder=binder, limits=configured)
    try:
        for path in sorted(catalog.artifact_dir.iterdir()):
            if path.is_dir() and (path / "PUBLISHED").is_file():
                catalog.load(path.name).publish()
        controller = RecipeSynthesis(catalog, capabilities)
        declared = capabilities.admitted_primitives
        loaded = catalog.capability_declarations()
        if any(key not in loaded or loaded[key] != value for key, value in declared.items()):
            raise ValueError("frozen baseline extensions need their complete hash-bound published artifacts in the run extensions directory")
        return controller, binder.declarations()
    except BaseException:
        catalog.close()
        raise

