"""CRE primary MinGap selection under the explicitly declared oracle panel."""
from recital.library_selection import LibrarySelectionPolicy, PrimaryLibrarySelector
from .score_contract import PROMPT_TO_COLUMN, TARGET_CELLS, panel_gap, panel_scope


def build_library_selector(contract, *, active_cells, oracle_panels, training_sequences):
    policy = LibrarySelectionPolicy.from_contract(contract)
    if policy is None:
        return None
    if policy.primary_oracle != "malinois" or policy.deduplication != "exact":
        raise ValueError("CRE delivery requires its declared Malinois primary and exact identities")
    panel = oracle_panels[policy.primary_oracle]
    panel_scope(**panel)
    columns = list(panel['oracle_columns'])
    mapping = {prompt: TARGET_CELLS[index] for prompt, index in PROMPT_TO_COLUMN.items()}
    mapping.update({cell: cell for cell in active_cells})

    def primary_scores(values, groups):
        if values.shape != (len(groups), len(columns)):
            raise ValueError("CRE primary output width must match the declared oracle panel")
        # A row with a missing declared head scores NaN and is left out of delivery.
        return panel_gap(
            values, groups, oracle_columns=columns,
            off_target_cells_by_target=panel['off_target_cells_by_target'], missing="nan",
        ).gap

    identities = frozenset(str(sequence) for sequence in training_sequences)
    if policy.exclude_training_copies and not identities:
        raise ValueError("CRE selected delivery requires a nonempty full training reference")
    return PrimaryLibrarySelector(
        policy, tuple(active_cells), mapping.get, primary_scores,
        lambda sequence: isinstance(sequence, str) and len(sequence) == 200 and set(sequence) <= set('ACGT'),
        str, identities,
    )
