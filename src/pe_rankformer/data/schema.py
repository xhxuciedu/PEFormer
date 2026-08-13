"""Canonical schema shared by every prime-editing source dataset.

One row is one measured experiment: a pegRNA design assayed in one experimental
context. Fields a source does not provide stay null; they are never imputed.

Orientation conventions
-----------------------
* ``unedited_sequence`` / ``edited_sequence``: protospacer strand, 5'->3', with the
  protospacer near the 5' end followed by the PAM.
* ``spacer`` / ``protospacer``: protospacer strand. ``spacer`` is the synthesized
  sequence (may carry a non-genomic 5' G); ``protospacer`` is the genomic match.
* ``pbs_sequence`` / ``rtt_sequence``: pegRNA orientation, 5'->3', i.e. the reverse
  complement of the protospacer-strand segments. The pegRNA 3' extension reads
  ``rtt_sequence + pbs_sequence``.
* ``editing_efficiency`` / ``indel_rate``: fractions in [0, 1].
"""

from __future__ import annotations

from typing import Final

IDENTITY_COLUMNS: Final[list[str]] = [
    "record_id",
    "source_study",
    "source_dataset",
    "source_row_id",
]

TARGET_COLUMNS: Final[list[str]] = [
    "target_id",
    "protospacer",
    "spacer",
    "pam",
    "unedited_sequence",
    "edited_sequence",
]

DESIGN_COLUMNS: Final[list[str]] = [
    "pbs_sequence",
    "pbs_length",
    "rtt_sequence",
    "rtt_length",
    "edit_type",
    "edit_length",
    "edit_position",
    "edit_position_from_nick",
    "edit_ref",
    "edit_alt",
]

CONTEXT_COLUMNS: Final[list[str]] = [
    "cell_type",
    "prime_editor",
    "pe_condition",
    "scaffold_type",
    "epegRNA_flag",
    "experimental_context_id",
]

OUTCOME_COLUMNS: Final[list[str]] = [
    "editing_efficiency",
    "indel_rate",
]

CANONICAL_COLUMNS: Final[list[str]] = (
    IDENTITY_COLUMNS + TARGET_COLUMNS + DESIGN_COLUMNS + OUTCOME_COLUMNS + CONTEXT_COLUMNS
)

# Fields whose combination defines one reproducible experimental context.
CONTEXT_KEY: Final[list[str]] = [
    "source_study",
    "cell_type",
    "prime_editor",
    "pe_condition",
]

STRING_COLUMNS: Final[set[str]] = {
    "record_id", "source_study", "source_dataset", "source_row_id", "target_id",
    "protospacer", "spacer", "pam", "unedited_sequence", "edited_sequence",
    "pbs_sequence", "rtt_sequence", "edit_type", "edit_ref", "edit_alt",
    "cell_type", "prime_editor", "pe_condition", "scaffold_type",
}

INT_COLUMNS: Final[set[str]] = {
    "pbs_length", "rtt_length", "edit_length", "edit_position",
    "edit_position_from_nick", "epegRNA_flag", "experimental_context_id",
}

FLOAT_COLUMNS: Final[set[str]] = {"editing_efficiency", "indel_rate"}
