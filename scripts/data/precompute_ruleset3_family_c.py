"""Compute RuleSet3 on-target Cas9 activity for every unique protospacer in the
official corpus (round-2 Family C, task spec §9: "Highest-priority engineered
feature: RuleSet3 / DeepSpCas9").

Must run in the isolated rs3 environment (scikit-learn<=1.0.2, incompatible with the
main .venv torch/numpy stack) -- see reports/baseline_reproduction_notes.md and the
identical pattern in scripts/evaluate/precompute_ruleset3_cache.py, which this reuses
almost verbatim but writes a plain spacer->score parquet instead of populating
OptiPrime's internal disk cache format.

Usage:
    /path/to/rs3env/bin/python scripts/data/precompute_ruleset3_family_c.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/srv/disk01/xhx/tmp/claude-8385/-srv-disk01-xhx-git-PEFormer/4e2fa2db-72ff-4f67-b286-f80299d50afa/scratchpad/rs3env/lib/python3.10/site-packages")
from rs3.seq import predict_seq  # noqa: E402

CORPUS = Path("data/processed/optiprime_official_318471.parquet")
OUT = Path("data/processed/ruleset3_scores.parquet")


def main() -> None:
    df = pd.read_parquet(CORPUS, columns=["spacer", "full_unedited"])
    print(f"loaded {len(df)} rows")

    spacer_dna = df.spacer.str.upper().str.replace("U", "T", regex=False)
    # proto30: 30nt window with the protospacer at a fixed offset, PAM set to NGG
    # (RuleSet3 doesn't model PAM variants), matching OptiPrime's own RuleSet3Score
    # feature (external/optiprime/scripts/pe/pe_inputs.py) and the round-1 cache build.
    proto30 = df.full_unedited.str.upper().str.slice(0, 30)

    unique = pd.DataFrame({"spacer": spacer_dna, "proto30": proto30}).drop_duplicates("spacer")
    unique = unique[unique.proto30.str.len() == 30]
    print(f"{len(unique)} unique spacers with a 30nt window to score")

    targets = []
    for p in unique.proto30:
        t = list(p)
        t[25] = t[26] = "G"
        targets.append("".join(t))

    scores = predict_seq(targets, sequence_tracr="Chen2013")
    unique = unique.assign(ruleset3_score=scores)
    print(unique.ruleset3_score.describe())

    out = unique[["spacer", "ruleset3_score"]].reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT} ({len(out)} unique spacers)")


if __name__ == "__main__":
    main()
