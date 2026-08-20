"""Build the round-5 internal lockbox (spec §4) -- and document why it is Liu-only.

The spec asks for a new round-5 lockbox "if practical". A *composition-matched* one
is not, and that is worth recording rather than quietly substituting something else.

Accounting of the 2,501 Liu+Kim protospacers in the 297,962-row training pool:

    used in a round-3 dev validation set   1,553
    consumed by the round-4 lockbox          367
    collide with Schwank (always training)    302
    ------------------------------------------------
    still untouched                          279   <- 98.5% Liu

The round-4 lockbox deliberately reproduced the official held-out set's 44.7% Liu /
55.3% Kim composition. The residue cannot: Kim protospacers were nearly exhausted by
the dev folds, leaving ~1.5% Kim here. A "matched" round-5 lockbox would need Kim
protospacers that no longer exist unused.

So this builds what is actually available -- a **Liu-only** lockbox of ~19k rows over
279 protospacers -- and labels it as such. It is a genuine, never-selected-on surface
and a valid check for Liu-side overfitting, but it says nothing about Kim, which is
precisely where round 5 is trying to improve (Kim 0.8124 vs Liu 0.8585). It must
therefore be read alongside the round-4 lockbox, not as a replacement for it.

Consequence for round-5 discipline, stated plainly: **Kim has no fresh gate left.**
Kim-side claims rest on out-of-fold dev folds plus the round-4 lockbox (used once),
and should be treated as the less-protected half of any result.

Like the round-4 lockbox, this is scored **out-of-fold**: its rows sit in official
folds 1-5, so every official-split model trained on roughly four fifths of them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_round5_lockbox")

CORPUS = Path("data/processed/optiprime_official_318471.parquet")
DEV = Path("data/processed/round3_dev_assignments.parquet")
LB4 = Path("data/processed/round4_lockbox.parquet")
OUT = Path("data/processed/round5_lockbox_folds.parquet")
SUMMARY = Path("results/round5/lockbox_accounting.json")
LIU_KIM = ("hsu2026", "deepprime")


def main() -> None:
    corp = pd.read_parquet(CORPUS, columns=["record_id", "spacer", "source_study", "fold"])
    dev = pd.read_parquet(DEV)
    lb4_spacers = set(pd.read_parquet(LB4).spacer)

    val_ids: set = set()
    for c in [c for c in dev.columns if c.startswith("round3_dev_fold")]:
        val_ids |= set(dev.loc[dev[c] == "val", "record_id"])
    dev_spacers = set(corp.loc[corp.record_id.isin(val_ids), "spacer"])

    train_pool = corp[corp.fold != 0]
    liukim = train_pool[train_pool.source_study.isin(LIU_KIM)]
    schwank_spacers = set(train_pool.loc[~train_pool.source_study.isin(LIU_KIM), "spacer"])

    all_sp = set(liukim.spacer)
    eligible = all_sp - dev_spacers - lb4_spacers - schwank_spacers
    sub = liukim[liukim.spacer.isin(eligible)]

    accounting = {
        "total_liu_kim_protospacers": len(all_sp),
        "in_dev_validation": len(all_sp & dev_spacers),
        "in_round4_lockbox": len(all_sp & lb4_spacers),
        "collide_with_schwank": len(all_sp & schwank_spacers),
        "eligible_remaining": len(eligible),
        "rows": int(len(sub)),
        "liu_fraction": float((sub.source_study == "hsu2026").mean()),
        "official_heldout_liu_fraction": 0.4474,
        "composition_matched": False,
    }
    logger.info("protospacer accounting: %s", json.dumps(accounting, indent=2))

    if accounting["liu_fraction"] > 0.90:
        logger.warning(
            "round-5 lockbox is Liu-only (%.1f%% Liu vs %.1f%% in the official held-out "
            "set): Kim protospacers were exhausted by the dev folds and the round-4 "
            "lockbox. This gate CANNOT validate Kim-side claims.",
            100 * accounting["liu_fraction"], 100 * accounting["official_heldout_liu_fraction"],
        )

    # Invariants -- this file gates promotion, so verify rather than trust.
    assert not (sub.fold == 0).any(), "round-5 lockbox contains held-out rows"
    assert not (set(sub.record_id) & val_ids), "overlaps a dev validation row"
    assert not (set(sub.spacer) & dev_spacers), "shares a protospacer with dev validation"
    assert not (set(sub.spacer) & lb4_spacers), "shares a protospacer with the round-4 lockbox"

    out = pd.DataFrame({"record_id": sub.record_id, "round5_lockbox_val": "val"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(accounting, indent=2))
    logger.info("wrote %s (%d rows, %d protospacers, folds %s)",
                OUT, len(out), sub.spacer.nunique(), sorted(sub.fold.unique()))
    logger.info("evaluate with --dev-folds-file %s --dev-fold-cols round5_lockbox_val --oof", OUT)


if __name__ == "__main__":
    main()
