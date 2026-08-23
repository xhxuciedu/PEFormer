# Audit: the repeated 0.9082 (round-6 spec §28)

The model report quotes Ordinal-SSM at **0.9082** for both out-of-fold development
Spearman and held-out Spearman. The spec correctly flags that as suspicious. It was
regenerated mechanically from the underlying artifacts.

## Result: genuine coincidence, not a reporting error

| Quantity | Source | n | ρ |
|---|---|---:|---:|
| dev fold 0 | `predictions_r4p2_ordSSM_oof_round3_dev_fold_0.parquet` | 35,649 | 0.909432 |
| dev fold 1 | `..._fold_1.parquet` | 35,692 | 0.906574 |
| dev fold 2 | `..._fold_2.parquet` | 35,700 | 0.908532 |
| **dev, mean of per-fold ρ** | — | 107,041 | **0.908179** |
| *(dev, pooled over concatenated rows)* | — | 107,041 | *0.908156* |
| **held-out** | `predictions_round4_final.parquet`, `member_ordSSM` | 20,509 | **0.908155** |

**The two numbers differ at the fifth decimal place** (0.908179 vs 0.908155) and agree
only when rounded to four. They are computed from:

- **different files** (three per-fold development prediction sets vs one held-out set),
- **different row counts** (107,041 vs 20,509),
- **disjoint rows** — the intersection of development and held-out `record_id`s is
  **0**, verified directly.

## Aggregation checks

- The development figure is the **mean of per-fold ρ**, not a pooled correlation over
  concatenated rows. Both were computed; they differ by 0.000023, so the choice of
  convention does not explain the coincidence either.
- The held-out figure is a single correlation over all 20,509 rows, taken from the
  member column stored inside the frozen ensemble prediction file, so it is the same
  vector that contributed to the reported ensemble.

## Conclusion

No error. The agreement to four decimal places is chance. Both numbers stand as
reported, and the report's practice of quoting them to four decimals is what creates
the appearance of an identity — worth keeping in mind when they are quoted together.
