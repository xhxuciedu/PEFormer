# Round-3 Research Log

Chronological experiment log per `claude_code_round3_pe_rankformer_experiments.md` §32.
Each entry: hypothesis, exact change, matched-dev result, Liu result, Kim result,
whether the hypothesis was supported, decision, next experiment.

---

## 2026-08-18 — Phase 0: inventory, baseline freeze, Liu+Kim-matched dev folds

**Hypothesis**: N/A (setup phase).

**Exact change**:
- `reports/round3_initial_inventory.md`: commits, corpus, architecture, existing
  checkpoints and which have known held-out scores (only round-1 baseline and
  round-2 Family C; Family A and D were Stage-A-only in round 2 and were never
  evaluated on held-out data).
- `configs/round3/baseline_round1.yaml`: frozen, byte-identical to round-1's
  actual winning config.
- `scripts/data/build_round3_dev_folds.py`: 3 Liu+Kim-matched, protospacer-
  disjoint development folds from the 297,962 training rows. Design: repeated
  random subsampling (not an exhaustive partition -- the 3 folds' validation
  sets may overlap each other, standard for repeated holdout), row-count-
  weighted (not protospacer-count-weighted -- Liu averages 65.8 rows/protospacer
  vs. Kim's 38.7, so a naive protospacer-count split skewed ~57% Liu instead of
  the target 44.7%; fixed by selecting protospacers via cumulative row count).
  Any protospacer also touching Schwank (62 Liu/Schwank + 241 Kim/Schwank
  collisions found) is pinned to always-train, since Schwank is otherwise
  unconditionally in every fold's training set and a shared protospacer sitting
  in both train (via its Schwank copy) and val (via its Liu/Kim copy) would be
  a real leak. Result: all 3 folds hit 44.6-44.7% Liu / 55.3-55.4% Kim (target:
  44.74%/55.26% exactly), verified protospacer-disjoint, zero Schwank in any
  validation set. 14 regression tests added (`tests/test_round3_dev_folds.py`).
- `scripts/train/train_pilot.py`: added `--dev-folds-file`/`--dev-fold-col` so
  training/evaluation can use the new dev folds instead of val_fold/train_folds.
  Smoke-tested (1 epoch, clean run).

**Development result**: N/A (infrastructure). 126/126 tests passing.

**Hypothesis supported**: N/A.

**Decision**: proceed to Stage 0's re-scoring sanity check (§6) -- evaluate the
4 existing checkpoints (round-1 baseline, round-2 Family A/C/D) on the new dev
folds and check whether the new ranking better predicts the 2 known held-out
outcomes (round-1 > Family C on held-out, despite Family C > round-1 on the
old Schwank-heavy validation).

**Next experiment**: Stage 0 sanity check (§6), then Phase 1 domain adaptation
(§8) if the sanity check passes.
