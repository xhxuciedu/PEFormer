# Round-2 Research Log

Chronological experiment log per `claude_code_round2_pe_rankformer_model_search.md` §37.
Each entry: hypothesis, exact change, development result, whether the hypothesis was
supported, decision, next experiment.

---

## 2026-08-18 — Phase 0: inventory, baseline freeze, test-set lock

**Hypothesis**: N/A (setup phase).

**Exact change**:
- Wrote `reports/round2_initial_inventory.md` (architecture, losses, protocol,
  known failure modes, current held-out results).
- Tagged `round1-final` at commit `04a6dbc`.
- Froze `configs/round2/baseline_round1.yaml` (= `configs/official.yaml` +
  simplex head + `lambda_rank=0` baked in as literal values). Verified against
  already-completed round-1 runs `cv1_simplex`..`cv5_simplex` rather than
  retraining: best-val-Spearman per fold matches exactly (0.9192, 0.9173,
  0.9161, 0.9174, 0.9202).
- Added `src/pe_rankformer/evaluation/heldout_guard.py` and wired it into
  `evaluate_ensemble.py` (`--split` default changed `test`→`val`;
  `--split test` now needs `--allow-heldout-evaluation`) and
  `evaluate_pe_rankformer.py` (previously evaluated test fold
  unconditionally; now always needs the flag). Verified both directions:
  refusal without the flag, success + audit-log entry with it.

**Development result**: N/A (infrastructure).

**Hypothesis supported**: N/A.

**Decision**: proceed to Stage A screening (§6): fixed development split =
`val_fold=1, train_folds=[2,3,4,5]` (matches the existing `cv1_simplex` run,
so it doubles as a same-config sanity check), same seed (20260812), same
epoch budget, same eval code, for every Stage-A candidate.

**Next experiment**: Family A — layerwise context conditioning
(FiLM/adaptive-LN applied at every block instead of once after pooling), per
§7.
