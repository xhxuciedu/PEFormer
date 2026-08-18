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

---

## 2026-08-18 — Family A: layerwise context conditioning (in progress)

**Hypothesis** (§7): context-dependent sequence interpretation, applied throughout
the network rather than only after pooling, should especially help heterogeneous
(Kim-like) conditions.

**Exact change**: `PERankFormerConfig.context_strategy: "late"|"layerwise"`.
Layerwise replaces the black-box `nn.TransformerEncoder` with an explicit
`ModuleList` of individual `nn.TransformerEncoderLayer`s so FiLM can be applied
before each one: `h'_l = (1+gamma_l(c))*h_l + beta_l(c)`, `h_{l+1} = block_l(h'_l)`,
applied to all 6 edit-encoder layers, all 4 pegRNA-encoder layers, and both
cross-attention blocks (14 FiLM modules total). The single late FiLM after pooling
is removed in this mode (context already permeates every layer). Params: 28.21M vs
26.64M baseline (+5.9%). 3 new unit tests (forward/backward, `use_context=False`
guard, param-count divergence) — 96/96 passing before launch.

**Stage-A run**: `configs/round2/baseline_round1.yaml` + `--context-strategy
layerwise`, run `r2_familyA_layerwise`, GPU 6. Same split/seed/epoch-budget as the
`cv1_simplex` baseline run for direct comparability.

**Development result (running)** — val Spearman by epoch, layerwise vs. baseline
(`cv1_simplex`, same val_fold=1):

| epoch | baseline | layerwise | Δ |
|---|---|---|---|
| 0 | 0.8439 | 0.8257 | −0.018 |
| 1 | 0.8624 | 0.8623 | −0.000 |
| 2 | 0.8825 | 0.8906 | +0.008 |
| 3 | 0.8873 | 0.8964 | +0.009 |
| 4 | 0.8903 | 0.8991 | +0.009 |
| 5 | 0.8906 | 0.9037 | +0.013 |
| 6 | 0.8913 | 0.9013 | +0.010 |
| 7 | 0.8984 | 0.9074 | +0.009 |

Layerwise starts behind (more parameters, presumably slower early optimization)
but overtakes by epoch 2 and holds a consistent +0.008-0.013 lead through epoch 7.
Still training; will record final best-epoch numbers here once complete.

**Hypothesis supported**: provisionally yes (pending full run + full 5-fold
confirmation per §6 Stage B, only after this and other Stage-A candidates are
ranked).

**Decision**: let it run to completion / early-stop. Given the lead is holding, if
it ends up in the top ~3 Stage-A candidates it proceeds to Stage B (all 5 official
CV folds).

---

## 2026-08-18 — Family C: feature-augmented branch (in progress)

**Hypothesis** (§9): the model currently uses zero continuous features (no
lengths, GC content, Tm, MFE, or Cas9-activity scores anywhere) -- adding a small
parallel MLP over externally-computed features should help, particularly the
RuleSet3 on-target activity score flagged as highest-priority in the spec.

**Features computed** (`scripts/data/compute_family_c_features.py`, 318,471 rows,
train+held-out uniformly since these are model *inputs* computable without any
label): PBS/RTT length, PBS/RTT/extension GC%, edit length/position/position-from-
nick/mismatch-count (reusing the already-tested `seqops.diff_window` /
`find_protospacer`), PBS/RTT melting temperature (Bio.SeqUtils RNA_NN3 table,
matching OptiPrime's own PBSMeltRNA feature), 4 MFE features via ViennaRNA
(protospacer, RTT, PBS, extension), and RuleSet3 on-target score (computed for all
36,731 unique protospacers in ~12s via the isolated rs3 env, joined onto every
row). RuleSet3 coverage: 96.4% (missing where `full_unedited` is under the 30nt
window RuleSet3 needs -- the same 151-row Kim truncation issue noted in round 1).

**Bug caught before use**: first RuleSet3 join used raw `spacer` (RNA, mixed-case
U6-`g`) against a DNA-uppercase-keyed score table -- 98.6% silently missing.
Fixed by normalizing both sides to DNA-uppercase before the join (96.4% coverage
after the fix). Worth flagging: a silent 98.6%-missing join would have trained a
model that saw RuleSet3 as noise for nearly every row without erroring.

**Exact change**: `PERankFormerConfig.n_features`/`feature_hidden_dim`; new
`FeatureBranch` module (`LayerNorm -> [features ++ missing_mask] -> Linear -> GELU
-> Dropout -> Linear`, concatenated onto the pooled sequence representation before
the head). Missingness is tracked, not silently imputed: `attach_family_c_features`
(new `src/pe_rankformer/data/family_c_features.py`) imputes NaN with the
**training-row-only** mean (satisfying §9's normalization rule) and passes a
parallel 0/1 mask into the branch. 5 new unit tests (branch forward/backward,
disabled-by-default, train-only normalization, missingness masking, record_id
mismatch guard) — 101/101 passing before launch.

**Stage-A run**: same frozen config + `--feature-branch`, run
`r2_familyC_features`, GPU 2 (parallel with Family A on GPU 6 -- more GPUs are
free than the single-GPU budget the spec assumes, used here to parallelize
independent Stage-A screens rather than serialize them).

**Development result**: running, not yet reported.

**Hypothesis supported**: pending.

**Decision**: let both run in parallel; compare against baseline and each other
once complete, then decide on Family D (A+C combined) per §10's rule of only
combining components that individually improved validation performance.
