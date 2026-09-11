# V3 execution protocol — initial tranche

Recorded 10 September 2026 before v3 fitting. The parent research plan remains
the proposal; this file records operational choices and deviations as they arise.

1. Acquire and audit OPED and ePRIDICT supplements without scoring models on them.
   Report actual same-edit candidate depth and overlap, not published row counts.
   Keep acquired panels outside model development. Missing eligible independent
   decisions stop the broad-confirmation claim, not the useful bounded Kim pilot.
2. Preserve all v2 data, checkpoints and results. Use the exact
   `r4p2_ordSSM_cv1` initialization and E31 nested total-label manifest, subset
   seed 20260910, budgets 200/1000. Record file SHA-256 hashes.
3. Model selection uses only budget-internal target validation. E25 validation
   is an exposed development evaluation, never early stopping. No new E25 test
   predictions in this initial tranche. Source fold 0 is an audit only.
4. Replay and source-validation components exclude any component touching fold 0
   and any allele/protospacer in the target panel. Do not infer absence of
   pretraining exposure from these new source roles. This is retention research.
5. Start from FP32 cached source-model representations for the frozen-head
   experiments; they must reproduce full-model predictions and use no comparator
   output. Confirm identity initialization, complete-group batching, deterministic
   ties and differentiable zero loss for constant groups.
6. Fixed optimization budget, not fixed epochs: 100 target updates per arm,
   batches of at most 256 rows containing whole groups; evaluate at step 0 and
   every 10 updates. Epoch zero means deployment of the original predictor for
   all arms, including fresh heads. Choose maximum inner achieved @1, retaining
   the earlier checkpoint on ties. AdamW, weight decay 0.01; fresh/residual head
   learning rate 0.001; native pretrained head and unfrozen backbone rate 0.00003.
   Hidden width 256; temperature 1 after fixed training-only score normalization.
   Replay weight 1; no hyperparameter sweep in the initial screen.
7. The primary training comparison matches target updates; replay adds source
   work, which is counted. Source-preservation selection is evaluated as a
   trade-off after fitting, not optimized on source fold 0. Report each arm's
   actual trainable modules and costs. A frozen-head screen cannot substitute
   for the planned final-block fine-tuning baselines.
8. Outcomes here are exploratory. Replications and matched causal controls are
   conditional on the screen, as specified in the plan. Log implementation
   failures and failed hypotheses; do not replace them silently.

No OptiPrime information enters training. No external uploads, model deployment,
wet-lab work, or external-study confirmation is authorized by this protocol alone.
