# Conditional acquisition-replication protocol

Written before replication, 11 September 2026. Execute only after the complete
32-fit screen, its report, artifact verification and a recorded go/no-go decision.
No independent-confirmation claim is possible from these development surfaces.

If the prespecified source-factorial promotion gate passes, compare A, the strongest
other conventional method chosen by target inner validation, E, and the single
promoted source method. Freeze each arm's target-inner-selected control LR/horizon;
the promoted method uses its factorial recipe. Do not retune by acquisition subset.
Use the exact v3.1 trainer and initial execution protocol; both target-only and
per-study source-constrained checkpoint policies are recorded for every run.

Matrix: budgets 200/1000 × acquisition seeds 20260910/20260911/20260912 × optimizer
seeds 20260910/20260911 × four methods = 48 configurations. Reuse the four identical
1000-group/first-subset/first-optimizer screen configurations, yielding at most
44 NEW fits and 76 total v3.1 training fits. Reuse requires identical full args and
unchanged code/cache/initial-checkpoint fingerprints. Count the reused observations
in the replication design, but do not call them new experiments.

Checkpoint opportunities remain step zero plus ten evenly spaced points. All target
labels for training and inner validation count toward each acquisition budget.
The source validation pool and per-study .001 point-loss filter stay fixed. Source
fold 0 and target outer validation remain evaluation-only within each fit. They
are already exposed research surfaces across the programme.

Primary replication question: does the promoted constrained recipe produce
nontrivial target adaptation while retaining source utility across acquisitions,
or was the first-screen result subset/optimizer-specific? Compare it with (a)
constrained A/E/conventional policies and (b) their unconstrained target-selected
policies. The latter comparisons expose any target cost hidden by fallback-only
baselines. Always include the original source checkpoint and released OptiPrime.

Report each of the six paired acquisition/optimizer combinations, the mean of
achieved outcomes (NOT a score ensemble), per-acquisition means, optimizer
variation, step-zero fallback frequency, and source domains separately. Component
bootstrap uses group outcomes averaged over matched runs and resamples the same
evaluation components across methods; its interval is conditional on these fitted
acquisitions/seeds, not a general confidence interval over unseen studies or all
possible acquisitions. Report run-to-run variability separately; do not treat the
six repeats as six independent biological cohorts. No significance testing of
optimizer seed means as if they were independent evaluation samples.

The original same-source, same-library exposure and unmatched target-label access
of released OptiPrime remain limitations. No test-set reopening, ensemble-weight
tuning, larger-budget fitting or architecture change in this tranche. Decide on
5,000/all-label curves only after this replication and the independent-data gate.
Use idle GPUs only; report realized hardware/compute and no automatic failed-run
overwrite. Do not substitute a post-hoc factorial winner if replication fails.
