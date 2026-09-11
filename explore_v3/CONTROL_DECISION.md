# Matched-control and ensemble checks

Recorded after 30 planned fits, before these additional computations.

E at 1,000 total groups improves target selection over A in all three optimizer
seeds, with substantially less source damage than the shared-score model. F does
not show a source-retention advantage over E. These development observations
justify the planned causal controls, not a superiority claim over OptiPrime.

Run four additional fits: C_frozen and F_replay at budgets 200/1000, seed 20260910.
C_frozen shares E's frozen encoder, 256-unit scalar head and target pairwise
objective, but has a fresh unanchored head. F_replay replaces F's soft teacher
preservation with true source-label pairwise replay, using the same source pool,
target updates, batches, loss weight and learning rates. This does not fully
match initial score variance or make differently scaled losses equivalent.
Record that limitation rather than interpreting either result as pure causality.

Also evaluate arithmetic-mean score ensembles for each replicated arm A/D/E/F
at both budgets, using all three predetermined optimizer seeds and equal weights.
This is an additional exploratory, no-training analysis, not the prespecified
primary mean-of-seed-outcomes comparison. No weight fitting or seed selection.
E/F share a frozen backbone, so their ensemble can reuse one representation;
A/D require distinct adapted backbone evaluations. No OptiPrime score is used
in any ensemble. Do not substitute an ensemble result for a failed primary claim.

No acquired independent panel or E25 target test is used in these checks.
