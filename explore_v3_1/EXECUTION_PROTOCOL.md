# V3.1 first-tranche execution protocol

Frozen before new fitting, 11 September 2026. Computational-only development,
not independent confirmation. Implements `explore_v3/NEXT_STEPS.md` Phases 0–2.

## Data and endpoints

Reuse the verified v3 FP32 cache and E31 total-budget manifests read-only.
Screen subset and optimizer seed: 20260910; nominal budget 1000, actual 1001
groups (839 train, 162 inner validation), 3848 labelled distinct candidates.
No E25 test scoring. Kim outer validation and source fold 0 are exposed audit
surfaces and cannot choose checkpoints or hyperparameters.

Retain component-isolated source replay and retention validation. Existing source
validation has 72 informative deepprime groups and 613 pridict_pridict2 groups;
this supports a noisy development filter, not precise per-domain noninferiority.
Both pools were seen in pretraining. Hsu singleton groups are reported separately
but do not enter selection training or the source-selection feasibility filter.
No new source validation subset is selected to improve a model's audit result.

Target primary: all eligible fixed-edit/context groups, including all-zero groups.
Source continuity endpoint: depth >=2 with nonconstant measured outcomes. Also
retain unfiltered per-candidate scores so all-zero/depth strata can be reported.
Ties resolve by lexicographically smallest design key. Same deployed score on
source and target; no routing to an unused old prediction head.

## Optimization controls: exactly 20 fits

A native ordinal, B direct standardized Huber, D shared Huber+pairwise,
C_frozen fresh frozen pairwise, E frozen anchored pairwise.
Cross learning-rate multiplier {1,3} with horizons {100,500} for each arm.
Base LR 3e-5, fresh head LR .001, AdamW WD .01; original v3 active-layer policy.
Whole target groups, at most 256 candidates/batch; gradient clipping norm 1;
FP32, TF32 off. Step zero plus ten evenly spaced validation checkpoints.
All fits, including freshly initialized heads, may deploy the original model.
Training updates are not equal compute or equal data passes.

For each trajectory save two checkpoint choices: (1) highest target inner @1;
(2) highest target inner @1 among checkpoints whose source-validation @1 drop
is <=.001 in EACH of the two eligible studies. Earlier checkpoints win ties.
This is a point-estimate feasibility filter, not a confidence guarantee.
Save final-horizon predictions as a secondary optimization diagnostic as well.
Across configurations choose by the corresponding inner score only, ties by
shorter horizon then smaller LR. Equal-study source mean and original pooled
source mean are reported; neither replaces the per-study feasibility test.
Phase 2 uses E's unconstrained inner-selected LR/horizon, held fixed for all arms.

## Source factorial: exactly 12 fits after control selection

Anchored frozen E, same target pairwise loss. Source objective {soft KL,
teacher top-choice margin, measured expected regret} × sampling {original
eligible-group mixture,equal-study} × weight {.1,1}.

All source arms use the SAME informative depth>=2 pool: 486 deepprime and 2461
pridict_pridict2 groups. Thus original mixture means that pool's 486:2461 group
mixture, NOT v3's singleton-heavy row mixture. Eligibility uses only source
training labels. This intentionally changes old F's replay protocol; new soft
KL is a within-v3.1 control, not an exact F replication.

Sample whole groups with replacement across steps, no duplicate group inside a
batch. Common maximum 256 source candidates/update; actual rows differ slightly
with indivisible group sizes and MUST be reported. This is a disclosed refinement
of the plan's fixed-candidate wording, not exactly matched source row counts.
The equal-study rule samples studies equally in expectation before packing;
record realized groups/rows per study to reveal depth-related packing effects.

Score standardization uses target-training q0 mean/SD, fixed for a fit. Temperature
1 throughout. Margin loss averages relu(min(q0_best-q0_other,cap) -
(s_best-s_other)) over competitors, then groups. Cap = source-training teacher
winner-versus-other gap 90th percentile. No positive floor on teacher ties.
Source expected regret = max(y)-sum(softmax(s)*y), divided by the source-training
mean positive within-group outcome range, then averaged over groups. Soft KL
uses existing v3 group-mean Bernoulli pairwise KL. Our own q0 only, never OptiPrime.
Weights are not claimed to equalize gradients. Log target and weighted source
gradient norms at the ten checkpoint updates, using the actual training graph.

Use the same two checkpoint policies as controls. Promote at most one factorial
recipe by constrained target inner @1 (then lower source weight, shorter/name
deterministic ties). Require at least +.0005 inner target improvement over the
constrained E reference for a promising promotion; this is a development screen,
not statistical superiority. Inspect all recorded results, including negative
audit results, without changing this threshold or picking by outer validation.

## Diagnostics, gating and artifacts

Record per-update losses/gradient norm and actual target/source exposure; save
per-candidate inner/source-validation checkpoint scores and selected/last scores
on outer validation/source audit. Hash code, manifests, cache metadata and all
saved checkpoints/predictions. Retain all results, failures and resource costs.
Training imports no OptiPrime scores; comparator join happens in analysis only.
Any code correction after a launched fit requires a versioned correction and
explicit accounting; never silently overwrite a fit.

After 32 fits, report optimized-control and factorial results and decide whether
acquisition replication is warranted. Independent data feasibility is a separate
gate. A failed preservation screen or unavailable credible confirmation can stop
the expansion under the approved plan. No automatic 98-fit sweep or architecture
search. Frozen recipes, manifests and a decision memo must precede any replication.

New files stay under explore_v3_1. Historical v2/v3 training and reports are
preserved. Use idle GPUs only, and do not stop unrelated processes.
