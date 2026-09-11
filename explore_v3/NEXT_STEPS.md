# V3.1 next-step plan: reliable adaptation, then independent confirmation

10 September 2026. **Proposed work, not new results.** This follows the
[completed v3 report](RESEARCH_REPORT.md), including all 34 fits and the
[matched controls](CONTROL_RESULTS.md). The original v3 protocol and results
remain unchanged.

Scope: computational research using existing measurements. Improve our own
model; no OptiPrime hybrid, teacher, features or parameters. Released OptiPrime
is an evaluation comparator only. No wet-lab programme is proposed.

## 1. Recommendation

Run a bounded **optimization and decision-preservation study**, not a broad
architecture search. Retain the frozen anchored selector E as a candidate,
strengthen its conventional controls, and test whether preserving useful source
choices works better than preserving soft score probabilities. In parallel,
resolve the independent-data gap before committing to a larger programme.

Keep two objectives separate:

1. **External-library performance:** improve achieved editing efficiency of the
   top choice over released OptiPrime, including at larger adaptation budgets.
2. **Methodological contribution:** establish a reproducible improvement in
   target utility, source retention and label cost over strong internal controls,
   with independent-domain evidence.

A win on the first does not establish the second. The intended paper remains
ambitious, but a small retrospective OptiPrime win alone would not establish the
substantial, generalizable contribution we are targeting for Nature Methods.

## 2. Evidence determining these priorities

All numbers below refer to the same v3 development populations, not the old v2
test or the published ensemble benchmark. Efficiencies are fractions.

| Completed finding | Implication for the next experiment |
|---|---|
| At 1,000 groups, E target @1 = 0.041199 versus OptiPrime 0.041409; paired difference CI includes zero | E is promising, not a validated OptiPrime winner. |
| E beats ordinary ordinal fine-tuning A by 0.001445, but all fits used only 100 target updates | Establish a stronger optimization baseline before attributing the gain to architecture. |
| E source change = -0.000687, CI [-0.002613, +0.001337] | Source noninferiority at margin -0.001 is unproven; a small point loss is not sufficient. |
| F reduces replay KL but loses source utility relative to E | Test decision-aware preservation and loss scale, not just a larger soft-KL penalty. |
| Frozen fresh-head control reaches target 0.041515 but source 0.127078 in one seed | Retain it as a strong target control; separate freezing, initialization and anchoring effects. |
| Replications vary optimizer seed, not label subset | Measure acquisition-subset variability before claiming label efficiency. |
| Replay/validation contain substantial PRIDICT data; the source selection audit is Kim-family | Evaluate and balance source domains explicitly; pooled retention can conceal conflicting effects. |
| Acquired K562 panel has only 15 shallow decisions; OPED lacks linked outcomes | Independent confirmation is an unresolved data prerequisite, not merely an unrun model evaluation. |

The source-validation comparison also matters: E trails A there despite beating
it on the source audit. We must not select the source population that makes a
method look best.

## 3. Phase 0 — diagnose the current runs and lock the next protocol

No new training is needed to begin this phase.

- Audit checkpoint histories: selection of step zero, early versus late winners,
  training loss trajectories, and variation in the 34-/162-group inner validation
  sets. Saved best-model predictions cannot reconstruct unsaved checkpoints;
  any missing diagnostic requires an explicitly counted rerun.
- Decompose selection losses by study/cell/editor, candidate depth, edit type,
  measured outcome gap and teacher score margin. Report denominators and paired
  changes. Use these as exploratory diagnoses, not independent subgroup claims.
- Distinguish harmful flips, beneficial flips and ties. Low teacher confidence
  is not automatically biological ambiguity; estimate measurement uncertainty
  only where genuine replicate data support it.
- Establish a source-training replay pool and component-disjoint retention
  validation pool with sufficient informative groups per represented study.
  Freeze equal study weights as a transparent development convention, report
  each study separately, and report the original mixture as a sensitivity check.
  Do not derive weights from source-fold-0 performance. Singleton-only domains
  need prediction endpoints, not invented selection endpoints.
- Record which retention-validation data were seen in pretraining. Exclude all
  source-fold-0-connected and held-out-target components from replay and selection.
- Predeclare the experiment matrix, label accounting, checkpoint rule, score
  scale, contrasts and stopping rules in a new versioned execution protocol.

Keep the historical source audit for descriptive continuity, but do not use it
for checkpoint selection, penalty tuning or claims of fresh confirmation.
Keep the already exposed Kim outer validation as a research evaluation, not an
untouched test. Do not reopen E25 test to choose the next method.

**Deliverables:** optimization/flip diagnostic, source-domain inventory, and
`explore_v3_1/EXECUTION_PROTOCOL.md`. New code, caches and runs belong under
`explore_v3_1/`; do not overwrite fingerprinted v3 artifacts.

## 4. Phase 1 — establish adequately optimized controls

Screen at 1,000 total labelled groups, including inner validation, using the
existing first acquisition subset and one paired optimizer seed.

Five arms: native ordinal A, direct Huber B, shared regression/ranking D,
frozen fresh pairwise C_frozen, and frozen anchored pairwise E.

Use a small, explicit grid: **two learning-rate multipliers (1 and 3) × two
training horizons (100 and 500 target updates)** per arm: at most **20 fits**.
Multipliers apply to each arm's existing parameter-group learning rates. Keep
group batching, FP32 precision and data roles matched. Include the original
checkpoint as an eligible deployment choice for every arm.

Use the same number of prespecified checkpoint opportunities per trajectory
(step zero plus ten evenly spaced checkpoints). Select within budget-inner
validation; record fixed-horizon outcomes as secondary diagnostics. Select
hyperparameters using inner validation only, not the Kim outer validation.
Report the selection rule's uncertainty rather than treating the maximum of
many noisy validation estimates as an unbiased performance estimate.

Track examples processed, passes through the training groups, active trainable
parameters, gradient norms and source replay cost. Equal update counts are not
equal compute. Do not call a loss comparison scale-controlled merely because
the scalar weights or learning rates match.

**Decision:** if longer or better-tuned conventional training removes E's
advantage, revise the claim accordingly. Retain the strongest conventional
control; do not keep an underoptimized A as the sole reference. If 500-update
curves are still clearly improving, report the baseline as unconverged and
amend the compute cap before claiming architectural superiority.

## 5. Phase 2 — isolate decision preservation and source sampling

Use E's frozen representation and anchored score throughout this phase. Hold
the selected target optimization recipe, target pairwise objective, head
capacity and source candidates processed per target update fixed.

Test a **3 × 2 × 2 factorial**, at most **12 fits** on the same development
subset/seed:

| Factor | Prespecified alternatives |
|---|---|
| Source objective | Existing soft pairwise KL; teacher top-choice margin; true-label expected source regret |
| Source sampling | Original study mixture; equal-study sampling of eligible decision groups |
| Source-loss weight | 0.1 and 1.0 under a fixed, documented score/loss normalization |

E without source regularization is the reference from Phase 1. Preserve the old
source-label pairwise replay results as historical evidence, not a scale-matched
replicated control for this new grid.

Operational definitions:

- **Top-choice margin:** for our frozen teacher's winner, penalize competitors
  crossing its ordering, using a nonnegative, capped teacher score gap as the
  desired margin. Estimate the cap from source-training scores only; add no
  arbitrary positive margin to teacher ties. This is explicitly a teacher-choice
  imitation control and can preserve teacher mistakes.
- **Label-aware utility:** minimize expected measured source regret under a
  softmax of the deployed score, averaged by group and normalized using a fixed
  source-training outcome scale. This permits correcting teacher mistakes.
  Actual argmax source utility, not surrogate loss, determines retention.
- Use one fixed score normalization and temperature convention across sampling
  controls. Log source/target gradient contributions after updates: some source
  penalties are zero at initialization, making initial-gradient matching invalid.
  Two weights bracket sensitivity; they do not prove gradient-scale equivalence.
- Report all-zero groups in evaluation. Document groups that cannot contribute
  a particular training loss rather than silently changing the denominator.

Use target inner validation and the frozen source-retention validation rule to
choose candidates. A proposed feasibility rule is to reject checkpoints whose
per-study source point loss exceeds 0.001, then maximize target inner @1 among
the remainder, with original deployment as fallback. This is an operational
development filter, **not** a statistical noninferiority certificate. Freeze
this rule before running it and apply it to all arms used for the trade-off
comparison; retain unconstrained target-selected controls as separate references.

**Promotion gate:** advance at most one new source-preservation recipe if it
shows a useful target/source trade-off relative to E on the permitted development
surfaces. If the gain exists only in training KL or a changed source mixture,
stop that preservation hypothesis. Do not repeatedly expand the penalty grid.

## 6. Phase 3 — replicate acquisition and trace the label frontier

### 6.1 Acquisition robustness

Freeze recipes before this stage. Compare at most four methods: A, the strongest
other conventional control, E, and the promoted decision-preserving method.
If no new method qualifies, use three methods and reduce the run count.

Use budgets **200 and 1,000 × three acquisition-subset seeds × two optimizer
seeds**, paired across methods: at most **48 fits**. The three existing E31
subset seeds can be used, subject to a manifest/cache coverage check. They are
repeated, potentially overlapping acquisitions from the same library—not three
independent biological datasets.

Count every labelled candidate and measurement, not only groups. Report inner
validation and development labels separately. Keep acquisition and optimization
variability separate; do not pool six runs as six independent evaluation cohorts.
Use paired component resampling shared across models/runs, with a separate
summary of subset/optimizer variability. A joint hierarchical uncertainty
analysis must preserve the shared evaluation population and paired run design.

### 6.2 Higher-label performance

For at most three finalists, add **5,000 and all-available total-budget curves
× three acquisition seeds × one paired optimizer seed**: at most **18 fits**.
Where full-data manifests are identical, deduplicate identical runs and identify
any remaining inner-split differences. Rebuild a versioned feature cache: the
v3 cache does not cover the full higher-budget training population.

This explicitly relaxes the 1,000-group objective for a separate question:
can our standalone model achieve a reliable target win with more target labels?
Do not present a 5,000/all-budget win as a 1,000-label success. Historical v2
full-budget results support testing this branch, but their different test
population prevents direct numerical comparison with the v3 validation table.

The learning curves determine the next move:

- If E catches the controls with more labels, prioritize an honest label-cost
  frontier and stable model selection, not extra backbone capacity.
- If frozen heads saturate while adapted-backbone controls keep improving,
  prioritize limited representation adaptation.
- If every method varies strongly by acquisition subset, consider a later,
  separately budgeted acquisition-policy study using training-pool information
  only. Do not select labelled examples by their known outcomes.

## 7. Architecture changes are conditional, not the first tranche

If a frozen-feature limitation survives the optimization and label-curve checks,
test one focused extension: **anchored final-block adaptation versus the same
anchored head on frozen features**, with the same retention objective and
optimization-search allowance. Do not simultaneously change pooling, loss,
replay and context inputs.

A true dual-branch model remains worth a controlled test, but the current D/E
comparison does not isolate it. Compare shared versus separate regression/ranking
readouts with matched backbone updates, loss weights, score scale and approximately
matched capacity. Both regression and ranking supervision must be present in
both arms. Apply retention to the score that actually selects designs; an unused
accurate prediction branch does not preserve deployed decisions.

Segment-aware pooling/edit alignment is a later branch only if development error
analysis identifies a persistent sequence-geometry failure. Prespecify this
branch's matrix and compute budget in an amendment; it is not included in the
98-fit main-path ceiling below. No general architecture sweep is proposed.

## 8. Independent-data work proceeds alongside development

1. **OPED:** seek the verified run/barcode-to-design, condition and replicate
   map in already available/public supplementary resources or a user-provided
   file. Do not download approximately 49.4 GB of reads before demonstrating
   that outcomes can be linked and the resulting decision population is useful.
   Author contact would require separate user authorization.
2. **ePRIDICT:** finish homology, editor-context, orientation and model-input
   eligibility checks. Freeze exclusions without viewing model performance.
   Score the 15 K562 decisions only after freezing a recipe; present these as a
   small retrospective case study, not decisive superiority evidence. HEK293T
   singleton alleles support prediction, not same-edit selection evaluation.
3. **Controlled transfer fallback:** inventory eligible source studies/contexts
   for leave-domain-out reconstruction. Any held-out domain and connected or
   homologous target components must be excluded from pretraining itself.
   Retrain ours and internal controls accordingly. This is useful transfer
   evidence but cannot manufacture an independently collected external library.
4. **Release feasibility:** document source-corpus permissions and whether a
   public-data reconstruction can reproduce the core claim. Do not redistribute
   privately supplied records or sequence-level derivatives without permission.

First deliver a feasibility memo with actual eligible groups, candidate depth,
independent study count and estimated precision. If no credible independent
confirmation population can be obtained, finish the bounded development study
and narrow the paper's claims; do not substitute endless model search for data.

## 9. Success criteria and manuscript consequences

Primary target endpoint: achieved efficiency of the top-ranked candidate among
the same eligible candidates for each fixed edit/context, retaining all-zero
groups. Report regret, @3/@5 where depth permits, within-group ranking, pooled
correlation and inference cost as secondary endpoints.

- **Reliable OptiPrime win:** a positive paired difference with a lower 95%
  confidence bound above zero on a frozen independent confirmation comparison.
  Current Kim development intervals remain exploratory. Disclose that our
  adapted model receives target labels while released OptiPrime does not;
  compare against equally adapted internal controls for method attribution.
- **Useful effect:** use approximately +0.001 absolute efficiency (+0.1
  percentage point) as a planning target, not a guaranteed minimum detectable
  effect. On current Kim validation this is about 10% of OptiPrime's remaining
  oracle regret: oracle 0.051199 minus OptiPrime 0.041409. Re-estimate precision
  for each real confirmation population before freezing its practical threshold.
- **Retention:** retain the working -0.001 noninferiority margin. On untouched
  confirmation data, require the interval lower bound to exceed it; for multiple
  prespecified source domains use simultaneous coverage or a declared primary
  source estimand. If data are too sparse, report retention as unresolved.
- **Transfer/label efficiency:** seek consistency across at least two independent
  studies and count all adaptation/inner-validation labels. This is a research
  objective, not an assertion about a journal's formal acceptance criteria.
- **Method contribution:** outperform the strongest tuned conventional control
  on a prespecified practical trade-off. Beating only the released unadapted
  comparator is insufficient evidence for a new adaptation method.

Freeze one primary method/budget/contrast before independent confirmation;
predeclare multiplicity handling for additional claims. Once confirmation is
examined, any redesign is a new development cycle requiring new confirmation.

Update the manuscript in two stages: first report v3's established results and
limitations without changing the historical benchmark claims; later add the
new method/learning curves/transfer evidence only as they are completed. Keep
single-model, mean-seed-outcome and score-ensemble results explicitly separate.

## 10. Execution budget and immediate hand-off

| Stage | Maximum new adaptation fits | Stop or advance decision |
|---|---:|---|
| Diagnostics and data feasibility | 0 initially | Lock protocol; identify missing data and baseline weaknesses. |
| Optimization controls | 20 | Establish credible controls before method attribution. |
| Preservation/sampling factorial | 12 | Promote at most one new recipe, or stop this hypothesis. |
| Acquisition replication | 48 | Assess whether the trade-off survives label selection. |
| Higher-label finalists | 18 | Locate label/capacity limits and test target superiority. |
| **Main-path ceiling** | **98** | Conditional ceiling, not a commitment to run every stage. |

The first executable tranche is **32 fits**, with a report and go/no-go decision
before replication. Diagnostic reruns, extra tuning, architecture experiments
and leave-domain-out pretraining must be separately counted; they are not hidden
inside this ceiling. Duplicate completed configurations may be reused only when
their full protocol and fingerprints match.

Estimate runtime from representative 500-update runs before scheduling the rest;
the previous 0.577 summed process-hours are not a valid blanket runtime estimate
for longer training, larger label pools or new pretraining. Use only idle,
approved GPUs and do not stop unrelated jobs.

**Immediate next work:** Phase 0 plus data feasibility, then the 20-fit control
screen. The most valuable next finding is whether a robust, adequately optimized
baseline already explains the apparent gain—or whether decision-aware adaptation
adds a reproducible benefit worth advancing to confirmation.
