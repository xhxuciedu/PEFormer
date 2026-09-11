# V3.1 research: stronger controls and acquisition replication change the anchoring claim

11 September 2026. **Completed computational research: 76 new training fits.**
No wet-lab work, OptiPrime hybrid, comparator teacher/features/parameters,
new target-test scoring, or independent-panel model evaluation.

[Updated manuscript PDF](../reports/paper/pe_rankformer_paper_v31.pdf).

## 1. Executive conclusion

**Ordinary fine-tuning becomes a strong target-selection baseline after tuning.
The unregularized anchored model's source advantage is acquisition-dependent.
Balanced teacher-margin preservation improves source robustness, but we still
have not established an external-library OptiPrime win at these label budgets.**

The strongest surviving new result is a preservation trade-off, not universal
architectural superiority. The study strengthens the paper's methodological
rigor and changes what we should claim; it does not yet deliver the independent,
generalizable methods contribution sought for Nature Methods.

The original benchmark prediction results and historical full-budget adaptation
win remain valid for their own populations and protocols. They are not replaced
by these different low-label development experiments.

## 2. What was completed

| Stage | New fits | Outcome |
|---|---:|---|
| Optimization controls: five arms × two LRs × two horizons | 20 | Tuned ordinal A closes the screened target gap to anchored E. |
| Preservation: three objectives × two samplings × two weights | 12 | Balanced teacher-margin, weight 1, passes the constrained-inner promotion gate. |
| Locked acquisition/optimizer replication | 44 | Four exact screen reuses complete the 48-configuration design. |
| **Total v3.1** | **76** | All fits completed and verified; no failed fit was silently replaced. |

Also completed: historical checkpoint/flip diagnostics; saved checkpoint
selection-stability analysis; edit-type, outcome-gap and teacher-margin strata;
depth-conditioned @3/@5, regret and correlation endpoints; RNA/DNA and overlap
audits; public-data feasibility; saved-model deployment parity and inference
timing; manuscript integration. Protocols were frozen locally before their
respective stages, not registered in a public preregistration registry.

The 76 fits consumed **2.273 summed process-hours**, including preparation inside
each process, evaluation and I/O; the 44 additional replication fits account for
1.317 hours. These are not pure GPU kernel-hours or elapsed wall-clock hours.
Training used idle RTX 2080 Ti/L40 GPUs; unrelated jobs were not stopped.
Historical fingerprinted v2/v3 artifacts were not overwritten.

## 3. Evaluation and label accounting

Target: the already exposed E25 outer validation, **5,561 fixed-edit/context
groups, 23,120 distinct candidates and 4,952 resampling components**. All-zero
groups remain included. Source retention: **1,463 informative source-audit
groups, 437 components**. Source-retention validation is a different mixture:
72 informative deepprime groups and 613 pridict_pridict2 groups, seen during
pretraining. Neither exposed audit is fresh confirmation.

Checkpoint selection uses target validation *inside* each acquisition budget.
The constrained policy additionally requires point loss no greater than 0.001
in each source-validation study, then maximizes inner target @1; original
deployment is always eligible. Source fold 0 never enters replay or checkpoint
selection. The released OptiPrime comparator receives no adaptation labels.

| Nominal budget | Acquisition seed suffix | Train groups | Inner groups | Total groups | Labelled candidates | Measurement records |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 910 | 168 | 34 | 202 | 788 | 788 |
| 200 | 911 | 159 | 41 | 200 | 695 | 695 |
| 200 | 912 | 152 | 49 | 201 | 750 | 750 |
| 1,000 | 910 | 839 | 162 | 1,001 | 3,848 | 3,848 |
| 1,000 | 911 | 807 | 193 | 1,000 | 3,555 | 3,555 |
| 1,000 | 912 | 776 | 226 | 1,002 | 3,698 | 3,699 |

Full seeds are 20260910/11/12. Two optimizer seeds reuse each acquisition.
Across all budgets/subsets, the union is 2,853 groups, 10,507 candidates and
10,508 measurement records, separate from outer validation and source labels.
The 1,000-group acquisitions share 35–74 groups pairwise. They are repeated
acquisitions from one library, not independent biological datasets.
See [acquisition audit](acquisition_audit.json).

## 4. Replicated results

Values are efficiency fractions and **means of achieved outcomes over six
paired acquisition/optimizer fits**, not metrics of a score ensemble. T means
target-only checkpoint selection; C means source-constrained selection.
M denotes v3.1 balanced margin preservation, not the earlier v2 multi-task arm.

| Method / policy | Target @1, budget 200 | Target @1, budget 1,000 | Source @1, budget 1,000 |
|---|---:|---:|---:|
| Starting checkpoint | 0.039627 | 0.039627 | 0.133635 |
| Released OptiPrime | 0.041409 | 0.041409 | — |
| A: tuned ordinal / T | 0.040158 | 0.041233 | 0.128679 |
| Fresh frozen pairwise / T | 0.039208 | 0.039800 | 0.117924 |
| E: anchored pairwise / T | 0.039381 | 0.041047 | 0.128462 |
| M: margin preservation / T | 0.039669 | 0.041126 | 0.133284 |
| M: margin preservation / C | 0.039853 | 0.040652 | 0.133858 |

The source column measures retention against our matched initialization; the
dash is not a claim that OptiPrime lacks source benchmark results.

Key paired component intervals at budget 1,000:

| Contrast | Difference | 95% interval |
|---|---:|---|
| A/T minus OptiPrime, target | -0.000177 | [-0.000726, +0.000352] |
| E/T minus A/T, target | -0.000186 | [-0.000506, +0.000157] |
| E/T minus A/T, source audit | -0.000217 | [-0.002714, +0.002329] |
| M/T minus OptiPrime, target | -0.000284 | [-0.000809, +0.000237] |
| M/T minus E/T, target | +0.000079 | [-0.000102, +0.000266] |
| M/T minus E/T, source audit | +0.004822 | [+0.003636, +0.006004] |
| M/C minus A/T, target | -0.000580 | [-0.000883, -0.000264] |
| M/C minus A/T, source audit | +0.005179 | [+0.002749, +0.007628] |
| M/C minus initialization, target | +0.001025 | [+0.000716, +0.001350] |
| M/C minus initialization, source audit | +0.000223 | [-0.000821, +0.001351] |

These intervals resample shared evaluation components after averaging matched
run outcomes. They are exploratory, unadjusted and **conditional on the six
fitted runs**, not confidence intervals over all acquisitions or unseen studies.
Nonsignificance is not equivalence or noninferiority.

The constrained margin model's averaged source-audit interval clears the working
-0.001 margin at 1,000 groups, but its target utility is below both tuned A and
OptiPrime (M/C minus OptiPrime: -0.000757 [-0.001324, -0.000219]). At budget 200,
its source interval is [-0.001359, +0.000390], so even that conditional retention
criterion is unresolved. For M/T at 1,000 groups, source change is -0.000351
[-0.001626, +0.001106], also insufficient to clear the retention margin.

All constrained A/fresh/E procedures fall back in 6/6 runs at each budget.
M/C adapts in 4/6 runs at budget 200 and 5/6 at budget 1,000. Its advantage over
fallback-only controls is real adaptation relative to initialization, but does
not erase the target cost relative to unconstrained tuned controls.

Complete outcomes, all policies, domains, confidence intervals, sensitivities
and secondary metrics: [REPLICATION_RESULTS.md](REPLICATION_RESULTS.md) and
[replication_results.json](replication_results.json).

## 5. What explains the changed conclusion?

**Optimization.** The original v3 E–A target gain was measured with a fixed
100-update recipe. In the new one-subset optimization screen, tuned A reaches
0.041101 and E 0.041304; their +0.000203 difference has interval
[-0.000295, +0.000736]. The broader grid does not prove global convergence, but
it prevents treating an under-tuned baseline as evidence of architectural
superiority. The highest-inner fresh frozen control performs poorly outside
inner validation; no post-hoc outer winner replaced it in replication.

**Acquisition sensitivity.** At budget 1,000, E's source-audit means by
acquisition are 0.133997, 0.117906 and 0.133483. The second acquisition causes
large source loss despite a frozen backbone. A is much more stable on target:
run SD 0.000131 versus E's 0.000356. Freezing representations does not protect
the decisions made by an unconstrained readout.

**Preservation helps, but its mechanism is not isolated.** M/T source means
are 0.134227, 0.132485 and 0.133140 across those acquisitions. This supports
the tested preservation recipe. It does not isolate margin geometry from
loss scale: at weight 1, median weighted-source/target gradient ratios are
0.632 for balanced margin and 0.270 for balanced KL. Balanced sampling also
changes deepprime's replay-group share from 16.4% to 49.6%. Equal nominal
weights are not gradient-matched controls, and source sampling changes the
dropout execution sequence. No claim of a uniquely identified mechanism follows.

**Selection and metrics.** Resampling fixed inner predictions selects the exact
screened E checkpoint in 35.7% of draws; this is a selection-noise diagnostic,
not retraining or independent validation. Pooled target Spearman at budget
1,000 is 0.732025 for A/T and 0.712110 for M/T, versus OptiPrime's 0.695996,
yet neither establishes better @1. Correlation remains insufficient evidence
for superior design selection. @3/@5 comparisons use only 4,014/2,056 eligible
groups and are not compared directly with the all-group @1 denominator.

The edit/gap/flip diagnostics are descriptive, with denominators retained;
measured outcome gaps are not estimates of biological measurement uncertainty.
See [screen report](SCREEN_REPORT.md), [selection stability](selection_stability.json),
[error strata](error_strata.json), [secondary diagnostics](secondary_results.json)
and the [screen trade-off plot](tradeoff.svg).

## 6. Data, deployment and verification

- Corrected RNA/DNA-normalized homology finds one additional target component
  connected by a shared 19-base spacer core. It affects one group/eight
  candidates; no matching source row enters informative replay. Excluding the
  entire component leaves screened and replicated qualitative conclusions
  unchanged. Primary frozen populations were not silently altered.
- The source corpus uses RNA U, historically assigned the token named N. This
  remains injective on its clean A/C/G/U alphabet and does not invalidate
  matched historical results. New DNA-formatted panels need a compatible
  adapter or consistently retrained tokenizer, not an inference-only swap.
- OPED metadata and primer supplements still do not provide verified
  barcode/design/condition linkage. No approximately 49.4-GB read download or
  unauthorized author contact occurred. Corrected ePRIDICT eligibility yields
  140 K562 designs but only 15 shallow decisions; context/deployment checks
  still precede any model-outcome evaluation. Independent confirmation remains
  unavailable; see [DATA_FEASIBILITY.md](DATA_FEASIBILITY.md).
- All 32 screen fits: 456 fingerprints and 288 endpoints verified. The 48
  replication configurations: 685 fingerprints, 1,056 checkpoint-surface
  reconstructions and 432 selected/final endpoints verified, including all
  budget manifests and per-study checkpoint rules. The four reused fits
  appear in both audits but count only once toward 76 new fits.
- 43 protocol/endpoint tests pass. Saved E/M deployments preserve all tested
  source/target choices across batch sizes. Full encoder/readout inference
  on 1,024 pretokenized GPU-resident candidates takes approximately
  0.753/0.758/0.761 ms per candidate for A/E/M on an RTX 2080 Ti. Timing excludes
  loading, tokenization and transfers; no OptiPrime speed advantage is claimed.
- The manuscript distinguishes original benchmark, external zero-shot,
  historical full-budget adaptation, fixed-recipe v3 and v3.1 replication.
  Existing utility prose was reconciled with the corrected fixed-allele table.
  Its 44 original numerical checks and 64 added-prose checks pass; the new
  ten-row replication table is generated directly from verified results.

## 7. Stop/advance decision and manuscript consequences

**Finish this bounded tranche at 76 fits. Do not expand the preservation grid
or start an architecture sweep.** The 98-fit figure in the proposed plan was
a conditional ceiling, not a requirement to exhaust it. The 5,000/all-label
branch, representation/dual-branch amendments and leave-domain-out pretraining
were not executed. Their prerequisites and new feature-cache/compute design
remain separate work; they must not be presented as negative experiments.

Why stop here: acquisition replication invalidates a simple anchoring advantage,
preservation buys source retention at an explicit target cost, and the independent
confirmation prerequisite is still missing. A higher-budget win on the same
exposed library could answer a useful engineering question, but would not
resolve the paper's central generalization gap; historical full-budget v2 already
demonstrated a target-supervised win on its different retrospective test.

Retain tuned A as the target baseline and margin preservation as the source-aware
candidate. Before another large fit tranche, obtain verified linked external
outcomes (or a genuinely held-out-domain reconstruction excluding that domain
from pretraining), resolve input compatibility and freeze the comparison.
If the separate priority is maximum performance on this particular library,
resume the explicitly higher-label branch with budget-aware optimization and
honest label accounting; do not relabel it a low-label or independent-transfer
success. A true dual-branch architecture remains untested here.

For the paper, retain the established prediction and full-budget results, remove
any broad claim that anchoring alone protects source decisions, and present
preservation as a tested trade-off. Independent computational confirmation and
a practical advantage over adequately tuned internal controls remain necessary
for the stronger methods claim we are pursuing.
