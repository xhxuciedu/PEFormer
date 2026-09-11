# PE-RankFormer: updated computational research report

Updated 10 September 2026, through E32. This is the current programme summary.
The previous narrative is preserved in [the archive](archive/SUMMARY_REPORT_E01_E28.md).
Detailed report: [PDF](../reports/adaptation_report.pdf), [LaTeX](../reports/adaptation_report.tex).

## 1. Current verdict

Our model improves pooled prediction accuracy over released OptiPrime on the
original benchmark and an external library. Without target adaptation it loses
the external fixed-allele top-choice decision. With target-library labels,
ordinary fine-tuning beats released OptiPrime on held-out loci within that library.
However, adapting a selector can reduce source utility while the original
prediction head remains strong.

This establishes a useful adaptation result, not architectural superiority,
general cross-domain transfer, or a demonstrated Nature Methods-level contribution.
The next study should establish a generalizable method for decision-focused,
label-efficient adaptation with retained deployment utility.

## 2. Prediction accuracy across evaluation populations

Spearman uses raw ranking scores. Efficiencies below are fractions.

| Population / our model | Rows | OptiPrime Spearman | Our Spearman |
|---|---:|---:|---:|
| Source benchmark / published final ensemble | 20,509 | 0.8690 | 0.9079 |
| Source Liu subset / same ensemble | 9,175 | 0.8365 | 0.8585 |
| Source Kim subset / same ensemble | 11,334 | 0.7320 | 0.8124 |
| External zero-shot / ordinal-S4D ensemble | 118,187 | 0.6931 | 0.7091 |
| External adaptation test / P, three-seed mean | 23,044 candidates | 0.6922 | 0.7771 |

Source final and external ordinal-S4D ensembles are different configurations.
Liu/Kim are benchmark subsets, not independent external tests. The zero-shot
panel and adaptation splits come from the same external library.

Calibrated Pearson is 0.8637 versus 0.8270 on source fold 0, and 0.6017 versus
0.5854 externally; the calibrator transfers unchanged. Source calibrated MAE is
0.0478 versus 0.0590 and RMSE 0.0912 versus 0.1026.
Sources: [manuscript benchmark](../reports/paper/pe_rankformer_paper.tex),
[E21](e21_correlation_surfaces.md), [E27](e27_adaptation_test.md).

## 3. Selection utility and adaptation

Achieved @1 is measured efficiency of the nominated distinct design, averaged over
fixed-allele/context groups. It is neither correlation nor best-design hit rate.

| Population / our model | Groups | OptiPrime @1 | Our @1 |
|---|---:|---:|---:|
| Source / published final ensemble | 1,463 informative | 0.1272 | 0.1364 |
| External zero-shot / ordinal-S4D ensemble | 30,475 eligible | 0.0369 | 0.0363 |
| External adaptation test / P, 3 seeds | 5,557 eligible | 0.04020 | 0.04178 |
| Same test / shared, 3 seeds | 5,557 eligible | 0.04020 | 0.04193 |
| Same test / utility S, 3 seeds | 5,557 eligible | 0.04020 | 0.04176 |
| Same test / multitask M, 3 seeds | 5,557 eligible | 0.04020 | 0.04173 |
| Same test / pairwise S, 1 historical seed | 5,557 eligible | 0.04020 | 0.04205 |

Absolute efficiencies across these populations are not directly comparable.
Informative-only external results (24,668 groups; 0.0456 versus 0.0448) must not
be mixed with all-eligible results above. Sources:
[selection accounting](../reports/paper/tables/tab_utility.tex),
[adaptation table](../reports/tables_adapt/tab_test.tex).

P improves by +0.00158, or **+0.158 percentage points**, approximately 3.9% relative
efficiency and 15.7% regret reduction. Each seed's locus interval is positive.
The test random/oracle efficiencies are 0.02323/0.05028, so selection is not solved.

Adaptation consumed **19,357 labelled training groups plus 5,561 labelled
validation groups**; released OptiPrime was not adapted. E25 components exclude
shared canonical alleles/protospacers across splits, but the library already
informed earlier research. This is retrospective within-library adaptation,
not independent cross-library confirmation.

## 4. What E29–E32 changed

### Matched initialization and label budgets

The exact initializing checkpoint scores 0.03919 on the old target test; the
published five-checkpoint ensemble scores 0.03952. The 200-training-group run gains
+0.000102 against its actual start, interval [-0.000041, +0.000255], p=0.158:
no established harm or benefit. Old small-budget runs additionally used 5,561
validation groups, had one seed/subsample, and were not consistently nested.
The “hundreds of labels cause harm” and universal label-threshold claims are withdrawn.

E31 constructed nested complete-component budgets for three subset seeds,
counting inner validation within total-label budgets. These corrected learning
curves have **not yet been trained**. [Budget protocol](e31_label_budgets.md).

### Replicated loss comparison

Selected-epoch validation means over three seeds are P 0.042713, utility S
0.042673, pairwise S 0.042998, and shared 0.043040. Direct cached-prediction
contrasts are:

| Three-seed validation contrast | Difference | 95% locus interval | p |
|---|---:|---|---:|
| Pairwise minus P | +0.000291 | [-0.000072, +0.000654] | 0.126 |
| Pairwise minus utility S | +0.000331 | [+0.000082, +0.000594] | 0.009 |
| Pairwise minus shared | -0.000023 | [-0.000270, +0.000233] | 0.888 |

These unadjusted development intervals condition on selected checkpoints.
Numerical/batch effects explain small discrepancies between training-log maxima
and re-evaluations. Seed-sign agreement does not establish superiority, and
nonsignificance does not establish equivalence. The two new pairwise seeds have
no target-test predictions. [Replication](e30_pairwise_replication.md),
[direct audit](e29_head_audit.md).

### Retention depends on the deployed head

| Source evaluation | Original prediction @1 | Deployed selection @1 |
|---|---:|---:|
| Exact initialization | 0.13350 | 0.13350 |
| Adapted P, 3 seeds | 0.12776 | 0.12776 |
| Adapted utility S, 3 seeds | 0.13323 | 0.12607 |
| Adapted pairwise S, 3 seeds | 0.13428 | 0.11460 |
| Frozen encoder Z, 1 seed | 0.13350 | 0.12697 |

E28 measured the original prediction head, not S/M/Z's deployed selector.
The four-fold retention advantage and recommendation to ship S across libraries
are withdrawn. Readout specialization is a plausible contributor, not an isolated
causal mechanism. Source fold 0 remains an exposed audit, not a tuning/replay set.

### Architecture controls and score scale

M and shared approximately match parameter counts, not architecture or score range.
At training temperature 1, mean maximum candidate probability is 0.9823 for M and
0.3409 for shared. Post-hoc positive temperature scaling cannot alter argmax;
a controlled training comparison is needed. One scalar-geometry run does not
reject segment pooling. Set attention was not tested.
[Score-scale diagnostic](e32_score_scale_diagnostic.md).

## 5. Completed work, limitations and manuscript implications

Twenty original adaptation runs and two pairwise replications are complete.
The follow-up contains 42 cached model-by-surface tables and 21 passing tests.
Original experiment JSON/checkpoints are preserved. Generated tables guard
numerical transcription, not scientific validity.

Keep the source/external correlation results and zero-shot selection reversal.
Add adaptation as a separate, caveated result; do not present it as zero-shot.
Keep P/shared as strong controls. Report deployed-selector and prediction-head
metrics separately. Withdraw blanket claims that specialized losses/dual heads
fail or that low-label adaptation harms.

The missing evidence for a stronger methods paper is a reproducible method that
transfers across independent domains, survives matched supervision/compute
controls, preserves deployment utility, and works at realistic total-label
budgets. A new split of this explored library is not an independent study.
No new architecture training, corrected learning-curve training, or independent-
library confirmation was completed in E29–E32.

Full numerical ledger: [ADAPTATION_FOLLOWUP_RESULTS.md](ADAPTATION_FOLLOWUP_RESULTS.md).

Next proposed programme: [v3 research plan](../explore_v3/RESEARCH_PLAN.md).
It prioritizes independent-data eligibility, deployed-selector preservation,
total-label efficiency and computational confirmation. The first 34 v3 fits are
now reported in [the v3 research report](../explore_v3/RESEARCH_REPORT.md);
independent confirmation remains incomplete. The v2 results above are unchanged.
