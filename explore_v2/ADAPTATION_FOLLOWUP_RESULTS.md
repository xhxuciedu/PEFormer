# Standalone adaptation follow-up

Date: 2026-09-10. See `ADAPTATION_FOLLOWUP_PLAN.md` for the declared first tranche.
Historical E25–E28 outputs and manuscript files are preserved. No OptiPrime
hybrid, teacher, feature borrowing, or wet-lab experiment is involved.

## Execution status

- E29: matched-start and both-head evaluation completed for the historical
  models and all three pairwise seeds, including seed-matched shared-control
  validation comparisons. Candidate predictions and full SHA-256 hashes of
  the recorded checkpoint/data inputs are cached separately.
- E30: both additional pairwise-loss seeds completed with the original E26 recipe.
  Validation selected epochs; no new pairwise test results were used for selection.
- E31: nested component manifests generated for three subset seeds; protocol
  construction is completed, but new learning-curve training is not yet run.
- E32: score-scale diagnostic completed; architecture/replay training is deferred
  to the next tranche.

The first tranche is complete: two new neural training runs, 42 cached
model-by-surface prediction tables, corrected budget manifests, and 21 passing
endpoint/protocol tests. No new architecture or learning-curve training is being
claimed, and no research GPU jobs are left running after this tranche.

## Findings already established by the audit

### 1. The old low-budget harm claim used the wrong starting reference

The exact single starting checkpoint achieves approximately **0.03919** on the
previously inspected target test split. The published ensemble achieves 0.03952.
The historical 200-group run achieves approximately 0.03929. Its paired gain
against the actual initialization is **+0.000102**, 95% locus-clustered interval
**[-0.000041, +0.000255]**, p=0.158. It provides evidence of neither harm nor a
reliable benefit. The historical ensemble comparison cannot identify the effect
of fine-tuning this starting checkpoint.

### 2. Source prediction retention is not deployed-selector retention

For seed 20260910, on source fold 0 (20,509 rows; 1,463 informative decision groups):

| Model/output | Source Spearman | Source achieved @1 |
|---|---:|---:|
| Exact starting checkpoint | 0.8984 | 0.13350 |
| P prediction/deployment output | 0.8698 | 0.12786 |
| S original prediction head | 0.8895 | 0.13294 |
| S deployed selection head | 0.7541 | 0.12433 |

The original head is relatively well retained; the deployed selector is not.
Even Z's frozen encoder does not protect its new selection head: the original
prediction head retains source @1=0.13350, but its deployed selector reaches only
0.12697. The original pairwise selector is more affected (@1=0.11483), while its
original prediction head remains at 0.13426. This points to readout specialization
as an important transfer problem, rather than attributing everything to encoder
forgetting. That mechanism remains a hypothesis for controlled training tests.
This does not support the historical recommendation to ship S as a single
cross-library selector. The other two S seeds show the same direction: deployed
source selection @1 is 0.12731 and 0.12655, versus matched P values 0.12768 and
0.12775. The three-seed mean source selection @1 is approximately 0.12607 for S
versus 0.12776 for P. These are descriptive cross-seed results, not a source
equivalence or superiority test. Source and target endpoint levels have different
eligibility and should not be compared directly.

On the previously inspected target test split, the direct three-seed mean
S-minus-P contrast is -0.000021, with a locus-clustered 95% interval
[-0.000436, +0.000378], p=0.939. This establishes neither an advantage nor
equivalence. It does not support claiming that every selection objective fails.

### 3. Label-budget accounting and subset nesting are now explicit

Historical 200/1,000/5,000 training-group runs all additionally used **5,561 labelled
validation groups**. The 200-group subset shares only 12 groups with the 1,000-group
subset. These are not consistently nested curves. The historical subsets include
130/580/2,099 partially sampled training components, respectively; this does not
create train/test overlap, but contradicts the report's component-budget caption.

The corrected protocol uses nested random prefixes of whole training components,
without outcome labels. It records both group and candidate-measurement counts.
For a total-label protocol, inner validation is allocated inside each selected
budget, without using the full E25 validation labels for checkpoint selection.
Nominal 200-group total budgets yield 200–202 groups across the three seeds.

### 4. The shared-head control has a substantially different training surrogate

On target validation at temperature 1, the mean maximum softmax probability is
0.9823 for M and 0.3409 for shared. Median within-group score ranges are 87.5 and
0.2451, respectively. S-utility is similarly sharp (mean maximum probability
0.9800). Thus the scale mismatch is observed in fitted models, not only a
theoretical concern. The result does not identify which architecture would win
after correcting the control. It motivates controlling score saturation as a
training intervention; post-hoc scaling leaves argmax decisions unchanged.

### Numerical verification

The recomputed test achieved-@1 values match the stored historical values exactly
for ten of eleven audited adapted models. M seed 20260910 differs by only
-0.000000291; the re-evaluation uses independent batch construction, and this
minor numerical difference does not change the reported rounded result or claim.
The follow-up test suite currently passes 21 tests, including direct agreement
with the original endpoint implementation under ties and duplicate measurements.

### 5. Pairwise loss replicates its validation point advantage, not general superiority

Two new training runs completed (seeds 20260911 and 20260912), giving three paired
seeds including the original experiment. Selected-epoch validation means are:

| Arm | Seeds | Validation achieved @1 |
|---|---:|---:|
| P, native ordinal fine-tuning | 3 | 0.042713 |
| S, utility loss | 3 | 0.042673 |
| S, pairwise loss | 3 | 0.042998 |
| shared control | 3 | 0.043040 |

Pairwise-minus-P differences are +0.000309, +0.000256 and +0.000292; mean
**+0.000286** (about +0.0286 percentage points of efficiency). Pairwise also beats
utility S in all three validation seed pairs. It does **not** consistently beat
shared (two negative contrasts, one positive; mean -0.000042).

These are development results after checkpoint selection, not independent
confirmation or proof of statistical superiority. Direct paired intervals use
the separately cached re-evaluation predictions; small inference/batch-related
differences mean these need not exactly equal training-log validation maxima.
No new pairwise test predictions were generated for seeds 20260911/20260912.

The direct, three-seed mean contrasts from cached validation predictions are:

- **Pairwise minus P:** +0.000291, 95% locus interval [-0.000072, +0.000654],
  p=0.126. Consistent seed signs do not establish an advantage over P.
- **Pairwise minus utility S:** +0.000331, interval [+0.000082, +0.000594],
  p=0.009. This supports a development-stage advantage over the tested utility
  loss, conditional on selected checkpoints; these are unadjusted comparisons.
- **Pairwise minus shared:** -0.000023, interval [-0.000270, +0.000233],
  p=0.888. No advantage over the shared control is established.

The source tradeoff is substantial across all three pairwise seeds: mean deployed
selection @1 is **0.11460**, versus P's **0.12776** and initialization's **0.13350**.
Yet the pairwise model's original prediction head retains @1=0.13428. This is not
a general-purpose selector improvement, even though target validation improves.

## Evidence files and reproduction

- `e29_head_audit.py`: neural inference, explicit original/deployment outputs.
- `e29_summarize_audit.py`: cached prediction metrics and direct clustered contrasts.
- `e30_replicate_pairwise.py`: isolated original-recipe replications.
- `e30_replication_summary.py`: validation-only replication status and seed contrasts.
- `e31_label_budgets.py`: corrected budget manifests and historical sampler audit.
- `e32_score_scale_diagnostic.py`: label-free score-range/softmax diagnostic.
- `test_adaptation_followup.py`: budget, endpoint, bootstrap and score-scale invariants.

Generated detailed results live in the correspondingly named `.json` and `.md`
files. Prediction caches and new checkpoints are in
`cache/adapt_followup_predictions/` and `cache/adapt_followup_runs/`.

Regenerate the main summaries without GPU inference:

```bash
.venv/bin/python explore_v2/e29_summarize_audit.py
.venv/bin/python explore_v2/e30_replication_summary.py
.venv/bin/python explore_v2/e31_label_budgets.py
.venv/bin/python explore_v2/e32_score_scale_diagnostic.py
.venv/bin/python -m pytest explore_v2/test_adaptation_followup.py explore_v2/test_endpoints.py -q
```

## Next model-improvement decision

Keep P as the established simple adaptation baseline and shared as a strong
target-selection comparator. Carry pairwise forward as a useful objective
control, not an overall winner. The current results do not justify replacing
these controls or claiming architectural superiority.

The source-selector audit makes **retention of the actual deployed score** a
priority. A useful next bounded comparison is an original-ranking-preserving
residual selector versus a fresh selector, with and without source-training
replay or a retention constraint from our own frozen model. The constraint must
act on the deployed selector, not merely an unused prediction head. Exclude
source fold 0 and target-held-out components from replay; match target updates.

The shared-versus-dual-head comparison also needs comparable selection logits or
a declared temperature protocol. Post-hoc temperature scaling cannot improve
argmax decisions; the intervention belongs in training. Geometry/segment pooling
remains a separate hypothesis, not ruled out by the prior scalar-only experiment.
The diagnostic also finds much less extreme pairwise score ranges (median 3.84)
than utility-trained S (96.38); whether controlling utility-score saturation
improves adaptation requires a matched training ablation, not a post-hoc rescale.

All additional target-test audits remain retrospective. A new independent
computational library evaluation is needed for confirmatory generalization.
