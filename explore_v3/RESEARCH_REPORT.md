# V3 research report: a better adaptation trade-off, not a confirmed OptiPrime win

10 September 2026. Computational-only; no OptiPrime hybrid, teacher, feature or
parameter borrowing. This reports completed experiments, not just a new plan.

## 1. Executive conclusion

**Source-anchored adaptation is worth retaining as a candidate method. The tested
soft-preservation penalty is not supported as an improvement over it. We have
not established a reliable external-library win over OptiPrime or reached the
evidentiary standard targeted for a Nature Methods submission.**

At a nominal total budget of 1,000 labelled groups, the anchored pairwise selector
improves target selection and source-audit retention over the ordinary ordinal
fine-tuning baseline across the replicated comparison. But its advantage over
released OptiPrime is not established, and source noninferiority to the original
checkpoint is also not established at the proposed -0.001 margin.

The independent-data audit changes the programme's feasibility: the acquired
ePRIDICT data provide only 15 shallow K562 decisions after sequence reconstruction;
the OPED design workbook lacks linked efficiencies. Two independent deep-choice
studies are not available yet. V3's confirmation stage therefore remains gated.

## 2. Completed work

- Acquired two public supplementary workbooks, two authors' sequence-analysis
  manifests and OPED sequencing-run metadata; hashed all inputs.
- Reconstructed intended alleles for 143 of 146 ePRIDICT designs without using
  outcome values to choose groups or resolve sequences. No model scored these
  independent-panel candidates.
- Built component-isolated source replay/validation data, excluding every source
  component connected to fold 0 or to target-panel alleles/protospacers.
- Prepared 79,335 FP32 candidate representations and corresponding tokenized
  inputs. The pretrained head reproduced cached source scores within 1.2e-7.
- Completed **34 training runs**: 14 initial fits, 16 optimizer-seed replications,
  and four matched frozen-head/source-label-replay controls. Every fit uses 100
  target updates, unchanged hyperparameters and budget-inner checkpoint selection.
- Evaluated uniform three-seed score ensembles separately from the primary
  mean-of-seed-outcomes analysis. No fitted ensemble weights or comparator scores.
- Completed precision and preservation-loss diagnostics, 32 protocol/endpoint
  tests, and an end-to-end saved-model deployment check.

Training runs consumed 0.577 summed process-hours, including evaluation and I/O;
0.331 summed hours were recorded through training completion. These are not pure
GPU kernel-hours. Preparation took about 166 seconds. The runs used idle GPUs 2
and 7; unrelated jobs were not stopped. Original v2 artifacts remain unchanged.

## 3. What the comparison actually measures

The target evaluation is **E25 outer validation: 5,561 fixed-allele/context
groups and 23,120 distinct candidates**, from the already explored Kim library.
All-zero groups remain included. No new E25 target-test predictions were made.
Source retention uses 1,463 informative source-fold-0 groups, clustered into
437 connected components for paired uncertainty. These populations differ from
the old adaptation test and from the published ensemble benchmark.

Checkpoint selection uses only validation inside each selected label budget:

| Nominal total groups | Actual total | Train groups | Inner-validation groups | Distinct labelled candidates |
|---:|---:|---:|---:|---:|
| 200 | 202 | 168 | 34 | 788 |
| 1,000 | 1,001 | 839 | 162 | 3,848 |

These are **per-fit adaptation budgets, not the label cost of the entire research
programme**. The larger outer validation informed arm-level research decisions;
the library also informed v2. A general label-efficiency claim still requires a
frozen recipe applied to another domain with its own budget counted honestly.
All replicated fits here use one subset seed and three optimizer seeds, not three
independent label acquisitions. Source fold 0 is an exposed audit that informed
research direction, not a fresh confirmation set or replay/checkpoint-selection set.

## 4. Replicated selection results

Efficiencies are fractions. Each model entry is the mean achieved outcome over
three fitted seeds, **not** an ensemble of their scores.

| Method | Target @1, budget 200 | Target @1, budget 1,000 | Deployed-source @1, budget 1,000 |
|---|---:|---:|---:|
| Exact starting checkpoint | 0.039627 | 0.039627 | 0.133635 |
| Released OptiPrime | 0.041409 | 0.041409 | — |
| A: ordinary native-ordinal fine-tuning | 0.039409 | 0.039754 | 0.129856 |
| D: shared regression-plus-ranking score | 0.038948 | 0.041465 | 0.112274 |
| E: frozen-backbone, anchored pairwise selector | 0.038642 | 0.041199 | 0.132948 |
| F: E with soft source-score preservation | 0.040148 | 0.041254 | 0.129052 |

The source column always evaluates the score that actually selects candidates,
not an unused prediction head. The dash does not imply that OptiPrime has no
source results; this table's source comparison is retention relative to our
matched initialization.

Key paired, three-seed-average development contrasts at budget 1,000:

| Contrast | Difference | 95% component interval |
|---|---:|---|
| E minus A, target @1 | +0.001445 | [+0.000975, +0.001950] |
| E minus A, source @1 | +0.003092 | [+0.000928, +0.005245] |
| E minus OptiPrime, target @1 | -0.000210 | [-0.000759, +0.000327] |
| D minus OptiPrime, target @1 | +0.000055 | [-0.000494, +0.000631] |
| F minus E, target @1 | +0.000055 | [-0.000194, +0.000308] |
| F minus E, source @1 | -0.003896 | [-0.006337, -0.001795] |

E's +0.001445 target advantage over A is +0.1445 percentage points. It is a useful
controlled development result under this optimization budget, not proof of
superiority over a fully tuned ordinal baseline in every regime.

E's source change relative to initialization is -0.000687, interval
[-0.002613, +0.001337]. The small mean loss is encouraging, but the lower bound
does not exceed -0.001: **the proposed source-noninferiority gate is not met**.
Nor does a nonsignificant E–D or E–OptiPrime contrast establish equivalence.

At budget 200, F improves target selection over A while losing more source utility;
E does not improve A. This is a budget/subset-dependent trade-off, not a universal
claim that adaptation with hundreds of labels helps or harms.

Complete initial matrix: [SCREEN_RESULTS.md](SCREEN_RESULTS.md).
All replicated seeds and contrasts: [REPLICATION_RESULTS.md](REPLICATION_RESULTS.md).

## 5. Matched controls and ensembles

The frozen fresh-head control reaches target/source @1 of 0.041515/0.127078 at
budget 1,000, versus anchored E's 0.041304/0.133085 in the same seed. Freezing the
encoder alone therefore does not reproduce E's source-retention point estimate.
This comparison matches the frozen encoder, head capacity and target objective;
initialization and initial score scale still differ.

True source-label replay reaches 0.040866/0.128611, versus soft-preservation F's
0.041157/0.129991 in that seed. Neither source replay variant establishes a better
overall trade-off than E. These are single-seed controls, not replicated causal
proof, and equal loss weights do not imply equal gradient scales.

Uniform score ensembles are an additional exploratory analysis. They use all
three predetermined seeds at equal weight, with no optimization on the evaluation
population. E/F can share one frozen backbone with three heads; A/D need three
adapted backbone evaluations. They must not replace the primary replicated
comparison when that comparison fails a claim.
At budget 1,000, E's ensemble reaches 0.041312 versus OptiPrime's 0.041409:
difference -0.000097, interval [-0.000696, +0.000489]. Its source @1 is 0.133306,
but its source-change interval [-0.002496, +0.001868] still fails the -0.001
noninferiority margin. D's ensemble has a positive target point difference
(+0.000256), also with an interval spanning zero. Ensembling does not establish
the missing OptiPrime win or resolve the source-retention claim.
Full ensemble numbers and matched-control intervals:
[CONTROL_RESULTS.md](CONTROL_RESULTS.md).

## 6. Why soft preservation did not solve retention

The diagnostic distinguishes agreement in probabilities from retained decisions.
For the first-seed, 1,000-group models, F reduces mean source-replay pairwise KL
from E's 0.0434 to 0.00578. Nevertheless, F has worse source-audit selection
efficiency. On source audit, the frozen teacher's median top-versus-runner-up
probability at the specified temperature is only 0.522; 63.2% of eligible groups
are below 0.55. Small score differences can therefore carry useful orderings
without producing a strong soft-distillation constraint.

The replay and source-validation pools also contain substantial PRIDICT-family
data, while the source audit contains Kim/Liu-family records. Improved retention
on the former does not guarantee retention on the latter. The source validation
pool was seen during pretraining; it is a retention surface, not unseen-domain
validation.

These observations support testing a **top-choice/utility-aware preservation
constraint with explicit source-domain balancing**, rather than assuming that
low soft KL preserves useful source choices. They do not isolate a single causal
explanation; loss scale, model selection and domain composition remain factors.
Numerical ledger: [retention_diagnostic.json](retention_diagnostic.json).

## 7. Independent-data gate and precision checks

The ePRIDICT workbook contains 146 designs; 143 resolve unambiguously using the
authors' unedited/edited amplicon manifests. After allele and conservative
spacer-core overlap checks, K562 has 114 alleles and **15 groups with two or three
designs; none has five**. HEK293T has 54 resolved designs over 54 different
alleles, hence no same-edit selection groups. Two ambiguous and one unmapped
design remain excluded. Full homology and deployment-input audits are not yet
complete. [Acquired-data audit and provenance](DATA_AUDIT.md).

OPED's workbook contains design alternatives but no efficiency columns. The
published accession has 30 runs totalling about 49.4 GB of compressed reads;
raw reads were not downloaded. We still need a verified mapping from runs/read
barcodes to designs, PE2 conditions and replicates before reconstructing outcomes.
[OPED data availability](https://www.nature.com/articles/s42256-023-00739-w).

An illustrative calculation using Kim development pair-difference SD 0.0244
would require about 4,669 independent decisions to detect a 0.001 gain with 80%
power under a simple normal approximation. Fifteen decisions would have an
illustrative 95% half-width near 0.0123. This is **not a formal power calculation
for K562**; domain-specific variability, dependence and heterogeneity may differ.
It shows why a few endogenous examples cannot substantiate a small universal gain.

FP32 versus historical BF16 changes the matched starting checkpoint's target
@1 from 0.039564 to 0.039627, with different selected outcomes in 39 of 5,561
groups. All v3 comparisons use the matched FP32 baseline. These precision changes
do not erase the source-only external deficit or invalidate the separate published
ensemble comparison. End-to-end reconstruction of E reproduces cached decisions
on 128 target and 128 source groups across batch sizes 31 and 128, with zero
choice changes and maximum score discrepancy below 1.8e-6.

## 8. Implications and next gate

Keep E as the best-supported **candidate for a target/source trade-off**, not as
a validated universally transferable selector. Keep D and frozen fresh-head
adaptation as strong target-utility controls. Do not advance the current soft
preservation term as the central methodological contribution.

The next computational work should be ordered as follows:

1. Resolve independent outcome provenance and input reconstruction. Do not use
   the tiny acquired panel for repeated method search or manufacture depth by
   grouping different desired edits.
2. Test direct source top-choice/utility preservation and source-stratified replay
   against E, with score/gradient-scale controls and a source validation population
   reflecting intended deployment. Do not tune these against source fold 0.
3. Repeat total-budget learning curves across independent subset seeds, then add
   5,000/all budgets for finalists. The present three optimizer seeds do not
   measure variability in which labels are acquired.
4. Freeze the method and comparator protocol before independent confirmation.
   Reconstruct source-domain holdouts with excluded-domain pretraining when making
   genuine domain-transfer claims. Resolve source-data redistribution permissions
   or supply a public reconstruction route before release.

No broad architecture sweep, full label curves, leave-study-out backbone retraining,
independent-panel model scoring, or new target-test confirmation was completed.
The present evidence improves the research direction but does **not** justify a
Nature Methods-level claim yet. Original benchmark and v2 adaptation results
remain separate, unchanged findings.

## Reproducibility

[Code and commands](README.md), [initial protocol](EXECUTION_PROTOCOL.md),
[replication decision](REPLICATION_DECISION.md), [control decision](CONTROL_DECISION.md),
[initial numerical ledger](screen_results.json), [replicated ledger](replication_results.json),
[control/ensemble ledger](control_results.json), [precision/feasibility ledger](protocol_audit.json).
All checkpoints, per-candidate predictions, logs and hashes are retained locally.
The final [artifact verification](verification.json) checks all 34 fits, 142 distinct
input/output fingerprints and the replicated aggregate values. The
[initial target/source trade-off plot](screen_pareto.svg) shows the single-seed
screen, not the replicated or ensemble comparison.
