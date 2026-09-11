# PE-RankFormer v3: transferable, label-efficient pegRNA selection

Proposed 10 September 2026. **Plan, not completed experiments.**

Execution update: the initial screen, seed replications and matched controls
(34 fits) are now complete. See [the research report](RESEARCH_REPORT.md) for
results and unmet gates. The proposal below is preserved as the original plan;
full learning curves and independent confirmation remain incomplete.

Scope: computational research using existing measurements. No wet-lab work;
no OptiPrime hybrid, teacher, features, scores or parameters in our model.
OptiPrime remains an evaluation comparator. This plan follows the corrected
[E01–E32 report](../explore_v2/SUMMARY_REPORT.md) and
[adaptation PDF](../reports/adaptation_report.pdf).

## 1. Recommendation and intended contribution

Make **reliable selection under domain shift and limited target labels** the
central methodological problem. Test whether a source-anchored decision score,
trained to preserve source decisions while learning target preferences, improves
actual pegRNA choice across independent studies. Do not center the paper on
another small improvement in pooled correlation or a one-library fine-tuning win.

The intended advance is a validated adaptation procedure with a demonstrably
better target-utility/source-retention/label-cost trade-off. Its useful biological
output is the choice of a measured design for a fixed desired edit, not merely a
more accurate pooled regression score.

This is a proposed route toward the requested Nature Methods standard, not a
prediction of acceptance. The journal emphasizes methodological advances with
practical relevance and strong validation; our inference is that a modest
within-library gain alone is insufficient. [Nature Methods aims and scope](https://www.nature.com/nmeth/aims).
OptiPrime's published contribution includes mechanistic interpretation and
prospective applications, not just a benchmark score. Our computational study
needs its own substantial contribution; it need not imitate those experiments.
[OptiPrime primary paper](https://www.nature.com/articles/s41587-026-03261-7).

Residual heads, replay, adapters and distillation are not individually novel.
In particular, preserving old predictions and parameter-efficient adaptation
have established precedents. A combination of these ideas is not automatically
a methods contribution. V3 must identify what is specifically necessary for
pegRNA decision transfer and demonstrate a reproducible advantage over those
standard alternatives. [Learning without Forgetting](https://arxiv.org/abs/1606.09282);
[parameter-efficient transfer](https://proceedings.mlr.press/v97/houlsby19a.html).

## 2. What the present evidence does—and does not—justify

| Observation | Consequence for v3 |
|---|---|
| Source Spearman 0.9079 versus OptiPrime 0.8690; external zero-shot 0.7091 versus 0.6931, but external achieved @1 0.0363 versus 0.0369 | Optimize and evaluate decisions directly; pooled correlation is secondary. Source and external ensembles differ. |
| Ordinary native-ordinal fine-tuning P reaches 0.04178 versus released OptiPrime 0.04020 on the old adaptation test | Adaptation is feasible. This is not a matched-label comparison or independent cross-library confirmation. |
| Three-seed pairwise-minus-P validation difference is +0.000291, CI [-0.000072, +0.000654]; pairwise does beat utility S on that development comparison | Pairwise is promising, not an established winner over ordinary fine-tuning or shared. |
| Pairwise source prediction @1 is 0.13428 but deployed-selector @1 is 0.11460; initialization is 0.13350 | Preservation must act on the deployed score, not an unused prediction head. |
| Frozen encoder Z still loses source selection utility | Readout specialization is a plausible contributor; encoder forgetting alone cannot explain all observed losses. This is not yet a causal isolation. |
| Old small-budget runs used an additional 5,561 labelled validation groups; shared and multitask logits had very different scales | Repair label accounting and score-scale controls before interpreting architecture or sample efficiency. |

The Kim large library and source fold 0 are now exposed research/audit surfaces.
A new random split cannot make either an untouched confirmation dataset.

## 3. First gate: secure valid evaluation populations

### 3.1 Create an exposure and eligibility ledger before more model search

For every proposed dataset and comparator, record study, accession, assay,
cell/editor context, candidate sequences, intended allele, strand, protospacer,
replicates, outcome definition, candidate depth, token-length support and known
training/development exposure. Hash source files and split manifests. Distinguish
new loci, new contexts and genuinely new studies; none implies the others.

Build decision groups using the same desired allele and experimental context.
Collapse repeated measurements of a design without counting them as alternative
designs. Retain all-zero groups for the primary endpoint and report informative
groups separately. Report depth >=2, >=5 and >=8 rather than allowing the many
binary source decisions to stand in for deep library selection.

Audit exact sequence/allele/protospacer overlaps and near-homologous target
sequences. Prespecify and record similarity rules without selecting thresholds
from outcomes. Use allele–protospacer connected components for partitioning and
a stronger homology-cluster sensitivity analysis. A nominal study holdout with
shared designs is not clean transfer.

### 3.2 Candidate acquisition queue—not a claim of available benchmarks

| Priority / existing-data candidate | Intended use | Required eligibility check |
|---|---|---|
| 1. OPED authors' original measurements, accession PRJNA882795 and supplementary/source data | Independent-laboratory, preferably endogenous transfer | Acquire sequences and measured outcomes; establish non-overlap and multiple designs for the same edit. Reused public training data are not new evidence. |
| 2. PRIDICT2/ePRIDICT measured endogenous panels, supplementary tables 7/8; data accession PRJNA1025026 | Reporter-to-endogenous and cell-context transfer | Some related data are already in our source corpus. Establish panel-level novelty and decision depth; otherwise use prediction only. |
| 3. OptiPrime/Hsu endogenous source data | Retrospective measured-design application | The local Endo_gRNAs sheet contains designs, not an efficiency benchmark. Recover matched outcome data and provenance first; disclose the relationship to comparator development. |
| 4. Leave-study/cell/editor-out reconstruction of our source corpus | Controlled domain-shift development and mechanism tests | Retrain excluding the held-out domain and overlapping components from pretraining. Existing checkpoints may already have seen these contexts. |

Acquisition sources: [OPED primary paper and data availability](https://www.nature.com/articles/s42256-023-00739-w),
[PRIDICT2/ePRIDICT primary paper and supplements](https://pmc.ncbi.nlm.nih.gov/articles/PMC7617539/),
[Hsu et al. source data](https://www.nature.com/articles/s41587-026-03261-7).
Local starting inventory: [external eligibility audit](../explore_v2/e06_external_eligibility.md),
`data/manifests/optiprime_context_counts.csv`, `data/raw/hsu2026/`,
`external/pridict2/`, and `external/deepprime/data/`. Historical E06 group counts
are superseded; use current canonical grouping code to recount.

Important comparator exclusion: released PRIDICT2 model B was trained using the
large Library-ClinVar dataset. It cannot be presented as a clean zero-shot
competitor on our current Kim large panel. Audit DeepPrime exposure as well.
The PRIDICT2 paper documents its training composition; audit each actual
checkpoint, rather than assuming every version has identical exposure.
[Training-data provenance](https://pmc.ncbi.nlm.nih.gov/articles/PMC7617539/).

TRIP measurements across integration sites are not automatically alternative
pegRNAs for one edit. Fitness screens are not editing-efficiency labels.
Long-insertion datasets with differing intended alleles or incompatible outcome
scales are secondary/out-of-scope unless those problems can be resolved. Do not
manufacture a selection benchmark by pooling different desired edits.

**Data go/no-go:** aim for at least two independent studies outside the current
Kim development library, with multiple contexts and a measured endogenous
application. This is our research target, not a journal rule. First count
eligible decisions and estimate attainable precision using development-based
cluster simulations. If independent deep-choice data are too sparse, retain
them as limited prediction/transfer evidence and narrow the paper's claims.
Do not run an expansive architecture search while assuming missing validation
data will eventually materialize.

### 3.3 Separate development, adaptation and confirmation

- Use current Kim data for method development, explicitly retrospective. Use
  source fold 0 only as an exposed audit, never replay or early stopping.
- Establish a source-training replay pool and a distinct source validation pool
  without fold 0 or held-out target components. For existing checkpoints,
  disclose whether the source validation pool was previously seen in pretraining;
  it measures retention, not unseen-source generalization.
- For genuinely held-out source domains, retrain all relevant backbones with
  those domains excluded. Removing labels only during fine-tuning is insufficient.
- Keep at least one eligible independent study entirely outside architecture and
  hyperparameter development. Freeze the adaptation recipe before applying it.
  Its allowed adaptation subset can supply inner validation; confirmation
  components must not influence model selection.
- Freeze outcome-independent manifests, endpoint definitions and analysis code
  before scoring confirmation. An adverse result is reported, not followed by
  tuning against the same confirmation set.

## 4. Main method hypothesis: source-anchored deployed-score adaptation

Start with a small, interpretable change rather than replacing the backbone.
Let `q0(x)` be our frozen source score and `h0(x)` its representation. Propose:

`s(x, context) = standardize(q0(x)) + r_theta(h0(x), context)`

Use a fixed positive affine standardization estimated from permitted training
data. Initialize the residual's last layer to zero, so step zero exactly
preserves the original ordering. Avoid clipping transformations that introduce
ties. The **same final score s** is evaluated for source and target decisions;
silently returning the old head on source data is not selector retention.

Initial objective:

`L = L_target_selection(s) + lambda_ret * L_source_preservation(s, q0)`

The preservation term constrains within-group decisions of the actual deployed
selector on permitted source groups. Compare soft pairwise distillation from
our own frozen model with a source-label ranking/replay control; an old model's
mistakes need not become ground truth. Use only our checkpoints and data—never
OptiPrime predictions. Optional prediction loss is a subsequent ablation, not
an unexamined addition to every variant.

Test frozen representations first. This isolates readout adaptation and supports
cached, inexpensive pilots. If capacity limits target utility, test a small
adapter or final-block update under the same source-preservation constraint.
Compare against fresh-head adaptation with the same capacity, feature inputs
and encoder update policy. Do not attribute a benefit to anchoring when the
comparison also changes trainable layers or supervision.

Record both prediction-head and deployed-selector metrics, residual magnitude,
score range, within-group probability entropy, gradient norms and near-tie
decision flips. Report source replay's extra compute and the cost of retaining
a frozen model if required at inference.

### Loss and architecture controls

- Retain native ordinal fine-tuning P. Add direct expected-efficiency regression
  (one prespecified Huber baseline), not an incorrectly relabelled ordinal loss.
- Compare pairwise ranking and smooth expected-utility training at comparable
  logit scales. Fix a small temperature/score-gain search on development inner
  validation, then freeze its selection rule. Post-hoc temperature cannot
  improve argmax; it is the training dynamics being tested.
- Repair the shared-head control using an unbounded decision-score parameterization
  with a separate specified transformation for the prediction loss. Match inputs,
  losses and update policy where making a shared-versus-dual-head claim.
- Treat direct source-label replay, own-model distillation and residual anchoring
  as distinct factors. Test whether the proposed component improves on standard
  replay, rather than claiming replay itself as the invention.

## 5. Bounded first experiment tranche

Use the E31 **total-label** manifests, not the old fixed-5,561-validation curves.
Budgets of approximately 200 and 1,000 groups include inner validation. Select
epoch zero when adaptation is not supported by that validation set.

| Arm | Question |
|---|---|
| A. Native ordinal P | Can ordinary fine-tuning explain the gain? |
| B. Direct-efficiency Huber regression | Is ordinal supervision the relevant limitation? |
| C. Fresh pairwise selector | Strong rank-based adaptation baseline |
| D. Corrected shared score | Does a shared head compete once scale is controlled? |
| E. Source-anchored residual, pairwise | Does identity initialization/readout anchoring help? |
| F. E plus deployed-score source preservation | Does explicit source preservation improve the trade-off? |
| G. Residual plus preservation, utility loss | Is any benefit specific to pairwise training? |

A–D reproduce their clearly specified training policies; E–G initially freeze
the encoder. Consequently the first table is an operational screen, not a pure
causal architecture comparison. In the follow-up, give C/E/F identical encoder
policies and head capacity, and add source-label replay as a matched control.
Log target-update counts, source updates, wall time and GPU memory separately.

Initial screen: **7 arms × 2 budgets × 1 paired seed = 14 fits**, plus no-update
reference predictions. Use a fixed, small development-selected configuration
per arm; any extra tuning fits must be logged and counted, not hidden outside
the budget. Promote P, the strongest other conventional baseline, E and F to
two additional seeds at both budgets: at most 16 additional fits. If E/F fail,
stop this hypothesis rather than substituting a post-hoc winning variant.

These approximately 30 fits are a development gate, not enough evidence for a
final method claim. Estimate GPU-hours from the first measured fits before
authorizing larger runs. Preserve v2 checkpoints and write all v3 artifacts to
new paths. No v3 fits were launched when this plan was written.

**Promotion rule:** require a promising target/source Pareto improvement over
P and the strongest conventional control across seeds, not just the largest
target validation score. The matched replay and encoder-policy ablations must
then retain that advantage before independent confirmation.

## 6. Conditional research branches, in priority order

1. **Representation adaptation:** if frozen residual capacity is insufficient,
   compare last-block updates and a small adapter with identical deployed-score
   preservation. Reject a larger model if its benefit disappears against a
   compute-matched conventional baseline.
2. **Segment-aware representation:** if residual errors implicate PBS/RTT/edit
   geometry on development data, test explicit segment pooling and canonical
   edit alignment. Validate insertion/deletion masks and sequence orientation.
   Include scalar-only, pooling-only and feature-only tree/ranker controls.
   The old geometry concatenation experiment did not test this hypothesis.
3. **Domain-aware training:** use leave-domain-out development episodes, context
   conditioning and source-group balancing if failures track cell/editor/assay.
   Compare a universal deployed score with explicitly routed domain-specific
   models; routing is a separate deployment assumption, not universal retention.
4. **Uncertainty-aware acquisition:** only after passive budget curves work,
   simulate buying complete training groups using our model's uncertainty and
   sequence diversity. Reveal only acquired labels, include validation cost,
   and compare random acquisition at the same measurement budget. This is an
   offline label-efficiency experiment, not a wet-lab proposal.

Set attention and additional omics are not first-line work. A physical design's
score should not arbitrarily change when irrelevant alternatives are added.
If a set model is tested, require permutation and candidate-set perturbation
checks. Context features constant across a locus cannot by themselves explain
within-locus preference; richer features require a specific testable interaction.
Do not launch all four branches simultaneously.

## 7. Label efficiency, fair comparators and statistical contract

### Total-label curves

Evaluate budgets 0/200/1,000/5,000/all when supported by the dataset. Report actual
complete-component counts, groups, distinct labelled candidates and replicate
measurements; groups are not individual experimental measurements. Include
inner validation in each total budget. Whole components may modestly exceed
nominal caps; disclose actual costs. Budget zero permits no target-labelled
validation for checkpoint choice.

Use three nested subset seeds and, for finalists, three optimizer seeds per
subset where feasible. Distinguish variation in acquired data from variation in
training. Compare equal target updates as the primary optimization control and
equal epochs as a sensitivity analysis; log replay's extra cost. The existing
three E30 optimizer seeds do not substitute for these corrected curves.

### Comparators

First validate our standalone method against frozen initialization, ordinal P,
direct regression, rank/shared controls and standard source replay. Compare
released OptiPrime separately, clearly stating unequal target-label exposure.
After our recipe is frozen, a matched-total-label OptiPrime adaptation comparator
can be considered as a separate fairness experiment if technically feasible;
it is not a hybrid and does not feed anything into our model. If not performed,
do not claim architecture superiority from beating its frozen release.

Add appropriate PRIDICT2/DeepPrime checkpoints only after training-exposure and
input-support audits. Report native coverage and a fixed common-support subset;
never remove difficult cases solely to favor our architecture. Where checkpoints
have seen the evaluation domain, mark that comparison as exposed or retrain
under matched exclusions. Match tuning opportunities and report compute.
[Computational benchmarking guidance](https://pmc.ncbi.nlm.nih.gov/articles/PMC6584985/).

### Endpoints and uncertainty

- **Primary:** achieved @1, the average measured efficiency of the nominated
  distinct design across eligible fixed-allele/context groups. Report differences
  in efficiency fractions and percentage points. Keep all-zero groups included.
- **Secondary:** regret versus the measured best candidate; relative regret
  reduction `(regret_baseline - regret_method) / regret_baseline` when the
  denominator is meaningful; within-group rank concordance; achieved @3 and
  success-at-budget; pooled Spearman/Pearson; deployed-source retention.
- Use identical candidate sets and prespecified deterministic ties. Evaluate
  decision scores consistently in full precision and test batch/permutation
  invariance, especially near ties.
- Bootstrap paired allele–protospacer components within study, not individual
  measurements. Report seed-specific effects and seed variability. Macro-average
  study-level effects rather than letting Kim's size dominate. Thousands of loci
  in one study are not thousands of independent domain-shift replications.
- Prespecify the primary budget, control and contrast before confirmation. Use
  multiplicity adjustment for a declared secondary family; selected development
  comparisons remain exploratory. Nonsignificance is not equivalence.
- Where replicates exist, assess label reliability and, where feasible, use
  different replicates for determining preferences and evaluating them. Report
  the noise sensitivity of the measured oracle; do not infer a noise ceiling
  from arbitrary synthetic noise or treat the best noisy observation as truth.

### Proposed practical success gates

These are working research criteria to freeze on development data, **not Nature
Methods requirements or universally meaningful biological thresholds**:

1. At a total budget of at most 1,000 groups, seek at least 10% relative regret
   reduction versus the strongest matched conventional adaptation control in
   two eligible independent studies. Report absolute effects as well. Require
   a positive primary macro-effect interval and investigate study heterogeneity;
   do not hide a failing study behind the aggregate.
2. For the same deployed score, seek source achieved-@1 noninferiority: the lower
   bound of the paired difference interval above -0.001 efficiency fraction
   (-0.1 percentage points). Report Spearman loss separately; correlation cannot
   rescue failed decision retention. Confirm this margin's usefulness on
   development distributions before freezing it.
3. For a label-efficiency claim, seek a twofold reduction in measured candidate
   label cost to reach a prespecified utility target relative to P. Set that
   target from development curves and freeze it before confirmation. Do not
   select whichever point on a crossing curve favors our method.

Failure of these ambitious gates need not mean the model is useless. It means
the corresponding strong contribution has not been demonstrated. If only
ordinary fine-tuning remains competitive, report that result and reassess the
paper around a rigorous benchmark/resource contribution if sufficiently broad;
do not imply that this is an automatic Nature Methods fallback.

## 8. Computational biological application and paper structure

Use existing measured endogenous designs for disease-associated edits, if the
data gate establishes adequate coverage. Hold the intended allele fixed and
nominate only candidates with observed outcomes. Define variant strata and
success thresholds before looking at model advantages. Quantify achieved
efficiency and the number of measured candidates needed to find an effective
design. Simulated experimental savings must be labelled retrospective.

Unmeasured alternative designs are hypotheses, not validated improvements.
Editing efficiency is not therapeutic efficacy or safety. If uncertainty or
abstention is offered, report coverage, utility and cost together; dropping hard
cases must not masquerade as an overall accuracy gain. If eligible endogenous
selection sets are unavailable, provide limited prediction evidence and state
that the proposed application could not be established computationally.

Proposed six-figure narrative, contingent on results:

1. Decision benchmark, provenance and the correlation–selection mismatch.
2. Deployed-score adaptation method with decisive initialization/loss/replay
   controls, not only an architecture diagram.
3. Independent-study selection utility and domain-specific failures.
4. Target utility versus source retention, explicitly distinguishing both heads.
5. Total-label efficiency, compute cost and optional acquisition benefit.
6. Retrospective measured endogenous application and calibrated limitations.

Keep the original benchmark as supporting evidence. The main claim must come
from the new method and independent decision-level validation, if successful.
Release grouping/split code, exposure ledger, full configurations, seeds,
checkpoints or permitted access instructions, hashed predictions, all failed
arms, and a CPU-runnable evaluation entry point.

## 9. Execution order and concrete deliverables

| Stage | Deliverable | Gate before the next expense |
|---|---|---|
| V3-01: data and exposure audit | `data_inventory`, group-depth/overlap tables, immutable split manifests, feasibility/power memo | Credible independent validation route; otherwise narrow scope |
| V3-02: protocol and precision | Total-budget loader; epoch-zero, leakage, score-scale and deployed-head tests; frozen primary analysis specification | Reproduce E29–E32 reference values and validate all split invariants |
| V3-03: focused development | 14-fit screen, up to 16 replication fits, complete result ledger and measured compute | Promising target/source trade-off against conventional controls |
| V3-04: causal controls | Matched encoder/head policies, source-label replay versus distillation, anchoring ablations; one conditional branch only if justified | Advantage cannot be explained by scale, labels or unmatched compute |
| V3-05: budget curves and confirmation | Frozen method; nested total-budget curves; independent study results; fair comparator audit | Predeclared effects, uncertainty and coverage support the intended claim |
| V3-06: application and release | Measured retrospective application, six-figure draft, reproducible package and limitations | Claims match completed evidence, including negative results |

Immediate next action is **V3-01**, with V3-02 preparation while eligibility is
resolved. The highest-value first training question is whether an identity-
initialized, source-preserved deployed selector can retain utility that the
current pairwise head loses—not whether a larger model wins another Kim sweep.

## 10. Status at proposal

The v2 summaries and adaptation report have been corrected through E32; historical
narratives are archived. V3 data acquisition, model training, total-budget curves
and independent confirmation remain proposed. Neither the desired independent
benchmark depth nor the proposed method's advantage is established yet.
