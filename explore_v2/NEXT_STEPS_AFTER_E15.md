# Next research steps after E15

Prepared 8 September 2026 from `SUMMARY_REPORT.md`, E09–E15, and the evaluation, batching, ranking-loss and checkpoint-selection implementations. This is a proposed research programme, not a record of experiments already run. It supersedes the forward-looking recommendations in `NEXT_STEPS_PLAN.md`; that document remains the historical review that motivated several completed repairs.

**Scope: purely computational.** Use existing measured datasets, public sequencing data, retrospective evaluation and computational experiments. No new biological measurements, wet-lab programme or experimental collaboration is required by this plan.

## 1. Direction: make training and model selection match the actual decision

The next question should be:

> Can a decision-aligned training and model-selection procedure reliably improve which pegRNA is nominated for a fixed intended edit, across unseen loci and independent public studies?

Do not make another architecture the starting point. The strongest opportunity is the mismatch between the biological decision, the training comparisons, and the validation criterion. E13 repairs one part of that mismatch, but the training code still selects checkpoints by pooled Spearman—the metric that preferred the worse top-one selector in E11.

For the Nature Methods ambition, my assessment is that a grouping bug fix and one benchmark reversal are not yet a sufficiently broad methods contribution. The stronger target is a reusable decision-aligned procedure, a causal explanation of its benefit, and transport across datasets and model backbones. Generic pairwise/listwise ranking is not itself novel; novelty would need to come from the demonstrated methodological problem and its generalizable solution in prime-editing design. Journal-level significance cannot be guaranteed by any particular performance margin.

### What the current evidence does and does not establish

| Observation | Consequence for the next phase |
|---|---|
| Published PE member: pooled Spearman 0.7091 versus OptiPrime 0.6931, but achieved efficiency @1 lower by 0.00062 | Optimize and select for within-edit decisions; retain pooled prediction metrics as secondary outcomes. The difference is 0.062 percentage points, not 0.062 efficiency units. |
| Canonical grouping improves the matched single-model control by 0.00178 on the panel | Replicate this repair first; it is the strongest intervention found so far. |
| Repaired single model still trails OptiPrime by 0.00117 | The repair is useful but does not establish superiority. “60% recovered” refers to the fresh single-control gap, not the published ensemble gap or a universal ceiling. |
| Both training arms use one seed; the published member averages five checkpoints | Match seeds, training data and ensemble size before attributing differences to the method. |
| Higher-depth and higher-observed-worth groups show larger deficits | Diagnose these strata, but do not use observed worth to select deployment cases: it requires outcomes unavailable at nomination time. |
| Fold 0 and the Kim panel have already been examined | Keep them as disclosed diagnostic/reference surfaces. A new split of the spent panel is not fresh external confirmation. |

Keep the corrected canonicalizer and E09 representation repair. Do not repeat E01–E15 wholesale. Park reference-panel adaptation: its current evidence does not justify another investment, although the measured interaction regret is not a universal limit on possible methods.

## 2. E16: a short endpoint and exposure check before more training

**Time box: one working day.** This is targeted cleanup, not another open-ended audit.

### Fix candidate accounting and statistical invariants

The current E11 group table has 13 groups with more rows than unique designs. Two groups with only two distinct designs receive a top-three result because `group_endpoints` checks row count while reported depth uses unique `design_key` count.

1. Define one candidate as one distinct pegRNA design in one decision context. Resolve repeated rows using source-defined replicate aggregation; report a sensitivity analysis if their provenance is ambiguous. Never spend multiple nomination slots on the same molecule.
2. Require at least k unique candidates for achieved efficiency @k. Use one fixed eligible population for budget curves; show its own oracle and random baseline. Do not compare an all-group oracle with @3 computed on a deeper subset.
3. Make tie handling consistent. With full credit for selecting any measured maximum, random top-one hit probability is the number of maximizers divided by candidate count, not always `1/n` as currently coded.
4. Test identical predictions/differences, all-zero outcomes, tied scores, duplicate candidates, row permutation and k exceeding depth. The current bootstrap tail calculation assigns its minimum p-value to an all-zero difference distribution; identical arms must instead have zero difference and no evidence against equality. Prefer paired, locus-clustered intervals; if testing is retained, implement an appropriate paired cluster-level test and verify its null behavior.
5. Recompute affected descriptive tables and disclose changes. These narrow issues do not by themselves overturn the top-one reversal. Passing `verify_report.py` establishes agreement with saved artifacts, not correctness of the endpoint algorithms.

### Measure the ranking comparisons actually used

Separate `batch_group_key` from `rank_group_key`. At present, the same key controls both `GroupedBatchSampler` and pair eligibility; E13 therefore changes both mechanisms.

Log distinct eligible design pairs, sampled pair instances, unique pairs actually visited, target-gap distribution, groups with a ranking update, and per-group exposure. Replay the actual sampler, minimum difference of 0.02, and four-pair cap. Measure occasional loss-gradient magnitudes if interpreting the objective balance.

Two wording corrections follow regardless of new training: 14,277/368,307 is a ratio of possible within-group design pairs, not measured training exposure; `lambda_rank=0.25` is a coefficient, not evidence that ranking contributes exactly a quarter of the loss or gradients.

**Deliverable:** a tested endpoint module, corrected tables where needed, and an exposure report that specifies what changed in E13.

## 3. E17: replicate the repair and separate its mechanisms

### Evaluation layout

Build connected-locus partitions before training, joining overlapping representations across edits and studies using validated identities. Keep all candidates and replicates for a decision together. Do not rely on the official row folds for a locus-generalization claim.

Use one development partition with separate training, inner validation and diagnostic test loci; reserve two additional blocked partitions for robustness of the selected comparisons. The existing corpus is already explored, so call these internal blocked evaluations, not independent confirmation. Publish component sizes, exclusions and candidate-depth distributions. All learned transformations and checkpoint selection must respect the same partition.

### Matched intervention matrix

Use the existing ordinal-S4D backbone, unchanged inputs and optimizer, paired initialization seeds, and fixed training budgets.

| Arm | Batch grouping | Ranking grouping/exposure | Question |
|---|---|---|---|
| A | Original | Original | Matched historical control |
| B | Canonical | Original | Does changing batch composition explain the gain? |
| C | Canonical | Canonical, exposure-budget matched to B | Does allocating comparable ranking effort to true alternatives help? |
| D | Canonical | Canonical, normal exposure | What is the complete repair effect? |
| E | Canonical | Ranking loss disabled | Does ranking add value beyond the non-ranking objective with repaired batches? |

Replay identical row-batch schedules for B–E. Define C's exposure matching before fitting: match total sampled-pair budget and, where feasible, group-depth and target-gap strata. Report unmatched strata rather than claiming exact equality. Keep loss normalization fixed. C versus B tests allocation under a matched budget, not a mathematically perfect isolation of every pair-distribution property.

Run A and D with three paired seeds first. If their direction is unstable on clean development loci, diagnose that before expanding. Otherwise add B, C and E with the same three seeds: 15 fits total on the first partition. Carry A and D, plus any genuinely better selected intervention, to the additional partitions. Report every seed; do not choose an ensemble's members by test performance.

### Checkpoint-selection experiment from the same trajectories

Save validation predictions and recoverable candidate checkpoints during a fixed epoch schedule. Compare selection by:

- Pooled validation Spearman, the current rule.
- Mean achieved efficiency @1 over eligible canonical validation decisions, the proposed rule.

Use identical trajectories and epoch budgets so a selector comparison is not confounded by different early-stopping durations. Include all eligible decisions, with ties handled consistently; report informative-group results secondarily. For multiple source studies, predeclare whether the primary mean weights decisions or studies equally, and report the other weighting as sensitivity.

Compare single models and matched three-seed ensembles. The published five-checkpoint PE member remains a reference, not the causal control. Released OptiPrime can be an external comparator only on data whose non-exposure is established; its predictions on an internally repartitioned training corpus are not a clean held-out baseline.

**Gate:** proceed to more complex objectives only after estimating how much benefit comes from grouping, batching, ranking exposure and checkpoint selection. A bootstrap over test loci does not replace training-seed replication.

## 4. E18: develop one decision-aligned objective, not a broad architecture search

Start with the best validated E17 recipe. Test the hypothesis that equal-weight ordering comparisons and the hard 0.02 target-gap cutoff are poorly matched to the cost of selecting the wrong candidate. E16 must establish which useful comparisons the cutoff removes; reducing it indiscriminately may mostly add measurement noise.

Prioritize a **group-normalized, utility-weighted pairwise loss**:

- Compare only distinct candidates for the same intended allele and experimental context.
- Weight ordering errors by their measured efficiency consequence, with a prespecified cap to limit outliers.
- Normalize within each decision group, then average groups, so candidate-rich loci do not dominate merely by producing quadratically more pairs.
- Retain the ordinal/prediction objective as an anchor; do not discard singleton training data.
- Use existing replicates to assess whether reliability weighting is supported. Do not transfer the mostly-Kim noise estimate to all studies as an established noise model.

The primary ablation is repaired uniform ranking versus utility-weighted ranking with matched batches, pair exposure, checkpoint rule and compute. Add a standard listwise/top-one utility surrogate as a comparator if the pairwise improvement is reproducible. Differentiable ranking already has substantial methodological precedent; treat it as a baseline/tool, not a novelty claim. See [Blondel et al., ICML 2020](https://proceedings.mlr.press/v119/blondel20a.html).

Before any second modeling branch, examine development-only residuals: how often is the true winner already among the model's first three candidates, and what distinguishes rank-one from rank-two mistakes? Stratify by PBS/RTT geometry, edit class, candidate depth and measured efficiency scale. Near-equal average @3 results do not prove that the two models nominate the same candidates.

Limit the initial search to a small, recorded set of loss weights and gap treatments. If this branch does not improve blocked validation beyond the corrected uniform-ranking baseline, stop it. A candidate-set reranker or new backbone should require a specific residual failure that this simpler intervention cannot address.

**Generalization test:** apply the winning recipe to one existing alternative backbone, such as the repository's Transformer configuration. Match the original-versus-repaired procedure within each backbone. This tests whether the contribution is a training/evaluation principle rather than an S4D-specific accident.

## 5. E19: qualify independent public data immediately

Start this alongside E16–E17; it is the main dependency for a strong paper, not a last-stage decoration.

### Public-data qualification

First inspect the original measurements from OPED, separating them from the public datasets OPED reused. The primary paper provides source code and sequencing accession PRJNA882795; this establishes availability, not adequate candidate depth. See [OPED data and code availability](https://www.nature.com/articles/s42256-023-00739-w).

For every candidate source, produce a compact eligibility manifest:

- Originating laboratory, assay, cell/editor configuration, and reporter versus endogenous readout.
- Full candidate identities and intended alleles; context-compatible efficiency definitions and replicate information.
- Counts of distinct decisions at depths 2, 3, 5 and 8 after overlap removal.
- Exposure to each comparator's training data, including data reused under different study names.
- Availability of alternatives measured under the same conditions, rather than only a published winner or incomparable candidates.

Spend at most two working days on initial source qualification and decide within the first week whether a usable independent panel exists. Reprocess public reads only when necessary and feasible; do not start an unbounded sequencing reconstruction project. An independent deep reporter panel and a smaller endogenous panel would provide complementary tests. Endogenous data are desirable, not a reason to reject an otherwise valid independent deep panel.

If only binary independent groups are available, evaluate binary transfer honestly, but do not present it as validation at large candidate depth. If no adequate independent source is available, complete internal blocked evaluation and a benchmark/resource paper; do not relabel a subset of Kim as an independent study.

### Revisit PRIDICT2 compatibility, narrowly

The current exclusion based on PBS enumeration at lengths 7–15 is not yet a demonstration that the scoring model cannot accept other explicit designs. The local `external/pridict2/pridict2_pegRNA_design.py` exposes the PBS enumeration range as a parameter. The [official PRIDICT2 repository](https://github.com/uzh-dqbm-cmi/PRIDICT2) is the implementation reference.

Time-box a check of explicit-candidate scoring through the official feature pipeline, reproducing supported-range predictions first. Establish encoding and trained-domain limits before extending the range; changing a generator parameter alone does not validate extrapolation. If full-panel scoring is unsupported, a common-supported-candidate subset is still a valid secondary comparison when every method gets exactly that same subset and coverage is reported. It must not silently replace the full-panel estimand. Continue excluding released DeepPrime from held-out evaluation on its own training library.

### Freeze the final comparison

After metadata/overlap qualification, separate final outcomes from model-development scripts. Freeze input reconstruction, candidate eligibility, models, seeds, checkpoint selection, endpoints and analysis code before generating final predictions. This is an analysis freeze on existing public outcomes, not prospective biological validation or a claim that nobody has previously seen public labels.

Use group-mean achieved efficiency @1 as the primary endpoint and regret as its paired complement on the same population. Include @3 and fixed-population budget curves, tie-aware hit rate, pooled correlation and geometry/depth strata as secondary endpoints. Resample connected loci; report training-seed variation separately. With few studies, show study-specific effects rather than implying that locus bootstraps establish broad study-level transport.

Report absolute efficiency differences, percentage-point differences and relative regret reduction with uncertainty. Define a practically worthwhile gain using development data before the final evaluation; statistical significance alone is not the success criterion. Claim superiority over OptiPrime only where the final eligible comparison supports it.

## 6. Execution order, budget and stopping rules

| Window | Priority | Output / decision |
|---|---|---|
| Days 1–2 | E16; start E19 qualification | Endpoint tests, exposure measurement, public-source shortlist |
| Days 3–7 | E17 A/D replication; finish source qualification | Decide whether the repair generalizes across seeds and clean loci |
| Week 2 | Complete E17 mechanism arms and checkpoint comparison | Identify which components actually reduce decision regret |
| Weeks 3–4 | Bounded E18 search; selected blocked-partition and second-backbone checks | Freeze one method, or stop adding model complexity |
| Weeks 4–6 | Frozen external evaluation, robustness, manuscript and release | Methods-level evidence package or an explicitly narrower benchmark contribution |

These are planning estimates, not a commitment to launch jobs. The completed E13 single-model runs took approximately two GPU-hours each on their hardware; 15 matched fits therefore suggest roughly 30 GPU-hours before extra partitions, objective candidates and alternative backbones. Reserve a staged budget, not an unrestricted sweep; repeat the timing estimate after adopting fixed epochs and new partitions. Reuse each trajectory for the two checkpoint selectors, while accounting for checkpoint storage.

Stop or narrow the claim if any of the following occurs:

1. Canonical regrouping does not reproducibly help under matched seeds and locus-disjoint evaluation.
2. Utility-aware training does not outperform the simpler corrected recipe. Publish the simpler method rather than adding components to rescue the story.
3. Independent public measurements cannot support the intended decision-depth or transfer claim.
4. Gains disappear against matched ensembles or on the second backbone. Report the dependence rather than claiming generality.

Do not resume context adaptation, increase model scale, collect new biological data, or search indefinitely for a favorable external subset to keep the project moving.

## 7. Intended paper and release

The ambitious paper should have four linked results:

1. **The evaluation problem:** pooled prediction performance can disagree with the quality of fixed-edit candidate choices; document when this occurs, not just one aggregate reversal.
2. **The mechanism:** quantify how decision identity, batching, pair exposure and checkpoint selection affect those choices through matched interventions.
3. **The method:** a minimal decision-aligned recipe that improves retrospective nomination across clean loci, independent public data and more than one backbone.
4. **The resource:** versioned allele/candidate identities, invariance tests, exposure and split manifests, adapters, frozen predictions, endpoint tests and reproducible figures.

Preserve the negative OptiPrime comparison and unsuccessful context branches in the scientific record. Update the manuscript's deployment claim now, but postpone an ambitious new headline until E17–E19 justify it. If the general method does not materialize, a rigorous benchmark and repair remain valuable; they should not be inflated into universal deployment superiority.

**Immediate next action:** implement E16, prepare the three-seed A/D comparison with both checkpoint selectors, and qualify one independent public dataset. Those three bounded tasks resolve the most important uncertainties before another substantial modeling investment.
