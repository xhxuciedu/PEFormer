# PE-RankFormer: an independent research plan for a Nature Methods submission

Scope update: the user has specified a purely computational project. The active roadmap is [explore_v2/NEXT_STEPS_PLAN.md](explore_v2/NEXT_STEPS_PLAN.md). Its computational validation program supersedes all recommendations below for new wet-lab work or biological measurements. The original proposal is retained here as research history.

Prepared 8 September 2026. This is a research proposal, not a report of completed new model experiments. `explore_v1.md` is treated as a collection of suggestions. The current manuscript, later experimental logs, evaluation code, and frozen data take precedence over its premises.

## Recommendation

Build a method for **choosing effective pegRNAs in a new experimental context with a small, explicitly budgeted amount of context-specific measurement**. Test whether a compact panel of reference edits can characterize the context well enough to transfer design preferences to unrelated edits and loci. Compare that functional measurement with molecular annotations and with ordinary fine-tuning under the same measurement budget.

The paper's strongest plausible claim is: **a reusable assay and model can adapt pegRNA selection to a new cell/editor setting, reducing the screening needed to obtain effective edits**. The current sequence model supplies a strong starting representation. The new contribution must be the transferable experimental capability and its validation.

I would not center the next paper on another architecture, a slightly higher pooled Spearman, or simply adding RNA-seq embeddings. Nature Methods describes its Articles as requiring strong validation of performance, reproducibility, general applicability, and potential biological discovery. That is a useful standard for this plan, not a promise of acceptance. [Journal criteria](https://www.nature.com/nmeth/content)

The program should proceed through explicit evidence gates. If reference measurements only calibrate overall efficiency and cannot improve design selection, the central hypothesis has failed. If public molecular annotations do not beat cell identifiers or random context vectors in held-out contexts, they should not be promoted into the main model.

## 1. What the present evidence actually establishes

The manuscript has a credible matched-benchmark result and an unusually useful record of negative experiments. It does not yet establish a new capability in experimental design.

| Finding | My interpretation | Consequence for the plan |
|---|---|---|
| Pooled Spearman 0.9079 versus OptiPrime 0.8690 | Strong on this benchmark; mixes variation between loci, edits, and conditions | Preserve as a historical benchmark, not the main deployment endpoint |
| Reported within-spacer Spearman 0.6356 versus 0.5472 | A real advantage, but the grouping still mixes experimental decisions | Rebuild evaluation around a fixed desired edit in a fixed context |
| Reported top-pick efficiency improves by 0.015 | About 1.5 percentage points, under the current mixed grouping | Do not equate the larger best-design retrieval gain with a large experimental benefit |
| S4D contributes about 0.0192; ordinal supervision about 0.0092 | Architecture matters here, but this does not establish a general biological principle | Keep the best single ordinal-S4D model as the main backbone |
| Weight matching costs PE-RankFormer 0.0136 pooled Spearman on average across three development folds | Training weights materially affect the comparison | Use a crossed architecture-by-weight experiment before attributing the margin |
| 649 metadata-identical replicate groups support an empirical repeatability estimate | Evidence worth using, with restricted support and assumptions | Estimate reliability for the decision being evaluated, not a universal ceiling |
| Context predictions are more correlated than observed outcomes | Compatible with missed interactions, measurement noise, shrinkage, and assay confounding | Test individual reproducible effects; do not optimize agreement with a correlation summary |
| Many generic changes failed | Strong reason to stop indiscriminate search | Spend effort on identifiability, new information, transfer, and prospective utility |

Local evidence: [manuscript](reports/paper/pe_rankformer_paper.tex), [revision summary](revision/SUMMARY.md), [round-6 interaction experiments](reports/round6_research_log.md), [round-9 results](revision/task_2_round9_wave_results.md).

### A new evaluation issue found during this review

Both `revision/task_1_1_per_target.py` and `revision/task_1_2_utility.py` group candidates by `spacer`. They do not hold experimental condition or intended edit fixed.

I checked the frozen data directly:

- **742 of 750 held-out spacers span more than one** `source_study × cell_type × pe_type` **condition**. These groups contain 20,500 of the 20,509 held-out rows.
- Reproducing the existing grouping gives the reported 0.6356 versus 0.5472 mean Spearman over 670 eligible spacer groups.
- Grouping instead by spacer and that three-field condition gives 0.5942 versus 0.4597 over 997 eligible groups. This is descriptive, with no new confidence interval; it is a different population and still does not fix the intended edit or every context field.
- Of 6,358 spacer/condition groups, 1,180 contain more than one exact WT/edited sequence pair. Exact sequence pairs can differ because of flanking-window conventions, so this is a warning about edit identity, not a count of distinct biological edits.
- An intentionally conservative exact WT/edited-pair plus all seven categorical fields produces 20,441 groups, only 17 with at least five rows. This is **not** the correct final key: it can oversplit equivalent edits and treats some design choices as fixed context. It shows why simply adding columns to the current grouping is inadequate.

The new benchmark must distinguish three tasks:

1. **Choose a pegRNA:** fixed locus, desired final allele, cell state, editor, delivery, and assay endpoint; vary allowed pegRNA designs.
2. **Choose a design and editor configuration:** fixed locus, desired allele, and biological system; allow only predeclared editor choices.
3. **Compare different intended edits:** a sequence-to-outcome prediction task, not automatically a choice a user is free to make.

Canonicalize edits to genomic coordinates and reference/alternate alleles where coordinates exist. Otherwise use validated strand-normalized local alignments and distinguish indel-equivalent representations. Retain original coordinates and assay provenance. Group across spacers when they implement the same desired allele. Treat optional silent changes as a separate, explicitly allowed design task; their products are not identical alleles.

Aggregate true replicate measurements before ranking candidates. A replicate is not another pegRNA choice. Where an edit has too few alternative designs, retain it for prediction evaluation but do not manufacture a design-selection benchmark.

### Claims that need correction or qualification

**FiLM can reorder designs.** For a linear readout, the score difference is

`s(d1,c) − s(d2,c) = wᵀ diag(1 + gamma(c)) [h(d1) − h(d2)]`.

Changing context can change its sign. The failed layerwise-FiLM experiment shows that this implementation did not learn useful additional reordering. It does not demonstrate a structural inability to reorder. The round-9 log recognizes this, but stronger language persists elsewhere.

**Cell identifiers can encode average cell-specific biology.** With enough examples, an embedding can learn that a particular cell behaves differently for particular designs. Molecular features are most compelling for sharing information across sparsely measured or unseen contexts, or for describing variation within a named context. They are not automatically additional information for a well-sampled fixed cell line.

**The effective molecular-context sample size is small.** The current corpus has nine cell labels and 43 combinations of the seven model context fields, not 300,000 independently observed cell states. NIH3T3 is a mouse line. Several human lines occur in only one source study. A high-dimensional expression encoder could memorize study or cell identity while appearing to benefit from hundreds of thousands of rows.

**The repeatability estimate is not a mathematical Spearman ceiling.** Independent-error attenuation formulas are not automatically valid for heteroscedastic, tied ranks. Technical repeatability also excludes shared systematic biases and some biological variation. Verify experimental replicate identities beyond identical recorded metadata, assess representativeness, and distinguish technical from biological replicates. The reported gap of 0.1040, with interval [0.0529, 0.1204], is a useful model-dependent estimate for the Kim held-out surface; it is not a budget of guaranteed recoverable accuracy, nor a within-edit selection ceiling.

**A zero with a positive replicate does not identify censoring.** Finite sampling, rounding, thresholds, and actual biological variability are alternative explanations. A censoring likelihood needs assay evidence for a detection/reporting rule. The harmful threshold-masking experiments do not prove label noise is irrelevant, but they do rule against repeating that particular treatment.

**An adaptively reused test set remains exposed even for a paired comparison.** Pairing improves precision; it does not eliminate selection bias when only the new model is repeatedly developed. The old held-out data should become a historical reference. Fresh external data or prospective measurements must carry the confirmatory claim.

**Neither broader capacity nor context models have been universally ruled out.** Their tested implementations failed at the observed resolution. Likewise, two selective-SSM implementations both underperforming S4D does not isolate the entire deficit to training recipe without a further controlled comparison. These results justify prioritization, not impossibility statements.

## 2. Diagnose the remaining error before selecting an intervention

Use a conceptual decomposition on an appropriate latent scale:

`outcome = locus/edit effect + design effect + context effect + design×context + locus×context + assay/batch effect + measurement variation`.

The current data do not independently identify all terms. Many design/context combinations are missing, and study, cell, delivery, and library are partly confounded. Fit deliberately small models to assess what the support can identify before training a richer neural model.

| Possible bottleneck | Evidence today | Discriminating experiment |
|---|---|---|
| Missing interaction signal | Excess cross-context similarity; feature-based rank-shift result | Predict replicated pair reversals on common candidate sets and held-out loci |
| Observation noise | Replicate discrepancies; many low measurements | Read-depth-aware held-out-replicate prediction and reliability by context/outcome range |
| Missing assay state | Sparse categorical descriptions; cell/study confounding | Hold out batches or studies; add a measured reference panel from the new batch |
| Missing molecular state | Biologically plausible, not established by existing failures | Molecular profiles versus identifiers, random profiles, and measured reference panels on held-out cells |
| Geometry representation | S4D improvement is suggestive | Small relative-coordinate ablation with capacity/optimization controls |
| Optimization | Generic sweeps are mostly negative | Check training versus validation fit and recovery of known synthetic interactions on the observed data support |
| Objective mismatch | Existing conditional quantiles and shift losses failed | Only test fixed-edit decision losses after validating candidate groups and repeatable differences |
| Domain shift | No confirmed independent evaluation | Study, cell, assay, and jointly unseen locus/context tests reported separately |

The diagnostic should have four components:

1. **Matched support.** Compare exactly the same complete designs in both contexts. Keep motif, linker, scaffold, and epegRNA identity in the design key. Do not interpret ranks computed against different candidate populations as biological reordering.
2. **Replicated effects.** For two designs implementing the same desired edit, test whether the sign of `mu(d1,c) − mu(d2,c)` changes between contexts, and whether each sign is supported by independent replicates. Report coverage and effect sizes, not just an accuracy on a conveniently filtered subset.
3. **Predictive identifiability.** Compare an additive model, small feature-by-context interactions, a frozen-embedding ridge interaction model, and a partially pooled context model. Hold out locus families, not only exact designs. The historical rho≈0.275 probe needs this stricter replication and a structured shuffle control.
4. **Noise-aware counterexample.** Simulate observations from an additive or rank-preserving latent model with the measured noise structure. Quantify how much of the observed cross-context correlation gap appears without genuine rank reversal. Check sensitivity to using different reliability estimates for each condition.

For a quartet measured in two contexts, one useful diagnostic is

`D = [mu(d1,c1) − mu(d2,c1)] − [mu(d1,c2) − mu(d2,c2)]`.

A nonzero `D` is an interaction on this scale; it is not necessarily a rank reversal. Report both `D` and the two order signs. Analyze uncertainty at the locus/experiment level, since thousands of derived pairs do not constitute thousands of independent experiments.

**Gate A:** proceed with an interaction-centered paper only if reproducible, practically meaningful interaction effects are predicted on held-out loci better than additive and structured-shuffle controls. Merely lowering the model's cross-context correlation is not success; adding noise would also do that.

Distinguish a negative result from an unidentifiable experiment. If canonicalization reveals too few alternative designs or repeated contexts to test this with current data, Gate A is **inconclusive**. In that case, a small crossed feasibility panel is the next information-gathering step; it should precede elaborate adaptation models. The historical corpus can still pretrain the sequence model without being adequate to validate the new decision claim.

## 3. My five priorities

Ranked by contribution to the proposed paper. Execution begins with priority 3 because the other results need its evaluation foundation. These are connected workstreams, not five unrelated model searches. Numerical targets below are proposed planning thresholds, not performance forecasts.

| Priority and direction | Core hypothesis / why the current model misses it | Implementation and critical experiment | Success / negative-result meaning | Effort and resources | Scientific and publication value |
|---|---|---|---|---|---|
| **1. Adapt using a small reference-edit panel** | Actual experimental state is partly revealed by how a common panel edits; identifiers do not observe a new batch | Frozen sequence encoder plus low-dimensional context inference from measured references; support/query-disjoint transfer at equal measurement budgets | Better selections on new loci; failure suggests the panel mostly measures global efficiency or lacks transferable information | 2–4 weeks retrospective; new panel measurements needed for confirmation; moderate compute | Highest methodological upside; medium, currently unquantified chance of useful transfer |
| **2. Crossed biological validation** | Some design preferences respond reproducibly to repair state | Same candidate designs across controlled contexts, followed by blind selection in unseen contexts | Predict intervention-dependent order changes and better selections; failure narrows the biology and prevents a false mechanistic claim | Pilot in 4–8 weeks if assays exist; definitive study likely several months; substantial wet-lab work | Highest biological and validation value; assay feasibility determines risk |
| **3. Fixed-edit selection and transfer benchmark** | Current aggregate scores overstate what is known about deployment | Canonical decision groups, source provenance, external panels, independent test reservation | Establishes where an advantage actually exists; failure may change the paper's central claim | 1–2 weeks audit; several more for external adapters/baselines; low compute | Essential; high probability of an informative answer, no guaranteed positive result |
| **4. Molecular priors for new contexts** | A compact mechanistic state description helps transfer beyond identity | Small pathway/genotype descriptors, optionally chromatin at actual assayed loci; leave-cell/study-out comparison with matched random features | Improves zero-shot or low-budget transfer beyond identifiers and references; failure means public profiles are insufficient here | 2–4 weeks metadata work; same-culture assays for strongest test | High only with transfer and biological validation; lower expected success for public expression alone |
| **5. Replicate-aware outcomes and selection uncertainty** | Measurement uncertainty matters for learning/adaptation and choosing near-tied designs | Count-based likelihood where supported, shared-replicate latent outcomes, calibrated decision probabilities | Better held-out-repeat prediction and budgeted selection; failure means noise modeling adds little practical value | 1–3 weeks after count/provenance recovery; uncertain data availability | Supporting method; modest standalone novelty and uncertain ranking benefit |

### Priority 1: infer functional experimental state from reference edits

**Hypothesis.** A small panel spanning different design sensitivities identifies a low-dimensional experimental state that transfers to designs at other loci. This state may combine repair, expression/delivery, and culture effects. Do not call it a mechanistic repair measurement until interventions establish that interpretation.

Start with

`eta(d,c) = f_theta(d) + a_c + u_theta(d)^T z_c`.

`f_theta` is the existing ordinal-S4D representation/readout; `a_c` captures a mean shift; `z_c` is a small context vector. Begin with rank 2 or 4, not a large hypernetwork. Estimate `a_c,z_c` from support measurements with shrinkage. A later molecular prior can set `z_c ~ N(g(m_c), Sigma)`.

The first implementation should be a regularized linear adaptation on frozen features, with the prediction likelihood fitted to the label type. It is both a baseline and a potential final method. Add an episodically trained support-set encoder only if the simple method leaves a reproducible advantage to capture. A new neural architecture is not required for the proposed capability.

**Inputs and loss.** Complete design sequence, explicit editor components, support designs with observed outcomes and measurement quality, optional molecular annotations. Retain the existing ordinal loss initially, learning an ordinal-logit correction from support data. Use a compatible count likelihood when counts are recovered. Optimize query prediction under a support-only context estimate; a context-specific calibrator must also use support labels only.

**Training protocol.** Outer split by context; within the test context partition support and query loci before observing labels. Exclude query loci from training in every context for the strongest test. Train adaptation hyperparameters on source contexts only. Evaluate budgets of 0, 12, 24, 48, and 96 measured designs, also counting replicates, assay batches, and required loci. Pseudo-episodes do not increase the number of independent biological contexts.

**Critical controls.** No adaptation; intercept-only calibration; nearest known context; support-fitted ridge regression; head-only and full-model fine-tuning; available DeepPrime-FT adaptation; random versus learned reference selection; support outcomes shuffled across designs while preserving their distribution; reference design identities without measurements. All receive the same support labels and budgets. Test whether a fixed reusable panel works across contexts, not only an optimally reselected panel per test set.

**Evaluation.** Success@k, selected efficiency, and regret for fixed-edit query groups; learning curves versus measurement budget; context-by-context results. Count reference-panel overhead when comparing with directly screening more query pegRNAs. It may be worthwhile for a 100-edit project and wasteful for one edit; identify the break-even workload.

**Planning target.** Seek at least 20% relative reduction in decision regret versus the strongest equally adapted baseline, or a clear reduction in measurements needed to reach a prespecified success level, reproduced in two independently held-out contexts. A +0.02 to +0.05 improvement in within-edit rank correlation is a secondary desirable effect, not a forecast or substitute for utility.

**If successful:** a portable calibration assay transfers design-dependent information. **If unsuccessful:** reference measurements may only improve absolute calibration, or too many context dimensions may be required. Do not rename calibration success as context-specific design transfer.

**What is new relative to failed low-rank conditioning?** The low-rank algebra is not the novelty. The failed model inferred context from the same categorical labels. This experiment obtains additional measurements from the actual new condition and tests transfer at a fixed acquisition budget.

### Priority 2: crossed experiments that can establish the biology

**Hypothesis.** Controlled repair-state changes alter particular design preferences, and the learned context model predicts those changes at unmeasured loci. Showing only that MMR modulation changes overall efficiency would reproduce established biology. OptiPrime already studies MMR-sensitive design effects and prospective applications; MinsePIE also models repair-dependent insertion behavior. [OptiPrime](https://www.nature.com/articles/s41587-026-03261-7), [MinsePIE](https://www.nature.com/articles/s41587-023-01678-y)

**Pilot design.** An illustrative feasible panel is 12 intended edits × 8 alternative pegRNAs × 4 contexts × 3 biological replicates = 1,152 design-context-replicate measurements. Use two cellular backgrounds and a matched repair-state intervention in each. Keep editor backbone, delivery, time point, and experimental handling matched when testing repair. Randomize plate position, balance batches, and measure delivery/editor abundance and relevant repair-state controls. Select locus/edit classes for interpretable hypotheses before measuring outcomes.

Use the exact same candidate sets across contexts, varying PBS/RTT geometry within an intended edit. Separate an optional silent-edit experiment, since it changes the final allele. A mechanistic lead should be confirmed by an orthogonal intervention or rescue where feasible; correlation with expression or a latent factor is insufficient. Do not initially add a large panel of repair-gene perturbations.

**Model and loss.** Compare the priority-1 model with additive and molecular-prior variants. In a strictly training-only subset, an optional loss can supervise paired order probabilities or quartet differences, weighted by replicate support. Retain the global outcome objective. Never force a minimum number of reversals. A quartets loss is worth testing only after Gate A, and only on fixed-edit, matched-candidate observations.

This differs from the failed generic pairwise and same-design rank-shift losses because it uses replicated comparisons of interchangeable designs on common support, potentially new measured context information, and a decision endpoint. If those differences cannot be supplied, do not rerun it.

**Evaluation and target.** Out-of-locus intervention-effect prediction; correctly predicted reproducible order reversals; effect sizes; and selection gain. An illustrative mechanism gate is reversal-direction accuracy at least 0.10 above the strongest comparison, with uncertainty excluding no improvement and an explicit comparison against the majority/no-reversal rule. Because reversals may be rare, include precision–recall, sensitivity, coverage, and pairwise probability calibration.

**If successful:** establish a reproducible design-by-state relationship that helps choose designs. **If unsuccessful:** the global effects may dominate, or the selected intervention/panel lacks informative interactions. This falsifies the chosen mechanistic story, not every possible cellular-context effect.

### Priority 3: benchmark the actual decision and its transfer

**Hypothesis.** PE-RankFormer retains a meaningful advantage for a fixed desired edit in a new context or assay, after provenance, candidate availability, and comparison fairness are controlled.

**Modification and loss.** No model modification is necessary initially. Freeze existing predictions and rebuild metadata and evaluation. Retrain baselines only on newly declared training partitions, using their published objectives and validated official preprocessing. The gradient-boosted feature model is useful but does not substitute for DeepPrime, PRIDICT2, or OPED.

Use four evaluation surfaces:

| Surface | Training/test separation | Claim it can support |
|---|---|---|
| New loci, familiar conditions | All related targets and replicate groups held together | Interpolation to new targets |
| New condition, familiar locus families | Entire condition withheld | Context transfer with possible target familiarity |
| New loci and new condition | Both withheld | Stronger deployment transfer |
| New study/assay or prospective experiment | Provenance-disjoint acquisition | Transport across experimental workflows |

Report zero-shot prediction separately from adaptation with measured labels. Leave-cell-out and leave-study-out are different tests; when a cell occurs in one study, the effects cannot be disentangled. Leave-editor-out needs compositional editor inputs or explicit unknown handling, not a magically trained embedding for an unseen category. Leave-edit-type-out is a useful optional stress test, not a mandatory primary endpoint. A publication date alone does not establish dataset independence.

**Primary utility estimand.** For a candidate set `D(g)` implementing edit `g`, evaluate the best measured outcome among the `k` designs selected before outcomes are revealed. Average over edits, then show each context. Report success@1 and success@3 at a threshold selected for the application before testing; selected efficiency; regret relative to the best tested candidate; and assay cost. Use absolute regret primarily because normalized regret becomes unstable when every design performs poorly.

A shortlist can estimate utility for that shortlist. It cannot establish regret relative to all possible pegRNAs. To claim a reduction in screening effort, compare budgets including the reference measurements and replicate requirements. Do not compute probability of at least one success as a product of marginal failure probabilities unless dependence has been addressed.

**Statistics.** Resample loci/edits as dependence units, with shared contexts kept paired; assess between-context, laboratory, donor, and batch variability separately. Biological replication does not turn two cell types into six independent cell types. Handle ties with midranks and tied-optimum credit; report near-optimal selection within a prespecified practical tolerance. Split replicates used to select an observed oracle from those used to assess it where possible, reducing winner's-curse bias.

**Planning target.** Establish confidence intervals for a meaningful fixed-edit utility advantage on an untouched external panel. There is no defensible expected numerical gain before decision groups are reconstructed. The old 0.6356 and 0.015 are not baselines for the revised estimand.

**If successful:** the existing performance advantage translates into a useful capability. **If unsuccessful:** revise the headline even if pooled Spearman remains excellent. This workstream must be allowed to change the research direction.

### Priority 4: molecular annotations as priors, not a substitute for measurement

**Hypothesis.** A compact biological description improves transfer to unseen contexts or reduces the number of reference edits needed.

Start with explicit editor components and a small, prespecified set of repair-related descriptors. MLH1/MSH2/MSH6/PMS2 expression, genotype, protein information, pathway scores, and measured intervention status are candidates. They are distinct measurements: RNA abundance does not equal repair competence, and MLH1dn does not imply zero MLH1 RNA. Record assay source, batch, species, uncertainty, and missingness. The existing flags that are deterministic functions of context cannot add information on their own, though a compositional encoding may improve transfer.

**Public resources.** DepMap provides processed molecular datasets and model metadata; ENCODE explicitly distinguishes biosamples and their experimental provenance. Audit availability for each actual line and assay instead of assuming coverage for HEK293T, HeLa, every perturbation, or mouse NIH3T3. Roadmap tissue profiles should not silently stand in for cultured cancer lines. [DepMap data](https://depmap.org/portal/data_page/?tab=customDownloads), [ENCODE metadata](https://www.encodeproject.org/help/data-organization/)

**Chromatin has a separate role.** Endogenous chromatin belongs to the genomic locus in the assayed cells. A target sequence copied into a randomly integrated reporter does not inherit its native locus's chromatin. Annotate actual mapped reporter integration sites or genuine endogenous sites only. Chromatin can improve locus accessibility prediction without improving the ordering of pegRNAs for the same locus; test those contributions separately. PRIDICT2/ePRIDICT and Li et al. already establish relevant sequence/chromatin modeling and context effects. [PRIDICT2/ePRIDICT](https://www.nature.com/articles/s41587-024-02268-2), [Li et al.](https://www.sciencedirect.com/science/article/pii/S0092867424003118)

**Model and loss.** Use a regularized linear or small-MLP map from a compact molecular vector to the prior over `z_c`; optionally add a separate locus/context accessibility term. Keep the outcome loss unchanged for attribution. Compare identifiers, molecular descriptors, reference measurements, and both together in a factorial experiment. Do not start with transcriptome-to-sequence cross-attention, context-conditioned SSM kernels, or a large mixture of experts.

**Training and ablations.** Leave-cell/study-out; train preprocessing on source samples; random equal-dimensional context vectors; permuted profiles assigned at the context level; unrelated matched gene panels; missingness-only control; editor metadata alone; same-culture versus public profiles where feasible. Many rows sharing a cell profile must stay in the same outer fold. Public profiles from the intended deployment context are permissible auxiliary inputs if available at deployment; their outcomes are not.

**Planning target.** A reproducible +0.02 or larger within-edit rank improvement in unseen contexts, accompanied by selection benefit, or achieving the same utility with half as many reference measurements. Expect the effect on the old pooled benchmark to be small or zero; these are targets, not predicted effects.

**If successful:** molecular state supplies a useful transfer prior. **If unsuccessful:** available profiles may be mismatched, insufficiently diverse, or redundant. Do not infer that molecular biology is irrelevant.

### Priority 5: use replicates to support decisions

**Hypothesis.** Modeling measurement reliability improves context estimation and calibrated selection probabilities, even if aggregate Spearman hardly moves.

**Inputs.** Edited and total usable read counts, biological-replicate identity, technical replicates, batch, and documented assay definitions. Do not reconstruct integer counts from rounded percentages or reinterpret OptiPrime's row weights as read depths.

**Model and loss.** Where the denominator supports it, use a beta-binomial observation model for edited counts, with dispersion partially pooled by assay. Multiple reads are sampling observations, not independent biological experiments. For mutually exclusive, exhaustive product categories, use an appropriately overdispersed multinomial model, including an “other product” category if needed. If only replicate fractions exist, use a bounded measurement model selected through held-out-replicate checks and sensitivity analysis; do not invent a known detection limit.

Keep replicates as observations of a shared latent design/context efficiency, rather than making a full-data denoised label and splitting it afterward. Compare raw-label training, train-only replicate aggregation, a hierarchical latent model, and clipped reliability weighting with matched effective sampling. Avoid allowing inverse-variance weighting to remove low-efficiency or rare contexts from the objective.

**Ablations and evaluation.** Counts versus percentages; binomial versus overdispersion; true versus shuffled depth/reliability metadata; source-specific versus pooled dispersion. Test independent replicate log likelihood, interval calibration, pairwise order probabilities, calibration under domain shift, and utility. Calibration fitted in one domain does not guarantee coverage in another; conformal prediction needs an appropriate exchangeability assumption or fresh target-context calibration data.

The current unconstrained ordinal head is a ranking score, not automatically a coherent predictive distribution. Use a coherent distribution for probability claims. Additional tasks are worthwhile only if they add actual measured products or states; an unedited fraction defined as `1 − intended − indel` is not an independent biological label.

**Planning target.** A practically useful improvement in interval/order calibration and roughly 10% relative regret reduction on unreliable measurements; +0.00 to +0.01 pooled Spearman would not be surprising but is not an evidence-based forecast. Stop this branch if it only improves a likelihood with no relevant calibration or decision benefit.

**If successful:** measurements and uncertainty improve experimental choices. **If unsuccessful:** noise modeling is primarily a reporting tool here. The failed censor-threshold masking should remain failed; this is an observation-model experiment, not a renamed repeat.

## 4. External data: a targeted acquisition queue

These candidates were checked against primary publication/data-availability descriptions. I have **not** downloaded and normalized all external supplements, established row-level independence, or verified compatible counts for every candidate. Published library sizes below are not counts of usable evaluation rows. This is a prioritized acquisition plan, not a certified external benchmark or an exhaustive census.

| Candidate | Published size/context information | Labels and input compatibility | Independence and use |
|---|---|---|---|
| **MinsePIE / Koeppel et al.** | 3,604 insertion sequences; initial screens include HEK293T/HAP1 and multiple target/repair settings, with broader follow-ups | Read-count tables available. Pooled insertion rates use library-abundance normalization and are not interchangeable with per-design edited-allele fractions. Longer inserts may exceed the current input scope | Distinct named source from the current three-study mix, pending sequence/experiment overlap audit. Best candidate for repair-dependent transfer and within-assay ranking; limited independent loci |
| **OPED's original prospective experiments** | Publication covers PE2, PE3/PE3b and ePE designs; usable row counts require supplement extraction | Original sequencing accession PRJNA882795; recover exact guide/target/context fields. Initially restrict to supported single-pegRNA configuration | Reused public datasets are not new validation. Only original, nonoverlapping measurements qualify; priority candidate for endogenous external selection |
| **PRIDICT2/ePRIDICT endogenous and chromatin panels** | Measured rates in Supplementary Tables 2, 7, 8, 12; sequencing PRJNA1025026; exact eligible counts pending | Recover guide fields, coordinates, and assay identity. Distinguish endogenous experiments from mapped reporters | Source family already contributes training data. Qualifies only for panels proven absent from training; valuable assay-shift test but not an independent-laboratory claim |
| **Li et al. chromatin/repair experiments** | Mapped reporter integrations and repair perturbations; BioProject PRJNA949965; usable design/site count pending | Appropriate for position/state effects; few shared reporter designs may not support alternative-pegRNA selection | Different study, pending overlap audit. Mechanistic validation or accessibility model test, not a large pegRNA ranking benchmark by default |
| **OptiPrime's endogenous prospective panels** | Local inventory records 283 design rows in Supplementary Table 3, without efficiency labels in that sheet | Need source-data outcomes and sample mapping. Exclude unsupported dual-guide modalities unless modeled explicitly | Same source as part of training; potentially useful assay transfer after audit, not independent-study confirmation |
| **StopPR functional screen** | Approximately 240,000 epegRNAs, approximately 17,000 codons | Main readout is guide abundance/fitness, not direct editing efficiency | Useful only for downstream application validation with phenotype-aware controls or genotype measurements. Do not use fitness as an efficiency label |

Sources: [MinsePIE](https://www.nature.com/articles/s41587-023-01678-y), [OPED](https://www.nature.com/articles/s42256-023-00739-w), [PRIDICT2 data availability](https://www.nature.com/articles/s41587-024-02268-2), [Li et al. data project](https://www.ncbi.nlm.nih.gov/bioproject/949965), [OptiPrime](https://www.nature.com/articles/s41587-026-03261-7), [StopPR](https://www.nature.com/articles/s41592-024-02502-4).

For every acquired file, make a manifest containing accession/version, experiment ID, biological replicate, cell/editor/delivery/time, native-versus-reporter status, sequence orientations, reference genome/coordinates where relevant, outcome denominator, raw counts, candidate-set membership, and exact/near-overlap flags. Check full constructs and target families across all source files, not just dataset names. Newly mapped legacy rows are still legacy data.

Reserve at least one suitable external panel before looking at comparative performance. Any external panel used to choose the model becomes development data. Supplementary panels selected by earlier authors using their own predictor are not unbiased population samples; disclose ascertainment and prefer prospective confirmation.

**Comparison methods.** Prioritize frozen PE-RankFormer, OptiPrime, DeepPrime/DeepPrime-FT, PRIDICT2, OPED, and a strong simple feature/embedding model. Include ePRIDICT for the chromatin task and MinsePIE for compatible insertion tasks. Check PrimeNet's applicable output/scope as a contemporary additional comparator; its reported multi-outcome results are not directly comparable with this benchmark. [DeepPrime primary study](https://www.sciencedirect.com/science/article/pii/S0092867423003318), [PrimeNet](https://academic.oup.com/bib/article/26/3/bbaf293/8169298)

Keep two tracks: released tools used as practitioners would use them, and matched-data retraining/adaptation where reproducible code is available. Report each tool's prediction coverage and a common-supported subset. Never silently discard rows a tool cannot score. Verify official preprocessing on author-provided examples before large comparisons.

## 5. What I would retain, defer, or reject from explore_v1

| Recommendation family | Decision | Reason or prerequisite |
|---|---|---|
| Rich biological context | Retain, sharply constrain | Most plausible value is unseen-context transfer; few independent cell states and assay mismatch limit public profiles |
| Conditional ranking | Retain only behind Gate A | FiLM, low-rank conditioning, within-condition quantiles, and direct rank shifts have already been tested; common-support decisions and new measurements are the substantive differences |
| Replicate-aware training | Retain as supporting work | Need defensible observation models; the existing censor-masking intervention was harmful |
| Domain generalization, external and prospective tests | Highest execution priority | These supply evidence of a new useful capability |
| Decision-focused evaluation | Essential correction | Existing within-spacer grouping is insufficient for the claimed deployment task |
| Uncertainty and active learning | Tie to reference-panel acquisition | Test against random, diversity, and equally budgeted adaptation; uncertainty is useful only if it improves decisions |
| Relative geometry | One bounded diagnostic | Add nick/edit/PBS/RTT-relative coordinates to S4D and a matched attention baseline; distinguish geometry from regularization and optimization |
| Minimal physical features | One small conditional test, if residual evidence supports it | Thermodynamic calculations are derived from sequence, so they change sample efficiency/inductive bias rather than add a new measurement; current feature residual gains are tiny |
| Generic DNA foundation model | Defer | No demonstrated missing sequence grammar; short engineered pegRNAs differ from genomic pretraining distributions |
| Synthetic edit pretraining | Defer behind learning-curve evidence | Alignment and edit reconstruction may be nearly deterministic and need not teach efficiency. Synthetic efficiencies must not be invented |
| Contrastive learning | Do not impose context-invariant outcome embeddings | Same-target or cross-context positives can erase exactly the design/state differences under investigation |
| Large MoE, hypernetworks, context-conditioned SSM, new Mamba sweep | Do not prioritize | Expensive ways to reuse the same information, with close negative precedents |
| Full outcome prediction | Only if new product labels are recovered | Recreating the existing simplex adds no information; PRIDICT/PrimeNet already cover related outputs |
| A second biological modality | Optional, late | Prime-editing depth can be sufficient. Broaden only if it directly tests the same adaptation principle |

A cheap geometry experiment can test whether an attention model with explicit relative coordinates closes part of the S4D gap. Match parameters, training budget, and seeds; assess length/position extrapolation. Even a positive result would support a restricted inductive-bias explanation, not prove S4D is generally superior for biology. It is secondary to the transfer program.

If pursuing a broader principle, the coherent extension is the same small-reference-panel adaptation protocol for base editing or CRISPRi/a across cell states, with its own appropriate outcomes. A protein-variant or MPRA dataset added merely to increase benchmark count would dilute this paper. Claiming “SSMs outperform Transformers for biological sequences” would require a separate controlled multi-task study including strong convolutional baselines.

## 6. The prospective study that would change the paper's significance

The first pilot identifies assay reliability and whether transferable interaction signal exists. The definitive study must then be prospective at the level of model choice, reference-panel choice, target selection, and predictions.

**Illustrative validation scale:** 40 new desired edits × 12 predeclared candidate pegRNAs × 2 unseen contexts × 3 biological replicates = 2,880 measurements, plus 48 reference designs × 2 contexts × 3 replicates = 288 reference measurements. With the earlier 1,152-measurement mechanistic pilot, that is 4,320 design-context-replicate measurements before controls, exclusions, and reruns. This is an illustrative scope, not a powered final protocol or a two-month promise.

Choose at least one biologically useful endogenous setting beyond the dominant benchmark cell lines; a primary or differentiated cell model would be particularly valuable if feasible. Donor generalization requires multiple donors and donor-level analysis, not just repeated cultures of one donor. Independent laboratory execution would strengthen transportability and requires its own replication.

Construct a prespecified candidate panel that spans the allowed design space, includes each comparator's top recommendations, and includes randomly sampled candidates. Keep a representative evaluation component distinct from an enriched model-disagreement/mechanism component. Evaluate all candidates in the defined shortlist so that within-shortlist regret is identifiable. If the union of comparator recommendations exceeds 12 designs, expand the panel or fix a symmetric inclusion rule before outcomes are seen.

Freeze zero-shot and adapted predictions before query outcomes are available. Permit adaptation only from the declared reference panel. Blind experimental processing to model identity and predicted ranking; randomize batches. Quantify intended editing and relevant unintended products, rather than implying that higher efficiency alone establishes an improved outcome profile.

Predeclare one primary budget and utility endpoint; for example, intended efficiency achieved by the best of three nominated candidates per edit. Add success above an application-specific threshold and screening cost as secondary endpoints. Analyze the same edit across methods as paired data, while preserving dependence across contexts and replicates.

Power the definitive experiment from pilot variation in the paired, per-edit endpoint. As an illustration only, a paired standard deviation of 0.08 and a target absolute improvement of 0.04 require approximately `((1.96+0.84)×0.08/0.04)^2 ≈ 32` independent edits for a simple two-sided 80%-power calculation. Forty edits leaves limited margin for losses; multiple primary comparisons, donor/batch clustering, or a smaller target effect can require substantially more. The thousands of wells/measurements are not the power denominator.

**Gate B:** run the definitive study only after the retrospective/pilot method beats ordinary fine-tuning or calibration on disjoint query loci at equal total cost, and after the effect size warrants the validation expense.

## 7. A realistic Nature Methods version of the paper

**Working title:** “Reference-edit profiling enables context-adaptive prime-editing design.” Use “enables” only if the prospective results support it; until then this is the proposed research question.

**Central claim:** A small reusable panel of editing measurements identifies experimental context sufficiently to improve pegRNA selection for unmeasured edits in new biological settings.

**Methodological contribution:** Joint design/context inference, a standardized reference-panel acquisition strategy, and a rigorously evaluated adaptation procedure with transparent measurement costs. The method must outperform simple equally informed adaptation; a bilinear layer alone is not novel enough.

**Biological insight:** Particular design sensitivities change predictably with experimentally verified state. The general observation that MMR or chromatin affects prime editing is already established. Latent-factor associations alone should be described as associations, not pathways discovered by the model.

**Evidence package:** The corrected legacy benchmark; at least two suitable, provenance-audited external settings where feasible; a crossed mechanistic panel; and a blinded prospective endogenous selection study. The number of independent contexts and studies matters more than the row count.

Five main figures would tell the story:

1. The experimental decision, assay provenance, and reproducible context-dependent design effects on matched candidate sets.
2. The reference-panel method and equal-budget adaptation comparisons, including learning curves and break-even costs.
3. New-locus/new-context and external-study results, with failure contexts and prediction coverage visible.
4. Molecular annotations and controlled perturbations explaining which state-dependent preferences transfer.
5. Prospective fixed-edit selection: achieved efficiency, success, product quality, and screening effort.

Move the S4D-versus-attention factorial, ordinary architecture negatives, and the historical pooled benchmark to supporting figures unless they become essential to the transfer result. Release a reproducible scorer, design generator, frozen model versions, standardized panel definition, raw outcome processing, split manifests, and an explicit unknown-context workflow.

| Likely reviewer objection | Required answer |
|---|---|
| “This is fine-tuning with extra labels.” | Equally budgeted fine-tuning/ridge/nearest-context controls, panel reuse, and disjoint query-locus transfer |
| “The expression vector is a cell/study ID.” | Entire-cell/study holdouts, random-profile controls, and matched experimental measurements |
| “You optimize pooled correlations, not experimental choices.” | Fixed desired allele/context candidate groups and prospective utility |
| “Your test set influenced development.” | Newly reserved external data and preregistered prospective predictions |
| “The biology is already known from OptiPrime/ePRIDICT/MinsePIE.” | A specific new transferable design rule or experimentally useful adaptation capability, compared directly with those methods |
| “Chromatin annotations are assigned to the wrong site.” | Native-versus-reporter provenance and actual assayed-locus mapping |
| “Interactions are noise or experimental batch effects.” | Matched crossed designs, independent biological replication, batch balancing, and uncertainty-aware controls |
| “Your panel costs more than it saves.” | Total-measurement cost curves and an explicit workload where the method breaks even |
| “Only one laboratory or cell background works.” | Context-specific results and external transport; independent laboratory/donor replication where the claim requires it |

If the reference-panel hypothesis fails but the corrected benchmark and external validations remain strong, the defensible paper is a rigorous prime-editing prediction method with demonstrated practical improvements. It should not be expanded into an unsupported biological-context story to pursue a journal label.

## 8. Execution plan

### A. Experiments I would run in the next two weeks — current data first

**Days 1–3:** reconstruct decision keys; audit which rows are alternative designs, distinct intended edits, and true replicates; create a label/provenance dictionary; freeze the historical model. Recalculate selection metrics descriptively and report candidate coverage. Use the old held-out set for this audit only, with no new model selection against it.

**Days 3–6:** rerun the interaction diagnostics on common support with locus-disjoint splits, replicate checks, and additive/shuffle controls. Check whether the existing rank-shift probe survives those controls. Use simple models to assess identifiability and an injected-interaction control to verify the analysis/training pipeline can detect an effect of practical size.

**Days 5–10:** implement frozen-embedding adaptation and intercept-only/ridge/fine-tuning controls on newly defined development episodes. Evaluate several reference budgets. Begin with two predeclared low ranks and a small regularization choice tuned only on source validation episodes; do not open a broad search.

**Days 8–14:** identify and reserve external panels through metadata inspection; validate baseline adapters on official examples; simulate pilot power from available grouped residuals. Produce a Gate A report and a concrete experimental panel specification if Gate A passes.

Expected deliverables: canonical decision manifest, corrected metric report, context-overlap/identifiability audit, reference-budget curves, external-data eligibility manifest, and a go/no-go decision. A handful of full-backbone controls may be necessary; most early work should use frozen embeddings. Existing S4D run logs suggest hours per full run, but hardware availability and current throughput must be checked before scheduling.

### B. Experiments I would run in the next two months — public data and a focused pilot

**Weeks 3–4:** acquire and normalize the highest-priority external panels; complete official baseline preprocessing; compare cell identifiers, compact molecular descriptors, functional references, and both on source-development holdouts. Recover read counts/replicate provenance where available.

**Weeks 4–6:** if Gate A passes, start the crossed pilot with collaborators and source the most important same-culture state measurements. If Gate A is inconclusive because current data lack the required crossing, first run a reduced feasibility panel to resolve that question. In parallel, test the smallest observation model and establish query-locus-disjoint adaptation learning curves wherever the data support them.

**Weeks 6–8:** evaluate pilot reliability and effect sizes, finalize the reference panel, and freeze a candidate method. Use untouched external data for the predeclared comparison. Prepare the powered, blinded prospective validation; do not promise that a large primary-cell study will finish in this window.

If wet-lab work is unavailable, complete the computational and public-data program and narrow the claim accordingly. The strongest version of this plan depends on new controlled measurements; that dependency should be resolved before committing most of the effort to complex modeling.

### C. Experiments required before the proposed Nature Methods submission

1. A reproducible advantage on the correct fixed-edit decision, with baseline coverage and total acquisition cost reported.
2. Transfer that survives jointly unseen loci and contexts, plus a suitable untouched external assay or study.
3. Prospective selection improvement against strong equally informed methods, measured before query labels influence modeling.
4. Controlled evidence for any claimed biological mechanism, with independent replication and appropriate intervention controls.
5. Ablations establishing that the reference measurements add transferable design information beyond a mean shift, and whether molecular annotations add further value.
6. Raw-data/provenance release, usable software, frozen splits, negative contexts, and calibrated limitations.

These are requirements for the claim proposed here, not a statement that Nature Methods universally requires wet-lab work or a fixed number of datasets.

### D. Experiments I would not spend time on now

More generic SSM/Mamba/MoE sweeps; repeated FiLM or low-rank categorical conditioning; another conditional-quantile objective; censor-threshold masking; learned ensemble weights; generic transcriptome cross-attention on nine cell identities; millions of synthetic “efficiency” labels; native-chromatin annotations for unmapped reporter sites; new precision claims from the repeatedly examined held-out set; or another modality added solely to make the paper look broad.

## Audit provenance for this review

Repository commit at inspection: `83ffae1c5b8fac2b5b9363ca57962e9c52ea7f37`. The existing untracked `explore_v1.md` was left intact. New computations in this review were read-only descriptive grouping checks using `revision._common.load_heldout()` and pandas on the frozen files; no training, calibration fitting, or checkpoint selection was performed.

| Input | SHA-256 |
|---|---|
| `data/processed/optiprime_official_318471.parquet` | `fe62229d50a08fa337f3a6f01a4a7d435f40ba9e2ca0648a4fb2288d4d5fe297` |
| `results/heldout_full_head_to_head.parquet` | `23d833dfcccb6f609d0b84de264a318ef04d89f1a9c234cc77fbe5ffdbc93326` |
| `results/round5/heldout_calibrated.parquet` | `9cbd7ce15f3e0c5ce747a062ff8a9536cdae0437fcab8d88b951a178edc5ddf6` |

The grouping checks used at least five rows and nonconstant observed outcomes for mean Spearman, matching the existing analysis. Those row counts are not certified counts of unique alternative designs. The conservative exact-pair diagnostic grouped by spacer, full WT/edited strings, and the seven model context fields; its purpose was to expose the need for biological canonicalization, not to define the final benchmark.
