# Next steps after explore_v2

**Status:** the current plan following `SUMMARY_REPORT.md` and E09–E15 is [NEXT_STEPS_AFTER_E15.md](NEXT_STEPS_AFTER_E15.md). The document below is retained as the earlier review; several repairs it proposes are now complete.

Prepared 8 September 2026 after reviewing `WORK_SUMMARY.md`, `GATE_A_REPORT.md`, E01–E08, the embedding extraction and canonicalization code, and the reserved-panel declaration. This is a proposed work program; the experiments below have not been executed in this review.

**Scope, revised following the user's clarification:** this is a purely computational research project. All validation uses existing datasets and previously measured outcomes. New wet-lab experiments, experimental collaborators, and acquisition of new biological measurements are outside scope. This revision supersedes the wet-lab recommendations in earlier plans.

## Recommendation

**Develop a computational method for budgeted, fixed-edit pegRNA selection, supported by rigorous transfer benchmarks on existing measurements. First repair the evaluation foundation, then develop and test the method on clean development splits while preparing independent public-data evaluation.** Defer further context-model development until a properly separated experiment establishes an opportunity worth pursuing.

I agree with the work summary's move away from a reference-panel headline. I do not agree that the present analyses conclusively establish either a small universal adaptation ceiling or a clean negative result for unseen-context adaptation. Several implementation and interpretation issues need resolving before they guide a publication-level decision.

The Nature Methods ambition should drive three computational contributions: a method that directly improves design decisions, a reproducible benchmark separating locus/context/study transfer, and evidence that the method works across independent datasets. Quantify selection benefit retrospectively from measured candidate panels. Describe simulated screening savings as retrospective estimates, not as savings demonstrated by new experiments. A corrected benchmark and a larger margin alone do not establish a major methodological advance.

## 1. What to retain from the completed work

- The unit of evaluation must be the desired allele in a specified experimental setting. The old within-spacer headline mixed distinct choices.
- The current canonicalization reports only 2,412 fold-0 groups with at least two candidates, 85 with at least three, and none with at least five. These counts need confirmation after the identity audit, but the candidate-depth limitation is substantial.
- On the currently defined 1,390 nontied binary decisions, the ensemble improves accuracy by about 0.091 and achieved efficiency by 0.0091 over OptiPrime. These are useful exploratory results, conditional on that eligible population.
- The ordinal-S4D member is a sensible primary model to carry forward. Its binary accuracy is about 0.750 versus 0.747 for the final ensemble. Confirm exactly how many checkpoints constitute this “member” before claiming a single-checkpoint computational advantage.
- Context interactions deserve analysis, but their existence and predictability are different questions from whether exploiting them changes the selected design.
- The deeper Kim panel is valuable as a reserved comparison for models that have not trained on its records. It is not an independent-laboratory or unseen-context dataset.

I ran `verify_report.py`: all **47 numerical checks passed**. This verifies agreement between the asserted values and stored JSON; it does not validate the scientific estimands, data separation, or implementations that generated them.

## 2. Findings from this review that change the next steps

### 2.1 Independently trained embeddings are being treated as one coordinate system

`extract_embeddings.py` puts outputs from five separately trained networks into one embedding matrix. E04 then subtracts the selected representation of design 2 from that of design 1 and fits one shared PCA/ridge model.

A metadata-only check matching E04's representative-row selection found:

- 15,718 of 55,090 decision groups with at least two designs span more than one checkpoint.
- **57,821 of 110,321 E04 quartet design contrasts use different checkpoints** for their two embeddings.

Hidden coordinate 17 in one network has no guaranteed correspondence to coordinate 17 in another. Scalar out-of-fold scores can be combined under a defined scoring procedure; hidden vectors cannot automatically be subtracted or pooled this way. This may distort positive and negative findings. It does not establish that the reported signal is entirely artifactual.

**Required correction:** within each outer evaluation split, use one source-trained encoder to embed every row used by that split's downstream model. For ensembles, fit a separate downstream model per encoder and combine predictions, not unaligned hidden coordinates. Alignment would require its own source-only procedure and validation; it is unnecessary for the first clean test.

### 2.2 The outer split does not extend through the representation learner

E04 holds loci out from the ridge fit, but its backbone was trained on different official folds. Related designs and measurements from a nominally held-out locus can therefore have influenced the representation. E05 excludes the target context from PCA/hyperparameter selection, but its pretrained backbone and context-dependent base predictions have already learned from other rows in that context. It tests additional calibration/adaptation in a familiar-context representation, not fully unseen-context transfer.

There is also a scope discrepancy: E03 reads the full decision manifest with no fold-0 exclusion, and E04 fits from that quartet cache. Thus the declaration that fold 0 was used only for descriptive analysis is not true for this downstream fitting pipeline. The old fold-0 results should remain explicitly exploratory.

**Required correction:** exclude outer-test loci and, for a transfer claim, the outer-test context from all backbone training, checkpoint selection, feature transformations, and downstream tuning. Use inner validation for early stopping. An embedding taken before FiLM can still have learned from context-associated outcomes during training.

### 2.3 The canonical allele key has a demonstrable strand-equivalence failure

I checked the same repeat-associated deletion in two strand orientations:

`WT = GGGACACACTTT; edited = GGGACACTTT`.

`canonical_keys(WT, edited)` and `canonical_keys(reverse_complement(WT), reverse_complement(edited))` return different keys. The analogous insertion also fails. The code left-normalizes an indel on one orientation before choosing between that key and its reverse complement; the reverse representation is not independently normalized. The same construction is used in the extended-window and external-panel paths.

This can split equivalent edits and undermine overlap exclusion. It does not quantify how many real records are affected. Other identity risks include short flanks, truncated edits, ambiguous reporter-to-genome mapping, and use of target names rather than connected locus identities.

**Required correction:** normalize both orientations consistently, validate edits against source metadata, flag unresolved/truncated alleles, and measure sensitivity to identity rules. Recompute candidate depth and overlap after the correction. Do not automatically equate every shared 24-base local signature with one uniquely mapped genomic locus.

### 2.4 The strongest interaction model lacks its matched shuffle comparison

E04 reports embedding-by-context R²≈0.397, but its shuffled-context and shuffled-target controls use the 17 engineered features, not the embedding features. The quoted 0.397-versus-0.012 comparison changes representation as well as the intended mechanism. Its PCA and scaling are also fitted before outer splitting.

The positive-control implementation adds a synthetic signal to the already predictable observed target and reports performance on their sum. That alone does not show that the added 5%-SD component was recovered, nor establish a power threshold for every negative adaptation arm.

**Required correction:** use exactly the same embedding features and model capacity for real and shuffled controls; fit transforms within each outer split; inject known signal into an appropriate null/noise target and directly measure recovery of that signal. Test power at the final selection endpoint as well as at D prediction.

### 2.5 The reference-panel and correction nulls have narrower scope than stated

Additional issues to fix if these branches are revisited:

- E05 tunes a shared penalty/rank for the low-rank arm using only `src[:3]`, then applies it to other arms. Each serious comparator needs suitable source-only tuning.
- E05 seeds episodes with Python's process-randomized `hash(ctx)`. A recorded integer seed alone does not reproduce the episodes. It also changes query sets across budgets; fix the query set and nest support panels to make budget curves interpretable.
- Its random support sampling is not a test of a standardized, deliberately informative reference panel with repeated within-edit design contrasts.
- E07 chooses blend strength and selectivity using in-sample correction predictions. These choices need inner out-of-fold predictions; choosing zero in the grid does not remove optimism in in-sample tuning.
- E05's summaries pool repeated support draws; they do not establish a context-clustered equivalence bound. “Statistically identical” requires a direct paired comparison and a defined practical equivalence margin.

These points justify a bounded clean retest only if the main program leaves a useful reason to pursue adaptation. They do not justify another broad search.

### 2.6 A transfer regret is not a universal ceiling, and large D is not a reversal

The 0.0073 estimate is the measured gain available over **one particular chooser**: transfer the observed winner of a two-design pair from one context into the other, averaged over a filtered quartet population. It is not an upper bound on improvement over PE-RankFormer, which may make other errors. It also does not bound gains at greater candidate depth, in other contexts, or on prospectively identifiable subgroups.

Further, `D = Delta(c1) − Delta(c2)` can be large while both contrasts have the same sign. For example, 0.30 versus 0.10 gives D=0.20 without a reversal. The reported 94.1% accuracy for the sign of large predicted D is **not** 94.1% accuracy for predicting whether designs reverse order. Evaluate those as separate prediction tasks using the existing matched measurements.

**Required correction:** calculate remaining regret for the actual baseline on the same decision sets, predict both design contrasts, and assess reversal probability and expected improvement from switching. Keep the reported transfer regret as a descriptive quantity for its stated population.

### 2.7 The reserved panel is part of DeepPrime's training source

`external/deepprime/train_base.py` explicitly loads `DeepPrime_dataset_final_Feat8.csv`, the source of the reserved panel. This is consistent with DeepPrime's published data/training description. Exact exposure still depends on the checkpoint and fold-handling implementation. [DeepPrime primary study](https://www.sciencedirect.com/science/article/pii/S0092867423003318)

**Required correction:** build a model-by-evaluation-data exposure matrix. PE-RankFormer versus OptiPrime can remain the prespecified primary comparison if their lack of exposure is verified. A released DeepPrime checkpoint exposed to this library cannot be described as an independent held-out comparator there. Use auditable out-of-fold predictions or matched retraining, or explicitly label an exposed released-tool result and rely on another dataset for the clean comparison.

## 3. Proposed work program

| Order | Workstream | Concrete deliverable | Decision rule | Estimated effort |
|---|---|---|---|---|
| **1** | Validate identities and claims | Versioned canonical manifest, corrected overlap/coverage tables, claim-status addendum | No unresolved identity problem allowed into the primary benchmark | 2–4 working days |
| **2** | Establish a clean development protocol | Locus-component outer split, one encoder per split, complete provenance, inner tuning | Evaluation separation must hold through all learned components | 3–7 days plus training |
| **3** | Prepare and execute the deeper comparison | Frozen adapters/checkpoints, protocol addendum, one comparative report on the Kim panel | Evaluate practical gain and robustness at candidate depth ≥5 | About 1 week after adapters work |
| **4** | Obtain independent endogenous evidence | At least one verified original dataset with useful candidate sets and strong baseline predictions | A larger claim requires transport beyond the Kim assay | 1–3 weeks; starts alongside 1–3 |
| **5** | Develop and test decision-focused learning | Controlled group-sampling/objective experiment and budgeted selection evaluation | Continue only if it improves utility beyond the unmodified model and simple controls | 1–2 weeks after the clean baseline |
| **6** | Establish computational generality and reproducibility | Independent-data results, task/architecture ablations, reusable benchmark and software | The claimed principle must transfer beyond one model on one split | Weeks 5–8 |

### Workstream 1 — identity, provenance, and metric audit

Add meaningful checks for repeated-sequence indels, reverse complements, multiple spacers, changing WT-window length, masked DeepPrime windows, no-op/invalid edits, and incomplete alleles. Round-trip recovered alleles through sequence reconstruction and compare with source edit type, length, and position. Audit stratified real examples as well as synthetic cases.

Join locus components using normalized spacer identity, validated alleles, target-site relationships, and coordinates/sequence overlap where available. The existing target-name/allele graph omits a direct shared-spacer edge; it should not be assumed to remove all site overlap. Keep distant paralog/sequence-family exclusion as a separate, documented sensitivity analysis rather than merging all approximate matches blindly.

Recompute E01/E02 with corrected keys and connected-locus uncertainty estimates. E02 currently assigns a multi-spacer decision to its first spacer for bootstrap purposes; use a dependence unit that keeps all connected decisions together.

Report two populations explicitly: all eligible fixed-edit groups for achieved efficiency, regret, and threshold success; nontied/informative groups for order discrimination. Both-zero groups must contribute zero achieved efficiency to the population-level utility calculation. A group whose all designs fail has no ranking information but matters to users.

Preserve original reports and create a dated correction/addendum identifying changed claims. Cache keys must include raw input hashes, canonicalizer version, checkpoint hashes, and split manifests; current caches are not automatically invalidated when the code changes. Archive old caches instead of silently mixing versions.

**Gate 1:** the main fixed-edit advantage remains useful on high-confidence canonical groups, with the direction checked on groups/loci absent from training. If that advantage is unstable under plausible identity rules, resolve this before any new modeling or panel scoring.

### Workstream 2 — one clean benchmark backbone

Start with one predeclared development split that holds complete locus components out, and reserve an inner validation set within the training components. Train the unchanged ordinal-S4D recipe. This first run tests the impact of correct separation rather than a new modeling idea. Freeze the recipe before expanding to three splits or seeds for the final selected comparisons.

For a downstream probe, use that same encoder on its source training and outer query sets. All PCA/scaling/adaptation choices belong inside source training/validation. Report the important distinction between source-trained frozen representation transfer and adaptation to a context the backbone has already seen.

Rerun the interpretable feature interaction probe first, followed by the corrected embedding probe and its representation-matched controls. Use source-only variance estimates for learned noise weighting or reliability filters. Global descriptive noise summaries can remain clearly labeled historical analyses.

If an unseen-context experiment remains justified, choose two well-supported target settings using metadata and candidate coverage, not favorable outcomes. Exclude each entire target setting from backbone training. Use an explicit, predeclared unknown-context input or compositional representation; do not supply a target context embedding learned elsewhere from its outcomes. Keep support and query locus components disjoint and count all measurements.

Do **not** run full fine-tuning on the current mixed-checkpoint cache and interpret it as the missing clean experiment. The representation and split problem must be fixed first.

### Workstream 3 — ready the reserved Kim comparison without using it for model selection

The panel remains valuable, but “sealed” needs precise wording: E06 has already inspected aggregate labels, including mean efficiency and within-allele outcome range. No comparative predictions have been evaluated in this program. Call it **reserved for comparative evaluation**, not a dataset whose outcomes have never been examined.

Before comparative scoring:

1. Record a versioned addendum to the existing declaration. Corrected allele mapping and overlap removal are justified preprocessing changes; document them before examining comparative results. Preserve the original panel/hash and create a newly versioned panel rather than overwriting it.
2. Verify model exposure, checkpoint identity, preprocessing, orientation, PBS/RTT extraction, and sequence-length coverage on existing development examples. The declared comparator set must not be chosen after inspecting results.
3. Choose one primary endpoint: mean achieved editing efficiency of the top-ranked candidate per fixed-edit group. Its paired improvement is exactly the paired reduction in absolute regret on the same groups; these are not independent discoveries. Keep best-of-three as a secondary budget endpoint.
4. Rename the current declaration's “success@1,” which means selecting the measured optimum, to **best-design hit rate**. Separately define threshold success as obtaining at least one design above a predeclared editing threshold. Give tied optima full credit, but do not present an all-zero tie as experimental success.
5. Include all outcome groups in deployment utility; use nontied groups for discrimination and report the difference. Use connected-locus bootstrap units and depth/edit-type strata fixed in advance.
6. Freeze both the original ensemble and the designated ordinal-S4D member as comparisons. Designate one primary model beforehand; do not promote whichever looks best after scoring. Record whether the member averages five checkpoints.

Run the primary PE-RankFormer/OptiPrime comparison once after these steps. Different tools' failures to score candidates must be visible. Report a common-supported comparison and tool coverage, with a predeclared policy for unscorable recommendations. An arbitrarily tiny but significant advantage on 30,000 groups is not enough; assess the absolute improvement, regret reduction, and number of designs saved at a fixed utility target.

**Gate 2:** a useful advantage persists at candidate depth ≥5. If it vanishes, the binary-choice result should not be extrapolated to practical multi-design screening. If it survives, proceed to evaluation on independently acquired public datasets; do not tune further against this panel. If its results inform later method choices, report it as development evidence for those later methods and reserve a different dataset for final confirmation.

### Workstream 4 — independent endogenous validation and complete baselines

Start acquisition now, alongside the evaluation repairs. Prioritize OPED's **original** prospective experiments; its reused public training/evaluation datasets are not independent evidence. The publication makes original sequencing available under PRJNA882795. Recover supplementary outcome tables first; use raw sequencing only when the necessary measurements are absent. [OPED data availability](https://www.nature.com/articles/s42256-023-00739-w)

Second, recover original endogenous PRIDICT2 panels and OptiPrime source-data outcomes. These can establish assay transfer after exact provenance checks, although some share a source laboratory with training. PRIDICT2's primary study explicitly compares endogenous predictions and distinguishes its sequence and chromatin tasks. [PRIDICT2 study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7617539/)

For each file, report the actual usable number of independent edits, alternatives per edit, biological settings, assay denominator, missing fields, source-model ascertainment, and exposure for every comparator. Do not substitute a paper's headline row count for eligible fixed-edit groups. If an independent panel has only one tested design per edit, use it for outcome prediction and state that it cannot validate selection regret.

Complete official adapters for DeepPrime, PRIDICT2, and OPED, keeping two tracks distinct: published tools used as released, and matched-data retraining where feasible. A feature tree does not replace these comparisons. MinsePIE is a lower acquisition priority for this specific question because it largely compares insertion sequences and normalized rates, not interchangeable pegRNAs for an identical final allele.

**Gate 3:** independent evidence shows that the advantage is more than a same-assay benchmark effect. If no suitable public panel exists for a particular claim, use the available data for the tasks they actually support and narrow that claim. A single-design-per-edit panel can test prediction transfer, while a deeper same-assay panel tests selection. Neither should be mislabeled to substitute for the other; no new experiments are a dependency.

### Workstream 5 — decision-focused learning as the main methodological hypothesis

Once clean development evaluation establishes meaningful remaining selection regret, test whether the existing row-level training objective underweights the decision groups that matter. The central question is whether training on interchangeable designs improves top-k utility and transfer more consistently than training to predict a pooled rank. If little regret remains, test harder existing-data transfer settings before expanding model complexity.

Use a small factorial:

| Sampling | Loss | Purpose |
|---|---|---|
| Existing row sampling | Existing ordinal loss | Unmodified control |
| Balanced sampling of fixed-edit/context groups | Existing ordinal loss | Isolate task weighting |
| Existing row sampling | Ordinal plus within-group decision loss | Isolate supervision |
| Balanced groups | Same combined loss | Test whether the components complement each other |

One candidate within-group loss is a soft selection loss `sum_i softmax(s_i / tau) * (max_j y_j − y_i)`, retaining ordinal supervision as the anchor. Select temperature and a small auxiliary-weight grid on source validation only. Near-tied noisy outcomes require replicate-aware sensitivity checks, and a lower training loss is not the endpoint. Include a structured within-group label shuffle for any apparent gain.

This differs from older generic pairwise objectives by fixing the intended allele and experimental context, and targeting the utility of selecting a candidate. It is still an incremental training change unless it produces substantial gains across independently evaluated datasets and settings. Do not call it a new general principle on the basis of one positive split.

Promote only an effect that improves achieved efficiency/regret across predeclared development splits, survives the sampling-only control, and exceeds a practical threshold declared from the target application. Otherwise retain the unmodified model. Keep molecular embeddings, SSM changes, and reference-panel fine-tuning out of this factorial.

## 4. A computational evidence package for the paper

### 4.1 Evaluate budgeted selection using existing outcome panels

For each held-out fixed-edit group, hide the measured outcomes, nominate k candidates using frozen predictions, and then evaluate the previously measured outcomes of those candidates. Evaluate k=1, 3, and 5 only where candidate depth supports the comparison. Keep the same groups when comparing budgets, or explicitly report how population changes affect the curves.

Report achieved efficiency, threshold success, absolute regret relative to the best measured candidate, and the budget needed to reach a fixed retrospective utility level. Compare with random selection, simple design rules, the unchanged ordinal-S4D model, and applicable published predictors. An oracle over measured candidates is not an oracle over every possible pegRNA.

If testing adaptive selection, reveal existing labels sequentially and refit only from revealed support measurements. Freeze query loci across budgets, include reference-label overhead, and compare with equally informed ridge/head/full fine-tuning and random or diversity-based acquisition. This is a retrospective simulation; it does not require new reference-panel measurements. Do not run it unless the corrected transfer diagnostics support the hypothesis.

### 4.2 Test generalization at the level claimed

The core evaluation matrix is new loci in known contexts, new contexts, jointly new loci and contexts, and independent studies/assays. Withhold each outer-test unit from the entire learned pipeline. Existing public endogenous measurements add useful assay diversity even when they only support prediction rather than selection.

Report per-dataset and per-context effects as well as pooled summaries. Show sensitivity to canonicalization, candidate depth, edit type, zero-heavy groups, source weighting, and training-data exposure. Obtain uncertainty from independent locus components and contexts, not from the number of derived pairs or repeated support draws. Use existing replicates to evaluate sensitivity to observed-best selection and label noise where possible; metadata-identical measurements alone do not establish biological replication.

### 4.3 Show that the contribution is more than a favorable backbone

If decision-focused learning works, apply the same objective/sampling changes to one strong alternate backbone, such as the existing attention model or a validated feature model. Use a matched factorial to separate sampling, loss, representation, and data size. Add learning curves to test whether the method improves sample efficiency, and compare compute, inference cost, and model size.

The proposed main claim becomes: **decision-focused learning improves fixed-edit pegRNA selection and transfer across existing prime-editing datasets**. Support that claim with external data and controlled ablations before broadening it.

A second modality is conditional on the scope of the claimed principle. If claiming a general method for budgeted biological design, select one public base-editing or CRISPR guide dataset only after verifying that it contains comparable alternative designs for a fixed biological objective and independent evaluation units. Do not treat different intended edits or fitness labels as interchangeable candidates simply to add a benchmark. A strong prime-editing-specific contribution need not include an unrelated modality.

### 4.4 Release the benchmark and method as usable computational resources

Release versioned data adapters, exposure matrices, canonicalization with tested invariances, locus/context split manifests, baseline wrappers, checkpoints, and a reproducible evaluation command. Provide a tool that accepts a desired edit and experimental metadata and ranks valid candidates, with explicit handling of unsupported contexts and incomplete inputs. Any candidate-generation component must validate that returned designs implement the requested allele.

Suggested main figures: (1) task definition and benchmark audit; (2) method and sampling/loss ablations; (3) utility-versus-budget curves; (4) context/study transfer and failure cases; (5) sample efficiency, optional cross-backbone/general-task validation, and computational cost. Context-interaction analyses belong in the main story only if they explain a reproducible method benefit.

## 5. Schedule and stop rules

**Next 3 working days:** versioned identity/overlap audit; fix claim scope; choose the clean development split; inventory comparator exposure and validate adapters. Begin independent-data extraction. Deliver `E09_identity_and_claim_audit` and a revised analysis declaration before scoring the reserved panel.

**Remainder of week 1:** train the unchanged backbone on the clean split; generate one consistent embedding space; rerun the smallest feature/embedding interaction probes with matched controls. Deliver `E10_clean_split_baseline` and `E11_interaction_retest`. This is enough to decide whether the context branch deserves any more attention.

**Week 2:** complete and freeze reserved-panel preprocessing, model versions, and metric definitions; execute the comparison when adapters are validated. Produce the comparative report plus a concrete external-eligibility table. If preprocessing or baseline reproduction takes longer, finish those dependencies rather than score an incomplete comparison to meet a date.

**Weeks 3–4:** complete public-data adapters and the four-arm sampling/loss experiment on clean development data. Run retrospective utility-versus-budget evaluation and the corrected context branch only if justified. Keep at least one external evaluation panel unused for model selection until the final method is frozen.

**Weeks 5–8:** replicate the selected method across clean splits/seeds, test the claimed principle on an alternate backbone, freeze the final method, and evaluate the reserved external data. Complete robustness and compute-cost analyses, release reproducible artifacts, and draft the computational-methods paper. Add one further task only if it directly tests the central methodological claim.

Stop the context-adaptation branch if a clean, appropriately powered test rules out an improvement of the prespecified practically useful size. An inconclusive interval is not evidence of equivalence. Stop additional model search if the frozen model already captures nearly all useful regret on the target assay. In that case, invest in external validation and usability. If clean independent comparisons show little practical advantage, reassess the Nature Methods target rather than expanding the narrative around the same benchmark.

## Review provenance

This review added this plan only. It did not modify E01–E08, change the manuscript, train a model, or score/open the reserved panel for comparative evaluation. The checks performed were the existing 47-value verifier, metadata joins over the existing embedding/decision/quartet caches, synthetic strand-equivalence examples, and read-only inspection of source code and primary publications. The checkpoint-mixing counts above describe the current caches; their impact on performance remains to be measured.

The subsequent scope revision changes this plan only and removes all dependencies on new biological measurements. Earlier wet-lab proposals are superseded by the computational evidence package above.
