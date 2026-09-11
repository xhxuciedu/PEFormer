# Improving standalone PE-RankFormer after E24

10 September 2026. Proposed computational experiments in response to the request to explore fine-tuning and separate ranking/selection branches, with the aim of exceeding OptiPrime on held-out targets from the Kim large library. No new fits or dataset partitions have been executed for this plan.

**Scope updated following the user's direction:** improve and validate PE-RankFormer itself through architecture, training, loss design and domain adaptation. OptiPrime is an evaluation comparator only in this phase. Hybrid prediction, OptiPrime-derived training features, teacher distillation from OptiPrime, and other OptiPrime-assisted methods are deferred until the standalone model has been validated. Historical hybrid results remain in their original reports but are not active experiments here.

## Recommendation

Prioritize target-library fine-tuning and a shared encoder with distinct prediction and selection heads. Compare both against ordinary efficiency fine-tuning, then test whether explicit segment geometry or candidate-set attention improves the resulting selector. Develop each change as an intervention with a matched control, not as one package whose source of improvement cannot be determined.

The main hypothesis is that a modest amount of library-specific supervision can correct candidate ordering under the geometry shift. A second hypothesis is that separating output heads lets us improve nomination while retaining broad prediction performance. Neither hypothesis has been established by E01–E24.

Target-library fine-tuning is different from the earlier context-calibration experiments. Those tested small reference panels or corrections to existing predictions; the proposed experiment updates a predictor using many fully measured candidate groups from the target library. Their negative findings do not rule out this experiment. Transfer learning is already established in prime-editing prediction, so fine-tuning alone is not a novelty claim: see [OPED](https://www.nature.com/articles/s42256-023-00739-w).

## 1. Define the result we are trying to obtain

Distinguish the two active comparisons and one later benchmarking question:

| Comparison | What a positive result supports |
|---|---|
| Original-corpus-only PE-RankFormer versus released OptiPrime on the library | Generalization without target-library training labels |
| PE-RankFormer adapted on part of the library versus released OptiPrime on held-out library loci | Added target-library supervision improves the practical tool |
| Later: both methods adapted on identical library training loci and label budgets | A stronger comparison of adaptation procedures; comparator development is deferred during the current standalone-model phase |

Training on a subset changes the library's role: it becomes an adaptation dataset with a held-out evaluation subset. It cannot retain the claim that neither model trained on that library. Training on all its outcomes and evaluating on those same outcomes would not establish predictive superiority.

The entire panel has already informed E11–E24. A partition made now can prevent new training leakage and support retrospective adaptation evaluation, but cannot erase earlier inspection or become a pristine independent confirmation. Preserve the original model comparison in the manuscript and report adaptation as a new result.

Primary success: higher group-mean achieved efficiency @1 than OptiPrime on the identical eligible test candidates, with a paired locus-clustered interval excluding zero and a consistent direction over training seeds. Also report absolute percentage-point gain, regret reduction and candidate-depth strata. A pooled-Spearman win alone does not meet this objective. A loss of pooled prediction accuracy can be acceptable for a dedicated selector, but must be visible; a claim that multitask learning preserves both abilities requires both to be measured.

For practical scale, provisionally target at least a 10% reduction of OptiPrime's regret, rather than merely a nominal p-value. On the old full-panel scale, regret 0.00922 makes that approximately 0.00092 efficiency above OptiPrime, or 0.092 percentage points. This is a proposed development target, not an expected result, and the actual comparator baseline must be recomputed on the new held-out population.

## 2. Partition once before adaptation

Start from the existing reconstructed panel, aggregate repeated measurements into distinct candidates, and retain all eligible decision groups, including all-zero groups in evaluation. Do not expand to the entire raw 288,793-row source until the initial experiment warrants the additional reconstruction and exposure audit.

Build connected locus components across canonical alleles, protospacers and validated equivalent sequence representations. Keep every design, allele variant and repeated measurement from a connected locus in one partition. Stratify approximately by edit class and candidate depth without separating components. Report connected-component counts and size distribution before choosing percentages.

Default development allocation: 60% training, 20% validation, 20% test by components, with realized row/group counts reported. Select architectures, losses, fine-tuning depth and early stopping on validation only. Do not use the final component split to choose among every option in this document.

For the selected recipes, predeclare a five-fold outer evaluation with inner validation in each training fold if a full-library comparison is needed. Every reported candidate prediction then comes from a model that never trained or selected checkpoints on that candidate's locus. Pool these out-of-fold predictions for the full-library adaptation estimate, disclose prior exploratory use, and do not keep optimizing against that pooled estimate. Report fold and seed variation separately; the uncertainty of the whole model-development process is larger than a test-locus bootstrap alone captures.

Learning curves use nested subsets of training components only: approximately 200, 1,000, 5,000 and all available training decision groups, taking whole components. Count both groups and candidate measurements, since labeling a deep group costs more than labeling a pair. These curves test whether adaptation pays with modest data rather than only after absorbing most of the library.

## 3. Options ranked by priority

| Option | Hypothesis and value | Main control / limitation | Priority |
|---|---|---|---|
| Ordinary efficiency fine-tuning | Target labels correct the shifted sequence-to-efficiency mapping | Same pretrained checkpoint, rows, updates and validation rule as selection models | First |
| Frozen encoder plus a small selector | Existing representations contain enough information; only the final ordering needs adaptation | Geometry-only and frozen-score residual models | First |
| Shared encoder, distinct prediction and selection heads | Separate outputs reduce a conflict between broad prediction and nomination | Single selection head and one shared score trained with both losses | First |
| Utility-weighted pairwise or listwise loss | Penalizing consequential top-choice errors transfers better than uniform pair accuracy | Same batches/encoder as ordinary fine-tuning | First, bounded |
| Candidate-set attention | Relationships among candidate representations add useful information | Independent candidate scorer trained with the identical group loss | Second |
| Segment-specific pooling and explicit edit/extension geometry | Preserve PBS/RTT information that a single pooled representation may lose | Same selector and loss with generic pooling or scalar features alone | Second |
| Source replay and small target-specific adapters | Adapt the shifted mapping without excessive forgetting | Target-only fine-tuning with equal updates and trainable capacity reported | First, staged |
| Training on a source/target mixture | Target examples benefit from continued exposure to diverse source examples | Sequential fine-tuning from the same initial checkpoint and matched data exposure | Second |
| Two independent full encoders or a larger backbone | Task separation needs substantial additional capacity | Much higher compute; no current evidence it is needed | Defer |

A geometry-based boosted ranker is a useful low-cost control, even though earlier source-only feature baselines were weak: this experiment supplies different training data and optimizes a different endpoint. [LambdaRank/LambdaMART](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/) provides an established ranking comparator. If using its standard NDCG objective, label it as such; NDCG is not achieved efficiency @1.

## 4. Fine-tuning strategy

Initialize from the actual published ordinal-S4D checkpoint family, with exact checkpoint hashes and source-fold metadata. Do not initialize from arm A merely because it is called the reconstructed original recipe: E24 showed that arm A's performance does not reproduce the published model. During pilot comparisons use the same predetermined starting checkpoint for every arm; select that checkpoint using source-development information, not panel performance. Match ensemble construction for the final comparison.

Use staged unfreezing:

1. Freeze encoders and fit the new or existing output head.
2. Unfreeze pooling/cross-attention and the last sequence blocks with a smaller learning rate than the head.
3. Try full-encoder fine-tuning only if validation gains justify it.

Use group-complete batches with canonical context keys. Keep the row/group sampling schedule identical across objective comparisons. A group that exceeds memory must use a documented subgroup strategy; do not silently treat a partial list as its full candidate set. Evaluate full candidate sets.

Retain an original-corpus replay stream or distillation to the frozen original prediction head to limit forgetting. Compare target-only adaptation against replay for the selected recipe, rather than assuming preservation is free. Source replay must exclude any outer-test locus represented in the original corpus; the current panel exclusions should already make that true, but assert it.

Here any distillation teacher is our own frozen source-trained PE-RankFormer. It supplies a retention constraint on source examples, not OptiPrime information. Keep an unconstrained target-only arm so that preservation is not imposed at the cost of the very selection gain being tested.

Ordinary adaptation should include an efficiency loss, such as MSE or Huber, as a strong baseline. A sufficiently accurate estimate of each candidate's conditional mean efficiency is enough to make the optimal top-one choice; joint candidate scoring is not mathematically required. Do not fill missing indel labels with zero. Keep ordinal thresholds and all standardization constants versioned and fitted on allowed training data only. If re-estimating ordinal thresholds, treat that as a separate intervention because it changes the prediction task.

## 5. What the dual branch should actually do

Use the existing sequence/context encoder to produce one representation h_i per candidate. Feed it into:

- A prediction head retaining the cumulative-threshold objective, with a separately fitted calibrator when an efficiency estimate is needed.
- A selection head producing a scalar s_i used to nominate the best candidate within an allele/context group. Initially use a small MLP on h_i and the candidate's PBS/RTT geometry. A residual formulation initialized to reproduce the original ranking is also worth a bounded comparison.

At inference, select with the selection head; report efficiency with the prediction head. Do not average the two outputs by default: that would partially undo their intended specialization. Report each head's selection and prediction performance so specialization can be inspected.

This differs from the current implementation, which applies a pairwise loss to the primary prediction-derived score. Earlier auxiliary simplex-head experiments also kept the primary head as the inference score. Neither is a direct test of a separate deployment selector.

Train with:

`L = lambda_prediction * L_prediction + lambda_selection * L_selection + lambda_replay * L_replay`

Keep the number of loss weights small. Normalize losses using fixed training-derived scales; a coefficient is not a gradient contribution. Compare gradient magnitudes and directions on the shared encoder. If the objectives interfere, first try a frozen encoder or a small branch-specific adapter. Only then consider an established gradient-conflict method such as [PCGrad](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html), with a matched control.

### Selection losses worth testing

The primary candidate is a smooth expected-regret objective. For measured efficiencies y_i in group g, define:

`p_i = softmax(s_i / T)`

`L_selection(g) = max_i(y_i) - sum_i(p_i * y_i)`

Average over groups, not all candidate pairs. This surrogate weights errors by efficiency consequence and gives constant-outcome groups zero selection gradient. It optimizes a stochastic relaxation; deployment still takes argmax(s), so validation must evaluate that actual decision. Control score scale/temperature and monitor early saturation. Because efficiencies are small, use one fixed training-derived scale to balance this loss against prediction; per-group range normalization would change how much high-consequence decisions matter.

Use a ListNet-style soft-target loss as the main alternative, `q_i = softmax(y_i / T_y)` with cross-entropy against p_i. Select T_y on training/validation, not test. This is an established listwise approach, not a new loss family: [Cao et al., 2007](https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/). Skip constant-outcome groups for this loss, or state explicitly if its uniform target is being used as additional regularization.

A utility-weighted pairwise logistic loss is a simpler fallback. Weight by capped efficiency gap and normalize by sampled-pair count within each group. Do not automatically retain the old 0.02 cutoff: inspect train-only outcome gaps and replication first. Neither hard winner labels nor very low temperatures should turn negligible, noisy differences into large penalties. Where actual independent replicates exist, examine replicate-level robustness; the corpus-wide noise estimate is not a known per-design uncertainty.

Crucially, listwise training does not require set-based inference. The same group loss can train independent scores s_i = f(h_i), which is the necessary control for the set branch.

## 6. Candidate-set attention: a conditional second step

If the independent selector leaves a stable residual gap, allow each candidate representation to attend to the others within its allele/context group. Use a small permutation-equivariant set encoder followed by one score per candidate; do not introduce candidate-order positional embeddings. [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) is an established starting point.

Test permutation equivariance, duplicate removal, candidate subsampling, and the effect of adding an obviously poor candidate. Changes in candidate inventory must not arbitrarily reverse good-versus-good preferences. Train and evaluate the independent and set-based selectors with identical group losses, labels and update budgets.

E23's failed global length sweep does not prove that a set model is necessary or that the physical efficiency of one pegRNA depends on alternatives listed beside it. It only shows that the tested additive correction did not help on development data. Set attention is a hypothesis about statistical prediction, not a biological interaction claim.

## 7. Architecture and domain shift: targeted follow-up experiments

### A. Make candidate geometry accessible to the selector

The current architecture embeds segment identity and absolute position, then pools the pegRNA stream once. It therefore has access to sequence length in principle; the results do not prove that S4D cannot count or that geometry is absent. The testable hypothesis is that explicitly preserving segment information makes the relevant relationships easier to learn under distribution shift.

Compare the following on top of the best independent selector, with the same labels and selection loss:

1. Existing pooled sequence representation, the baseline.
2. Baseline plus PBS/RTT lengths and validated edit-offset/RTT-overhang features through a small nonlinear branch.
3. Separate masked pools for spacer, PBS and RTT, plus the same geometry features, projected back to a matched selector width.

Candidate features may interact nonlinearly with the allele/context representation. Insertion of a fixed global length penalty is not the intended intervention. A feature-only branch already had little benefit on the unadapted panel in E22; adding target supervision and separating segment pools are the specific new hypotheses. Include scalar-only and pooling-only controls if the combined arm wins.

If residual errors concentrate near the nick or RTT boundaries, consider explicit segment-relative positions or a validated edit-to-RTT alignment bias as a later intervention. Reuse the established molecular orientation and reconstruction conventions. Do not assume that reverse-complementing a pegRNA preserves its label or that arbitrary sequence alterations are label-preserving augmentation.

### B. Separate allele/context baseline from candidate differences

An optional structured head can express an efficiency logit as `b(allele, context) + d(candidate, allele, context)`. The common baseline cancels when ranking alternatives within one decision. Train the global output against efficiencies and the design term against within-group contrasts or selection utility; all candidate-dependent information must remain available to the design term. Centering design residuals within training groups can impose an identification convention, but no test outcomes may be used to compute that centering at inference.

This tests whether explicitly allocating capacity to within-edit variation helps. It is a model assumption, not a proven decomposition of prime-editing biology. Build it only if the simpler two-head model suggests that shared global/selection representations interfere. Its matched control is the same-capacity ordinary two-head model.

### C. Locate the adaptation bottleneck

Profile distributions using source training and target training data: geometry ranges, edit classes, efficiencies, zero fractions, and distances in one fixed encoder's representation. Avoid treating a source classifier's success as proof of harmful shift; it may recognize a harmless assay identifier. The decisive evidence is which adaptation changes improve held-out validation decisions.

- If a frozen-encoder head closes the gap, prioritize readout adaptation and data efficiency.
- If final-block updates help substantially more, test a small adapter versus the same blocks fully unfrozen.
- If gains require full-encoder updates, examine whether source retention suffers and whether replay recovers it.
- If source replay improves source metrics but reduces target utility, report the tradeoff and test a modest target-heavy mixture rather than assuming equal source/target weight.

For the selected recipe, compare target-only training and replay fractions of 25% and 50% of updates as a small initial grid. Match the number of target updates when attributing a benefit to replay, and report the additional source compute. Also compare sequential fine-tuning with mixed-source/target continuation from the same initial checkpoint if forgetting remains consequential.

A bounded unlabeled adaptation experiment may later use masked-sequence learning on target-training sequences, followed by identical supervised fitting. Keep target-validation and target-test inputs outside that pretraining in this inductive protocol. This is lower priority than supervised fine-tuning because target labels are available and representation mismatch has not yet been isolated. Generic domain-adversarial alignment is also low priority: matching marginal representations need not correct a changed outcome relationship.

### D. Separate supervision gains from architectural gains

For the baseline architecture and the best modified architecture, run both source-only training and source-plus-target adaptation. The resulting two-by-two comparison asks whether the architecture helps without target labels, whether target labels help without the architecture, and whether they interact. Source-only model decisions must use source validation; they cannot be selected for the cleanest zero-target-label claim using target validation outcomes.

On the adaptation path, inspect learning curves at fixed label budgets and depth strata. Once a winner is selected, add a harder geometry-extrapolation analysis using locus-disjoint partitions defined from input metadata. Specify whether the holdout excludes those geometry ranges from target adaptation only or from all pretraining; the former is not a claim of geometry never seen during training. Keep this secondary analysis distinct from the primary full-coverage test.

## 8. Smallest informative experiment matrix

First compare these four arms with identical data and a predetermined initial checkpoint:

| Arm | Updated components | Objective | Question |
|---|---|---|---|
| P | Head and final encoder blocks | Ordinary efficiency prediction | Does library supervision alone close the gap? |
| S | Same encoder blocks and one selection head | Selection loss | Is the specialized objective useful? |
| M | Same encoder blocks, prediction plus selection heads | Multitask prediction + selection | Does preserving prediction help selection or prevent forgetting? |
| Z | Frozen encoder, small selection head | Same selection loss | Is encoder adaptation needed? |

Add frozen PE-RankFormer and geometry-only adaptation as internal controls. Released OptiPrime supplies a frozen evaluation reference; its scores do not enter any PE-RankFormer inputs, losses, teachers or ensemble. Match extra head capacity in a shared-score control before attributing any M-over-S gain specifically to task separation. A source-only M control tests whether the benefit requires target labels; prioritize it if the goal remains generalization without target adaptation.

Pilot one seed per neural arm, using the same training/validation components. Promote the two most useful candidates and ordinary fine-tuning to three paired seeds. Select on validation achieved @1, and report pooled correlation, source-retention performance and validation trajectory stability. The previous selector null does not justify selecting the new target-library experiment using an unrelated metric, but it also provides no reason to expect a large gain from checkpoint selection alone.

Next compare explicit segment geometry and candidate-set attention separately on the best training recipe. For loss design, compare ordinary efficiency, smooth selection utility, and the selected alternative group loss on the same architecture before combining winning changes. Learning curves and a matched ensemble comparison follow for the selected recipe. Begin with a timed pilot before estimating GPU-hours; cached frozen embeddings make Z cheap, whereas full adaptation and five-fold ensembles require separate budgets.

The execution order is: establish a reproducible starting checkpoint and grouped splits; compare P/S/M/Z; replicate selected arms; isolate one geometry or set interaction change; test replay/adapter choices and label efficiency; freeze and evaluate the standalone model. Report each arm's trainable parameter count, target-label budget, target updates, source replay, wall time and inference cost.

## 9. Interpretation and stopping decisions

- If P beats released OptiPrime and M does not improve on P, use ordinary adaptation and report its sample efficiency. A dual branch is unnecessary for that result.
- If M improves selection over P and S while retaining prediction quality, it supports the multitask design. Replicate that tradeoff across label budgets and held-out loci.
- If an architectural addition does not improve on the same fine-tuning/loss recipe without it, keep the simpler architecture.
- A win against released OptiPrime after target adaptation is a valid practical comparison with unequal target-label exposure; it does not by itself establish architectural superiority. A matched-label comparator can be addressed after standalone PE-RankFormer is validated.
- If no method wins, report uncertainty and the learning curves rather than changing test eligibility or iterating against the test set. The next branch should follow a measurable residual error.

The most promising manuscript extension is a reproducible, data-efficient improvement to standalone PE-RankFormer with clear prediction/selection tradeoffs, supported by matched internal ablations and subsequently stronger comparator benchmarking. A win achieved only by training on the evaluation outcomes would not increase the paper's significance. All work here uses existing computational resources and measured public data. Combining models is deferred until our model has been validated.
