
u are acting as a senior machine-learning researcher, computational biologist,
and Nature Methods reviewer.

I have an existing research project on prime-editing efficiency prediction,
PE-RankFormer. Read the attached manuscript/research report carefully before
making recommendations.

The current work is already strong:

- PE-RankFormer substantially outperforms OptiPrime under matched data, splits,
  and evaluation.
- The best model uses:
    * paired wild-type / edited-sequence tokenization,
    * a segment-aware pegRNA encoder,
    * bidirectional cross-attention,
    * experimental-context conditioning,
    * bidirectional S4D state-space sequence mixing,
    * ordinal/rank-based supervision.
- The S4D architecture is a larger contributor than the ordinal objective.
- The best single ordinal-S4D model essentially matches the ensemble.
- A large number of straightforward architecture, loss, capacity, domain-
  weighting, and ensemble modifications have already been tested and many did
  not improve performance.
- Replicate analysis nevertheless suggests substantial genuine remaining
  predictive headroom.
- The current diagnostic suggests the major unresolved problem is
  design × experimental-context interaction: the model learns rankings that
  are too invariant across cell types / editing systems.
- Current experimental context is represented mainly through categorical
  metadata.
- External validation on genuinely independent datasets has not yet been done.

My goal is NOT simply to perform another broad hyperparameter sweep.

The goal is to identify the strongest research directions that could produce a
substantial methodological advance and potentially elevate the work to the
level of Nature Methods.

Please independently assess the ideas below. Do not assume they are correct.
Reject ideas that are unlikely to work, refine promising ones, and propose
better ideas where appropriate.


============================================================
PART I. DIAGNOSE THE CURRENT MODEL
============================================================

Begin by analyzing the manuscript and answering:

1. What types of error remain?
2. Which remaining errors appear reducible?
3. What evidence suggests the bottleneck is:
   - representation,
   - optimization,
   - label noise,
   - missing biological covariates,
   - domain shift,
   - inadequate sequence-context interaction,
   - inadequate objective,
   - or something else?

Distinguish these carefully.

In particular, investigate the observation that the model seems to preserve
design ranking too strongly across experimental contexts.

Ask whether this implies that the appropriate next problem is not simply

    efficiency = f(sequence, categorical_context)

but instead something closer to

    efficiency = f(sequence,
                   edit geometry,
                   pegRNA,
                   molecular cell state,
                   editor configuration,
                   sequence × context interactions).

Determine what experiments would falsify this hypothesis.


============================================================
PART II. RICH BIOLOGICAL CONTEXT REPRESENTATIONS
============================================================

This should receive especially serious consideration.

Current categorical labels such as "HEK293T" or "A549" tell the model very
little about why editing differs between those cells.

Explore replacing or augmenting categorical labels with biological context
vectors representing, when obtainable:

- mismatch-repair state,
- DNA repair pathway activity,
- MMR genes such as MLH1/MSH2/MSH6/PMS2,
- DNA damage response,
- replication state,
- chromatin accessibility,
- local genomic chromatin state,
- DNA methylation,
- transcriptional activity,
- cell-cycle state,
- expression of prime-editing machinery,
- Cas9/RT expression or delivery level,
- cell-line transcriptomic profiles,
- perturbations such as MLH1dn,
- PE2 / PE3 / PE4 / PE5 biology.

Investigate public resources such as:

- CCLE / DepMap,
- ENCODE,
- Roadmap epigenomics where applicable,
- ATAC-seq / DNase-seq,
- RNA-seq,
- cell-line-specific chromatin annotations.

Ask whether every experimental condition can be represented by a continuous
molecular-state embedding rather than an arbitrary category ID.

Develop candidate models for:

    sequence/design representation
          ×
    molecular-context representation

using explicit interaction mechanisms rather than simple FiLM.

Candidates might include:

- bilinear interaction layers,
- low-rank multiplicative interactions,
- cross-attention between molecular context and sequence,
- context-conditioned SSM parameters,
- context-conditioned kernels,
- hypernetworks,
- conditional adapters,
- conditional LoRA,
- mixture-of-experts with biological gating,
- latent-factor models,
- tensor-product representations.

Give special attention to mechanisms capable of actually REORDERING pegRNA
designs across contexts rather than merely shifting their average efficiency.


============================================================
PART III. CONDITIONAL / CONTEXT-AWARE RANKING
============================================================

The current ordinal head is well matched to global Spearman correlation.

Ask whether the more scientifically meaningful target is instead

    P(y_i > y_j | context)

or a context-conditioned latent utility

    s(design, context).

Develop principled approaches that allow ranking to change across context.

Consider:

1. conditional ordinal regression;
2. hierarchical ordinal models;
3. listwise ranking within experimental conditions;
4. context-conditioned pairwise ranking;
5. differentiable rank-correlation objectives;
6. multi-level ranking:
       global ranking
       + within-cell ranking
       + within-editor ranking;
7. partially pooled hierarchical ranking;
8. learning a global design score plus a context-specific residual:
       s(d,c) = s_global(d) + delta(d,c).

Do NOT simply retry the pairwise ranking loss already reported as unsuccessful.

Explain why any new ranking formulation is fundamentally different from the
failed experiment.

Design diagnostics that directly quantify whether a method recovers
context-specific rank reversals.


============================================================
PART IV. LABEL NOISE AND REPLICATE-AWARE TRAINING
============================================================

The manuscript contains valuable replicate information that is currently used
mainly to estimate a performance ceiling.

Explore using replicates during training.

For example:

- estimate heteroscedastic measurement uncertainty;
- identify low-confidence zero measurements;
- infer latent editing efficiency from repeated measurements;
- train against posterior/latent efficiencies rather than raw observations;
- uncertainty-weight individual observations;
- use censoring-aware likelihoods for exact-zero measurements;
- explicitly model detection limits;
- learn aleatoric uncertainty;
- construct replicate-consistency objectives.

Investigate models such as:

    y_observed = y_true + epsilon(x, context)

or bounded / zero-censored alternatives appropriate for efficiency values.

Ask whether part of the apparent 0.10–0.12 headroom could be recovered by
denoising training targets.

Design experiments comparing:

    raw labels
    vs replicate-denoised labels
    vs reliability-weighted training.

Avoid assumptions that conflict with the empirical replicate distribution.


============================================================
PART V. PRETRAINING AND REPRESENTATION LEARNING
============================================================

Investigate whether PE-RankFormer is currently learning sequence grammar from
only ~300k supervised observations when much larger unlabeled sequence resources
could be exploited.

Evaluate several distinct strategies.

A. Domain-specific self-supervised pretraining

Pretrain the paired sequence encoder using tasks such as:

- masked nucleotide prediction,
- masked paired-token prediction,
- reconstruction of edited sequence from WT + edit description,
- edit localization,
- spacer-target alignment,
- PBS-target complementarity,
- RTT alignment,
- reverse-transcription reach prediction.

B. Contrastive pretraining

Construct positive pairs representing:

- same genomic target,
- related pegRNAs,
- same edit under different conditions,
- same construct across replicates.

Develop carefully chosen negative pairs.

C. Genomic foundation-model initialization

Evaluate whether pretrained DNA encoders such as current genomic foundation
models can provide useful representations.

Do NOT assume that generic DNA language-model embeddings will help short,
engineered pegRNA sequences.

Formulate experiments that test this directly.

D. Prime-editing-specific pretraining

Ask whether millions of synthetically constructible sequence/edit combinations
can be used for self-supervised structural tasks without inventing efficiency
labels.


============================================================
PART VI. MORE EXPRESSIVE SEQUENCE MODELS
============================================================

The current S4D result is scientifically interesting because state-space
mixing beats self-attention even for short sequences.

Rather than indiscriminately testing larger models, identify architectures
whose inductive bias directly matches prime editing.

Explore, among others:

1. Selective state-space models / modern Mamba variants.

   Would input-dependent state transitions provide an advantage over S4D's
   content-independent convolution?

2. Sparse attention + SSM hybrids.

   The manuscript reports that a naive 1:1 S4D/attention hybrid was harmful.
   Explore whether sparse placement of attention, as in successful hybrid
   architectures, could capture a small number of sequence-specific long-range
   interactions without destroying the SSM inductive bias.

3. Context-conditioned SSMs.

   Make the state-space dynamics depend on:
       edit type,
       PBS length,
       RTT length,
       cell state,
       editor.

4. Relative-position / geometry-aware models.

   Explicitly encode biologically meaningful distances:
       nick → edit,
       edit → PAM,
       PBS boundary,
       RTT boundary,
       mismatch position,
       RT termination location.

5. Dual-coordinate sequence representations.

   Represent both:
       nucleotide sequence
   and
       coordinates relative to nick / edit / PBS / RTT.

Ask whether S4D's success is actually evidence that relative geometry is the
missing inductive bias.

Design ablations capable of demonstrating this mechanistically.


============================================================
PART VII. MULTI-TASK LEARNING
============================================================

Editing efficiency is only one observed outcome.

Investigate whether auxiliary biological outputs can improve the learned
representation:

- intended editing rate,
- indel rate,
- unedited fraction,
- product purity,
- by-product profiles,
- editing failure,
- pegRNA stability if available.

The existing simplex formulation already uses some of this information, so
propose multi-task objectives that add genuinely new information rather than
simply recreating the existing simplex head.

Consider shared representation + outcome-specific heads and uncertainty-
weighted multi-task objectives.

Ask whether predicting the entire editing outcome distribution is a better
scientific problem than predicting one scalar.


============================================================
PART VIII. BIOPHYSICAL AND MECHANISTIC FEATURES
============================================================

The current paper makes an interesting point that a minimally mechanistic model
can outperform a heavily engineered mechanistic model.

Do not destroy that contribution by simply copying OptiPrime's feature set.

Instead ask whether a small number of universally meaningful physical
quantities can complement the learned representation.

Examples:

- RNA/DNA hybrid thermodynamics,
- PBS melting energy,
- RTT secondary structure,
- pegRNA secondary structure,
- GC content,
- local folding energies,
- edit/nick distances,
- PAM geometry,
- sequence accessibility.

Test whether these provide information not recoverable from sequence at the
available sample size.

A particularly interesting model would be:

    learned sequence representation
          +
    minimal physical measurements
          +
    molecular cellular context

rather than a hand-engineered reaction simulator.


============================================================
PART IX. DOMAIN GENERALIZATION
============================================================

A Nature Methods-level claim should be much stronger than performance on one
random held-out partition.

Construct harder tests such as:

- leave-one-cell-line-out;
- leave-one-PE-system-out;
- leave-one-editor-out;
- leave-one-study-out;
- leave-one-edit-type-out;
- leave-one-genomic-locus-family-out;
- train on old datasets and test chronologically on a newly published dataset.

Measure both:

    in-domain interpolation
and
    out-of-domain generalization.

Determine whether context-aware models improve the second even if the first
changes only modestly.

This may be scientifically more valuable than another +0.005 on the existing
benchmark.


============================================================
PART X. EXTERNAL AND PROSPECTIVE VALIDATION
============================================================

Search for all independent prime-editing datasets that could provide genuine
external validation.

For every candidate dataset determine:

- number of observations;
- cell types;
- PE systems;
- overlap with training data;
- available sequence fields;
- compatibility with current inputs;
- whether raw efficiency measurements are available.

Prioritize datasets that are genuinely out-of-distribution.

Also design a prospective wet-lab validation study if experimentally feasible.

For example:

1. Choose one or more contexts unseen during training.
2. Generate a large candidate pegRNA set.
3. Rank with:
       PE-RankFormer,
       OptiPrime,
       DeepPrime,
       PRIDICT,
       other relevant methods.
4. Experimentally test:
       predicted high,
       intermediate,
       low,
       and model-disagreement designs.
5. Measure:
       Spearman,
       top-k enrichment,
       success rate,
       calibration,
       regret relative to the experimentally best design.

A particularly persuasive validation would demonstrate that the model selects
better pegRNAs in a NEW biological context rather than merely predicting an
existing benchmark.


============================================================
PART XI. DECISION-FOCUSED EVALUATION
============================================================

Spearman is useful but pegRNA design is a selection problem.

Develop metrics that directly answer:

    "If a scientist can test only k pegRNAs, how much does the model help?"

Evaluate:

- Precision@k;
- enrichment among top 1%, 5%, 10%;
- expected efficiency of the selected top-k;
- normalized regret;
- probability that at least one selected pegRNA exceeds a desired threshold;
- calibration of success probability.

Determine whether PE-RankFormer provides substantially larger practical gains
than suggested by the difference in Spearman correlation.

This could strengthen the Methods story considerably.


============================================================
PART XII. UNCERTAINTY-AWARE DESIGN
============================================================

Explore whether prediction uncertainty can be made useful.

Compare:

- deep ensembles;
- MC dropout;
- heteroscedastic output models;
- conformal prediction;
- replicate-informed uncertainty estimation.

Evaluate whether uncertainty predicts model error under domain shift.

If so, explore selective prediction:

    predict only when confident

and active-learning strategies:

    which pegRNA measurements would most improve the model?


============================================================
PART XIII. DATA-CENTRIC IMPROVEMENTS
============================================================

Audit the entire dataset rather than assuming architecture is the limiting
factor.

Investigate:

- inconsistent normalization across studies;
- duplicated constructs;
- near duplicates;
- target leakage;
- assay-specific biases;
- measurement floors;
- different definitions of zero;
- batch effects;
- study-specific normalization;
- missing covariates;
- inconsistent metadata.

Ask whether harmonizing labels across studies could produce a larger gain than
another architectural change.

Develop a quantitative model of study/batch effects where possible.


============================================================
PART XIV. ANALYZE FAILED EXPERIMENTS
============================================================

The project contains approximately seventy controlled negative experiments.

Do not treat these merely as failures.

Cluster them according to the hypothesis they tested.

Infer what classes of explanations have effectively been ruled out.

For example:

- more generic capacity;
- naive mixture-of-experts;
- naive source weighting;
- naive context FiLM;
- additional ordinal variants;
- naive zero-inflation modeling;
- naive S4D-attention hybrids.

Use these negative results to constrain the next hypothesis space.

Identify what fundamentally NEW information or inductive bias each proposed
next experiment adds.


============================================================
PART XV. LOOK FOR A GENERAL METHODOLOGICAL CONTRIBUTION
============================================================

Nature Methods significance should preferably extend beyond prime editing.

Ask whether the project can establish one or more general ideas such as:

A. Metric-matched ordinal learning for heavily zero-inflated biological
   screening data.

B. State-space sequence models as a better inductive bias than Transformers
   for short position-governed biological sequences.

C. Learning biological intervention outcome as a conditional ranking problem.

D. Molecular-context embeddings for transferring experimental predictors
   across cell states.

E. Replicate-aware latent-target learning for noisy functional-genomics assays.

For each candidate, determine what additional benchmark(s) would be necessary
to show that the principle generalizes beyond this one PE dataset.

For instance, can the same methodological principle improve:

- CRISPR guide efficiency prediction,
- base editing,
- CRISPRi/a,
- MPRA,
- protein variant functional assays,
- other sequence-to-function tasks?

Do not broaden the paper gratuitously. Determine which generalization is
scientifically coherent.


============================================================
PART XVI. PRODUCE A PRIORITIZED RESEARCH ROADMAP
============================================================

Do not return a long unordered collection of ideas.

Rank proposed directions by:

1. expected scientific importance;
2. expected performance improvement;
3. probability of success;
4. novelty;
5. ability to explain the current residual errors;
6. value for a Nature Methods submission;
7. computational cost;
8. wet-lab/data requirements.

Create a table:

Direction
Core hypothesis
Why current model misses it
Specific implementation
Critical experiment
Expected outcome if hypothesis is correct
Negative result interpretation
Estimated effort
Nature Methods value
Priority

Then choose only the TOP 5 research directions.

For each of the top five provide:

- precise hypothesis;
- model modification;
- loss;
- required inputs;
- training protocol;
- ablations;
- evaluation;
- expected numerical gain;
- scientific conclusion if successful;
- scientific conclusion if unsuccessful.

Separate experiments into:

TIER 1 — immediately executable with current data
TIER 2 — requires public auxiliary data
TIER 3 — requires new experimental measurements


============================================================
PART XVII. DEFINE A "NATURE METHODS VERSION" OF THE PAPER
============================================================

Finally describe what the strongest realistic final paper would look like.

Give:

1. proposed title;
2. one-sentence central claim;
3. major methodological innovation;
4. main biological insight;
5. datasets;
6. external validation;
7. prospective experiment;
8. core figures;
9. essential ablation experiments;
10. comparison methods;
11. generalization experiments;
12. likely reviewer criticisms;
13. experiments needed to neutralize each criticism.

Be demanding.

Do not assume that improving benchmark Spearman alone is sufficient for
Nature Methods.

Explicitly tell me which experiments would merely produce incremental
engineering improvements and which could materially change the publication
level of the work.

At the very end provide:

A. "Experiments I would run in the next 2 weeks"
B. "Experiments I would run in the next 2 months"
C. "Experiments required before Nature Methods submission"
D. "Experiments I would NOT spend time on"
