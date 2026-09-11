# Gate A report and go/no-go — explore_v2

Executed 8 September 2026 against `explore_v2_research_plan.md`, section 8A ("experiments I
would run in the next two weeks — current data first") plus the computational parts of
section 8B. Repository commit at start: `83ffae1`. Every number below is produced by a
script in `explore_v2/` that writes its own JSON with git commit, seed and input hashes.
All numbers are from canonicaliser version 2 (`CORRECTIONS.md` C3); the version-1 values
they replace differ in the third decimal and change nothing.

> **Update 8 September 2026 — one conclusion here is now wrong.** The reserved panel has
> been built, validated and scored under its pre-declared analysis: 30,475 fixed-allele
> decision groups, 8,775 at candidate depth five or more, on a surface neither model was
> trained on. **OptiPrime chooses better pegRNAs than PE-RankFormer there** (achieved
> efficiency 0.0369 vs 0.0363, difference −0.00062, 95% CI [−0.00088, −0.00037], p = 0.0005),
> with the deficit growing with candidate depth and with how much the choice is worth — while
> PE-RankFormer keeps the higher pooled correlation (0.709 vs 0.693). Section 1's claim that
> "the margin over OptiPrime roughly doubles" holds only on fold 0's binary choice and does
> **not** generalise. See **`RESULT.md`**, and `e13_regroup_training.md` for a diagnosed
> mechanism inside our own training objective.

> **Corrected 8 September 2026.** A review (`NEXT_STEPS_PLAN.md`) found real defects in E04.
> The clean retest is E09 and the corrections of record are in `CORRECTIONS.md`. Two numbers
> in this document are superseded and one conclusion is withdrawn; both are marked inline
> below. The Gate A verdict is unchanged.

**Nothing was trained.** The frozen ordinal-S4D backbone is used out of fold: round 4's
five checkpoints each held one official fold out, so every row is embedded and scored by
the checkpoint that never saw its fold. Held-out fold 0 was used for audit and descriptive
re-scoring only; every choice that could constitute model selection was made on
development folds.

---

## The verdict in one table

| Gate A criterion (plan section 2) | Result | Where |
|---|---|---|
| Reproducible design-by-context interaction effects exist | **Yes.** 28.8% of the variance of the design contrast, CI [27.9%, 29.8%]; supported order reversals at 4.1% against 0.1% under an additive null; survives the noise estimate being wrong by up to 2.5x | E03 |
| Predicted on held-out loci better than an additive model | **Yes.** R² = +0.184 against 0 for predict-zero, on quartets with one coordinate system and no encoder exposure | E09 |
| Better than a structured shuffle | **On average, and unstably.** +0.184 against +0.077 for a representation-matched permuted-context control, negative on one of five folds. *(E04's "+0.392 against +0.012" compared different feature sets and pooled mixed embedding coordinate systems — see `CORRECTIONS.md` C1.)* | E09 |
| **Practically meaningful** | **No.** Perfect knowledge of another context's true ranking is worth 0.0070 efficiency; two independent attempts to realise any of it deliver at most +0.0023 top-1 accuracy, which a context-free control matches | E03, E05, E07 |

**Gate A passes on the science and fails on the utility.** The interaction is real,
reproducible and predictable. It is also small enough that no method tested here converts
it into a better experimental choice. The plan's instruction — "proceed with an
interaction-centred paper only if reproducible, practically meaningful interaction effects
are predicted on held-out loci" — therefore says: do not centre the paper on the
interaction.

The plan's own fallback is the right move, and it is now on firmer ground than the plan
assumed: *"If the reference-panel hypothesis fails but the corrected benchmark and external
validations remain strong, the defensible paper is a rigorous prime-editing prediction
method with demonstrated practical improvements."* The corrected benchmark does not merely
survive; the margin over OptiPrime **roughly doubles** on the decision a user actually
faces.

---

## 1. The evaluation issue the plan found is real, and worse than described (E01, E02)

The plan noted that the manuscript groups candidates by `spacer` without holding condition
or intended edit fixed. Canonicalising the intended allele shows the size of the problem.

Canonicalisation collapses 130,921 distinct WT windows into **51,766 canonical alleles**:
32,181 target sites carry several WT windows that are one allele written at different RTT
lengths, and 8,024 alleles are reachable by more than one protospacer. Crossing allele with
the nine context fields gives 175,668 decision groups — "one intended allele, one fully
specified experimental context, choose among the alternative pegRNAs".

**The held-out surface cannot support the manuscript's decision metric at its own candidate
depth.**

| surface | rows | decision groups | ≥2 designs | ≥3 | ≥5 |
|---|---:|---:|---:|---:|---:|
| whole corpus | 318,471 | 175,668 | 54,390 | 18,332 | 9,543 |
| **held-out fold 0** | 20,509 | 17,872 | **2,412** | **85** | **0** |
| development fold 1 | 59,581 | 35,457 | 11,559 | 4,633 | 912 |

The manuscript's 735 protospacer groups of five or more rows are a different object: 100%
of them span more than one source/cell/editor condition and 26% contain more than one
intended allele. Ranking inside such a group is mostly *which edit, in which cell line, is
easier* — not *which pegRNA should I order*.

Two further audit findings:

- **The official folds are not locus-disjoint under the canonical keys.** 80.7% of fold-0
  protospacers and 20.3% of fold-0 decision groups also appear in training folds; 24.0% of
  fold-0 rows sit in a decision group the training folds also contain. The manuscript's
  leakage check counted 196 exact design-and-condition twins (1.0% of rows) and concluded
  the issue was negligible. That count was correct and the conclusion was too narrow: the
  larger channel is alternative designs for the same allele at the same site.
- **40% of fold-0's two-candidate decisions are unscoreable ties, all of them both-zero.**
  For those alleles no available design works at all. This belongs in any honest deployment
  claim.

### The corrected numbers

Same frozen predictions, four groupings, replicates averaged first:

| grouping | eligible groups | OptiPrime | ordinal-S4D | final ensemble | Δ |
|---|---:|---:|---:|---:|---:|
| spacer (manuscript) | 670 | 0.5471 | 0.6353 | 0.6357 | +0.0885 |
| spacer × source/cell/editor | 988 | 0.4606 | 0.5845 | 0.5955 | +0.1349 |
| **allele × context, ≥2 designs** | 1,463 | **0.3292** | 0.5063 | 0.5028 | **+0.1735** |

And on the decision fold 0 can actually score — 1,390 scorable binary choices between two
alternative designs for one allele in one context:

| predictor | accuracy | efficiency of the pick | regret |
|---|---:|---:|---:|
| random | 0.500 | 0.1092 | 0.0316 |
| OptiPrime | 0.657 | 0.1252 | 0.0156 |
| **ordinal-S4D member** | **0.750** | 0.1340 | 0.0068 |
| final ensemble | 0.747 | 0.1343 | 0.0065 |
| oracle | 1.000 | 0.1408 | 0 |

Paired, protospacer-clustered: **+0.0906 accuracy [+0.0603, +0.1230]** and **+0.0091
efficiency [+0.0065, +0.0127]**. PE-RankFormer removes **79%** of the regret a random pick
incurs; OptiPrime removes 51%.

Three things follow that the manuscript should say:

1. The advantage is **larger** on the corrected estimand, not smaller, and it is larger on
   the decision groups training never saw (+0.111 accuracy) than on those it did (+0.070).
   Leakage makes the absolute numbers optimistic; it does not manufacture the margin.
2. **The single ordinal-S4D member matches the five-member ensemble on the decision**
   (0.750 versus 0.747 accuracy; regret 0.0068 versus 0.0065). The ensemble buys pooled
   Spearman, not decisions. The plan's instinct to keep the single model as the backbone is
   supported.
3. The advantage concentrates where it matters: on choices whose two designs differ by more
   than 0.20 efficiency, the S4D member is right 92.4% of the time against OptiPrime's
   72.7%. OptiPrime falls **below chance** on Kim A549/PE4 (0.452).

---

## 2. The interaction is real and is worth about 0.007 efficiency (E03)

111,862 quartets — two alternative designs for one allele, both measured in both contexts —
over 10,487 alleles, 10,136 target sites and 128 ordered context pairs. Analysis on the
arcsine-root scale, which the 654 metadata-identical replicate groups confirm is
variance-stabilising here.

| component of the design contrast | SD | share of variance |
|---|---:|---:|
| design main effect (context-independent) | 0.1890 | 66.0% |
| **design × context interaction** | **0.1250** | **28.8%** |
| measurement noise | 0.0535 | 5.2% |

Under any additive latent model the quartet contrast D is exactly measurement error, so its
SD would be 2σ = 0.0757; the observed SD is 0.1997. Restricting to quartets where both
design contrasts exceed two standard errors, 4.1% reverse order against 0.1% expected from
noise. The interaction estimate would only vanish if the replicate groups understate the
per-measurement SD by a factor of **2.54**.

So the round-6 diagnosis was right that something real is there. What was missing is its
price:

> Order the design that measured better in one context and evaluate it in the other. On the
> 25,961 quartets where both contrasts are statistically supported, this context-blind
> transfer achieves 0.2412 against an oracle's 0.2482 — a **transfer regret of 0.0070
> efficiency, CI [0.0060, 0.0081]**, which is **92.5%** of the way from a random pick to
> the oracle.

That is the ceiling for any context-adaptation method on this data, and it requires the
true ranking in another context, which is more information than any adaptation method gets.
For scale: the whole random-to-oracle gap on fold 0's pairwise decision is 0.0316, and
PE-RankFormer's entire advantage over OptiPrime is 0.0091. **Perfect context awareness is
worth about the same as the existing architecture margin, and about a quarter of the total
decision.**

The interaction is largest for a cell-line change (SD 0.1385) but reverses order most often
for an editor change, PE2 versus PE4 (5.3% of supported quartets against 2.8%).

---

## 3. The interaction carries some predictable structure on held-out loci (E04, corrected by E09)

> **Superseded in part.** The table and the round-6 paragraph below are E04 as published.
> E04 subtracted embeddings from different networks in 57,821 of 110,321 contrasts, ran its
> shuffles on a different feature set from its headline, and fitted its PCA before splitting.
> E09 repairs all three; the corrected figures are +0.184 (range [−0.056, +0.299]) for
> embedding × context against **+0.077** for a representation-matched shuffle and +0.066 for
> the same features with no context term at all. See `CORRECTIONS.md` C1 and C2.

Target: the quartet contrast D. Model class chosen to be antisymmetric in both the design
swap and the context swap by construction, so a context-free solution is inexpressible.
Locus-grouped 5-fold, where a locus is a connected component of the (target site, canonical
allele) graph; ridge penalty chosen inside each training split.

| model | Spearman | Pearson | R² vs predict-zero |
|---|---:|---:|---:|
| additive (predict zero) | — | — | 0 |
| 17 interpretable features × context | 0.311 | 0.324 | +0.105 |
| **frozen S4D representation × context** | **0.489** | **0.626** | **+0.392** |
| design features, no context term | 0.072 | 0.045 | +0.003 |
| context labels permuted within locus | 0.093 | 0.105 | +0.012 |
| D permuted within context pair | 0.066 | 0.086 | +0.007 |

An injected synthetic interaction is recovered down to 5% of SD(D), so the controls' near-zero
scores are a real null and not a power failure.

The signal is concentrated, and selectable in advance. Sign agreement for the frozen-
representation model, on held-out loci:

| subset, ranked by **predicted** \|D\| | sign accuracy |
|---|---:|
| all quartets | 0.549 |
| top 50% | 0.701 |
| top 19% | 0.848 |
| top 9% | 0.917 |
| **top 5%** | **0.938** |

Round 6 concluded: *"What distinguishes A549 from DLD1 for a given design ... is not in the
input. The model receives a categorical label, and no objective over a categorical label can
recover biology the label does not encode."*

**I withdraw the claim that this is wrong as stated.** After the E09 repairs, what survives
is narrower: the categorical label plus a frozen representation carries *some* predictable
interaction structure beyond a context-free baseline — +0.184 against +0.077 — but it is
unstable across folds and roughly 40% of what E04 attributed to context is design-pair
structure needing no context at all. And the stratum the model flags as high-interaction
contains almost no actual order reversals (0.000–0.057 across folds), so "predicting the
interaction" and "predicting that the better design changes" are not the same skill. What
failed in rounds 6, 7 and 9 was the conversion into a better score; the identifiability is
real but smaller and shakier than E04 implied.

---

## 4. Two ways of spending the predictability, both of which fail (E05, E07)

### Reference-panel adaptation in an unseen context (E05) — negative

The plan's priority 1: `eta(d,c) = f_theta(d) + a_c + u_theta(d)^T z_c`, with `f_theta`
frozen, `a_c` and `z_c` estimated from a measured reference panel inside the target context.
Leave-one-context-out over 20 contexts, support and query alleles disjoint, budgets 0/12/24/
48/96/192 measured designs, five support draws each, every arm given the same support
labels, penalty and rank chosen on source contexts only.

No arm beats no adaptation. The intercept-only arm comes out at exactly +0.0000 at every
budget, which is the required arithmetic: a monotone map of the score cannot reorder
candidates inside a decision group. **A calibration success is not a design-transfer
success, and this experiment separates them cleanly.** The affine arm is exactly zero too,
except at the smallest budget, where a fitted slope came out negative and inverted the
ranking (−0.0068 averaged over eighty episodes). Even a two-parameter recalibration can damage a ranking it was meant to leave
alone when the support set is that small.

Arms fitted on the raw representation get monotonically worse as the budget grows
(residual ridge: −0.0186 accuracy at 12 measurements, −0.0258 at 192), which is what
over-fitting a correction to a small support set looks like. Arms fitted in the
allele-centred (design-contrast) subspace — the subspace the interaction has to occupy, so
the one that removes the most obvious representational excuse for a null — land on zero
instead: `within_ridge` runs from −0.0035 to −0.0255 accuracy across budgets, and its own
shuffled-label twin `within_ridge_shuffled` is no worse (−0.0024 to −0.0058) — the arm given
real support labels is in fact the one that degrades faster as the budget grows.
**Support labels drawn from the target context carry no more design information than the
same labels permuted.** That is the same signature that killed the round-6 shift loss.

| arm | B=12 | B=24 | B=48 | B=96 | B=192 |
|---|---:|---:|---:|---:|---:|
| within-subspace ridge, real support labels | −0.0035 | −0.0098 | −0.0175 | −0.0166 | −0.0255 |
| within-subspace ridge, **shuffled** labels | −0.0024 | −0.0012 | −0.0040 | −0.0043 | −0.0058 |
| raw-representation residual ridge | −0.0186 | −0.0250 | −0.0293 | −0.0247 | −0.0258 |

Paired change in top-1 accuracy against no adaptation, mean over 80 episodes. Full table in
`e05_reference_panel.md`.

### An explicit context correction for contexts that *are* in training (E07) — negative in the way that matters

51,895 decision groups over 24,170 locus groups and 16 contexts, locus-grouped 5-fold. The
correction is fitted on within-group deviations, in the allele-centred representation, with
the centred base score carried as its own covariate so the correction cannot smuggle in a
rescaling of the frozen model. Its blend weight and its selectivity are both chosen on
training loci by the decision metric, and zero recovers the frozen model exactly.

| score | top-1 accuracy | Δ vs frozen | 95% CI | Δ regret |
|---|---:|---:|---|---:|
| frozen base | 0.6702 | — | — | — |
| design correction, **no context** | 0.6718 | +0.0016 | [+0.0004, +0.0029] | −0.00029 |
| **design × context** | 0.6725 | +0.0023 | [+0.0010, +0.0038] | −0.00038 |
| design × context, **labels shuffled** | 0.6721 | +0.0019 | [+0.0008, +0.0029] | −0.00023 |

The residual prediction behaves exactly as E04 predicts — R² +0.103 with context against
+0.046 without and +0.040 shuffled — and the decision does not care. The context term's
incremental value over a context-free correction is +0.0007 accuracy, which the shuffled
control matches. Selecting only the groups with a large predicted reordering was allowed and
the training loci chose to touch everything, so the E04 concentration does not rescue it
either.

**Why both fail, quantitatively.** The design main effect carries 66% of the contrast
variance and the frozen model already captures most of it; the interaction carries 28%, and
its entire realisable value is 0.0070 efficiency. A correction has to be estimated with
error smaller than that, and neither 192 support measurements in a new context nor a
32-dimensional bilinear map over 150,000 training rows manages it.

---

## 5. Deliverables produced

| plan deliverable (section 8A) | artefact |
|---|---|
| canonical decision manifest | `cache/decision_manifest.parquet`, 318,471 rows with allele/design/context/decision/replicate keys; `e01_decision_manifest.{json,md}` |
| corrected metric report | `e02_corrected_metrics.{json,md}`, `e02_pairwise_table.csv` |
| context-overlap and identifiability audit | `e01` (fold structure), `e03` (variance components), `e04` (predictive identifiability) |
| reference-budget curves | `e05_reference_panel.{json,md}` |
| external-data eligibility manifest | `e06_external_eligibility.{json,md}` |
| go/no-go | this document |
| pilot power from grouped residuals (8B) | `e08_pilot_power.{json,md}` |
| — additionally — | `e07_context_correction.{json,md}`; `reserved_panel_kim_large.parquet` + `.sha256`; `PREREGISTRATION_reserved_panel.md` |

### The candidate-depth problem has a solution already on disk (E06)

`external/deepprime/data/DeepPrime_dataset_final_Feat8.csv` is Kim's large variant library,
288,793 measured pegRNAs at 58,217 target windows, and it is **not** in OptiPrime's training
mix, which took only Kim's `LibSmall` files. Its released `Edited74_On` column is masked
outside the RT-PBS footprint, so the raw strings cannot identify an allele; aligning the
unmasked block back onto the WT window with each row's own edit type recovers it. Removing
every row that shares a protospacer or a canonical allele with the training corpus leaves
222,488 rows in **32,925 decision groups with two or more candidates, 23,345 of them with
five or more** — against fold 0's 2,412 and zero.

It is the same laboratory, library chemistry and readout, so it cannot support a
transportability claim. It can support the fixed-allele selection estimand at real candidate
depth, which nothing else available can. It has been written, hashed and **left unscored**,
with the analysis declared in advance in `PREREGISTRATION_reserved_panel.md`.

### Power (E08)

The plan's illustrative paired SD of 0.08 can be replaced with a measurement: the per-edit
paired difference between PE-RankFormer and OptiPrime in achieved efficiency has SD
**0.0497** (the two methods nominate the same design in 71% of groups, where the difference
is exactly zero). That gives 48 edits for 80% power at an 0.02 absolute effect and 12 edits
at 0.04 — so the plan's 40-edit design is comfortable at 0.04 and thin at 0.02, before
clustering and attrition. Measurement error contributes 0.0265 per observation, falling to
0.0153 at three replicates; the edit-to-edit spread does not fall with replication.

---

## 6. What I would change in the plan

**Promote priority 3 to the spine of the paper — and accept what it found.** It was ranked
third for execution reasons and it is the workstream that produced the decisive result. The
reserved Kim large-library panel made the estimand testable at candidate depth five and
above, which fold 0 never could, and the answer went against the model: OptiPrime's single
nominated candidate delivers 0.0369 achieved efficiency against PE-RankFormer's 0.0363
(p = 0.0005), and the gap widens with depth (−0.00249 at depth eight or more) and with the
value of the decision (−0.00329 where perfect selection is worth ≥0.05). Both models beat
random by a wide margin — random needs four candidates to match either model's first pick —
and at a three-candidate budget they are indistinguishable. **The manuscript can claim a
pooled-correlation advantage; it cannot claim a deployment advantage over OptiPrime.** The manuscript's headline should become the fixed-allele decision, with the
pooled 0.9079 demoted to a historical benchmark exactly as the plan proposes.

**Demote priority 1.** Reference-edit panels do not beat an equally informed no-adaptation
baseline at 12-192 measurements with a frozen encoder, and the shuffled-support control
matches the real one. If it is revisited, the bar must be *beating a context-free design
correction*, not beating no adaptation, and the ceiling to aim at is 0.0070 efficiency.
Full-backbone fine-tuning at these budgets is the one variant not yet tested and is worth
one clean run, not a programme.

**Keep the interaction as a characterisation result, not a capability claim.** E03 and E04
together are a genuinely new, well-controlled quantitative finding: the design-by-context
interaction is 28.8% of the design-contrast variance, its direction is predictable on unseen
loci at 94% accuracy for the top 5% by predicted magnitude, and its total worth to a user is
0.0070 efficiency. Published as such — with the two failed conversions — it corrects the
literature's implicit assumption that context-dependent design preference is a large
untapped gain, and it corrects this project's own round-6 conclusion that the context labels
are the binding constraint.

**~~Re-aim priority 2's wet-lab pilot by enriching for high predicted |D|.~~ Withdrawn.**
The supported reversal rate is 4.1%, so an unselected crossed panel does spend most of its
wells on quartets that will not reorder. But E09 shows the high-predicted-|D| stratum
contains essentially *no* order reversals (0.000–0.057 across folds): predicted |D| ranks
margin changes, not changes of choice. Enriching on it would have been enriching for the
wrong event. Predicting reversal probability is a separate model that has not been built
(`CORRECTIONS.md` C2).

**Deprioritise priority 5 further.** Measurement noise is 5.2% of the variance of the design
contrast that selection depends on. An observation model cannot be worth much for ranking
when the quantity it cleans up is that small a share of the signal, and E03's own numbers
now bound what it could deliver.

**One correction to the plan's premises.** It reports the manuscript's grouping as giving
0.5942/0.4597 over 997 spacer-and-condition groups; with replicates averaged and alleles
canonicalised I get 0.5955/0.4606 over 988. Same conclusion, and the plan's reading of the
issue was right. It also cites 649 metadata-identical replicate groups; the nine-field
context key gives 654, of which 649 are Kim — which is why every noise-derived number here
is explicitly a Kim-calibrated one applied elsewhere for want of anything better.

---

## 7. Honest limitations of this program

- **No new model was trained.** Every negative result is a negative about a frozen encoder
  plus a linear adaptation, at the budgets and ranks tested. Full fine-tuning, a jointly
  trained support encoder, and non-linear context maps remain untested here.
- **The noise model rests on 654 replicate groups, 99% of them Kim.** Every variance
  decomposition and every "supported" filter inherits that. E03 reports the multiple of σ at
  which its conclusion would flip (2.54x) rather than asserting the estimate is right.
- **`best.pt` checkpoints were selected on the fold they hold out**, so epoch choice saw one
  scalar per epoch from the embedded fold. This is small but it is not zero.
- **Fold 0 was read repeatedly.** E01, E02 and E08 are descriptive re-scorings of frozen
  predictions with no fitting, but the surface has now been examined many times across nine
  rounds and cannot carry a confirmatory claim. That is the reason the Kim large-library
  panel was sealed rather than used.
- **The canonical allele key is a stand-in for genomic coordinates.** It is 12 bp of WT
  context on each side plus the allele, strand-normalised, with flanks read off the site's
  longest observed window; 89% of rows get the full flank on both sides. Where the stored
  window truncates the edit itself, the recovered allele is the truncated one.
- **No external, independent-laboratory data were acquired.** The acquisition queue in E06
  is a plan, not a result, and published library sizes there are not counts of usable rows.
