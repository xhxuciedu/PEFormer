> Historical record, archived 10 September 2026. Adaptation interpretations are
> superseded by ../SUMMARY_REPORT.md and ../ADAPTATION_FOLLOWUP_RESULTS.md.
> Relative links below refer to the original explore_v2 location.

# PE-RankFormer: research report, explore_v2 programme

**Status 2026-09-09 23:10.** Twenty-four experiments (E01–E24). One confirmatory test on a
reserved panel, pre-registered before unblinding; everything after it is development
work and labelled as such. One test is still running and is marked pending in §7.

Every number below is reproducible from `explore_v2/e*.py`, is written to a JSON artifact
beside its script, and is checked by `explore_v2/verify_report.py` (210 claims, all passing).
Endpoint invariants are pinned by 13 tests in `explore_v2/test_endpoints.py`.

---

## 1. Verdict in one page

**The manuscript's headline claim survives, on a second surface it did not previously have.**
PE-RankFormer's pooled rank correlation beats OptiPrime on held-out fold 0 (+0.0389) *and* on
a 288,793-pegRNA library neither model was trained on (+0.0159), both with intervals excluding
zero. The frozen isotonic calibrator transfers to that unseen library, lifting Pearson by
+0.0495. This claim is now stronger than when the programme started.

**A second claim the manuscript makes does not survive, and must be withdrawn.** On the actual
deployment task — choose one pegRNA for an allele you have already decided to install, in the
cell line and editor you already work in — OptiPrime obtains more efficiency from its single
nominated design than PE-RankFormer does: −0.00064, CI [−0.00089, −0.00038], p = 0.001. The
deficit widens with candidate depth (−0.00249 at depth ≥8) and with how much the choice is
worth (−0.00329 where the decision is worth ≥0.05).

**Both statements are true at once because they are different estimands.** Pooled correlation
asks how well the whole list is ordered and mixes "which locus is easy" with "which design is
best". The fixed-allele decision isolates the second. They rank the two models in *opposite*
order at the margin, and E18 shows the same reversal inside our own ablations: the ordinal head
is better on the decision (p = 0.009) while pooled ρ prefers the simplex head.

**The size of the disagreement is small and should be stated as such.** A random pick achieves
0.0268 and a perfect chooser 0.0570, so the whole decision is worth 0.0302. PE-RankFormer
captures 59.6% of that and OptiPrime 62.2%. **The gap is 2.6% of the available gain** (E23).

**Training changes recovered most of the deficit but did not close it.** An eight-arm ablation
matrix over 18 runs found a recipe (arm F) worth +0.00164 on fold 0 with all three seeds
agreeing in sign, which transfers to the panel as +0.00115 over its matched control (p = 0.001)
and cuts the OptiPrime deficit from −0.00198 to −0.00083. It does **not** beat the published
model: −0.00019, p = 0.044. Part of the reason is ensemble construction — the published model
is five per-fold checkpoints, arm F was three same-data seeds — and the fair per-fold
comparison is running now (§7).

**A defect in our own scoring changed that number.** Family C feature statistics are refit at
attach time and are not stored in the checkpoint, so E22 originally standardised the panel by
the panel's own mean and SD — concealing that panel `pbs_length` sits 1.35 training SDs low
with twice the spread. Scored honestly, arm F loses ground (+0.00140 → +0.00115 over its
control; +0.00003 → −0.00019 against the published member). See `CORRECTIONS.md` C3a.

**Scoring every arm then overturned the conclusion we drew from that.** Canonical batching
alone (arm E: no features, ranking loss off) is worth +0.00141 over the released recipe and
draws level with the published model. The feature branch in isolation is +0.00018 (p = 0.068)
— close to inert, not a liability. And **F − E, which was +0.00064 on fold 0 with all three
seeds agreeing in sign, is −0.00025 (p = 0.006) on the panel**: an ablation conclusion that
reversed on an external library. That is the strongest practical argument in the programme
for validating component choices outside the development corpus.

---

## 2. The estimand was wrong (E01, E02, E20)

The manuscript grouped candidate pegRNAs by protospacer. A user does not choose a protospacer;
they have already chosen the allele they want to install and the cell and editor they work in.
The decision unit is **one canonical allele × one experimental context**, and the candidates are
alternative pegRNA designs for it.

`explore_v2/canon.py` builds that key: minimal edit, VCF-style indel left-normalisation applied
independently **in both orientations**, strand normalisation, and 12 bp flanks read off the
site's longest observed window (`CANON_VERSION = 2`).

Regrouping changes what the evaluation set is:

| | protospacer grouping | allele × context grouping |
|---|---:|---:|
| fold-0 groups with ≥2 distinct designs | 670 | 1,463 |
| with ≥5 distinct designs | 146 | **0** |

The manuscript's 735 "targets with ≥5 designs" are groups of ≥5 **rows**, not designs; their
median distinct-design count is 2 (E20). All 670 protospacer groups span more than one
source/cell/editor condition and 26% contain more than one intended allele — so a ranking inside
one of them is partly a ranking of conditions, not of designs.

Four endpoint defects were found and fixed in a tested module (`endpoints.py`): candidates were
counted as rows rather than distinct designs; achieved-efficiency@k was scored on groups
shallower than k; the random-choice hit rate ignored tied maximisers (0.3354 → 0.4442); and the
paired bootstrap returned p ≈ 0 for a degenerate zero difference.

---

## 3. The reversal, on the reserved panel (E10, E11, E16, E22)

Fold 0 has no groups at deployment depth, so the decision question could not be answered on it.
E10 built a reserved panel from Kim's large variant library (288,793 pegRNAs), which is absent
from OptiPrime's training mix — that mix took only `LibSmall`. The panel was pre-registered
(`PREREGISTRATION_reserved_panel.md`) and unblinded once.

On 24,668 informative groups over 19,642 sites:

| model | achieved @1 | share of the available gain | hit rate @1 |
|---|---:|---:|---:|
| random choice | 0.0268 | 0% | 0.3133 |
| OptiPrime | 0.0456 | **62.2%** | 0.5858 |
| PE-RankFormer, published member | 0.0448 | **59.6%** | 0.5795 |
| perfect chooser | 0.0570 | 100% | 1.0 |

| paired difference @1 | value | 95% CI | p |
|---|---:|---|---:|
| published member − OptiPrime, all eligible | −0.00064 | [−0.00089, −0.00038] | 0.001 |
| published member − OptiPrime, depth ≥8 | −0.00249 | [−0.00416, −0.00083] | 0.004 |
| published member − OptiPrime, worth ≥0.05 | −0.00329 | [−0.00451, −0.00209] | 0.001 |

Intervals are cluster bootstraps resampling target sites. This is the programme's one
confirmatory result; every subsequent panel use is development and is disclosed as such in the
artifact that reports it.

---

## 4. The headline correlation claim replicates (E21)

| surface | metric | OptiPrime | PE-RankFormer | margin | 95% CI | p |
|---|---|---:|---:|---:|---|---:|
| fold 0, 750 clusters | Spearman | 0.8690 | 0.9079 | +0.0389 | [+0.0290, +0.0495] | 0.0005 |
| fold 0 | Pearson, calibrated | 0.8270 | 0.8637 | +0.0366 | [+0.0203, +0.0557] | 0.0005 |
| panel, 30,302 clusters | Spearman | 0.6931 | 0.7091 | +0.0159 | [+0.0136, +0.0182] | 0.0005 |
| panel | Pearson, calibrated | 0.5854 | 0.6017 | +0.0163 | [+0.0119, +0.0207] | 0.0005 |

The ordinal head emits rank estimates, not efficiencies, so Pearson is reported on the
isotonic-calibrated scale. The calibrator was fitted on development out-of-fold predictions and
applied **unchanged** to the panel, where it lifts Pearson from 0.5522 to 0.6017 while leaving
the rank correlation at 0.7089 — a monotone map cannot change a rank correlation, and the
third-decimal difference is out-of-range clipping creating boundary ties. That the frozen map
transfers to an unseen library is itself a result worth reporting.

---

## 5. What actually caused the deficit — and the causal claim I withdrew (E13, E17a, E19)

The first diagnosis was that the training ranking key differed from the deployment decision
unit, so the model was trained on comparisons it would never face. Regrouping the ranking key
recovered +0.00178 on the panel (E15). I attributed the gain to ranking-pair exposure.

**That attribution was wrong, and E19 withdraws it.** The matrix separates two things the
regrouping changed at once: which rows share a *batch*, and which rows form a *ranking pair*.
Eight arms, 18 runs, three seeds on the load-bearing ones, scored on fold 0's 1,463 informative
groups; no arm trains on fold 0 or early-stops on it.

| contrast | what it isolates | mean | all seeds same sign |
|---|---|---:|:--:|
| F − A | the full recipe | **+0.00164** | 3/3 |
| E − A | canonical batching, **ranking disabled entirely** | **+0.00099** | 3/3 |
| F − E | adding the feature branch on top | +0.00064 | 3/3 |
| H − F | dropping ranking from the best arm | −0.00050 | 3/3 |
| D − E | canonical ranking pairs vs. no ranking at all | −0.00006 | ✗ |
| F − D | features, with ranking on | +0.00071 | ✗ |

Arm E turns the ranking loss **off** and still captures 60% of the total gain. D − E, the
contrast that isolates ranking-pair exposure with batching held fixed, is −0.00006 and flips
sign across seeds. **The mechanism is batch composition, not ranking-pair exposure**: grouping
comparable candidates into the same batch changes the gradient through batch statistics,
regardless of whether they are then formed into ranking pairs. Arm G (16 pairs per group,
146,088 pairs/epoch — 2.4× arm D) is *worse* than arm D, which is the same conclusion from the
other direction.

E17a, which built the exposure replay this claim originally rested on, contained two index bugs
(sampler indices are into the training subset, not the corpus) that produced an impossible "100%
of canonical pairs span two alleles" and a negative mean target gap where the minimum is 0.02.
Both are fixed and recorded in `CORRECTIONS.md`.

**Checkpoint selection is not a lever.** Selecting on the decision metric rather than pooled ρ
gains +0.00004 over the same 18 trajectories, despite the two selectors choosing different
epochs in 16 of 18 runs.

### 5a. Only part of the matrix survives an external library (E22, all five arms)

Every arm's 3-seed ensemble, scored on the panel with ensemble size matched and feature
standardisation frozen:

| arm | what it changes | achieved @1 | pooled ρ | vs. published |
|---|---|---:|---:|---:|
| A | released recipe | 0.0349 | 0.6908 | −0.00134 |
| D | canonical batching, ranking on | 0.0362 | 0.7019 | −0.00006 |
| E | canonical batching, ranking off | 0.0363 | 0.7070 | +0.00006 |
| F † | canonical batching, ranking on | 0.0361 | 0.7071 | −0.00019 |
| H † | canonical batching, ranking off | **0.0365** | **0.7113** | **+0.00024** |
| | OptiPrime | 0.0369 | 0.6931 | |
| | published member | 0.0363 | 0.7091 | |

† carries the Family C feature branch.

- **Canonical batching is the whole gain and it transfers**: E − A = +0.00141, p = 0.001, and
  arm E draws level with the published model while carrying no features and no ranking loss.
- **The ranking loss costs accuracy here**, on both metrics: F − H = −0.00043, D − E = −0.00010,
  with pooled ρ lower in both cases too.
- **The feature branch is close to inert**: H − E = +0.00018, p = 0.068.
- **F − E reverses sign between surfaces**: +0.00064 on fold 0 with 3/3 seed agreement,
  −0.00025 (p = 0.006) on the panel. Seed agreement measures optimisation noise, not transfer.
  This is the strongest practical argument in the programme for validating component choices
  outside the development corpus.

Arm H beats the published member on both metrics, but that is **not** a result: it comes from
ten contrasts scored on a surface being used for development, and adopting it would be
selecting a component on the evaluation surface.

---

## 6. Where the remaining error lives, and one fix that fails (E23)

On the groups where the models nominate different designs, the nominated designs differ
systematically in geometry — measured against the best-measured design:

| model | PBS length bias | RTT length bias |
|---|---:|---:|
| OptiPrime | +0.519 | **−0.531** |
| PE-RankFormer, published member | +0.766 | +0.162 |
| arm A, released recipe | +0.943 | +0.483 |
| arm F, canonical + features | +0.812 | +0.428 |

Every PE-RankFormer variant prefers a longer PBS **and** a longer RTT than the best design.
OptiPrime is the only model preferring a shorter RTT, consistent with its treating RTT length as
an explicit synthesis repeat count. Supplying the lengths as explicit scalars (A → F) moves the
PBS bias by −0.131 and the RTT bias by −0.055: both shrink a little, neither is removed, and
both stay on the opposite side of zero from OptiPrime. Handing the model the length is not the
same as modelling what the length does.

**A global length correction does not work.** If the preference were an additive score offset, a
linear length term would remove it. Swept on 43,425 development decision groups, the optimum is
exactly α = 0 on both axes (best gain +0.00000). The miscalibration is conditional on allele and
context, not a global constant — which is an argument against a per-candidate correction and for
a model that scores the candidate set jointly.

### 6a. The two models are complementary, on the deployment-like surface (E23)

If the two models err in opposite directions on RTT geometry, their errors may partly cancel.
An equal-weight rank average of the two scores, on the same 24,668 informative groups, with
**nothing fitted** (the weight is fixed at one half):

| model | achieved @1 | share of available gain |
|---|---:|---:|
| OptiPrime | 0.0456 | 62.2% |
| PE-RankFormer, published member | 0.0448 | 59.6% |
| **equal-weight rank average** | **0.0461** | **64.1%** |

| paired difference @1 | value | 95% CI | p |
|---|---:|---|---:|
| blend − OptiPrime | +0.00055 | [+0.00034, +0.00077] | 0.001 |
| blend − published member | +0.00133 | [+0.00109, +0.00156] | 0.001 |

Sweeping the weight on PE-RankFormer, **every** value from 0.2 to 0.7 beats both endpoints,
so 0.5 is not cherry-picked.

**But it does not replicate on fold 0.** There the blend beats OptiPrime (+0.00429) and
*loses* to PE-RankFormer alone (−0.00486, p = 0.001), because on fold 0 PE-RankFormer leads
OptiPrime by 0.0092 and averaging in the weaker model dilutes the stronger one. On the panel
the two are within 0.0008 and the blend pays.

**What survives:** on the surface that resembles deployment — external library, realistic
candidate depth, the two models within a thousandth of each other — combining them beats
either alone, robustly across weights, with nothing fitted. The mechanistic baseline is
complementary to the neural model wherever the two are comparably strong. That is better
supported than a superiority claim in either direction, and more useful to a practitioner.

---

## 7. Can the improved recipe beat the published model? No (E24)

E22 left arm F's 3-seed ensemble marginally behind the published member (−0.00019, p = 0.044)
and conjectured that ensemble construction was the reason: the published member is five
per-fold checkpoints, arm F was three same-data seeds. E24 tests that by rebuilding arm F the
same way — `--val-fold 1..5`, combined by the same plain mean of predicted efficiencies, with
a fresh isotonic calibrator fitted on this model's own development out-of-fold predictions and
frozen before either surface was scored.

| contrast, achieved @1 | value | 95% CI | p |
|---|---:|---|---:|
| per-fold F − published member | −0.00017 | [−0.00034, +0.00001] | 0.062 |
| per-fold F − arm F, 3 seeds | +0.00002 | [−0.00015, +0.00020] | 0.812 |
| per-fold F − arm A (released) | +0.00117 | [+0.00095, +0.00140] | 0.001 |
| per-fold F − arm E | −0.00023 | [−0.00043, −0.00004] | 0.014 |
| per-fold F − arm H | −0.00041 | [−0.00059, −0.00023] | 0.001 |
| per-fold F − OptiPrime | −0.00081 | [−0.00107, −0.00055] | 0.001 |
| *on fold 0:* per-fold F − published member | +0.00007 | [−0.00152, +0.00173] | 0.952 |

**The ensemble-construction conjecture is refuted.** Per-fold construction is worth +0.00002
(p = 0.81) — nothing. E22's explanation was wrong.

**The recipe does not beat the published model**, on either surface or either metric: −0.00017
(p = 0.062) on the panel, +0.00007 (p = 0.95) on fold 0; pooled ρ 0.7096 vs 0.7091 on the
panel and 0.9084 vs 0.9079 on fold 0.

**An unexplained gap that bounds what the matrix can claim.** Arm A — our reconstruction of
the released recipe — scores 0.0349 against the published member's 0.0363. That 0.00135 gap
is larger than any effect in the ablation matrix, and per-fold construction cannot explain it
(just measured at +0.00002). So arm A is not a faithful reproduction of the shipped model, and
every improvement the matrix reports is an improvement over *our reconstruction*. Arm F's
+0.00117 over arm A is real and replicates; it simply returns the model to where the published
member already was. **The gain the manuscript could claim from this recipe is zero.**

---

## 8. Corrections to this programme's own work

Recorded in full in `CORRECTIONS.md`. The load-bearing ones:

- **Strand-invariance bug in `canonical_keys`** — left-normalised on the forward strand then
  reverse-complemented, so 61.7% of 51,012 indel pairs could key differently by orientation.
  Fixed by normalising both orientations independently. Fold-0 counts unchanged.
- **The E13 causal claim withdrawn** — see §5.
- **Four endpoint defects**, including a random hit rate of 0.3354 that is correctly 0.4442.
- **`aggregate_candidates` defaulted to grouping by allele alone**, pooling up to 14 conditions
  on fold 0. Caught because a table returned 546 groups where E02 independently says 1,463.
- **Non-reproducible `hash()` seeding** in E05; **a degenerate PCA direction** giving fold-0
  R² = −4×10¹¹; **a per-epoch decision metric assuming sorted rows** (`reduceat` invented 14,680
  boundaries against 11,559 real groups). The last was fixed and then validated against a
  brute-force per-group loop — exact match.

---

## 9. Limits that bound every claim above

- **One reserved surface, used once confirmatorily.** E12, E15, E18, E19, E22 and E23 are
  development uses of the same panel. Any component decision taken from them needs a newly reserved
  surface to become confirmatory.
- **The panel is one library, one laboratory, HEK293T-dominated.** Cross-cell-type and
  cross-editor generalisation is not tested by it.
- **Achieved efficiency @1 is measured against the best *measured* design**, so it is bounded by
  what the library happened to contain, not by what is designable.
- **No wet-lab validation.** Every claim here is retrospective.
- **Differences at the third and fourth decimal place.** The programme's own resolution on fold 0
  is about 0.005 in pooled ρ; the decision differences that separate methods are an order of
  magnitude smaller, which is exactly why the panel's 24,668 groups were necessary.

---

## 9a. Follow-on: target-library adaptation (E25–E28)

A separate programme, run against `ADAPTATION_SELECTION_PLAN.md`, asked whether supervision
from part of the external library corrects the decision deficit. It does: on held-out locus
components, ordinary efficiency fine-tuning reaches 0.04178 against OptiPrime's 0.04020
(+0.00158, clustered interval excluding zero, 3/3 seeds, 15.7% regret reduction) while also
lifting pooled ρ from 0.7102 to 0.7771. None of the proposed elaborations — a selection
objective, separated heads, explicit geometry — beats its own control. Adaptation costs
0.029 source Spearman, where a selection-head-only variant gets the same target gain for
0.008. Details in `ADAPTATION_RESULTS.md` and `reports/adaptation_report.pdf`.

**This does not change §1–§9.** Those results are about models with no target-library labels,
which is the manuscript's claim; the adaptation result is an additional, separately caveated
one, and it is not a matched-label comparison — OptiPrime was not adapted.

---

## 10. What the manuscript should say

1. **Keep the correlation claim and strengthen it** with the panel replication and the
   transferring calibrator (§4). It is now a two-surface result.
2. **Withdraw the deployment-superiority claim.** Replace it with the honest comparison: on the
   fixed-allele decision OptiPrime is ahead by 2.6% of the available gain, and both models are
   far closer to each other than either is to random selection (§3, §6).
3. **Report the decision under the correct grouping**, with the row-versus-design accounting
   stated plainly — the ≥5-design population the manuscript describes does not exist in fold 0
   (§2). `tables/tab_utility.tex` is already regenerated with all three panels.
4. **Report the metric disagreement as a finding, not a caveat.** Two defensible metrics rank the
   same models in opposite order, and E18 reproduces the reversal inside our own ablations. This
   is the most transferable thing the programme found.
5. **Report the ablation matrix honestly**: batch composition, not ranking supervision, is what
   pays (§5) — including the withdrawn claim, which is what makes the matrix credible.
