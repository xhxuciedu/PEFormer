# The result: the manuscript's comparative claim does not replicate on the decision that matters

Prepared 8 September 2026. Companion to `GATE_A_REPORT.md` (research direction) and
`CORRECTIONS.md` (what earlier analyses got wrong). Every number is produced by a script in
`explore_v2/` that writes its own provenance; `verify_report.py` re-checks the ones quoted
here against their artefacts.

**The headline.** On a 30,475-group panel that neither model was trained on, scored once
under a pre-declared analysis, **OptiPrime chooses better pegRNAs than PE-RankFormer** —
while PE-RankFormer retains the higher pooled rank correlation. The deficit grows with
candidate depth and with how much the choice is worth. This inverts the manuscript's
central comparative claim on the estimand a user actually faces, and it was found by the
first test of that claim at real candidate depth.

---

## 1. Why a new surface was needed

The manuscript groups candidates by protospacer. A user has already fixed the allele they
want and the cell and editor they work in, so the group they face is **one intended allele
in one experimental context**. Rebuilding the corpus on canonical alleles collapses 130,921
distinct WT windows into 51,766 intended alleles and 175,668 decision groups, and under
that grouping the held-out fold 0 contains 2,412 groups with two or more alternative
designs, 85 with three, and **none with five**. Its 735 protospacer groups of five or more
rows are a different object: 100% span more than one condition and 26% contain more than one
intended allele.

So fold 0 could only score a binary choice. There, PE-RankFormer wins clearly: 0.750
accuracy against OptiPrime's 0.657 (+0.0906, 95% CI [+0.060, +0.121]) on 1,390 scorable
groups, with the Spearman margin roughly doubling from +0.0885 to +0.1735. That is what
motivated finding a deeper surface — and the deeper surface does not agree.

## 2. The panel

Kim's large variant library (`DeepPrime_dataset_final_Feat8.csv`, 288,793 measured pegRNAs
at 58,217 target windows) is **not** in OptiPrime's training mix, which took only Kim's
`LibSmall` files. It was unusable until now because its released `Edited74_On` column is
masked outside the RT-PBS footprint. E10 recovers it and validates the recovery three
independent ways:

| check | result |
|---|---|
| nick position, from DeepPrime's own `AfterRTT_left4` column | 153,974 / 153,974 substitution rows |
| PBS and RTT rules reproduce the corpus's stored pegRNAs | 100% of Kim rows |
| reconstructed PBS vs the corpus, locus and frame fixed | **1.0000** of 1,575 rows |
| reconstructed RTT, same key | **0.9968** |

After removing every row sharing a protospacer or a canonical allele with training:
118,187 rows in **30,475 decision groups**, 18,660 at depth ≥3, **8,775 at depth ≥5**,
2,400 at depth ≥8. 24,668 groups have a non-constant outcome; 5,807 have every design at
exactly zero and contribute zero achieved efficiency rather than being dropped.

Three properties make this a fair test:

- **Pre-declared.** `PREREGISTRATION_reserved_panel.md` fixed the endpoints, eligibility,
  strata, dependence unit and primary comparison, and the panel file was hashed, before any
  model saw it. Scored once.
- **Matched exposure.** PE-RankFormer and OptiPrime were trained on the same corpus and
  neither saw this library. Released DeepPrime *does* train on it
  (`external/deepprime/train_base.py:26`) and is therefore excluded as a comparator, not
  reported as a held-out baseline.
- **Identical inputs.** Both predictors receive the same reconstructed sequences and the
  same assigned context. E12 shows the endpoints barely move under three alternative
  context assignments (achieved efficiency 0.0447–0.0453, hit rate 0.579–0.583), so that
  assignment is not load-bearing.

Sanity check before any endpoint: pooled Spearman is 0.7091 for PE-RankFormer and 0.6931
for OptiPrime. A reconstruction or context error would have shown up as a near-zero
correlation, and PE-RankFormer is *ahead* on this metric — which is what makes the decision
result below informative rather than a suspected artefact.

## 3. The result

All 30,475 eligible groups, 22,931 target sites, oracle 0.0461:

| predictor | achieved efficiency @1 | @3 | @5 | best-design hit rate | regret @1 |
|---|---:|---:|---:|---:|---:|
| random choice | 0.0217 | 0.0469 | 0.0674 | 0.4442 | 0.02443 |
| **OptiPrime** | **0.0369** | 0.0558 | 0.0738 | **0.6647** | **0.00922** |
| PE-RankFormer ordinal-S4D | 0.0363 | 0.0557 | 0.0738 | 0.6597 | 0.00986 |

Paired, resampling target sites:

| difference (ours − OptiPrime) | value | 95% CI | p |
|---|---:|---|---:|
| achieved efficiency @1 | **−0.00064** | [−0.00089, −0.00038] | 0.001 |
| achieved efficiency @3 | −0.00003 | [−0.00019, +0.00013] | 0.674 |
| best-design hit rate | −0.00482 | [−0.00968, −0.00007] | 0.044 |
| regret @1 | +0.00062 | [+0.00037, +0.00088] | 0.0005 |

Both models are far better than random — PE-RankFormer removes 60% of random's regret,
OptiPrime 62% — and at a three-candidate budget they are indistinguishable. The difference
is in the single top pick, and it favours OptiPrime.

**The deficit tracks exactly the two axes that make a decision matter.**

| stratum | groups | OptiPrime @1 | ours @1 | difference | p |
|---|---:|---:|---:|---:|---:|
| depth 2 | 11,815 | 0.0243 | 0.0240 | −0.00023 | 0.082 |
| depth 3–4 | 9,885 | 0.0356 | 0.0352 | −0.00034 | 0.093 |
| depth 5–7 | 6,375 | 0.0520 | 0.0509 | −0.00110 | 0.001 |
| depth 8+ | 2,400 | 0.0645 | 0.0620 | −0.00246 | 0.006 |
| decision worth < 0.005 | 12,553 | 0.0027 | 0.0027 | +0.00001 | 0.53 |
| decision worth 0.005–0.02 | 6,346 | 0.0226 | 0.0227 | +0.00007 | 0.52 |
| decision worth 0.02–0.05 | 5,961 | 0.0554 | 0.0552 | −0.00019 | 0.53 |
| decision worth ≥ 0.05 | 5,615 | 0.1099 | 0.1066 | **−0.00328** | 0.0005 |

Where the candidates are nearly indistinguishable the two models are identical; where the
choice is worth something, OptiPrime is better. The pattern holds for substitutions
(−0.00062, p=0.0005) and insertions (−0.00079, p=0.002) and is not significant for
deletions (−0.00038, p=0.16).

On a fixed population eight candidates deep, random selection must test four candidates to
match PE-RankFormer's first pick, and OptiPrime's first pick already beats it:

| candidates tested | PE-RankFormer | OptiPrime | random |
|---|---:|---:|---:|
| 1 | 0.0620 | **0.0645** | 0.0293 |
| 3 | 0.0842 | 0.0845 | 0.0587 |
| 8 | 0.0934 | 0.0933 | 0.0876 |

## 4. What this means, and one diagnosed mechanism

This is the plan's own thesis, confirmed against the model that motivated it: **pooled rank
correlation and the deployment decision are different quantities, and this model was built
and selected on the first.** PE-RankFormer leads on pooled Spearman on both surfaces
(+0.0389 on fold 0, +0.0160 here) and on the narrow binary choice fold 0 supports, and loses
the fixed-allele choice on the wider design space of Kim's large library — where PBS lengths
run 1–17 and RTT lengths 1–50, against the benchmark's much narrower geometry.

E13 identifies a concrete mechanism inside our own training objective rather than leaving
this as a mystery, and E14/E15 test the repair. The pairwise ranking loss, weighted at 0.25, builds its groups from the
raw `full_unedited | full_edited` window pair — whose extent depends on the row's RTT
length. Alternative designs for one allele differ *precisely* in PBS/RTT geometry, so the
key meant to group them is the one that splits them:

| grouping, 238,381 training rows | groups | singletons | rows in a multi-design group | within-group design pairs |
|---|---:|---:|---:|---:|
| current ranking key | 209,161 | 93.8% | 16.2% | 14,277 |
| canonical decision group | 132,066 | 68.0% | 62.3% | **368,307** |

The term that exists to rank alternative designs has seen **3.9%** of the available
comparisons, and the missing 96% are the ones the deployment estimand scores.

E13 writes a re-grouped corpus differing only in `group_key`. Two arms were trained on it,
identical in architecture, seed, recipe and code. The repair works, on both surfaces:

| | fold 0 (E14) | panel, all groups (E15) | panel, depth 8+ | panel, decision worth ≥0.05 |
|---|---:|---:|---:|---:|
| control, current ranking key | 0.1330 | 0.0340 | 0.0559 | 0.0978 |
| canonical decision-group key | **0.1356** | **0.0357** | **0.0611** | **0.1049** |
| difference | +0.00258 | +0.00184 | +0.00561 | +0.00729 |
| p | 0.01 | 0.001 | 0.001 | 0.001 |

Achieved efficiency; the panel differences are paired on target sites. The effect grows
along the same axes the deficit did, and on the panel it **recovers about 60% of the
control's gap to OptiPrime at every stratum** (all groups −0.00310 → −0.00126; depth 8+
−0.00930 → −0.00369). Regret on fold 0 falls 25%, and 70.6% of random's regret removed
becomes 78.1%.

**It does not close the gap.** The repaired model still loses to OptiPrime: −0.00126 overall,
−0.00369 at depth 8+, −0.00540 where the decision is worth ≥0.05, all p = 0.001. And two
caveats bound how much weight this can carry. It is a single seed. And the published member
is a five-checkpoint ensemble while both retrained arms are single models, so the valid
comparison is canonical-versus-control — not either arm against the published number.

## 5. What this does not establish

- **Not that OptiPrime is the better tool overall.** It wins the single top pick on this
  library; it ties at k=3, loses on pooled correlation, and its released checkpoints' fold
  assignments are unknown, so its fold-0 numbers carry a leakage caveat this panel does not.
- **Not an independent laboratory or assay.** Same group, same library chemistry, same
  readout, one cell line, one editor configuration. No transportability claim follows.
- **Only two comparators.** DeepPrime excluded by exposure. PRIDICT2 measured and rejected:
  it enumerates PBS 7–15, covering 65% of the panel's designs, so scoring with it restricts
  candidates by geometry and changes the estimand; its released 23k library also fails as a
  second surface (46% allele overlap with training, no group deeper than two).
- **Nothing prospective.** Every outcome was measured before this analysis existed.
- **The panel is now spent.** It was scored once, as declared. Any model choice informed by
  it makes it development data, and the next confirmation needs a different dataset.

## 6. What follows

1. **The manuscript cannot claim a deployment advantage over OptiPrime.** It can claim a
   pooled-correlation advantage, and it must report this panel.
2. **The repair is real but partial.** Regrouping the ranking loss recovers ~60% of the
   gap, replicated on two surfaces, with the effect scaling in depth. That is a mechanism
   plus a repair plus a corrected benchmark — a better contribution than the original claim,
   and it needs multiple seeds and a sealed surface before it is confirmatory. The remaining
   40% is unexplained and is the obvious next question.
3. **Acquire an independent panel** before any further confirmatory claim; the queue is in
   `e06_external_eligibility.md`.
4. **Power is known** for a prospective test: measured paired between-method SD 0.0497, so
   48 edits give 80% power at an 0.02 absolute effect, 12 at 0.04 (E08).
