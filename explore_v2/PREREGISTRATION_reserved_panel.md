# Pre-declared analysis for the reserved Kim large-library panel

Written 8 September 2026, before any model was scored on the panel.
Panel: `explore_v2/reserved_panel_kim_large.parquet`, SHA-256 in
`explore_v2/reserved_panel_kim_large.sha256`.

This document exists because the research plan's requirement is unambiguous: *"Reserve at
least one suitable external panel before looking at comparative performance. Any external
panel used to choose the model becomes development data."* The panel below has been built,
hashed and left unscored. What follows is what will be computed on it, once, when the
comparison is ready.

## What the panel is, and what it is not

`external/deepprime/data/DeepPrime_dataset_final_Feat8.csv` is Kim's large variant library:
288,793 measured pegRNAs at 58,217 target windows. It is not part of OptiPrime's 318,471-row
training mix, which took only Kim's `LibSmall` files. After removing every row that shares
a protospacer or a canonical allele with the training corpus, 222,488 rows remain in 32,925
decision groups with two or more alternative designs for one intended allele, 23,345 of them
with five or more.

It is **not** an independent-laboratory or independent-assay test. Same group, same library
chemistry, same readout, one cell line and one editor configuration. It cannot support a
transportability claim. What it can support, and what fold 0 cannot, is the fixed-allele
selection estimand at real candidate depth: fold 0 offers 2,412 two-design groups and none
with five.

## Primary endpoint

For each decision group `g` with candidate set `D(g)`, one intended allele, one context:

- **success@1**: the model's top-ranked design is the measured best in `D(g)`.
- **selected efficiency@1**: the measured efficiency of that design.
- **regret@1**: `max_{d in D(g)} y(d) - y(top-ranked)`, in absolute efficiency units.

Primary comparison: PE-RankFormer's ordinal-S4D backbone against OptiPrime, paired per
decision group, with a percentile bootstrap resampling **target windows** (not rows, not
groups). Report success@3 and regret@3 as secondary.

## Declared in advance

1. **Eligibility.** Groups with two or more distinct designs and a non-constant measured
   outcome. Groups whose designs all measure exactly zero are counted and reported
   separately, never silently dropped: on fold 0 that class is 40% of two-design groups.
2. **Stratification.** Results reported for the whole panel and, separately, by candidate
   depth (2, 3-4, 5-7, 8+) and by edit type (substitution, insertion, deletion). No
   post-hoc subset becomes the headline.
3. **Overlap removal.** Any row sharing a protospacer or a canonical allele with the
   training corpus is already excluded from the panel file. No further filtering.
4. **Scale.** DeepPrime reports efficiency as a percentage; the panel file stores fractions.
   No re-normalisation, no calibration fitted on this panel.
5. **Prediction coverage.** Every candidate a model cannot score is reported, and the
   common-supported subset is reported alongside the full panel. Rows are never dropped
   silently.
6. **No selection.** Model weights, ensemble membership, calibration and hyperparameters
   are fixed before the panel is opened. If any of them is changed afterwards, the panel is
   spent and this document is void.

## What would count as a result

The plan's Priority 3 hypothesis is that PE-RankFormer keeps a meaningful advantage for a
fixed desired allele once provenance and candidate availability are controlled. On fold 0's
pairwise decision that advantage is +0.091 in accuracy and -0.0091 in regret. A comparable
advantage here, at candidate depth 5 and above, would be the first such evidence on a
surface with real candidate depth. A materially smaller advantage would be the more
important finding, and would have to be reported as such.

## What this panel cannot settle

Transport across laboratories, assays, cell types or editors; anything about unseen
contexts; anything prospective. Those need the acquisition queue in
`e06_external_eligibility.md` and, for the strongest claims, new measurements.
