# E06 - external eligibility, and one reserved panel

## 1. Kim's large library is on disk and is not in the training mix

`external/deepprime/data/DeepPrime_dataset_final_Feat8.csv` holds 288,793 measured rows over 58,217 WT windows, 43,744 canonical alleles and 287,981 distinct designs. Mean efficiency 0.0218, 41.0% exact zeros.

The released `Edited74_On` column is masked outside the RT-PBS footprint, so the raw string pair cannot identify an allele: three designs for one deletion give three different masked strings. Aligning the unmasked block back onto the WT window with each row's own edit type recovers the allele and makes it comparable with the training corpus.

| candidate depth | alleles |
|---|---:|
| >= 2 designs | 40,755 |
| >= 3 designs | 37,787 |
| >= 5 designs | 29,609 |
| >= 8 designs | 13,114 |

Overlap with the 318,471-row training mix: 13,294 rows share a protospacer and 49,276 rows share a canonical allele (6,499 alleles). Removing every row with either kind of overlap leaves:

- 233,223 rows, 36,538 alleles
- **33,897 decision groups with two or more alternative designs**, 31,248 with three or more, 24,133 with five or more
- within those groups the best design averages 0.0544 and the spread between best and worst averages 0.0536

For comparison, the manuscript's held-out fold 0 offers 2,412 two-design groups and none with five. This panel is the same laboratory and the same assay, so it is **not** an independent-study test; it is an untouched surface with the candidate depth the corrected decision estimand needs.

## 2. The OptiPrime/Hsu endogenous panel has designs but no outcomes

`data/raw/hsu2026/41587_2026_3261_MOESM3_ESM.xlsx`, sheet `Supp Table 3 Endo_gRNAs`: 283 rows over 60 spacers, median 1 designs per spacer, editors {'PEmax': 267, 'PE6b': 4, 'PE6a': 2, 'PE6c': 2, 'PE6d': 2, 'PE6e': 2, 'PE6f': 2, 'PE6g': 2}. No efficiency column.

Confirms the plan's inventory: 283 endogenous design rows with spacer, scaffold, RTT, PBS, motif and nicking guide, and no measured outcome in this sheet. It is a design panel, not an evaluation panel, until per-figure source data are recovered.

## 3. Acquisition queue

| candidate | status | why it is wanted | what blocks it |
|---|---|---|---|
| MinsePIE / Koeppel et al. | not acquired | repair-dependent insertion behaviour; read-count tables released | pooled insertion rates are library-abundance normalised, not per-design edited-allele fractions; long inserts may exceed the 90-token pegRNA input |
| OPED original prospective experiments (PRJNA882795) | not acquired | original endogenous measurements from a different group; the plan's first choice for external selection | usable row counts require supplement extraction; restrict to the single-pegRNA configuration this model supports |
| PRIDICT2 / ePRIDICT endogenous and chromatin panels (PRJNA1025026) | partially on disk | assay-shift test; endogenous rather than reporter | the same source family already contributes 174,067 training rows, so only panels proven absent from training qualify; external/pridict2/dataset holds the 23k processed library and a ranking-percentile table, both of which need an overlap audit against the training mix before use |
| Li et al. chromatin/repair experiments (PRJNA949965) | not acquired | position and chromatin-state effects at mapped integration sites | few shared reporter designs, so probably cannot support alternative-pegRNA selection; mechanistic validation only |
| OptiPrime/Hsu endogenous panel (Supp Table 3) | on disk, no outcomes | matched-design endogenous panel from a training-adjacent source | no efficiency column in the released sheet; needs per-figure source data |
| StopPR functional screen | not acquired | downstream application validation | readout is guide abundance/fitness, not editing efficiency; do not use as an efficiency label |

Nothing in this queue has been downloaded, normalised or scored in this program. Published library sizes are not counts of usable evaluation rows.

## 4. The reserved panel

`explore_v2/reserved_panel_kim_large.parquet` - 230,581 rows, 33,897 decision groups with at least two candidates.
SHA-256 `679d68f02e4d76b1fe2c4df84ec31ea3...`

Written and hashed without any model being run on it. The first evaluation on this panel must be a pre-declared comparison; any earlier use turns it into development data.
