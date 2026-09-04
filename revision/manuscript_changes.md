# Manuscript changes, and why

Every change made to `reports/paper/pe_rankformer_paper.tex` during this revision, with
the evidence behind it. Grouped by whether it corrects an error, adds a result, or
changes framing.

## Corrections — things that were wrong

| # | What was wrong | Corrected to | Evidence |
|---|---|---|---|
| 1 | Headroom table labelled "all Kim rows" was **development fold 0 alone** and was compared against a Kim score from a different surface | All four surfaces reported with intervals | `task_1_6` |
| 2 | Abstract quoted "+0.12" headroom | +0.10, with CI [+0.0529, +0.1204] | `task_1_6` |
| 3 | Zero-mass 28.4% used to motivate a *training* objective; it is the dev-fold figure | 16.0% training / 26.8% held-out / 28.4% dev, each labelled at point of use | `audit_00` |
| 4 | Kim zero fraction given as both 49.9% and 50.7% two lines apart | Labelled: 49.9% training, 50.7% dev folds | `audit_00` |
| 5 | "K=20 yields 17–18 distinct thresholds" | **17** in all twenty official checkpoints; 18 on dev folds. Read back from the checkpoints | `audit_00` |
| 6 | Residual correlation "separates the two families cleanly: ordinal 0.686–0.708, simplex 0.76–0.78" | Neither bound held. Group means 0.705 vs 0.756, with the overlap stated | `diversity_dev0/wave2` |
| 7 | Early-stopping survey ("25 runs, median 21, max 28, 36%, cost 0.0012") not reproducible | All 64 official-fold runs: median 24.5, max 29, 50%, cost ≤0.0031 | `early_stopping_survey` |
| 8 | "Held-out set — evaluated once" contradicted a table showing three generations | Four accesses, from the audit log, with the adaptive-selection exposure bounded | `logs/heldout_evaluations.log` |
| 9 | Development resolution quoted as both ±0.002 and ≈0.005 | Standardised on ≈0.005, with the two calibrations that fix it | new §Resolution |
| 10 | "Eight-fold MAE reduction" measured against a quantity with no interpretation | Head-to-head against the baseline, plus the trivial floor | `task_1_5` |
| 11 | Empty `\author`; placeholder comparator citation | Real citation (Hsu A, … Liu DR, *Nat Biotechnol* 2026). **Author list still a TODO** | verified via publisher |
| 12 | Prose quoted headroom +0.1043 from a superseded script | +0.1040 from the authoritative analysis; the old script is marked superseded | caught by `verify_manuscript` |
| 13 | Two numbers in the stratification table written from memory (row count, weighted mean) | 19,334 rows and +0.0758; tables are now generated so this cannot recur | caught on review |

## Additions — new results

| What | Where | Headline |
|---|---|---|
| Per-target evaluation as the deployment-relevant metric | §Stratification, Table 3 | 0.6356 vs 0.5472; margin **+0.0884**, ahead on 78.4% of 670 targets |
| Deployment utility | new §, Table 4 | precision@1 0.367 vs 0.224, but top-pick success only +0.011 (n.s.) |
| Tie-robust and decomposed metrics | new §, Table 5 | τ-b +0.0547; detection +0.0292; quantification +0.0469 |
| Leakage-free subset | new §, Table 6 | 196 rows (1.0%); headline moves 0.0004 |
| Calibration floors and tail | §Calibration, Table 8 | floor MAE 0.1227; top 1% over-predicted by +0.065 |
| Ceiling intervals | Table 11 | held-out gap +0.1040 [+0.0529, +0.1204] |
| **Training-weight asymmetry** | new §, Tables 9–10 | comparator gives Kim 2.0% of its gradient for 55.3% of the evaluation; matched-weighting costs us +0.0136 (3/3 folds), **all of it on Kim** |
| Feature baseline | new §, Table 2 | tuned trees on engineered features reach 0.7413 |
| Partition-wise significance | Table 2 | **Liu +0.0220, p=0.028** — the paper's weakest claim, previously without an interval |
| Round-9 negatives | Table 12 | context-conditioned low-rank map (null vs own control), cross-attention 2→4 (−0.0012), censoring-aware loss (control beat it by 0.0091), selective SSM (+0.0031) |
| Multiplicity and power | Table 12 notes | negatives reframed as bounds; Holm correction |

## Framing changes

- **Scope statement added to the Introduction.** The comparator predicts PE3 and twinPE
  outcomes, exposes interpretable biochemical pseudorates, and was validated *in vivo*.
  We compare on one of its axes and say so.
- **The mechanism claim is qualified** three ways: task scope, the weighting asymmetry,
  and interpretability — a mechanistic parameterisation is an inspectable hypothesis and
  ours is not.
- **Pooled Spearman demoted.** It is presented as conflating locus difficulty with design
  ranking, with the per-target figure given as the honest deployment number.
- **The Liu partition is foregrounded** as the conservative result: it is the surface the
  comparator itself reports, and it is weighting-independent.
- **Data availability rewritten** to describe the reconstruction path and to state
  plainly that the missing public corpus is the main limit on reproducibility and is not
  ours to remove.
- **Interaction term demoted** from a finding to a direction, being below the stated
  resolution.

## Still open

1. **Author list and affiliations.** Marked `TODO(authors)` in the preamble. Only you can
   supply this.
2. **Title.** Unchanged, and defensible as it stands, but it names the method rather than
   the finding. Three alternatives that lead with the claim:
   - *A minimally mechanistic model outperforms a mechanistic one at predicting prime-editing efficiency*
   - *How much of a mechanistic predictor's accuracy comes from its mechanism?*
   - *Architecture and objective, not reaction mechanism, drive prime-editing efficiency prediction*
3. **A second external comparator.** PRIDICT2.0 and DeepPrime-FT remain unrun. The
   DeepSpCas9 weights the draft cited as the blocker are in fact present in `external/`;
   the real obstacle is byte-compatible Tm/MFE features. The limitation is reworded
   accordingly, but running them would strengthen the paper materially.
4. **External validation** on an independent dataset — still the single most valuable
   addition, and still unavailable.
