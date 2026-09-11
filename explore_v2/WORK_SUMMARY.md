# Work summary — executing explore_v2_research_plan.md

Session of 8 September 2026. Starting commit `83ffae1`, branch `round9-submission-revisions`.
Task: conduct the research the plan specifies. Scope executed: all of section 8A ("the next
two weeks — current data first") and the computational parts of section 8B. Nothing was
committed; everything is untracked in `explore_v2/`.

For the scientific conclusions read `GATE_A_REPORT.md`. This file is the record of what was
built, what was decided along the way, and what was deliberately not done.

**Later additions, in order.**

1. A review (`NEXT_STEPS_PLAN.md`) found real defects in E04. I verified its claims, ran the
   clean retest (`e09_clean_interaction_retest.py`), fixed three code bugs, and recorded
   what changed in `CORRECTIONS.md`. Two published numbers are superseded and one conclusion
   is withdrawn; the Gate A verdict is unchanged.
2. The canonicaliser fix cascaded, so E01-E09 were rebuilt under version-2 keys. Fifteen
   quoted numbers moved in the third decimal; the verifier caught every one; no conclusion
   changed.
3. The reserved panel was built, validated three ways, and scored under its pre-declared
   analysis (`e10`, `e11`, `e12`). This is the substantive new result and it has its own
   document, **`RESULT.md`** — and it went **against** the model: on 30,475 fixed-allele
   decision groups at real candidate depth, on a surface neither compared model was trained
   on, OptiPrime chooses better pegRNAs than PE-RankFormer (p = 0.0005), while
   PE-RankFormer keeps the higher pooled correlation.
4. E13 diagnosed a mechanism inside our own training objective: the pairwise ranking loss
   groups rows by the raw window pair, so it has seen 3.9% of the available within-allele
   design comparisons. Two arms differing only in that grouping were trained. Repairing the
   grouping improves achieved efficiency on fold 0 (+0.00258, p = 0.01) and on the panel
   (+0.00184 overall, +0.00561 at depth eight or more, p = 0.001), recovering about 60% of
   the control arm's gap to OptiPrime — but not closing it. Single seed; and the published
   member is a five-checkpoint ensemble, so the valid contrast is canonical versus control. Section 6 of this file, "what was
not done", should now be read alongside that file.

---

## 1. What was produced

**Eleven scripts, ~3,300 lines**, each writing a JSON with its own git commit, seed and
input hashes plus a markdown report, following the conventions already established in
`revision/`.

| file | role |
|---|---|
| `canon.py` | canonical intended-allele keys: minimal edit, VCF-style indel left-normalisation, strand normalisation, flanks read off each site's longest observed window |
| `_v2common.py` | provenance, tie-aware Spearman, cluster bootstrap, cached corpus loader |
| `extract_embeddings.py` | out-of-fold frozen representations for all 318,471 rows (GPU, ~4 min) |
| `e01_decision_manifest.py` | candidate-availability, replicate and fold-purity audit |
| `e02_corrected_metrics.py` | frozen predictions re-scored on four groupings + the pairwise estimand |
| `e03_matched_support.py` | interaction variance components, additive null, transfer regret |
| `e04_interaction_predictability.py` | held-out-locus predictability with three controls and an injected-signal positive control |
| `e05_reference_panel.py` | the plan's priority 1, leave-context-out, six budgets, eleven arms |
| `e06_external_eligibility.py` | on-disk external audit, acquisition queue, one reserved panel |
| `e07_context_correction.py` | explicit context correction for contexts that are in training |
| `e08_pilot_power.py` | pilot power from measured variability |
| `verify_report.py` | 47 numbers in the Gate A report checked against their artefacts |

**Documents:** `GATE_A_REPORT.md` (the go/no-go), `README.md` (run order and reproduction),
`PREREGISTRATION_reserved_panel.md`, and eight per-experiment reports.

**Data artefacts:** `cache/decision_manifest_v2.parquet` (318,471 rows with allele, design,
context, decision-group and replicate keys), `cache/quartets_full_v2.parquet` (111,862 matched
quartets), `cache/embeddings.npy` (318,471 × 768, out of fold),
`reserved_panel_kim_large.parquet` + `.sha256` (sealed, unscored), `e02_pairwise_table.csv`.

Everything under `cache/` is rebuildable and now git-ignored (549 MB).

---

## 2. How each plan deliverable was met

| plan deliverable (§8A) | how |
|---|---|
| canonical decision manifest | `canon.py` + `e01`; 130,921 raw WT windows reduced to 51,766 canonical alleles and 175,668 allele-by-context decision groups |
| audit of alternative designs vs distinct edits vs true replicates | `e01`; 654 metadata-identical replicate groups, 32,199 sites whose several windows are one allele, 7,702 alleles reachable by two protospacers |
| corrected metric report | `e02`; four groupings, replicates averaged before ranking, protospacer-clustered intervals |
| context-overlap / identifiability audit | `e01` (fold purity under canonical keys), `e03` (variance components), `e04` (predictive identifiability) |
| interaction diagnostics on common support with locus-disjoint splits, replicate checks, additive and shuffle controls | `e03` + `e04` |
| injected-interaction control | `e04`; recovers a signal at 5% of SD(D), so the nulls are nulls and not a power failure |
| reference-budget curves | `e05`; 364 episodes over 20 contexts × 6 budgets × 5 support draws |
| external-data eligibility manifest | `e06` |
| go/no-go | `GATE_A_REPORT.md` |
| pilot power from grouped residuals (§8B) | `e08` |

Two things were added that the plan did not ask for, because the results made them the
obvious next question: `e07` (the interaction is predictable — does using it help?) and the
reserved panel with its pre-registration (the corrected estimand needs candidate depth that
fold 0 does not have).

---

## 3. Decisions taken during the work

**Where to run what.** Fold 0 was used only for descriptive re-scoring of frozen
predictions, never for fitting. Anything involving a choice was run on development folds,
using round 4's five out-of-fold checkpoints so no row's label shaped the representation
that describes it. This is why `extract_embeddings.py` exists at all rather than reusing an
existing prediction file.

**The latent scale.** Efficiency is a bounded proportion with a 41% zero block, so
differences on the raw scale are heteroscedastic. The arcsine-root transform was adopted
after checking on the 654 replicate groups that it flattens the mean-variance relationship
(group SD's correlation with group mean falls from 0.53 to 0.28). Raw-scale results are
reported alongside throughout.

**Two nulls were rebuilt mid-analysis after they proved unfair.** The first additive null
compared observed reversal rates against a model whose design contrasts were shrunk toward
zero, which manufactured reversals the data would never show; it was replaced with one that
conditions on each quartet's own additive contrast estimate. The first version of `e07`
added the fitted correction at unit weight, which damaged the ranking for a reason unrelated
to the hypothesis; it was replaced with a blend weight and a selectivity fraction both
chosen on training loci, where zero recovers the frozen model exactly. Neither change
rescued the arm, and both make the negative credible rather than an artefact of scaling.

**The most obvious representational excuse for the null was closed.** `e05`'s first pass
used principal directions of the raw embedding, which are dominated by locus identity.
Arms fitted in the allele-centred subspace — where a design-by-context interaction has to
live — were added and behave the same.

**The reserved panel was sealed rather than spent.** Kim's large library turned out to be on
disk, not in the training mix, and to carry 32,925 fixed-allele decision groups against
fold 0's 2,412. The plan is explicit that a panel used to choose a model becomes development
data, so it was hashed and left unscored with the analysis declared in advance.

---

## 4. Problems hit and how they were handled

- **The corpus's edited-sequence windows are a design artefact, not an edit identity.** Four
  designs for one deletion produce four different WT/edited string pairs. Fixed by
  canonicalising and by extending flanks from each site's longest observed window; verified
  by hand on merged and unmerged examples.
- **DeepPrime's released `Edited74_On` column is masked with `x` outside the RT-PBS
  footprint.** A first pass treated the mask as sequence and concluded the library had one
  design per allele, which would have made it worthless. Recovered by aligning the unmasked
  block back onto the WT window using each row's own edit type and length.
- **A two-level MultiIndex `.loc` over 110,000 tuples ran for five CPU-hours without
  finishing.** Replaced with a single-string key and a `reindex`, which is seconds.
- **`pkill -f <pattern>` matched the shell running it**, killing two command batches
  mid-way; switched to killing by PID.
- **`e05`'s marginal-mean table is not a valid comparison** because the adaptation rank is
  chosen per target context, so two arms are averaged over different context sets. The
  paired within-episode table is the comparison and the report now says so.
- **One claim needed correcting after the fact.** The affine-recalibration arm was described
  as exactly zero at every budget; it is exactly zero except at the smallest budget, where in
  one episode of eighty the least-squares slope on twelve support points came out negative
  and inverted the ranking. Corrected in the report, the experiment's own markdown, and the
  verifier — and it is a useful observation in its own right about tiny support sets.

---

## 5. Result in brief

The interaction the last three rounds chased is real (28.8% of the design-contrast variance,
CI [27.9%, 29.8%]; supported reversals 4.1% against 0.1% under an additive null) and is
predictable at held-out loci (R² +0.392 against +0.012 for a structured shuffle; direction
correct on 94.1% of the top 5% by predicted magnitude). That overturns round 6's conclusion
that the categorical context labels are the binding constraint.

It is also worth 0.0070 efficiency — the regret of transferring another context's *true*
ranking, which is more information than any adaptation method gets. Reference panels of up to
192 measurements in an unseen context do not beat no adaptation, with shuffled support labels
performing as well as real ones; and an explicit context correction for known contexts buys
+0.0023 top-1 accuracy, which a context-free control matches.

Meanwhile the manuscript's comparison improves under correction rather than degrading: on
the fixed-allele decision the margin over OptiPrime roughly doubles, and the single
ordinal-S4D member matches the five-member ensemble.

So: Gate A passes on the science and fails on the utility. Promote the corrected decision
benchmark to the spine of the paper, demote the reference-panel capability claim, and publish
the interaction as a bounded characterisation with its two failed conversions attached.

---

## 6a. The panel result, added after the review

The gap the Gate A report identified — a corrected estimand with no surface deep enough to
test it — is now closed on retrospective data, and the answer was not the one I expected. Kim's large variant library was on disk,
absent from OptiPrime's training mix, and unusable until its masked edited-sequence column
was aligned back onto the WT window. Reconstruction validated three independent ways
(nick position on 153,974 rows, PBS/RTT rules on 100% of corpus Kim rows, and 1.0000/0.9968
agreement with the corpus where locus and frame match). After removing every protospacer and
allele overlap with training: 30,475 decision groups, 8,775 at depth five or more, against
fold 0's 2,412 and zero.

Released DeepPrime trains on that library, so the exposure audit excluded it as a
comparator. PRIDICT2 was measured and rejected too — it enumerates only PBS 7-15, covering
65% of the panel's designs, which would restrict the candidate set by geometry.

**The result inverts the manuscript's comparative claim on this estimand.** OptiPrime
achieves 0.0369 with its single nominated candidate against PE-RankFormer's 0.0363
(−0.00064, 95% CI [−0.00089, −0.00038], p = 0.001), and the deficit grows monotonically with
candidate depth (−0.00023 at depth two, −0.00246 at depth eight or more) and with the value
of the decision (−0.00328 where perfect selection is worth ≥0.05). Both beat random by a
wide margin, and at a three-candidate budget they are indistinguishable. PE-RankFormer's
pooled Spearman on the same panel is *higher* (0.709 vs 0.693), which is the cleanest
statement of the plan's thesis: pooled correlation and the deployment decision are different
quantities, and this model was selected on the first.

## 6. What was not done

- No model training of any kind. Every negative result is a negative about a frozen encoder
  plus a linear adaptation at the budgets and ranks tested. Full fine-tuning at matched cost
  is the one untested variant worth a single clean run.
- No external data acquired. `e06`'s queue is a plan; published library sizes in it are not
  counts of usable rows.
- No wet-lab work, and none of the plan's priority-2 or section-6 experiments.
- No baseline retraining. DeepPrime, PRIDICT2 and OPED were not run; the comparison remains
  PE-RankFormer against OptiPrime's frozen predictions.
- The reserved panel was not scored.
- Nothing was committed, and the manuscript was not edited.
