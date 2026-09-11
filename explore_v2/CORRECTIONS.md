# Corrections to E01–E08, after review

Dated 8 September 2026, following `NEXT_STEPS_PLAN.md`. Numbers labelled *as published* are
the version-1 values that were in the reports when the review was written; current values
are canonicaliser version 2 and differ in the third decimal. That review makes seven technical
criticisms of the work in this directory. I checked each one. **Five are correct, one is
correct with a smaller consequence than implied, and one is a fair reading that I want to
state more precisely.** Two of them change numbers I published, and one changes a
conclusion I drew. The original reports are left as written; this file is the correction of
record, and `GATE_A_REPORT.md` now points here.

Nothing below was found by re-reading my own prose. Every item is a measurement.

---

## C0. My own mechanism for the regrouping gain was wrong (E19)

E13 measured that the pairwise ranking loss groups rows by a design artefact, and E17a's
replay of the real sampler confirmed the consequence exactly: **100.0% of its sampled pairs
compare two designs of identical RTT length**, and repairing the key takes measured exposure
from 15,695 to 62,028 sampled pairs per epoch. The live training logs then reported those two
numbers to the digit, so the measurement is sound.

**The causal story I built on it is not.** `corpus.group_key` drives both the batch sampler
and pair eligibility, and E19's arm matrix separates them on fold 0 (1,463 informative
decision groups, 439 protospacer clusters, no arm trained or early-stopped there):

| arm | what it changes | seeds | mean achieved @1 | pooled rho | pairs/epoch |
|---|---|---:|---:|---:|---:|
| A | original batching, original ranking | 3 | 0.13360 | 0.8905 | 15,695 |
| D | canonical batching **and** ranking | 3 | 0.13453 | 0.8950 | 61,978 |
| E | canonical batching, **ranking disabled** | 3 | 0.13459 | 0.8949 | 61,975 |
| C | canonical both, 1 pair per group | 1 | 0.13463 | 0.8950 | 25,253 |
| G | canonical both, 16 pairs per group | 1 | 0.13357 | 0.8968 | 146,088 |
| **F** | canonical both **+ feature branch** | 3 | **0.13523** | **0.8974** | 61,976 |
| H | canonical batching, features, **no ranking** | 3 | 0.13473 | 0.8971 | 61,976 |
| B | canonical batching, **original** ranking key | 1 | 0.13205 | 0.8703 | 6,975 |

The repair itself replicates: **D − A = +0.00093, same sign in all three seeds**
[+0.00053, +0.00104, +0.00121]. But it does not work the way I said:

- **D − E = −0.00006**, and it flips sign across seeds: with batching held fixed, whether the
  ranking loss is on or off makes no reliable difference at all.
- **D − C = −0.00008**: cutting the pair budget from four to one changes nothing.
- **G − D = −0.00098**: raising it to sixteen makes things worse.

So pair exposure is not the mechanism, and more of it is not better. What helps is **batch
composition** — putting rows that share an intended allele and context into the same batch —
which arm E delivers with no pairwise term at all (E − A = +0.00099, same sign in all
three seeds). Arm B is the diagnostic
that makes this concrete: keep canonical batching but restore the original ranking key and
exposure *collapses* to 6,975 pairs per epoch, below arm A's, because the artefactual groups
are now scattered across canonically-composed batches; that arm is the worst of the seven.

**Withdrawn:** the claim that the gain comes from letting the ranking objective see the
geometry comparisons it was blind to. The blindness is real and worth reporting; it is not
what the repair fixes. `e13_regroup_training.md`, `RESULT.md` and `SUMMARY_REPORT.md` are
corrected accordingly.

**Retained, with the effect resized by the extra seeds:** the **feature branch** is the arm
motivated by the diagnosis that PE-RankFormer never receives PBS and RTT lengths as scalars
while OptiPrime models RTT length as an explicit repeat count, and arm F is the best arm in the
matrix (F − A = +0.00164, same sign in all three seeds, and the best pooled rho of the eight).
Its incremental contribution over arm D is +0.00071 but flips sign in one seed; the clean
statement of its value is F − E = +0.00064, same sign in all three seeds.

**A null worth recording:** selecting the checkpoint by the decision metric rather than by
pooled validation Spearman changes the chosen epoch in 16 of 18 runs and is worth
**+0.00004** in fold-0 achieved efficiency. The two selectors disagree about *which epoch*
and not about *what they deliver*, so this cheap intervention is a null, not a lever.

## Confirmed, and it changes a published conclusion

### C1. E04's headline was inflated, and its controls were not comparable (plan §2.1, §2.2, §2.4)

Three defects, all real, all in the same experiment:

**Mixed coordinate systems.** `extract_embeddings.py` writes five independently trained
networks' hidden vectors into one matrix, and E04 subtracts one network's embedding from
another's in **57,821 of its 110,321** version-1 quartet contrasts (the review's count; I reproduce it
exactly). I then measured whether that is harmless, by embedding the same 8,000 fold-0 rows
with three of the checkpoints:

| comparison | result |
|---|---|
| median per-coordinate correlation | **+0.48** |
| linear CKA | 0.91 |
| SD of a genuine design contrast within one checkpoint | 0.396 |
| SD of the pure checkpoint offset (same rows, two networks) | **0.322** |

The representations agree up to a rotation — which is exactly what coordinate-wise
subtraction cannot survive. Over half of E04's features carried a nuisance term about 80%
the size of the signal.

**The outer split did not extend through the representation learner.** 23.6% of locus
components span more than one official fold (36.2% of rows), so a locus held out from the
ridge could have had relatives in the encoder's training data.

**The controls changed the representation as well as the mechanism.** E04's shuffles were
run on the 17 engineered features while its headline used embeddings, so "+0.392 against
+0.012" is not a like-for-like comparison. Its PCA was also fitted before outer splitting.

**E09 repairs all three at once** by using only the 47,470 quartets whose four measurements
all lie in one official fold — both designs then embedded by the checkpoint that held that
fold out, so one coordinate system and no exposure — with locus-grouped CV inside each fold
and every transform fitted inside the training split:

| arm (identical features, dimension, penalty grid, splits) | E04 as published | E09, clean |
|---|---:|---:|
| embedding × context | +0.392 | **+0.184**, range [−0.056, +0.299] |
| embedding, design contrast only, no context | (not run) | +0.066 |
| **shuffled context, same features** | +0.012 *(wrong features)* | **+0.077** |
| shuffled D, same features | +0.007 *(wrong features)* | −0.005 |

So the honest statement is: the context term beats a representation-matched shuffle on
average, +0.184 against +0.077, but **one of five folds is worse than its own shuffle**, and
roughly 40% of what E04 called interaction predictability is design-pair-level structure
that needs no context at all. On the historical held-out fold 0 the model does not
generalise even nominally (R² −2.82 on 3,431 quartets over 508 loci); that surface is too
small to fit this model, which is consistent with its having no candidate depth either.

**What this changes in `GATE_A_REPORT.md`:** the third verdict row said "Better than a
structured shuffle: **Yes, decisively.** +0.392 against +0.012". It should read: *on average
and unstably, +0.192 against +0.078, negative on one of five folds.* And §3's claim that
round 6's conclusion "is wrong as stated" is too strong. What survives is narrower: the
categorical context labels plus a frozen representation carry *some* predictable interaction
structure beyond a context-free baseline, unstably, and much less than E04 implied.

**What it does not change:** the Gate A verdict. E03's variance decomposition is a
descriptive calculation on measurements and involves no embeddings, so the interaction's
existence and size are untouched. E05 and E07 remain null, and E07's null is now *more*
credible, since the residual-R² gap it was set against was inflated the same way.

### C2. Large D is not a reversal, and I let the two run together (plan §2.6)

`D = Δ(c1) − Δ(c2)` can be large with both contrasts the same sign — a change of margin,
not a change of choice. E09 separates the tasks. Across development folds, for the top 5%
of quartets by predicted |D|:

| quantity | E04 as published | E09, clean |
|---|---:|---:|
| accuracy on the **sign of D** | 0.938 | 0.70 – 0.90 (mean 0.81) |
| **actual order-reversal rate** in that stratum | — | **0.000 – 0.057** |

So the stratum the model flags as high-interaction almost never contains an order reversal.
The recommendation in `GATE_A_REPORT.md` §6 to "enrich the panel for high predicted |D|"
was therefore aimed at the wrong event: it would enrich for margin changes, not for changes
of choice. The reversal-classification numbers E09 reports are based on at most seven
reversals per fold and should be treated as uninformative rather than as a result.

Relatedly, the plan is right that the transfer regret is the gain over **one
particular chooser** — transferring the observed winner of a two-design pair between
contexts, on a filtered quartet population. It is not an upper bound on improvement over
PE-RankFormer, and it does not bound gains at greater candidate depth. It should be quoted
as a descriptive quantity for its stated population, and `GATE_A_REPORT.md`'s phrase "the
ceiling for any context-adaptation method on this data" overstates it.

---

## Confirmed, and fixed in code

### C3a. Feature standardisation was refit on the evaluation surface (found 2026-09-09)

`attach_family_c_features` computes each column's mean and standard deviation from whichever
rows are passed as `train_idx`, and those statistics are **not** stored in the checkpoint.
That is correct during training, where `train_idx` is the run's own training split. It is
wrong at evaluation: E22 scored the feature-branch arm on the reserved panel with
`train_idx = arange(len(panel))`, so the panel was standardised by its own mean and SD.

This restandardises the inputs away from the scale the model learned to read, and it erases
precisely the distribution shift the panel exists to measure. On this corpus the shift is
large:

| feature | panel mean, in training SDs | panel SD / training SD |
|---|---:|---:|
| `pbs_length` | -1.35 | 2.03 |
| `pbs_tm` | -0.86 | 2.16 |
| `rtt_length` | +0.62 | 1.17 |
| `edit_position_from_nick` | +0.55 | 1.29 |

**Fixed** in `explore_v2/featstats.py`, which recovers a checkpoint's training-fold statistics
and applies them frozen to any surface; E22 and E23 were re-scored. E19 was **not** affected --
it already standardised on `fold >= 2`, the correct training folds -- so the fold-0 matrix
conclusions in C0 stand unchanged.

**It changes a number, and then a second experiment changed its interpretation.** Under
honest scaling arm F's panel advantage over its matched control falls from +0.00140 to
+0.00115, and against the published member it moves from +0.00003 (p = 0.75, "draws level")
to **-0.00019 (p = 0.044, slightly behind)**.

My first reading of that was that the feature branch extrapolates badly on a shifted library
and costs accuracy. **That reading is withdrawn.** Scoring arms D, E and H on the panel
isolates the branch with the ranking loss held off: H - E = +0.00018 (p = 0.068), which is
close to inert rather than harmful. What the standardisation fix removed was an artefact that
had *flattered* the branch. The finding that survives is narrower and still worth reporting:
an evaluation that restandardises features on the test surface reports a better number than
the model would actually deliver, because it erases the distribution shift it exists to
measure.


### C3. The allele key was not strand-invariant (plan §2.3)

Confirmed exactly as described. Version 1 left-normalised the indel on the forward strand
and then reverse-complemented that single representation; an indel inside a repeat has its
own leftmost position in each orientation, so the two disagreed.

```
WT = GGGACACACTTT, edited = GGGACACTTT
  forward input   -> AAGTGT|GT>|CCC
  revcomp input   -> AAA|GT>|GTGTCC      (should be identical)
```

Scope: substitutions were always correct; **61.7% of the corpus's 51,012 distinct indel
sequence pairs** produced an orientation-dependent key, and indels are 35.1% of rows and
17,021 of 52,507 alleles.

Fixed in `canon.py` (`CANON_VERSION = 2`): each orientation is now normalised
independently and the minimum taken, which is invariant by construction; verified on
deletion, insertion and substitution cases. `_v2common.load_corpus` now puts the version in
the cache filename, so an older manifest cannot be silently reused.

**Measured consequence, which is smaller than the defect rate suggests:**

| | v1 (buggy) | v2 (fixed) |
|---|---:|---:|
| canonical alleles | 52,507 | 51,766 |
| decision groups | 176,554 | 175,668 |
| corpus groups, ≥2 designs | 55,090 | 54,390 |
| corpus groups, ≥5 designs | 9,208 | **9,543** |
| alleles reached by >1 protospacer | 7,702 | 8,024 |
| **fold 0 groups, ≥2 / ≥3 / ≥5 designs** | **2,412 / 85 / 0** | **2,412 / 85 / 0** |

The fold-0 counts — which is where E01's and E02's conclusions live — do not move at all,
and correct merging *increases* the deep-candidate count. The published E01–E09 artifacts were produced with v1 keys. The recompute under v2 is the
first item of the next work program rather than a silent edit here, because it cascades
through every downstream cache — and the cache itself was the hazard the review identified.
Both caches are now versioned (`decision_manifest_v1.parquet`,
`corpus_canonical_v1.parquet`), every downstream script reads through
`_v2common.require_manifest()`, and that helper **raises** rather than falling back if the
version it needs is absent:

```
decision_manifest_v2.parquet is missing. The canonicaliser version changed, so the cached
manifest is stale rather than absent: re-run explore_v2/e01_decision_manifest.py ...
To reproduce the published numbers instead, set CANON_VERSION = 1 in explore_v2/canon.py.
```

So E02–E09 now fail loudly against the fixed canonicaliser until E01 is rebuilt, which is
the intended behaviour: it is not possible to compare groups defined two different ways by
accident. Reproducing the published numbers requires `CANON_VERSION = 1`.

### C4. E05's episodes were not reproducible from the recorded seed (plan §2.5)

Confirmed: episodes were seeded with `hash(ctx)`, and Python randomises string hashing per
process unless `PYTHONHASHSEED` is set. The recorded integer seed therefore did not
reproduce them. Fixed with a stable BLAKE2b digest of the context name.

Also confirmed and **not** fixed, deliberately: `budget` is part of the episode seed, so the
support/query split is redrawn at every budget and the budget curves compare different query
populations. Fixing it changes the episode population, so it belongs in a rerun with a
nested support design, not an edit to a script whose outputs are already published. A
comment in `e05_reference_panel.py` records this.

### C5. A degenerate-direction bug that the retest exposed

Not in the review — E09 found it. Standardising PCA scores with
`np.maximum(sd, 1e-9)` turns a direction with negligible training variance into a feature
with enormous leverage. On fold 0, small enough for this to bite, the first E09 run returned
R² of −4 × 10¹¹. Fixed by dropping directions below `1e-6 × max(sd)` and widening the
penalty grid. Fold 0's honest value is −1.78: the model does not generalise there at all.

---

## Confirmed, with a smaller consequence than implied

### C6. DeepPrime trained on the reserved panel's source file (plan §2.7)

Confirmed: `external/deepprime/train_base.py:26` reads
`data/DeepPrime_dataset_final_Feat8.csv`, the source of `reserved_panel_kim_large.parquet`.
A released DeepPrime checkpoint therefore cannot be described as an independent comparator
on that panel.

The consequence is bounded because `PREREGISTRATION_reserved_panel.md` already names
PE-RankFormer versus OptiPrime as the primary comparison and neither trained on it. What is
missing is the explicit exposure matrix, and the wording needs to change: an exposed
released tool must be labelled as such rather than reported as a held-out baseline. The
pre-registration will be superseded by a versioned addendum before the panel is scored.

The review is also right that "sealed" was too strong for what E06 did: it computed
aggregate label statistics on the panel (mean efficiency, within-allele outcome range). No
comparative prediction has been evaluated, but **"reserved for comparative evaluation"** is
the accurate phrase and the reports now use it.

---

## Fair, and stated more precisely

### C7. E07's tuning was in-sample (plan §2.5)

Confirmed: the blend weight and the selectivity fraction were chosen using corrections
predicted on the same training rows they were fitted on, which is optimistic. That
optimism inflates the *positive* direction, and E07's finding was that the context
correction buys +0.0023 accuracy against a shuffled control's +0.0019 — so the bias runs
against the arm I was testing and the null is safe. It still needs inner out-of-fold
predictions before the number is quoted as an estimate rather than an upper bound. Likewise,
E05's "statistically identical" was loose: it describes overlapping means over pooled
support draws, not an equivalence test with a declared margin.

---

## Standing after these corrections

| claim | status |
|---|---|
| Fold 0 cannot support the manuscript's decision metric at its candidate depth (2,412 / 85 / **0**) | **unchanged**, and invariant to the key fix |
| The fixed-allele margin over OptiPrime roughly doubles; +0.0906 accuracy, +0.0091 efficiency | **holds on fold 0 only.** On the reserved panel's 30,475 groups OptiPrime wins the same estimand by +0.00064 achieved efficiency (p = 0.001), widening with depth. See `RESULT.md` |
| Interaction is 28.8% of the design-contrast variance, 4.1% supported reversals against 0.1% | **unchanged** (descriptive, no embeddings) |
| Transfer regret 0.0070 | **unchanged as a number, narrowed in scope**: it is the gain over one specific chooser on a filtered population, not a universal ceiling |
| Interaction predictable at held-out loci, R² +0.392 against +0.012 | **corrected to +0.192 against +0.078, unstable across folds** |
| Direction predictable at 93.8% on the top 5% | **corrected to 0.70–0.90 for the sign of D**, and that stratum contains essentially no order reversals |
| Round 6's "context labels are the binding constraint" is wrong | **withdrawn as stated**; the weaker claim is that some predictable structure exists beyond a context-free baseline |
| Reference panels do not beat no adaptation (E05) | **unchanged**, with the reproducibility and budget-nesting caveats above |
| An explicit context correction does not improve the decision (E07) | **unchanged**, and more credible now |
| Gate A: passes on the science, fails on the utility; do not centre the paper on the interaction | **unchanged** |
