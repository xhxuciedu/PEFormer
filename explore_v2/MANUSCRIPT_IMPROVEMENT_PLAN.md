# Plan: improving the manuscript, and improving the model

8 September 2026. Written against `reports/paper/pe_rankformer_paper.tex` (1,799 lines) and
the E01–E18 evidence in this directory. Purely computational; no new measurements required.

---

## 1. The problem in one sentence

**The manuscript optimised the model, selected its components, chose its ensemble and set its
stopping rule on pooled Spearman — a metric that ranks PE-RankFormer and OptiPrime in the
*opposite* order to the deployment decision the paper claims to serve.**

On the reserved panel, on 30,475 fixed-allele decision groups neither model was trained on:

| | pooled Spearman | achieved efficiency @1 |
|---|---:|---:|
| PE-RankFormer | **0.7091** | 0.0363 |
| OptiPrime | 0.6931 | **0.0369** |

Paired on target sites, the decision difference is −0.00064, CI [−0.00089, −0.00038],
p = 0.001, and it widens with candidate depth (−0.00249 at depth ≥8) and with how much the
choice is worth (−0.00329 where perfect selection is worth ≥0.05). Every performance decision
in the paper was therefore made through a lens that inverts the outcome of interest at the
margin. That, not any particular architecture, is the thing to fix first.

## 2. Why nothing could be decided before: a resolution mismatch

The paper states its development resolution honestly as **≈0.005 pooled ρ** and requires
three-fold replication above it. The decision differences that separate methods are one to two
orders of magnitude smaller:

| quantity | value | resolution available |
|---|---:|---|
| paper's promotion threshold | 0.005 pooled ρ | three dev folds |
| ordinal head's claimed contribution | +0.0092 pooled ρ | resolvable |
| ours − OptiPrime, decision | −0.00064 | panel: CI half-width 0.00026 |
| ranking-key repair, decision | +0.00184 | panel: resolvable |
| fold 0's decision population | 546 informative groups | CI half-width ≈0.0022 |

So fold 0 cannot resolve the effects that matter: asked there whether the simplex head beats
the ordinal head on the decision, the answer is +0.00075 with CI [−0.00156, +0.00284] and
p = 0.50 — no information. The panel, with 30,475 groups over 22,931 sites, has roughly
**8× tighter resolution** on the decision than fold 0 and 56× the group count.

**Consequence for the plan:** the evaluation surface and the selection criterion are
prerequisites, not refinements. Every lever below is measured as mean achieved efficiency @1
over canonical decision groups, paired and clustered on target sites, at panel scale.

## 3. Performance levers, ranked by evidence per GPU-hour

### L1. Repair the ranking loss's grouping — measured, replicate it

The pairwise ranking loss builds its groups from the raw `full_unedited | full_edited` window
pair, whose extent is set by the row's RTT length. Replaying the real sampler for one epoch:

| per epoch, measured | current key | canonical key |
|---|---:|---:|
| sampled pair instances | 15,695 | **62,028** (3.95×) |
| groups receiving any ranking update | 3.2% | **19.1%** |
| **pairs comparing equal RTT length** | **100.0%** | 6.8% |
| mean target gap of sampled pairs | 0.121 | 0.212 |

The objective has never once been asked to compare two pegRNAs of different RTT length — the
single axis on which alternative designs for one allele differ most. Repairing the key gives
+0.00184 on the panel (p = 0.001), +0.00258 on fold 0 (p = 0.01), scaling to +0.00561 at depth
≥8, and **recovers about 60% of the single-model gap to OptiPrime at every stratum**.

*Status:* one seed, and confounded — the same key drives batch composition and pair
eligibility. *Action:* the five-arm matrix (original/canonical batching × original/canonical/
exposure-matched ranking, plus ranking disabled), three paired seeds, ~15 fits, ~30 GPU-hours.
Run arms A and D first; if their direction is unstable across seeds, stop and diagnose.

### L2. Raise ranking exposure directly — untested, cheap, mechanistically motivated

`max_pairs_per_group = 4` and `min_pair_diff = 0.02` are what hold exposure to 15,695 pairs per
epoch. The paper's twenty-entry table of failed interventions contains **no ranking-loss
hyperparameter at all**, so this axis is unexplored, and it is now the axis with a measured
mechanism behind it. Sweep `max_pairs_per_group ∈ {4, 8, 16, all}` and
`min_pair_diff ∈ {0.005, 0.01, 0.02}` on top of the canonical key, with exposure logged rather
than assumed. Cheap: it rides on the L1 matrix.

*Caution:* lowering the gap admits pairs whose measured difference is within measurement noise
(per-measurement SD 0.0265 raw), so the sweep must be read against that floor, and the E03
replicate estimate is 99% Kim.

### L3. Give the backbone explicit design geometry — best-motivated architectural change

The deficit is specifically *within-allele geometry discrimination*, and OptiPrime's advantage
is a mechanistic one: its `syn` rate takes RTT length as an explicit repeat count, so it has a
structural model of RTT-length-dependent synthesis. PE-RankFormer's pegRNA encoder marks PBS
and RTT boundaries with segment ids but never receives their **lengths as scalars** — it must
count, which is exactly what a convolutional/state-space mixer does badly.

The instrument already exists: the Family C feature branch supplies 17 scalars including
`pbs_length`, `rtt_length`, `edit_position_from_nick`, PBS/RTT melting temperatures and MFEs.
The paper's own ensemble includes a feature-branch member and reports it as adding marginal
value — but the ordinal-S4D backbone that carries the headline has no feature branch.

*Action:* one arm, ordinal-S4D + feature branch + canonical key, evaluated on the decision.
This is a single fit and it tests a specific diagnosis rather than a general hope.

### L4. Weight the ranking loss by what the decision is worth

The deficit is entirely concentrated where the choice matters: −0.00329 where perfect selection
is worth ≥0.05, and **+0.00001 (p = 0.53) where it is worth <0.005**. The current loss weights
every eligible pair equally, so it spends most of its capacity on comparisons with nothing at
stake. Replace it with a **group-normalised, utility-weighted pairwise loss**: compare only
distinct candidates for the same allele and context; weight each ordering error by its measured
efficiency consequence with a pre-set cap; normalise within group before averaging groups so
candidate-rich loci do not dominate quadratically; keep the ordinal term as an anchor so
singleton rows still train.

*Primary ablation:* repaired uniform ranking versus utility-weighted ranking, with batches, pair
exposure, checkpoint rule and compute matched. Differentiable ranking is well-trodden ground;
the novelty claim here is the diagnosed problem and its fix, not the loss family.

### L5. Select checkpoints and ensemble members on the decision, not on pooled ρ — nearly free

E18 has now demonstrated the cost of the wrong criterion on a real component decision: pooled
ρ prefers the simplex head, the decision prefers the ordinal head, and the two disagree with
p = 0.009. A selection rule reading pooled ρ would have dropped the paper's titular component.

Checkpoints are still chosen by pooled validation Spearman. Save validation predictions on a
fixed epoch schedule and compare that rule against mean achieved efficiency @1 over canonical
validation decisions, **on the same trajectories** so the comparison is not confounded by
different stopping times.

For the ensemble, I ran the reselection already on fold 0: the pooled-best subset
(ordSSM+ssm+ordA, ρ = 0.8924) is *not* the decision-best (ssm+ordC), and the published
five-member system is neither. But the gain is +0.00051 achieved efficiency at n = 546 —
**underpowered, not a result.** Redo at panel scale before acting on it.

### L6. Match the ensembling before quoting any margin

The published member averages five checkpoints; both retrained arms are single models, and on
the panel both sit below the published member (0.0338 and 0.0356 against 0.0363). Part of every
apparent deficit is ensembling. No arm-to-published comparison should be quoted until ensemble
size is matched.

### What not to do

More architecture search: the paper already reports twenty matched-control failures. Context
adaptation and reference panels: closed by E05 and E07, with shuffled-label controls matching
the real ones, and E03 bounds the whole prize at 0.0070 efficiency. Model scale: three capacity
interventions are already in the harmful column.

## 4. Manuscript changes the evidence now requires

### Claims that are wrong as written

1. **Abstract, "Evaluated *within a protospacer* — the deployment-relevant task".** It is not
   the deployment task. Of the 735 protospacer groups with ≥5 rows, **100% span more than one
   source/cell/editor condition** and 26% contain more than one intended allele. Replace with
   the allele × context estimand; fold 0 has 2,412 such groups at depth ≥2, 85 at ≥3 and
   **none at ≥5**.
2. **Abstract, "the efficiency actually obtained from its top pick improves by only +0.015".**
   On the corrected estimand this is +0.0091 on fold 0's binary choice, and on the panel it is
   **−0.00064 — worse than OptiPrime.** The panel must be reported.
3. **§"What the model is worth at the bench" (`tab_utility`).** Computed by code with four
   defects: a candidate was a row rather than a distinct design (13 groups affected, 2 given a
   false top-three result), @k was computed wherever rows allowed, the random hit rate ignored
   tied maximisers (0.3354 where 0.4442 is correct), and the bootstrap returned p = 0.0005 for
   two identical predictors. Regenerate through `endpoints.py`, whose 13 tests pin those
   invariants.
4. **The title's "metric-matched ordinal supervision" — now supported, but for a different
   reason than the paper gives, and this is a result worth promoting.** E18 settles it at panel
   scale. The ordinal head is the better *chooser* and the pooled metric says the opposite:

   | | pooled Spearman | achieved efficiency @1 |
   |---|---:|---:|
   | ordinal + S4D | 0.7091 | **0.0363** |
   | simplex + S4D | **0.7126** | 0.0361 |

   Paired on sites: simplex − ordinal = **−0.00026, CI [−0.00044, −0.00006], p = 0.009**,
   widening to −0.00123 (p = 0.006) where the decision is worth ≥0.05. On fold 0 the same
   comparison gave **+0.00075 with p = 0.50** — pointing the wrong way with no power.

   So the paper's factorial (+0.0071 pooled ρ for the ordinal head on development folds) is
   not merely imprecise: pooled ρ **changes sign** for this component between the development
   folds and the panel, while the decision metric is consistent and favours the ordinal head at
   every stratum. The honest and stronger claim is that the ordinal head is matched to the
   *decision*, and that the pooled metric understates it. Rewrite the component analysis around
   the decision endpoint, and report the sign instability of pooled ρ as evidence for the
   paper's own methodological argument.

### Sections to add

5. **The corrected estimand and the canonical allele key** — 51,766 alleles from 130,921 raw
   windows; the fold structure is not locus-disjoint (80.7% of fold-0 protospacers appear in
   training); 40% of fold 0's two-candidate decisions are all-zero ties.
6. **The reserved panel and the reversal** — construction, the three-way validated
   reconstruction, the exposure audit that excluded released DeepPrime, the pre-registration,
   and the result with its depth and decision-value strata.
7. **The ranking-group defect and its repair** — the measured exposure table, the two arms, and
   the ~60% recovery, with the batch/pair confound stated.
8. **Extend `tab:neg`** with reference-panel adaptation (E05: shuffled support labels do as well
   as real ones at every budget) and the explicit context correction (E07: matched by its own
   shuffled control). Both are stronger negatives than most entries already there.

### Framing

The honest paper is no longer "a minimally mechanistic model matches a mechanistic one". It is
**"pooled correlation and the fixed-allele decision can disagree, here is where and why, and
here is a repair"** — a benchmark and diagnosis contribution with a partial method result. That
is a smaller claim than the current abstract and a more durable one. Keep the negative OptiPrime
comparison and the failed context branches in the record.

## 5. Execution order, gates, stop rules

| window | work | gate |
|---|---|---|
| days 1–2 | L5 checkpoint-selector harness; L6 matched-ensemble protocol; start public-data qualification | selector comparison runs on saved trajectories |
| days 3–7 | L1 arms A and D, three paired seeds | direction stable across seeds, else stop and diagnose |
| week 2 | L1 arms B, C, E; L2 exposure sweep; L3 feature-branch arm | know how much comes from grouping, batching, exposure, features |
| weeks 3–4 | L4 utility-weighted loss, bounded search; second backbone check | beats the corrected uniform baseline, or is dropped |
| weeks 4–6 | frozen evaluation on a newly qualified independent panel; manuscript revision | claim only what that surface supports |

**Budget:** ~2 GPU-hours per fit on the L40s, so L1's 15 fits are ~30 GPU-hours; L2 and L3 add
roughly 10 more. Stage it rather than opening a sweep.

**Stop rules.** If canonical regrouping does not reproduce under matched seeds, the mechanism
story goes and the paper is the benchmark plus the reversal. If utility-aware training does not
beat the corrected uniform recipe, publish the simpler one. If no independent panel qualifies,
narrow the claim rather than relabelling a Kim subset as independent.

**A standing constraint.** The reserved panel has been used once as pre-declared (E11) and twice
in disclosed development tests (E15/E16, E18). It is spent for confirmation. Every lever above
is therefore development evidence until it is re-tested on a surface that is still sealed, and
qualifying that surface is on the critical path — not a final decoration.
