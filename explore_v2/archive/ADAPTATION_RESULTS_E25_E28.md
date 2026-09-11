> Historical record, archived 10 September 2026. Adaptation interpretations are
> superseded by ../SUMMARY_REPORT.md and ../ADAPTATION_FOLLOWUP_RESULTS.md.
> Relative links below refer to the original explore_v2 location.

# Target-library adaptation: results (E25–E28)

Executed against `ADAPTATION_SELECTION_PLAN.md`, 10 September 2026. Twenty training runs.
Full write-up: `reports/adaptation_report.pdf`. All numbers verified by
`explore_v2/verify_report.py` (210 claims).

---

## 1. Verdict

**Adaptation works, and the simplest method is the one that works.** On held-out locus
components, ordinary efficiency fine-tuning takes PE-RankFormer from behind OptiPrime to
clearly ahead of it — and improves pooled correlation at the same time.

**None of the plan's three interesting hypotheses survives its own control.** A specialised
selection objective, a separated prediction/selection branch, and explicit candidate geometry
all fail to beat the simpler thing they were meant to improve on.

**One genuinely useful side result:** the selection-head arm reaches the same target gain as
ordinary fine-tuning for a quarter of the forgetting on the source corpus.

---

## 2. The partition (E25)

Alleles and protospacers form a bipartite graph — 13,803 alleles are reachable by more than
one protospacer and 9,893 protospacers install more than one allele — so decision groups are
**not** independent units. Its **20,276 connected components** are the assignment unit.

| split | components | groups | informative | depth ≥5 | candidates | mean RTT | mean eff. |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 10,370 | 19,357 | 15,337 | 4,658 | 72,010 | 24.45 | 0.0233 |
| val | 4,952 | 5,561 | 4,656 | 2,056 | 23,120 | 24.64 | 0.0258 |
| test | 4,954 | 5,557 | 4,675 | 2,061 | 23,044 | 24.55 | 0.0253 |

Balanced on edit class, depth **and RTT band** — the last because RTT geometry is the axis on
which these models are known to be biased, so an accidental imbalance there would confound
every comparison.

---

## 3. The primary endpoint (E27) — met

Test components: 5,557 groups over 4,954 locus clusters. Nothing was selected on this split.

| arm | seeds | achieved @1 | Δ vs OptiPrime | all seeds + | regret ↓ | pooled ρ |
|---|---:|---:|---:|:--:|---:|---:|
| S, pairwise | 1 | 0.04205 | +0.00185 | yes | +18.4% | 0.7107 |
| shared (control) | 3 | 0.04193 | +0.00173 | yes | +17.1% | 0.7624 |
| S, listnet | 1 | 0.04189 | +0.00169 | yes | +16.8% | 0.7164 |
| **P — ordinary fine-tuning** | 3 | **0.04178** | **+0.00158** | yes | **+15.7%** | **0.7771** |
| S | 3 | 0.04176 | +0.00156 | yes | +15.4% | 0.7224 |
| M | 3 | 0.04173 | +0.00153 | yes | +15.2% | 0.7492 |
| Sgeom | 1 | 0.04156 | +0.00136 | yes | +13.5% | 0.7183 |
| Z (frozen encoder) | 1 | 0.04112 | +0.00092 | yes | +9.1% | 0.6981 |
| G (geometry only) | 1 | 0.03968 | −0.00052 | **no** | −5.2% | 0.6052 |
| | | | | | | |
| OptiPrime (released) | — | 0.04020 | — | — | — | 0.6922 |
| PE-RankFormer, unadapted | — | 0.03952 | −0.00068 | — | −6.8% | 0.7102 |
| random | — | 0.02323 | — | — | — | — |
| perfect chooser | — | 0.05028 | — | — | — | — |

All full-budget arms: locus-clustered 95% intervals excluding zero at p = 0.001. Exceeds the
plan's pre-set 10%-regret-reduction target.

**In interpretable units:** the whole decision is worth 0.02705. OptiPrime captures 62.8%,
the unadapted model 60.2%, ordinary fine-tuning **68.6%**.

---

## 4. Every elaboration fails its control

Seed-paired contrasts on validation (where the arm choice was made):

| contrast | per seed | mean | seeds agree |
|---|---|---:|:--:|
| M − P | +0.00015, −0.00001, +0.00028 | +0.00014 | **no** |
| M − shared | −0.00026, −0.00037, +0.00007 | −0.00019 | **no** |
| S − P | +0.00005, −0.00017, −0.00000 | −0.00004 | **no** |
| M − S | +0.00010, +0.00016, +0.00029 | +0.00018 | yes |
| S − Z | +0.00087 | +0.00087 | yes |
| Z − G | +0.00059 | +0.00059 | yes |

- **A selection objective does not beat prediction.** S − P is sign-unstable; on test 0.04176
  vs 0.04178. A good conditional-mean estimate suffices for the top-one choice, as the plan
  itself anticipated.
- **Separating heads does not beat one score.** M − shared is sign-unstable, and `shared`
  spends 199,040 extra parameters against M's 198,657 — matched capacity, differing only in
  separation.
- **Explicit geometry hurts slightly:** Sgeom 0.04156 vs S 0.04176.
- **Candidate-set attention was not built.** The plan makes it conditional on a stable
  residual gap from the independent selector; there is none.
- **Controls confirm the encoder matters:** frozen-encoder Z gains only +0.00092, and
  geometry alone (G) does *not* beat OptiPrime (p = 0.134), ruling out "a few scalars fitted
  to the target library".

---

## 5. Data efficiency — adaptation is not cheap

| training groups | achieved @1 | vs OptiPrime | vs unadapted | p |
|---:|---:|---:|---:|---:|
| 200 | 0.03929 | −0.00091 | −0.00023 | 0.006 |
| 1,000 | 0.04021 | +0.00001 | +0.00069 | 0.971 |
| 5,000 | 0.04137 | +0.00117 | +0.00185 | 0.001 |
| 19,357 | 0.04178 | +0.00158 | +0.00226 | 0.001 |

**At 200 groups adaptation is worse than not adapting.** The crossover with OptiPrime falls
between 1,000 and 5,000 fully measured decision groups. A user with a hundred measured loci
should expect harm, not help.

---

## 6. What adaptation costs on the source corpus (E28)

Development fold 0 (20,509 rows), which no adaptation run touched. Start: ρ 0.8984.

| arm | source ρ change | source @1 change | target gain |
|---|---:|---:|---:|
| Z | +0.0000 | +0.00000 | +0.00092 |
| **S** | **−0.0075** | −0.00027 | **+0.00156** |
| P | −0.0287 | −0.00573 | +0.00158 |
| M | −0.0230 | −0.00619 | +0.00153 |
| shared | −0.0561 | −0.00766 | +0.00173 |

**Arm S reaches ordinary fine-tuning's target gain at roughly a quarter of the forgetting.**
This is where the dual-branch idea pays — not through better selection, but because a
selection-only objective perturbs the shared representation far less than a prediction
objective does. A laboratory adapting to one library it will keep working in should use
ordinary fine-tuning; a maintainer shipping one model across libraries should use arm S.

---

## 7. Limits

- **Not a matched-label comparison.** PE-RankFormer saw 19,357 labelled target groups;
  released OptiPrime saw none. This is a fair statement about the tool a user can build
  today, **not** evidence of architectural superiority. Adapting OptiPrime on identical loci
  and budgets is the comparison that would settle that, and it is deferred.
- **Not a confirmatory surface.** The panel informed E11–E24 before it was partitioned. The
  partition prevents new training leakage and the test components were untouched during
  selection, but a fresh library is what would make this confirmatory.
- **One library, one laboratory, one context** — every row is HEK293T with PE2.
- **The published model is a five-checkpoint ensemble; the adapted models are single
  checkpoints** started from `r4p2_ordSSM_cv1`. A matched-ensemble adaptation is the obvious
  next step and was not run.
