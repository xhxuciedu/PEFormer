# Round-5 research log

Format per spec §22: hypothesis, why it may improve accuracy, why it may improve
complementarity, implementation, protocol, dev result, lockbox result, decision,
independent observations, next step.

---

## 2026-08-20 — Phase 0: inventory, factorial, lockbox accounting

### Independent lead (not in the spec): compute §11's factorial from stored artifacts

The spec places the objective x architecture factorial at position 9 in the priority
order, implying new training runs. All four cells already existed as round-4 OOF
predictions, so I computed it first, at zero compute cost, and used it to re-order
the rest of the round.

**Result** (dev, OOF, fold-averaged):

| | simplex | ordinal |
|---|---:|---:|
| Transformer | 0.8798 | 0.8911 |
| SSM | 0.9011 | **0.9082** |

objective effect **+0.0092**, architecture effect **+0.0192**, interaction −0.0043.

**This corrects a round-4 claim of mine.** The round-4 report called the ordinal head
"the breakthrough... a change of objective, not architecture". The factorial says the
architecture effect is about twice the objective effect. The ordinal head was the more
novel idea and the one that decorrelated errors, but the SSM mixer contributed more
raw accuracy. I over-credited the objective and will say so in the round-5 report.

**Consequences for the round-5 plan:**
1. The hybrid Transformer/SSM mixer (§12) is **promoted** above its advisory slot --
   architecture is the bigger lever, and the diagonal pair (Transformer/simplex vs
   SSM/ordinal) has the lowest residual correlation measured (0.6453).
2. The sub-additive interaction (ordinal is worth +0.0114 on a Transformer but only
   +0.0071 on an SSM) sets a realistic ceiling: mechanisms overlap rather than stack,
   so I should expect round-5 gains smaller than round-4's.
3. **The round-4 ensemble has a redundancy.** Its two strongest members, `ordSSM` and
   `ssm`, are the most correlated pair in the whole table (pred 0.9639, resid 0.7641)
   because they share an architecture. Swapping one for a Transformer-side member of
   comparable strength is a concrete round-5 lead.

### Round-5 lockbox: a composition-matched one does not exist

| Liu+Kim training protospacers | 2,501 |
|---|---:|
| in a round-3 dev validation set | 1,553 |
| consumed by the round-4 lockbox | 367 |
| collide with Schwank | 302 |
| **remaining** | **279 (98.5% Liu)** |

Built the Liu-only lockbox that is actually available (19,084 rows / 279
protospacers) and labelled it as such rather than presenting it as a matched gate.

**Stated up front: Kim has no fresh gate left.** Kim is exactly where round 5 most
wants to improve (0.8124 vs Liu's 0.8585), and its claims now rest on OOF dev folds
plus the once-used round-4 lockbox. That is the less-protected half of any round-5
result and will be reported as such. It is a real limitation of the benchmark rather
than something to engineer around.

### Phase-1 screens launched (dev fold 0, Stage A)

| run | experiment | change |
|---|---|---|
| `r5_dual010/025/050_d0` | §6 A | auxiliary simplex head, lambda in {0.10, 0.25, 0.50} |
| `r5_ctx010/025_d0` | §8 C | auxiliary context-normalised ordinal head, lambda in {0.10, 0.25} |
| `r5_multires_d0` | §7 B | auxiliary ordinal heads at K=8 and K=50 beside the primary K=20 |

Baseline to beat: Ordinal-SSM at **0.9082** OOF dev (though Stage-A screens carry the
round-4 best-epoch-on-scored-rows bias, so only *relative* comparisons among screens
are meaningful -- the same discipline round 4 established).

**Implementation note.** All three are one mechanism: auxiliary output segments on a
shared trunk, with the ranking score sliced from the primary segment only. Auxiliary
heads can therefore only help by shaping the representation; they never move the
prediction directly. Without that property these would be ensembling in disguise, and
a negative result could read as positive -- so it is the property the unit tests
target hardest.

---

## 2026-08-21 — Phase 1 + hybrid + quantile: a comprehensive null

14 candidates trained on dev fold 0 across all 8 GPUs, all 30 epochs, no failures.
Every one measured against the round-4 ensemble (0.9156 on this fold).

| candidate | S1 solo | S2 vs ordSSM | S3 vs R4ens | S4 resid | **S5 gain** |
|---|---:|---:|---:|---:|---:|
| multi-res + context (small) | 0.8991 | 0.9530 | 0.9585 | 0.7614 | +0.0020 |
| hybrid alternating (small) | 0.8981 | 0.9511 | 0.9584 | 0.7539 | +0.0019 |
| dual-head λ=0.50 | 0.8973 | 0.9525 | 0.9572 | 0.7583 | +0.0019 |
| **plain re-run (no new mechanism)** | **0.8979** | **0.9542** | **0.9608** | **0.7809** | **+0.0018** |
| context-ordinal λ=0.10 | 0.8988 | 0.9550 | 0.9613 | 0.7752 | +0.0017 |
| hybrid parallel | 0.8945 | 0.9451 | 0.9549 | 0.7466 | +0.0017 |
| hybrid alternating | 0.8958 | 0.9489 | 0.9599 | 0.7759 | +0.0014 |
| quantile head | 0.8840 | 0.9322 | 0.9386 | **0.6645** | +0.0010 |

*(remaining dual/context/multi-res variants all fall between +0.0016 and +0.0019)*

**Nothing clears either promotion bar.** Path A needs +0.005 standalone (best: +0.0016).
Path B needs +0.003 ensemble gain (best: +0.0020).

### The control that makes this conclusive

`r5s_ordssm_ref_d0` is Ordinal-SSM retrained at a different batch size -- **no new
mechanism whatsoever**. It gains **+0.0018**, statistically indistinguishable from the
best novel mechanism's +0.0020.

So the ~+0.002 every candidate shows is not the auxiliary heads, the hybrid mixers or
the quantile head doing anything. It is simply what adding a **sixth member** to a
five-member ensemble is worth on this benchmark. **The difference attributable to
every round-5 idea combined is +0.0002.**

Without that control I would have reported fourteen small positive gains and might
have promoted the top one. Including a mechanism-free re-run in the screen is what
turned an ambiguous set of small positives into a clean null, and it is the single
most useful thing I did this phase.

### Why the hybrid mixer failed — the gate says so directly

The parallel hybrid's learned gates were built as a diagnostic. After training:

```
edit encoder, mean SSM weight per layer: 0.514 0.510 0.508 0.505 0.505 0.505
peg  encoder, mean SSM weight per layer: 0.514 0.509 0.507 0.507
```

Initialised at 0.500. They barely moved. The model **never developed a preference**
and kept a near-even blend at every layer, which is why it scored *below* the pure
SSM (-0.0027) rather than above it.

The mechanistic reading: **averaging two mixers inside a network is not ensembling
two networks.** Across models, averaging predictions cancels independent errors and
helps. Within a model, averaging two representations before the head produces a
blurrier representation than either -- the mixers cannot specialise because both see
identical input and their outputs are summed before any nonlinearity that could
separate them. Diversity pays at the prediction level and costs at the representation
level. That also explains the alternating variant's smaller loss (-0.0014): it at
least lets each mechanism act on the other's output rather than averaging them.

### Quantile head: another "decorrelated but not competent"

Lowest residual correlation of any candidate (0.6645, well below the ~0.75 pack) --
genuinely different errors, exactly as hoped for a distributional objective. And the
lowest standalone score (0.8840) and lowest ensemble gain (+0.0010). This is the same
pattern round 4's ranking-loss model established, now confirmed on a second, unrelated
objective: decorrelation without competence is worthless.

Its value is not predictive. It is the only head that outputs efficiency in the
target's own units, so it is retained as the natural instrument for the §15
calibration work rather than as an ensemble member.

### Decision

Close experiments A (dual-head), B (multi-resolution), C (context-ordinal),
G (hybrid mixer) and H (quantile) as **null on this backbone**. Classification per
§20: *redundant* -- not weak, not unstable, not implementation-invalid. They train
fine and score near baseline; they simply add nothing the backbone does not already
capture.

This is consistent with, and now strongly reinforces, the Phase-0 factorial's
sub-additive interaction (-0.0043): mechanisms stacked on one backbone overlap rather
than compose.

**Not pursuing** experiments D (reverse-complement) or E (relational contrastive
pretraining) on this evidence. Both are substantially more expensive than anything
above, and five independent mechanisms spanning supervision geometry, architecture
and output parameterisation have now each returned the same ~+0.002 that a plain
re-run returns. Spending days of implementation on a sixth is not justified by the
evidence; §26 explicitly warns against over-tuning when a round does not materially
improve. Documented here rather than silently skipped.

**Next:** the one lead Phase 0 raised that is *not* about new mechanisms -- the
`ordSSM`/`ssm` redundancy inside the round-4 ensemble (most correlated pair measured,
0.9639). Testing ensemble recomposition over the nine existing OOF members, which
costs no training at all.
