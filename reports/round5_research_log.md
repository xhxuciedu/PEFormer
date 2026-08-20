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
