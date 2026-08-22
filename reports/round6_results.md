# Round 6 results — single-model performance: a comprehensive null

**Verdict: no candidate is promoted. Nothing replicated across dev folds.** The
frozen round-4 system (0.9079 held-out) stands unchanged.

25 runs across four independent leads — data distribution, rank consistency,
zero-inflation, and capacity — plus MoE and four scaling axes. Every apparent Wave-1
gain disappeared under replication.

## The decisive table

Promotion required **≥ +0.005 and positive on all three dev folds**.

| Candidate | fold 0 | fold 1 | fold 2 | mean | verdict |
|---|---:|---:|---:|---:|---|
| SSM depth 9+6 | **+0.0038** | −0.0018 | *(see note)* | +0.0010 | sign flip → reject |
| Source weights, mild | **+0.0033** | −0.0016 | +0.0011 | +0.0009 | sign flip → reject |
| MoE-4 experts | **+0.0032** | +0.0001 | — | +0.0017 | does not replicate → reject |

Each was the leader of its axis on fold 0. Each collapsed on fold 1.

**This is exactly the failure mode Wave 2 was built to catch,** and exactly what round
3 documented: fold-to-fold sign disagreement at the ~0.003 scale. Had round 6 stopped
at Wave 1 it would have reported three promising leads on three orthogonal axes, and
all three would have been wrong.

## Complete Wave-1 results (fold 0 only — none replicated)

| Lead | Candidate | Δ vs control |
|---|---|---:|
| 4 capacity | SSM depth 9+6 | +0.0038 |
| 1 distribution | source weights, mild | +0.0033 |
| — | MoE-4 (small track) | +0.0032 |
| — | MoE-4 (batch-512 track) | +0.0034 |
| 1 distribution | source weights, moderate | +0.0008 |
| 2c consistency | monotonicity penalty | +0.0007 |
| 2a consistency | true CORAL | +0.0005 |
| 4a capacity | ssm_state_dim 256 | +0.0002 |
| 4 capacity | ffn_dim 2304 | −0.0001 |
| — | MoE-8 experts | −0.0013 |
| 4a capacity | ssm_state_dim 128 | −0.0022 |
| 3a zero-inflation | hurdle head | −0.0022 |
| 1 distribution | source weights, aggressive | −0.0059 |

## What was actually learned

### 1. Source weighting has an inverted-U, and Schwank's data is not wasted

Mild +0.0033, moderate +0.0008, aggressive **−0.0059**. Cutting Schwank from 58.4% to
4.3% of the training gradient produced the **worst run in the wave**.

Schwank contributes 174k rows and **0%** of the evaluation set, so removing it looked
like free efficiency. It is not: its volume carries transferable signal that outweighs
its distributional mismatch. The risk was flagged in the plan before running, and the
three-point sweep is what answered it — a single setting would have produced a false
positive or false negative depending on which was chosen.

### 2. Rank consistency is an implementation gap that does not matter empirically

The ordinal head violates monotonicity on **100% of rows** (8 of 16 threshold pairs
on average) — it is not CORAL despite the code saying "CORAL-style". Fixing it
properly:

- true CORAL (shared weights, ordered biases): **+0.0005**
- monotonicity penalty (constraint without reparameterisation): **+0.0007**

Both flat. Averaging an incoherent CDF is evidently a perfectly good ranking statistic.
Classification: **implementation gap, empirically inert** — worth fixing so the
predicted distribution is coherent, not for accuracy. This was the plan's predicted
"least likely to pay" lead, and that prediction held.

### 3. Zero-inflation is already handled implicitly

The hurdle head (−0.0022) targeted a real structure: 28.4% of rows are exactly zero,
49.9% within Kim. It still lost. Classification: **redundant** — the ordinal head's
lowest thresholds already separate the zero block, so an explicit P(y>0) gate adds a
parameter path without adding information.

Note this does **not** contradict round 5's tie-floor finding (+0.013–0.016 on Kim for
every model). That gain came from *refusing to rank* within the zero block, exploiting
Spearman's tie convention. It was available to every model regardless of quality and
was rejected as unfair. The hurdle head tried to capture the same structure
*legitimately*, by predicting which rows are zero — and there is no gain available that
way. Taken together: the zero block contains **metric-exploitable** structure but not
**predictable** structure.

### 4. Capacity does not help in any direction tested

state_dim 128 (−0.0022), 256 (+0.0002), ffn 2304 (−0.0001), depth 9+6 (+0.0038 →
sign-flipped), MoE-4 (+0.0032 → did not replicate), MoE-8 (−0.0013).

The plan's argument was that the only prior scaling experiment had scaled the
*Transformer*, which the factorial showed was the worse backbone, so the SSM deserved a
try. It got one, on four axes. **The model is not capacity-limited.**

### 5. Orthogonal axes do not compose

MoE-4 alone +0.0032; MoE-4 × source weighting **+0.0029**. Combining two of the three
Wave-1 leaders produced *less* than the better one alone — consistent with the
factorial's sub-additive interaction (−0.0043) and with round 5's finding that stacked
mechanisms overlap rather than add.

## The conclusion this forces

Across rounds 5 and 6, roughly **forty** experiments have now been run spanning
supervision geometry (auxiliary simplex, multi-resolution, context-relative, hurdle,
quantile, pairwise), architecture (hybrid mixers, depth, width, state dimension, MoE),
output parameterisation (CORAL, CORN-style constraints), and training scheme (source
weighting, bagging, seeds, batch size).

**Every one returns what a mechanism-free re-run returns.**

The reasonable inference is that **Ordinal-SSM at ~0.908 OOF dev is at the practical
ceiling for this backbone on this data**, and the limit is the data rather than the
model. Two specifics support that reading:

- **Kim is the weak partition** (0.7912 vs Liu's 0.8444) and holds 55% of the
  evaluation set, yet gets 19.6% of the training rows. Re-weighting toward it does not
  fix this, which means the problem is Kim's *quantity and noise*, not its
  gradient share.
- **50% of Kim rows are exactly zero.** Half of that partition carries no ordering
  information at all, and round 5 showed the recoverable structure there is a metric
  artifact, not signal.

Per the round-5 spec's own guidance — *"if a round does not materially improve, do not
over-tune; preserve the previous system"* — **round 4 remains the final system**.

## Recommendation for round 7

Not more modelling on this data. The productive directions are:

1. **More Kim-like data**, or better replicate-level filtering of it. The ceiling is
   most likely a label-noise ceiling in the partition that dominates the benchmark.
2. **Measure that noise ceiling directly** — replicate-to-replicate Spearman within Kim
   would establish what the maximum achievable score actually is. If it is ~0.82,
   round 4 is already at it and further modelling is pointless.
3. **External validation** on an independent PE dataset, which tests generalisation
   rather than chasing thousandths on a saturated benchmark.

Estimating the noise ceiling is cheap and would settle whether any future round is
worth running. It should come first.
