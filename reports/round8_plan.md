# Round 8 plan — Ordinal-SSM: sequence mixing and domain shift

Two tracks, both requested: **(A)** SSM/Transformer hybridisation in the style of recent
LLM architectures, and **(B)** explicit domain-shift and batch-effect modelling across
the three source studies.

**Baseline to beat:** Ordinal-SSM, 0.9082 OOF dev / 0.9082 held-out (verified distinct
in `round6_ordssm_metric_audit.md`), Liu 0.8444, Kim 0.7912.

---

## What the accumulated evidence constrains

Rounds 5–7 ran roughly sixty experiments. Three constraints matter for this plan:

1. **Architecture is the largest lever measured.** The factorial gives $+0.0192$ for
   SSM-over-Transformer against $+0.0092$ for the ordinal objective. Track A is
   therefore the higher-prior direction.
2. **A naive hybrid already failed** (alternating $-0.0014$, parallel-gated $-0.0027$).
   Any new hybrid must explain why it differs, not merely re-run.
3. **Every effect below $\approx0.005$ has failed to replicate**, and a mechanism-free
   control has twice out-scored the real mechanism. Controls and 3-fold replication are
   mandatory, not optional.

---

# Track A — sequence mixing

## A1. Selective SSM (Mamba-style). *Highest priority.*

**The gap.** Our mixer is **S4D: linear time-invariant.** `S4DKernel.forward(L)` takes
only the sequence length — the convolution kernel is *identical for every input*. The
model can learn "positions 8 apart interact" but never "attend to *this* position
because of what it contains".

This is exactly the limitation Mamba (S6) was introduced to fix: make $\Delta$, $B$, $C$
functions of the input, so the recurrence becomes content-dependent and can selectively
retain or forget.

**Why it should matter here specifically.** Prime editing is a *matching* problem. The
PBS must anneal to the nicked strand; the RTT must template a specific edit at a
specific offset. Which positions matter depends on *what the sequence contains* — the
edit position moves from design to design. An LTI kernel must average over all possible
edit positions; a selective one can gate on the edit itself.

**Note what this is not.** Round 6 scaled `ssm_state_dim` (64→128→256) and found
nothing. That varies *how much* the LTI kernel remembers. Selectivity changes *whether
memory is content-dependent at all* — a different axis, and the one that separates S4
from Mamba.

**Implementation.** Input-dependent $\Delta$, $B$, $C$ via linear projections of the
token representation; keep bidirectionality (two scans) and the existing block
topology, so the comparison isolates selectivity. Selective SSMs cannot use the FFT
convolution — they need a scan — but at $L\le102$ a simple sequential scan in PyTorch
is acceptable; if too slow, chunk it.

**Control.** Same parameter count with the selection projections *frozen at
initialisation*, so the extra parameters exist but cannot become content-dependent.
This isolates selectivity from capacity.

**Honest prior.** Highest of anything in this plan. It is the one mechanism inside the
SSM family we have not touched, and it is the mechanism that made Mamba work.

## A2. Sparse hybrid at LLM ratios (Jamba-style)

**Why this is not a repeat.** Round 5's "alternating" hybrid was **1:1** — three
attention layers among six. Successful LLM hybrids use attention *sparingly*: Jamba is
**1:7**, Zamba shares a single attention block globally, Griffin interleaves local
attention into a mostly-recurrent stack. The published finding is that a *little*
attention supplies content-based lookup that recurrence lacks, while *much* attention
erodes the recurrent inductive bias. We tested only the eroding regime.

**Design.** One full-attention layer in the 6-layer edit encoder and one in the 4-layer
pegRNA encoder ($\approx$1:5), with **placement varied**: early, middle, late. Placement
is the interesting variable — content lookup is plausibly most useful *after* local
structure has been built, which argues for late.

**Control.** All-SSM at matched depth and parameter count.

**Prior.** Moderate. The efficiency argument for LLM hybrids does not transfer (our
sequences are ~100 tokens, so attention is cheap), but the representational argument
might.

## A3. Local/sliding-window attention in the hybrid (Samba/Griffin-style)

Only if A2 shows signal. Replace the full-attention layer with a windowed one (window
16–32). For a problem where the relevant interactions are mostly local (PBS–target
annealing, edit-proximal context), a window may add the content-based lookup without
the long-range noise. Cheap once A2 exists.

## A4. Cross-stream attention capacity

Currently 2 cross-attention blocks join the two streams. **The pegRNA–target matching
that defines prime editing happens there**, and it has never been varied. Test 2→4
blocks. This is the least novel item but sits at the architecturally most meaningful
junction.

---

# Track B — domain shift and batch effects

## The problem, quantified

| Source | % train | % held-out | zero-mass | mean efficiency |
|---|---:|---:|---:|---:|
| Schwank | **58.4%** | **0.0%** | 10.3% | — |
| Liu | 22.0% | 44.7% | 0.8% | — |
| Kim | 19.6% | 55.3% | 49.9% | 0.061 |

These are not three samples from one distribution. They are three **different assays**
— different libraries, readouts and normalisations — whose "efficiency" columns are not
the same measurement. A 49.9% versus 0.8% zero-mass is not biology; it is substantially
protocol.

**What has already been tried.** Source loss re-weighting: inverted-U, best setting
$+0.0033$ on one fold but $-0.0016$ on another — failed replication. Aggressive
re-weighting ($-0.0059$) was the worst run of its round, showing Schwank's 174k
never-evaluated rows carry real transferable signal. So *re-weighting* is closed. What
follows are structurally different: they model the batch effect rather than reweighting
it away.

## B1. Source-conditional output head. *Highest priority in this track.*

Shared trunk; the final ordinal head gets **per-source parameters** (three sources,
each with ≥58k rows — ample support).

**Rationale.** If each source measures a different monotone function of the same
underlying editability, the right model is one shared representation read out through
three source-specific links. That is precisely a batch-effect model: shared biology,
per-batch measurement model.

**Why it should survive where ctx-primary died.** ctx-primary *replaced* the global
target with within-condition quantiles across 28 fine-grained contexts and lost badly,
including on its own within-condition metric — removing global signal destroyed
information. B1 removes nothing: global supervision is untouched, and only the readout
is allowed to differ across three coarse, data-rich groups.

**Control.** Same parameter count with the per-source heads tied (i.e. one head
replicated), so the gain must come from *differentiation*, not capacity.

## B2. Source-adversarial trunk (gradient reversal)

A source classifier on the pooled representation, trained through a gradient-reversal
layer so the **trunk** becomes source-invariant while the **head** stays source-aware
(pairs naturally with B1).

**Rationale.** The canonical batch-effect correction, and untested here. It targets a
real hazard: with 58.4% of rows from a study that is never evaluated, the trunk can
spend capacity on Schwank-specific artefacts. Adversarial invariance suppresses exactly
that.

**Risk, stated up front.** Source and biology are confounded — Kim *is* the multi-cell-
line data. Forcing source-invariance may erase real biological signal that happens to
correlate with source. Sweep $\lambda_{\text{adv}} \in \{0.01, 0.05, 0.2\}$ and watch
Kim specifically; a Kim drop with a pooled gain means the invariance is destroying
signal.

**Control.** Shuffled source labels — identical architecture and gradient scale, no
real domain signal. (This control caught the §9 shift loss, where the shuffled version
*beat* every genuine setting.)

## B3. Two-stage domain-adaptive fine-tuning on Ordinal-SSM

Pretrain on all three sources, then fine-tune on Liu+Kim with a fresh schedule.

**Why revisit.** Round 3 rejected DAPT because it correlated 0.997 with an existing
*ensemble member* — an ensemble criterion, irrelevant now that we optimise a single
model. Its standalone effect was **positive** ($+0.0015$). It has never been tried on
the Ordinal-SSM backbone, and the factorial showed architecture interacts with
everything.

Cheap: initialise from an existing checkpoint, fine-tune ~10 epochs. Sweep fine-tune LR
$\in \{1\mathrm{e}{-5}, 3\mathrm{e}{-5}\}$.

## B4. Per-source ordinal thresholds, with a global auxiliary head

Thresholds are currently global quantiles. Given the source distributions differ so
sharply, "the 8th of 18 global thresholds" means something different in Kim than in
Liu.

**Hedged deliberately.** ctx-primary showed that *replacing* global supervision is
harmful, so here the per-source thresholds are added **alongside** a retained global
head rather than instead of it. If B4 also fails, that closes the label-renormalisation
family for good — two clean attempts at different granularities.

**Lower priority** than B1–B3 precisely because its nearest neighbour already failed.

---

# Protocol

**Stage A** — dev fold 0, every candidate against a matched control run in the same
batch. **Stage B** — anything reaching $\ge +0.003$ goes to all three folds; promotion
needs **same-sign on 3/3** and $\ge +0.005$ mean. **Stage C** — official 5-fold OOF for
finalists. **Stage D** — freeze. **Stage E** — held-out only for a genuine improvement;
the set has informed several rounds already and should not be spent on thousandths.

Every wave includes a **plain mechanism-free re-run**. Two rounds have now produced
uniform small positives indistinguishable from run variation, and once the shuffled
control beat the real mechanism outright.

**Reported per candidate:** pooled, Liu, Kim, macro-context, and — for Track B —
per-source breakdown. A pooled gain that comes entirely from Schwank-like rows is not
useful, since Schwank is 0% of the evaluation set.

---

# Sequencing and honest expectations

| Wave | Contents |
|---|---|
| 1 | A1 selective SSM (+frozen-selection control); B1 source-conditional head (+tied control); B3 DAPT; plain control |
| 2 | A2 sparse hybrid × 3 placements (+matched-depth control); B2 adversarial × 3 λ (+shuffled-label control) |
| 3 | A3 windowed attention and A4 cross-blocks, only if A2 shows signal; B4 only if B1 shows signal |
| 4 | Combine only mechanisms that independently replicated |

**Priors, stated before running.** A1 is the most likely single win — it is the one
untested mechanism inside the family that the factorial identified as the dominant
lever. B1 is the most likely win in Track B, because it adds structure without removing
signal, which is the pattern that has distinguished the successes from the failures
here. A2 is genuinely uncertain: the LLM evidence is real but its motivation
(long-context efficiency) does not transfer to 100-token sequences. B4 is the least
likely, and is included mainly to close out the label-renormalisation family.

Given that ~60 consecutive experiments have failed to move this baseline, the
appropriate expectation for the round as a whole is **modest**: a single replicated
$+0.005$ would be a good outcome. If Wave 1 produces nothing, the evidence will favour
concluding that the benchmark's context labels and label noise — not the architecture —
are binding, and switching to the external-validation track.
