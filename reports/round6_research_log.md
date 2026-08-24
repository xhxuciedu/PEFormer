# Round-6 research log — context interaction

## Phase 0 audits

**§28, the repeated 0.9082.** Genuine coincidence. Dev (mean of per-fold, n=107,041)
is 0.908179; held-out (n=20,509) is 0.908155. Different files, disjoint rows
(intersection verified as 0), differing at the fifth decimal. Full audit in
`round6_ordssm_metric_audit.md`.

**§7 interaction dataset.** 49,897 designs measured in >=2 contexts, 205,585
observations, **506,115 context pairs**; mean |rank shift| 0.135, with 44.3% of pairs
shifting by more than 0.1. The design/context boundary was the load-bearing choice:
`motif`, `linker`, `scaffold`, `epegRNA` are DESIGN (they change the molecule);
`rt_name`, Cas9 fields, PEmax/MLH1dn/NRCH are CONTEXT (they change the machinery).
Round 5 had already shown that merging epegRNA with plain-pegRNA constructs
manufactures "context effects" that are really construct effects.

**§31 identifiability.** All 28 contexts have >=200 rows, so context-specific
parameters are supportable without hierarchical shrinkage.

## Experiment A (§9): same-design cross-context shift loss — **null**

| run | dev fold 0 | vs control |
|---|---:|---:|
| control, lambda=0 | 0.8990 | (ref) |
| **SHUFFLED control, lambda=0.10** | **0.9014** | **+0.0024** |
| shift lambda=0.25 | 0.9002 | +0.0012 |
| shift lambda=0.10 | 0.8987 | −0.0003 |
| shift lambda=0.05 | 0.8977 | −0.0013 |
| shift lambda=0.50 | 0.8925 | −0.0065 |

**The mechanism-free control beats every real setting.** The shuffled-target run --
identical computation, identical gradient scale, targets permuted so no interaction
signal survives -- scored higher than all four genuine λ values. Whatever the +0.0024
is, it is not interaction learning; it sits exactly in the ±0.002 band that rounds 4-6
established for ordinary run variation.

Without the §23 control, λ=0.25's +0.0012 would have read as weakly encouraging.

## Experiment: ctx-primary — **null, and it fails on its own terms**

Trains the ordinal head on within-condition quantiles as the *primary* target,
inverting through per-condition training CDFs at inference (`eval_ctx_primary.py`).

| metric | baseline | ctx-primary | delta |
|---|---:|---:|---:|
| pooled (dev fold 0) | 0.9094 | 0.8859 | −0.0235 |
| Kim | 0.7869 | 0.7391 | −0.0478 |
| **Kim within-condition** | **0.7650** | **0.7328** | **−0.0322** |

The last row is the important one. Removing the cross-condition mean shift was supposed
to *force* within-condition ranking to improve. It got **worse**. So the global signal
is not a shortcut the model hides behind -- it is **informative for within-condition
ranking too**, and training without it loses information rather than concentrating it.

(The raw within-condition quantile scores 0.3484 pooled, confirming the CDF inversion
is both necessary and working.)

## Correction: the excess-invariance diagnostic was inflated

The diagnostic compared the model's cross-context rank correlation against the
*observed* one. But observed correlations between two **noisy** measurements attenuate
by the reliability, while model predictions carry no measurement noise. Comparing them
directly overstates the excess.

With Kim's empirical reliability R = 0.9362:

| scope | observed | disattenuated | model | raw excess | **corrected** |
|---|---:|---:|---:|---:|---:|
| all 51 pairs | 0.7099 | 0.7583 | 0.8576 | +0.1477 | **+0.0993** |
| Kim-Kim pairs | 0.6897 | 0.7367 | 0.8571 | +0.1674 | **+0.1204** |

The excess is real but roughly **a third smaller** than reported. The headline
"0.683 vs 0.835" in earlier write-ups should be read as "0.73 vs 0.835".

## Assessment

Three independent attacks on the interaction have now failed:

1. layerwise FiLM (round 7) — null, and moved the mechanism *backwards*;
2. same-design shift loss (§9) — beaten by its own shuffled control;
3. ctx-primary objective — worse on every metric including its own target.

Combined with the attenuation correction, the honest reading is that the diagnosed
gap is **smaller than it first appeared and not addressable by supervising the
interaction directly**. The spec's §41 anticipated this case: *"reconsider whether
missing experimental covariates, not model structure, limit context reordering."*

That is now the most plausible explanation. What distinguishes A549 from DLD1 for a
given design -- MMR status, chromatin state, repair-pathway balance -- is not in the
input. The model receives a categorical label, and no objective over a categorical
label can recover biology the label does not encode. The ρ≈0.275 rank-shift
predictability from design features is consistent with a *small* learnable component
sitting inside a much larger unlearnable one.

**Next**, per §39 Phase 2, the bilinear head (§13) is the remaining distinct mechanism:
it changes the scoring *parameterisation* rather than the objective, and is the one
form of interaction capacity not yet tested. It is worth one clean test. If it also
fails, the conclusion is that the benchmark's context labels are the binding
constraint, and §37's external-validation track becomes the higher-value direction.
