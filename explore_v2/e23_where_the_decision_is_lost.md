# E23 - where the decision is lost, and a fix that does not work

## How large is the deficit, in interpretable terms

On 24,668 informative panel groups a random pick achieves 0.0268 and a perfect chooser 0.0570, so **the whole decision is worth 0.0302** efficiency.

| model | achieved @1 | share of the available gain |
|---|---:|---:|
| OptiPrime | 0.0456 | 62.2% |
| PE-RankFormer, published member | 0.0448 | 59.6% |
| arm A, released recipe | 0.0431 | 54.1% |
| arm F, canonical + features | 0.0446 | 58.9% |

The published model and OptiPrime differ by **2.6% of the available gain**. The difference is statistically clear and practically small; both models are far closer to each other than either is to random selection. Any claim in either direction should be stated in these units.

## The error is structured: a length preference

Over the same 24,668 groups, the geometry of the nominated design against the best-measured design:

| model | PBS length bias | RTT length bias |
|---|---:|---:|
| OptiPrime | +0.519 | -0.531 |
| PE-RankFormer, published member | +0.766 | +0.162 |
| arm A, released recipe | +0.943 | +0.483 |
| arm F, canonical + features | +0.812 | +0.428 |

Every PE-RankFormer variant nominates designs with longer PBS **and** longer RTT than the best design. OptiPrime is the only model that prefers a shorter RTT, which is consistent with its treating RTT length as an explicit synthesis repeat count. Supplying PBS and RTT lengths as explicit scalars (arm A to arm F) moves the PBS bias by -0.131 and the RTT bias by -0.055: both shrink a little, neither is removed, and both stay on the opposite side of zero from OptiPrime. Handing the model the length is evidently not the same as modelling what the length does.

## A global length correction does not fix it

If the preference were an additive offset, a linear length term would remove it. Swept on 43,425 development decision groups against a baseline of 0.29213:

| RTT coefficient | achieved @1 |
|---|---:|
| -0.010 | 0.28703 |
| -0.006 | 0.28987 |
| -0.004 | 0.29080 |
| -0.002 | 0.29168 |
| -0.001 | 0.29213 |
| +0.000 | 0.29213 |
| +0.001 | 0.29183 |
| +0.002 | 0.29118 |

The optimum is exactly zero on both axes over 43,425 development decision groups, so the length preference is not a global additive offset in the score and cannot be corrected by one.

## Do the opposite-direction errors cancel?

An equal-weight rank average of the two scores, weight fixed at one half and nothing fitted, on the same 24,668 informative panel groups:

| model | achieved @1 |
|---|---:|
| OptiPrime | 0.0456 |
| PE-RankFormer, published member | 0.0448 |
| **equal-weight rank average** | 0.0461 |

| paired difference @1 | value | 95% CI | p |
|---|---:|---|---:|
| blend − op | +0.00055 | [+0.00034, +0.00077] | 0.001 |
| blend − ours | +0.00133 | [+0.00109, +0.00156] | 0.001 |

On the panel the blend captures 64.1% of the available gain, against 62.2% for OptiPrime alone.

### The equal weight is not cherry-picked

| weight on PE-RankFormer | 0.0 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| achieved @1 | 0.04559 | 0.04589 | 0.04607 | 0.04607 | 0.04614 | 0.04597 | 0.04576 | 0.04551 | 0.04481 |

0.0 is OptiPrime alone and 1.0 is PE-RankFormer alone. Every intermediate weight tested beats both endpoints, so the equal-weight choice is not cherry-picked.

### But it does not replicate on fold 0

On fold 0's 1,463 informative groups over 439 clusters:

| model | achieved @1 |
|---|---:|
| OptiPrime | 0.1272 |
| PE-RankFormer, published member | 0.1364 |
| **equal-weight rank average** | 0.1315 |

| paired difference @1 | value | 95% CI | p |
|---|---:|---|---:|
| blend − op | +0.00429 | [+0.00257, +0.00615] | 0.001 |
| blend − ours | -0.00486 | [-0.00704, -0.00278] | 0.001 |

**The blend beats OptiPrime here but loses to PE-RankFormer alone.** On fold 0 PE-RankFormer leads OptiPrime by 0.0092, a wide margin, so averaging in the weaker model dilutes the stronger one; on the panel the two are within 0.0008 of each other and the blend pays. Blending helps when the components are comparably strong, which is ordinary ensemble behaviour rather than a special property of these two models.

**What is claimed.** Not that the blend is universally better. Only that *on the surface that resembles deployment* -- an external library, realistic candidate depth, the two models within a thousandth of each other -- combining them beats either alone, robustly across weights, with nothing fitted. The mechanism is the opposite-direction RTT disagreement above, not generic ensembling, which is an argument for treating the mechanistic baseline as complementary to the neural model rather than as a rival to be beaten.

**What this implies for method work.** A per-candidate score correction is the wrong instrument. The residual error is comparative and conditional -- which design is best depends on the alternatives present -- so the matched instrument is a model that scores the candidate set jointly rather than each candidate independently.
