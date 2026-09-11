# E03 - design-by-context interaction on matched candidate sets

Descriptive. The only fitted object is a two-way additive mean decomposition, used as a null.

## Support

- **111,862 quartets** over 10,482 canonical alleles and 10,132 target sites, on 128 ordered context pairs.
- A quartet is two alternative designs for one allele, both measured in both contexts. Design pairs are capped at 6 per allele (69,059 pairs dropped).

## Measurement noise

- 654 metadata-identical replicate groups, 858 degrees of freedom, 99% of them from Kim.
- pooled per-measurement SD: **0.0378** on the arcsine-root scale (0.0265 on the raw efficiency scale).
- a single design contrast therefore carries SE 0.0535, and a quartet's D carries SE 0.0757, before any real effect.

## What the observed contrasts contain

| component of the design contrast Delta | variance | SD |
|---|---:|---:|
| total spread of Delta across alleles, design pairs, contexts | 0.05420 | 0.2328 |
| design-pair main effect | 0.03571 | 0.1890 |
| design x context interaction | 0.01563 | 0.1250 |
| measurement noise (2 sigma^2) | 0.00286 | 0.0535 |

The interaction accounts for **28.8%** of the variance of the design contrast, against 5.3% for measurement noise and the rest for the context-independent design effect. It is estimated from 14,022 (allele, design pair) combinations seen in two or more contexts.

## Order changes

| quartet subset | n | reversal rate |
|---|---:|---:|
| all quartets | 111,862 | 0.129 |
| both contrasts > 2 SE | 26,061 | 0.041 |
| both > 2 SE and both raw gaps > 0.02 | 23,547 | 0.041 |
| both > 2 SE and both raw gaps > 0.05 | 16,958 | 0.042 |
| both > 2 SE and both raw gaps > 0.10 | 10,986 | 0.036 |

## The additive counterexample

Under any additive latent model D is exactly measurement error, so its SD is 2*sigma = 0.0757 analytically. Reversal rates are simulated conditional on each quartet's own additive contrast estimate (200 simulations), so the null carries the observed contrast sizes and no interaction:

| statistic | observed | additive null (mean) | null 95% range |
|---|---:|---:|---|
| mean \|D\| (arcsine) | 0.1201 | 0.0604 | [0.0601, 0.0606] |
| SD of D (arcsine) | 0.2004 | 0.0757 | [0.0754, 0.0759] |
| reversal rate, all | 0.129 | 0.241 | [0.239, 0.243] |
| reversal rate, both > 2 SE | 0.041 | 0.001 | [0.001, 0.002] |

## What the interaction costs a context-blind chooser

Order the design that measured better in one context, then evaluate it in the other. Raw efficiency units.

| quartet subset | n | oracle in c2 | transferred choice | random choice | transfer regret | share of random regret removed |
|---|---:|---:|---:|---:|---:|---:|
| all_quartets | 111,862 | 0.1060 | 0.0987 | 0.0730 | 0.0073 | 77.8% |
| supported_quartets | 26,061 | 0.2435 | 0.2365 | 0.1503 | 0.0070 | 92.5% |

## Locus-clustered intervals for the load-bearing numbers

Resampling 10,132 target sites, 400 draws.

| statistic | estimate | 95% CI |
|---|---:|---|
| interaction share of Var(Delta) | 0.288 | [0.279, 0.298] |
| SD of interaction (arcsine) | 0.1250 | [0.1219, 0.1281] |
| supported reversal rate | 0.0409 | [0.0355, 0.0476] |
| transfer regret, supported quartets | 0.0070 | [0.0060, 0.0081] |

## How wrong could the noise estimate be?

| sigma multiplier | SD interaction | interaction share of Var(Delta) |
|---|---:|---:|
| 1.0x | 0.1250 | 28.8% |
| 1.5x | 0.1098 | 22.2% |
| 2.0x | 0.0839 | 13.0% |
| 2.5x | 0.0243 | 1.1% |
| 3.0x | 0.0000 | 0.0% |

The interaction estimate reaches zero only if the true per-measurement SD is **2.54x** the replicate-based estimate, i.e. if the 654 replicate groups understate the noise by that factor.


## Which context change carries it

| context change | quartets | SD interaction | SD design main | reversal rate | supported n | supported reversal rate |
|---|---:|---:|---:|---:|---:|---:|
| cell and editor | 46,417 | 0.1223 | 0.2010 | 0.129 | 11,655 | 0.048 |
| cell line only | 44,267 | 0.1385 | 0.1808 | 0.125 | 9,297 | 0.027 |
| editor only (PE2 vs PE4) | 21,178 | 0.1169 | 0.1680 | 0.137 | 5,109 | 0.051 |