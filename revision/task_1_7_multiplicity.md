# Task 1.7 — multiplicity and power for the negative-results table

Commit `d47d183082cb`.

## The empirical null

Matched pairs in this repo where the candidate is mechanism-free by construction or was
established as null. These are draws from the null distribution the table's entries
should be read against.

| pair | Δ | what it is |
|---|---:|---|
| `r9_clr16c` vs `r9_ctrl` | -0.0009 | capacity control vs plain control, dev fold 0 |
| `r9_clr16` vs `r9_clr16c` | -0.0001 | context low-rank vs its own capacity control |
| `r8_dapt` vs `r8_dapt_ctrl` | +0.0006 | domain-adaptive fine-tuning vs its control |
| `r8_srchead` vs `r8_srctied` | -0.0018 | per-source head vs tied-head control |
| `r8_sel` vs `r8_selfrozen` | +0.0000 | selective SSM vs frozen-selection control |
| `r7_layerwise_f0` vs `r7_ctrl_f0` | -0.0001 | layerwise context on SSM, matched control, dev fold 0 (from reports/round7_diagnosis.md) |
| `r7_layerwise_f1` vs `r7_ctrl_f1` | +0.0001 | layerwise context on SSM, matched control, dev fold 1 (from reports/round7_diagnosis.md) |
| `r7_layerwise_f2` vs `r7_ctrl_f2` | +0.0003 | layerwise context on SSM, matched control, dev fold 2 (from reports/round7_diagnosis.md) |

Excluding the selective-SSM pair (a recipe change, not a mechanism-free control), the
null spread over **7 matched differences** has
SD = **0.0008**, mean -0.0003, range [-0.0018, +0.0006].

## This estimate contradicts the project's own experience, and the tension is the finding

Taken at face value, SD = 0.0008 implies the design detects 0.0013 on three
folds. But the manuscript records three interventions that reached +0.003 to +0.004 on
one development fold and **changed sign on another** — impossible if the SD of a
difference were really 0.0008, since +0.0035 would then be over four standard
deviations.

Both observations are real, and they measure different things. A *matched* pair cancels
the fold and the seed: both runs see the same rows in the same order from the same
initialisation, so the only thing that differs is the mechanism, and the difference is
correspondingly tiny. What governs whether an effect **replicates** is the variance
across folds and seeds of an *unmatched* comparison, which is larger and which no
artifact in this repo measures.

So the measured SD is a **lower bound** and the power figures below are optimistic:

| folds | detectable at the measured SD (0.0008) | detectable at the manuscript's working resolution |
|---:|---:|---:|
| 1 | 0.0023 | 0.0087 |
| 3 | 0.0013 | 0.0050 |
| 5 | 0.0010 | 0.0039 |
| 10 | 0.0007 | 0.0027 |

The right column back-solves the SD (0.0031) implied by the manuscript's own
promotion rule (detect +0.005 across three folds). **I recommend the paper keep quoting
the conservative figure**, and say explicitly that it is calibrated from observed
fold-to-fold sign changes rather than from a matched-pair SD.

## The negative-results table under Holm correction

Computed against the *measured* SD, so this table is **over-confident** and is shown to
bound the exercise rather than to license its conclusions.

| intervention | Δρ | z | p | p (Holm) | survives α=0.05 |
|---|---:|---:|---:|---:|---|
| context-relative ordinal (primary) | -0.0235 | -28.93 | 0.000 | 0.000 | yes |
| quantile regression head | -0.0140 | -17.23 | 0.000 | 0.000 | yes |
| training on evaluated sources only | -0.0294 | -36.19 | 0.000 | 0.000 | yes |
| selective (Mamba-style) SSM | +0.0031 | +3.82 | 0.000 | 0.002 | yes |
| hybrid S4D+attention, 1:1 | -0.0027 | -3.32 | 0.001 | 0.011 | yes |
| hurdle (zero-inflation) head | -0.0022 | -2.71 | 0.007 | 0.074 | no |
| S4D state dimension 128 | -0.0022 | -2.71 | 0.007 | 0.074 | no |
| per-source output head | -0.0018 | -2.22 | 0.027 | 0.241 | no |
| nonlinear stacking | +0.0013 | +1.60 | 0.110 | 0.877 | no |
| auxiliary simplex head | +0.0008 | +0.98 | 0.325 | 1.000 | no |
| multi-resolution ordinal | +0.0008 | +0.98 | 0.325 | 1.000 | no |
| monotonicity penalty | +0.0007 | +0.86 | 0.389 | 1.000 | no |
| residual learning on features | +0.0006 | +0.74 | 0.460 | 1.000 | no |
| rank-consistent CORAL head | +0.0005 | +0.62 | 0.538 | 1.000 | no |
| layerwise context conditioning | +0.0001 | +0.12 | 0.902 | 1.000 | no |
| context-gated ensemble weights | +0.0000 | +0.00 | 1.000 | 1.000 | no |

At the measured SD even +0.0031 is "significant", which is plainly wrong given that
effects of that size have flipped sign across folds in this project. At the manuscript's
resolution, only the four large entries (|Δ| > 0.01) clear correction.

## How the table should be reworded

Entries that do not survive are not "no effect" — they are **bounded**. The defensible
sentence is "no intervention produced a replicated gain above ≈0.005", not "these had no
effect", and the bound should be stated at the conservative resolution.

**Caveat, load-bearing, and the reason Phase 2.1 matters.** across-fold/run spread at fixed seed; a true across-seed estimate needs the Phase 2.1 factorial. These are optimistic. The
5-seed x 4-cell factorial is the only thing here that would measure the unmatched
across-seed variance directly, which is what both the resolution claim and this power
analysis actually need.
