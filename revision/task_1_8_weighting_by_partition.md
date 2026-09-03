# Task 1.8 — where the training-weight asymmetry costs performance

Two matched runs on development fold 0: `r9_ctrl` (uniform weights) and `r9_opw`
(the corpus `weight` column, i.e. OptiPrime's own per-row loss weights). Same seed, same
fold, same recipe, same code state, so the only difference is the weighting.
Protospacer-clustered bootstrap, 5000 resamples, seed 20260903.
Commit `d47d183082cb`.

Positive Δ means uniform weighting is better.

| partition | n | zero-mass | OptiPrime weights | uniform | Δρ | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| all | 35,649 | 28.4% | 0.8830 | 0.8978 | +0.0148 | [+0.0091, +0.0218] |
| deepprime | 19,747 | 50.8% | 0.7389 | 0.7636 | +0.0247 | [+0.0144, +0.0355] |
| hsu2026 | 15,902 | 0.7% | 0.8492 | 0.8556 | +0.0063 | [-0.0052, +0.0179] |

## By condition

| condition | n | OptiPrime weights | uniform | Δρ |
|---|---:|---:|---:|---:|
| deepprime|DLD1|PE2 | 1,157 | 0.7800 | 0.8261 | +0.0461 |
| deepprime|HeLa|PE2 | 1,142 | 0.7385 | 0.7752 | +0.0367 |
| deepprime|HEK293T|PE2 | 5,601 | 0.7495 | 0.7844 | +0.0348 |
| deepprime|A549|PE2 | 2,249 | 0.5553 | 0.5875 | +0.0322 |
| deepprime|DLD1|PE4 | 2,079 | 0.7946 | 0.8227 | +0.0281 |
| deepprime|MDA-MB-231|PE2 | 1,063 | 0.6272 | 0.6529 | +0.0257 |
| deepprime|HCT116|PE2 | 1,130 | 0.7439 | 0.7678 | +0.0239 |
| deepprime|HEK293T|PE4 | 1,983 | 0.7980 | 0.8170 | +0.0190 |
| deepprime|NIH3T3|PE4 | 1,119 | 0.7272 | 0.7436 | +0.0164 |
| hsu2026|HeLa|PE2 | 4,161 | 0.8209 | 0.8330 | +0.0121 |
| hsu2026|HEK293T|PE2 | 3,803 | 0.8056 | 0.8160 | +0.0104 |
| hsu2026|HEK293T|PE4 | 3,777 | 0.7944 | 0.8045 | +0.0101 |
| deepprime|A549|PE4 | 2,224 | 0.6661 | 0.6731 | +0.0070 |
| hsu2026|HeLa|PE4 | 4,161 | 0.8095 | 0.8148 | +0.0053 |
