# E08 - power for the proposed pilot, from measured variability

## 1. The paired between-method difference, measured

On 1,390 scorable two-candidate decision groups of the held-out surface, the per-edit difference in achieved efficiency between PE-RankFormer and OptiPrime has mean +0.0091 and **SD 0.0497**. The two methods nominate the same design in 71% of groups, where the difference is exactly zero; among the rest the SD is 0.0880.

Both methods pick the same design in most groups, so the paired difference is exactly zero there. That is a real feature of the endpoint, not a missing value: it lowers the SD and raises power, which is why the SD over all groups is the one to use.

## 2. What replication buys

- per-measurement SD from 654 metadata-identical replicate groups: 0.0265 efficiency units
- with three biological replicates per design: 0.0153
- the noise part of a paired difference at three replicates: 0.0216

A paired difference between two methods at one edit involves two different designs, so two independent measurement errors, unless the methods nominate overlapping candidates - in which case the shared measurement cancels and the noise term is smaller than this.

## 3. Edit-to-edit spread at real candidate depth

9,264 development decision groups with five or more candidates, over 16 contexts. Best-of-three achieved efficiency: model 0.5925, random 0.5254, oracle 0.6034. The model-minus-random paired difference averages +0.0671 with SD **0.0656**; the endpoint itself has SD 0.2598 across edits.

## 4. Edits required

| paired SD source | SD | target effect | edits at 80% | edits at 90% |
|---|---:|---:|---:|---:|
| plan_illustration | 0.0800 | 0.02 | 126 | 168 |
| plan_illustration | 0.0800 | 0.04 | 31 | 42 |
| plan_illustration | 0.0800 | 0.08 | 8 | 11 |
| measured_between_method_paired_sd_depth2 | 0.0497 | 0.02 | 48 | 65 |
| measured_between_method_paired_sd_depth2 | 0.0497 | 0.04 | 12 | 16 |
| measured_between_method_paired_sd_depth2 | 0.0497 | 0.08 | 3 | 4 |
| measured_model_minus_random_paired_sd_depth5plus | 0.0656 | 0.02 | 85 | 113 |
| measured_model_minus_random_paired_sd_depth5plus | 0.0656 | 0.04 | 21 | 28 |
| measured_model_minus_random_paired_sd_depth5plus | 0.0656 | 0.08 | 5 | 7 |
| measured_endpoint_sd_depth5plus_unpaired | 0.2598 | 0.02 | 1325 | 1773 |
| measured_endpoint_sd_depth5plus_unpaired | 0.2598 | 0.04 | 331 | 443 |
| measured_endpoint_sd_depth5plus_unpaired | 0.2598 | 0.08 | 83 | 111 |

## Caveats

- These are the simplest paired two-sided calculations. Clustering by locus family, donor or plate, more than one primary comparison, and attrition all raise the requirement.
- The between-method SD is measured where both methods choose from only two candidates. A twelve-candidate panel gives both methods more room to differ, so its paired SD will be larger than the depth-2 estimate and closer to the depth-5 figure.
- The number of wells is not the denominator. Replicates reduce the measurement term only; the edit-to-edit spread is irreducible by replication.
- Nothing here powers a mechanistic reversal endpoint. E03 puts the supported reversal rate at 4.2%, so an experiment whose primary endpoint is reversal direction needs its own calculation on that base rate.