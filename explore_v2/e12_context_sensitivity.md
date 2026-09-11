# E12 - sensitivity of the panel result to the assigned context

24,668 informative decision groups. PE-RankFormer re-scored under the assigned context and three alternatives.

| context supplied to the model | pooled rho vs observed | rank agreement with the assigned context | same top pick | achieved @1 | hit rate | regret @1 |
|---|---:|---:|---:|---:|---:|---:|
| assigned (Kim HEK293T PE2max, conventional pegRNA) | 0.7091 | 1.0000 | 1.0000 | 0.0448 | 0.5798 | 0.01216 |
| epegRNA motif instead of none | 0.7075 | 0.9725 | 0.8404 | 0.0452 | 0.5832 | 0.01181 |
| PE4max instead of PE2max | 0.7112 | 0.9977 | 0.9327 | 0.0447 | 0.5786 | 0.01231 |
| K562 instead of HEK293T | 0.6363 | 0.8379 | 0.7267 | 0.0453 | 0.5793 | 0.01170 |

If the decision endpoints barely move across these alternatives, the panel result does not rest on the metadata assignment. If they move a lot, the assignment is load-bearing and has to be stated as a limitation of the absolute numbers -- though not of the head-to-head comparison, since both predictors receive the identical context.
