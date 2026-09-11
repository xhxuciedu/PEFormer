# E32 preliminary diagnostic: selection-score scale

Label-free score-scale diagnostic, not a temperature-tuned performance result. Post-hoc positive temperature scaling cannot change argmax selection.

For eight candidates with scores in [0,1], T=1 bounds the largest softmax
probability by e/(e+7) = 0.2797. Unbounded heads do not share this constraint.

| Model | Temperature | Median within-group range | Mean max probability | Normalized entropy |
|---|---:|---:|---:|---:|
| M_s20260910 | 1.0 | 87.5000 | 0.9823 | 0.0379 |
| M_s20260910 | 0.1 | 87.5000 | 0.9959 | 0.0056 |
| M_s20260910 | 0.01 | 87.5000 | 0.9961 | 0.0051 |
| shared_s20260910 | 1.0 | 0.2451 | 0.3409 | 0.9929 |
| shared_s20260910 | 0.1 | 0.2451 | 0.6181 | 0.6806 |
| shared_s20260910 | 0.01 | 0.2451 | 0.8973 | 0.2009 |
| S_s20260910 | 1.0 | 96.3750 | 0.9800 | 0.0413 |
| S_s20260910 | 0.1 | 96.3750 | 0.9943 | 0.0089 |
| S_s20260910 | 0.01 | 96.3750 | 0.9946 | 0.0081 |
| S_pairwise_s20260910 | 1.0 | 3.8359 | 0.6595 | 0.6067 |
| S_pairwise_s20260910 | 0.1 | 3.8359 | 0.9480 | 0.0992 |
| S_pairwise_s20260910 | 0.01 | 3.8359 | 0.9935 | 0.0116 |

A clean training control needs comparable logit parameterization or a declared
validation-selected temperature protocol. These measurements do not establish
which architecture would win under that control. No E32 training was run.