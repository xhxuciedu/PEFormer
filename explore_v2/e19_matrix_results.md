# E19 - the decision-aligned training matrix

Fold 0, 1,463 informative decision groups over 439 protospacer clusters; no arm trains on fold 0 or early-stops on it.

## Arms

| arm | what it changes | seeds | mean achieved @1 | per seed | pooled rho | pairs/epoch |
|---|---|---:|---:|---|---:|---:|
| A | original batching, original ranking key | 3 | 0.13360 | [0.13402, 0.13345, 0.13332] | 0.8905 | 15,695 |
| B | canonical batching, original ranking key | 1 | 0.13205 | [0.13205] | 0.8703 | 6,975 |
| C | canonical both, 1 pair per group | 1 | 0.13463 | [0.13463] | 0.8950 | 25,253 |
| D | canonical batching, canonical ranking key | 3 | 0.13453 | [0.13455, 0.13449, 0.13453] | 0.8950 | 61,978 |
| E | canonical batching, ranking disabled | 3 | 0.13459 | [0.13496, 0.13516, 0.13365] | 0.8949 | 61,975 |
| F | canonical both, feature branch | 3 | 0.13523 | [0.13565, 0.13557, 0.13448] | 0.8974 | 61,976 |
| G | canonical both, 16 pairs per group | 1 | 0.13357 | [0.13357] | 0.8968 | 146,088 |
| H | canonical batching, feature branch, ranking disabled | 3 | 0.13473 | [0.13487, 0.13488, 0.13445] | 0.8971 | 61,976 |

## Seed-paired contrasts

| contrast | seeds | per seed | mean | same sign |
|---|---:|---|---:|---|
| D − A | 3 | [0.00053, 0.00104, 0.00121] | +0.00093 | True |
| B − A | 1 | [-0.00197] | -0.00197 | True |
| D − B | 1 | [0.0025] | +0.00250 | True |
| D − C | 1 | [-8e-05] | -0.00008 | True |
| D − E | 3 | [-0.00041, -0.00068, 0.00089] | -0.00006 | False |
| F − D | 3 | [0.0011, 0.00108, -6e-05] | +0.00071 | False |
| G − D | 1 | [-0.00098] | -0.00098 | True |
| E − A | 3 | [0.00094, 0.00172, 0.00033] | +0.00099 | True |
| F − A | 3 | [0.00163, 0.00213, 0.00116] | +0.00164 | True |
| H − A | 3 | [0.00084, 0.00144, 0.00113] | +0.00114 | True |
| H − F | 3 | [-0.00079, -0.00069, -2e-05] | -0.00050 | True |
| H − E | 3 | [-9e-05, -0.00028, 0.00081] | +0.00014 | False |
| F − E | 3 | [0.00069, 0.00041, 0.00083] | +0.00064 | True |

## Checkpoint selection: pooled Spearman versus the decision

Both selections come from the same 18 trajectories, so the comparison is not confounded by different stopping times. The selectors chose different epochs in 16 of 18 runs.

- mean gain in fold-0 achieved efficiency from selecting on the decision: **+0.00004**
- mean pooled-rho cost of doing so: -0.0004

Per run: {'m_A_s1': np.float64(0.00092), 'm_A_s2': np.float64(-0.00038), 'm_A_s3': np.float64(0.0), 'm_D_s1': np.float64(0.00042), 'm_D_s2': np.float64(-0.00088), 'm_D_s3': np.float64(-0.0003), 'm_B_batch': np.float64(0.00106), 'm_C_p1': np.float64(-0.00168), 'm_E_norank': np.float64(0.0), 'm_F_feat': np.float64(8e-05), 'm_G_p16': np.float64(0.00133), 'm_F_s2': np.float64(0.00068), 'm_F_s3': np.float64(-0.00054), 'm_E_s2': np.float64(-0.00017), 'm_E_s3': np.float64(0.00064), 'm_H_s1': np.float64(-5e-05), 'm_H_s2': np.float64(0.00028), 'm_H_s3': np.float64(-0.00076)}
