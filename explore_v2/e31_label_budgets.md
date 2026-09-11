# E31: corrected label-budget protocol

Protocol construction only: no new learning-curve training has been run.

Whole-component prefixes are nested; actual budgets may exceed nominal values.
Total-budget inner validation uses only selected E25 training components.

| Seed | Nominal groups | Actual total groups | Candidates | Inner train | Inner val |
|---|---:|---:|---:|---:|---:|
| 20260910 | 200 | 202 | 788 | 168 | 34 |
| 20260910 | 1000 | 1001 | 3848 | 839 | 162 |
| 20260910 | 5000 | 5002 | 18655 | 4027 | 975 |
| 20260910 | all | 19357 | 72010 | 15441 | 3916 |
| 20260911 | 200 | 200 | 695 | 159 | 41 |
| 20260911 | 1000 | 1000 | 3555 | 807 | 193 |
| 20260911 | 5000 | 5001 | 18360 | 4058 | 943 |
| 20260911 | all | 19357 | 72010 | 15559 | 3798 |
| 20260912 | 200 | 201 | 750 | 152 | 49 |
| 20260912 | 1000 | 1002 | 3698 | 776 | 226 |
| 20260912 | 5000 | 5002 | 18582 | 4035 | 967 |
| 20260912 | all | 19357 | 72010 | 15543 | 3814 |

Historical small-budget runs also used 5,561 labelled validation groups.
Their sampler retained whole decision groups, not necessarily whole components.
See JSON for overlap and partial-component counts. No new test results are used.