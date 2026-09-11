# E28 - what target adaptation costs on the source corpus

Development fold 0, 20,509 rows -- the source corpus's official held-out fold; no adaptation run trained or selected on it.

The starting checkpoint scores **0.8984** pooled Spearman and **0.13350** achieved efficiency @1 there.

| adapted model | arm | Spearman | change | decision @1 | change |
|---|---|---:|---:|---:|---:|
| M_s20260910 | M | 0.8768 | -0.0216 | 0.12730 | -0.00620 |
| M_s20260911 | M | 0.8772 | -0.0213 | 0.12719 | -0.00631 |
| M_s20260912 | M | 0.8723 | -0.0261 | 0.12742 | -0.00608 |
| P_n1000_s20260910 | P|n1000 | 0.8797 | -0.0188 | 0.12819 | -0.00531 |
| P_n200_s20260910 | P|n200 | 0.8982 | -0.0002 | 0.13358 | +0.00008 |
| P_n5000_s20260910 | P|n5000 | 0.8770 | -0.0214 | 0.12780 | -0.00570 |
| P_s20260910 | P | 0.8698 | -0.0286 | 0.12786 | -0.00564 |
| P_s20260911 | P | 0.8693 | -0.0291 | 0.12768 | -0.00582 |
| P_s20260912 | P | 0.8700 | -0.0284 | 0.12775 | -0.00575 |
| S_listnet_s20260910 | S|listnet | 0.8940 | -0.0045 | 0.13382 | +0.00032 |
| S_pairwise_s20260910 | S|pairwise | 0.8912 | -0.0072 | 0.13426 | +0.00076 |
| S_s20260910 | S | 0.8895 | -0.0090 | 0.13294 | -0.00056 |
| S_s20260911 | S | 0.8916 | -0.0069 | 0.13424 | +0.00074 |
| S_s20260912 | S | 0.8917 | -0.0067 | 0.13251 | -0.00099 |
| Z_s20260910 | Z | 0.8984 | +0.0000 | 0.13350 | +0.00000 |
| shared_s20260910 | shared | 0.8457 | -0.0528 | 0.12527 | -0.00823 |
| shared_s20260911 | shared | 0.8475 | -0.0510 | 0.12634 | -0.00716 |
| shared_s20260912 | shared | 0.8337 | -0.0647 | 0.12590 | -0.00760 |

## Mean source Spearman change by arm

| arm | change |
|---|---:|
| shared | -0.0561 |
| P | -0.0287 |
| M | -0.0230 |
| P|n5000 | -0.0214 |
| P|n1000 | -0.0188 |
| S | -0.0075 |
| S|pairwise | -0.0072 |
| S|listnet | -0.0045 |
| P|n200 | -0.0002 |
| Z | +0.0000 |

A negative change is forgetting. The plan asks for this tradeoff to be visible rather than assumed away, and for replay to be tested against target-only adaptation only if the cost is consequential.
