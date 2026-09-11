# V3.1 first-stage result and replication gate

11 September 2026. **All 32 planned screen fits completed and verified.**
This is one-subset/one-optimizer development, not independent confirmation.

## Main finding: baseline optimization matters

The anchored model's target-utility advantage over ordinary fine-tuning is no
longer established once the baseline receives the prespecified learning-rate
and training-horizon search. At 1,001 total labelled groups:

| Inner-selected recipe | Target @1 | Source audit @1 |
|---|---:|---:|
| Starting checkpoint | 0.039627 | 0.133635 |
| Released OptiPrime | 0.041409 | — |
| A: ordinal, 100 updates, 3x LR | 0.041101 | 0.128339 |
| B: direct Huber, 500 updates, 3x LR | 0.040664 | 0.119120 |
| D: shared score, 500 updates, 1x LR | 0.039265 | 0.111535 |
| C_frozen: fresh pairwise, 500 updates, 1x LR | 0.038883 | 0.122010 |
| E: anchored pairwise, 100 updates, 1x LR | 0.041304 | 0.133085 |
| Balanced top-choice margin, weight 1, 100 updates | 0.040636 | 0.134283 |

E minus optimized A is +0.000203 on target, with paired component interval
[-0.000295, +0.000736]. E retains more source-audit utility (+0.004746,
[+0.002327, +0.007184]), but again trails A on the different source-validation
mixture (-0.004050, [-0.006810, -0.001497]). The original v3 fixed-optimization
result remains correctly reported for that protocol; it should not be generalized
into universal architectural superiority.

The fresh frozen head has the largest inner-validation utility (0.042260), yet
its chosen checkpoint performs poorly on outer validation. The shared score also
selects a poor outer-validation checkpoint from the wider grid. Conversely, a
different shared-score configuration reaches outer @1 0.041977, but was NOT
chosen by inner validation. We do not promote it based on that exposed outer
score. These patterns motivate testing selection/acquisition robustness, not
presenting a post-hoc outer-validation winner as the tuned method.

## Preservation screen: one candidate, not a solved trade-off

Each trajectory also selects its best inner target checkpoint subject to <=0.001
source-validation point loss in EACH of deepprime and pridict_pridict2. Every
conventional/anchored control falls back to the original model under that rule.

Among the 12 source-loss/sampling/weight combinations, only balanced top-choice
margin at weight 1 selects a nonzero feasible checkpoint (step 10). Its target
inner improvement over constrained E is +0.0019055, above the predeclared +0.0005
promotion threshold. Its source-validation changes are approximately -0.000051
for deepprime and -0.000947 for pridict_pridict2. Those are point-estimate filters,
not confidence-based noninferiority. The latter is close to the cutoff.

The candidate improves on the starting model's target and source-audit point
estimates, but remains below OptiPrime and below unconstrained E on target.
Preserving source decisions has not solved external utility. Soft KL, true-label
utility, and margin loss at other tested sampling/weight settings are all retained
as negative or trade-off results, not omitted.

Complete ledgers: [optimization controls](CONTROLS_RESULTS.md),
[factorial](FACTORIAL_RESULTS.md), and the machine-readable selection files.

## Validation and data checks

- 32 fits, 456 distinct input/output fingerprints, all checkpoint choices and
  288 selected/final endpoint evaluations verified.
- 41 protocol/endpoint tests pass. End-to-end E and balanced-margin deployments
  reproduce cached choices for 128 target and 128 source groups across batch
  sizes 31/128; maximum tested score discrepancy is below 2.4e-6.
- The manuscript now distinguishes original benchmark, external zero-shot,
  historical full-budget adaptation, and low-label development results. Its
  utility prose agrees with the previously corrected fixed-allele table; 44
  numerical checks pass. Original pooled prediction results remain unchanged.
- Independent data are still insufficient: OPED lacks a verified barcode/design/
  condition linkage; corrected ePRIDICT input eligibility yields 140 K562 designs
  but only 15 shallow decisions. No acquired-panel model evaluation occurred.
  See [data feasibility](DATA_FEASIBILITY.md) and [audit corrections](RUNTIME_NOTES.md).

## Go/no-go: replicate the candidate, not expand the architecture

**Proceed with the prespecified acquisition replication.** The source candidate
passes the promotion rule, and the optimized-baseline/inner-selection findings
make it important to distinguish one lucky subset from a repeatable procedure.
This does not require pretending independent confirmation is available.

Freeze four methods: A (100 updates, 3x LR), C_frozen (500 updates, 1x LR;
highest-inner conventional control, not best observed outer performance), E
(100 updates, 1x LR), and balanced margin M (100 updates, 1x LR, weight 1).
Evaluate both checkpoint policies for every method. Repeat 200/1,000 budgets
over three acquisition seeds and two optimizer seeds. The 48 configurations
reuse four identical screen fits and require **44 new fits**, within the proposed
budget. No hyperparameter expansion, OptiPrime hybrid, test reopening, or
architecture change. Higher-label curves and independent confirmation remain
separate conditional gates after replication.
