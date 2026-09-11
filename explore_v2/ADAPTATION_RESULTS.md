# Target-library adaptation: current results through E32

Updated 10 September 2026. Earlier interpretations are preserved in
[the historical archive](archive/ADAPTATION_RESULTS_E25_E28.md).
Original experiment JSON and checkpoints are unchanged.

## Supported findings

- Native ordinal fine-tuning P achieves test selection efficiency 0.04178 versus
  released OptiPrime's 0.04020 across three seeds: +0.00158 and 15.7% less regret.
  It used 19,357 target training groups plus 5,561 validation groups; OptiPrime
  received no target adaptation.
- The actual initializing checkpoint scores 0.03919, not the ensemble's 0.03952.
  The 200-group experiment does not establish harm against its matched start.
- Three-seed pairwise replication improves validation over utility S, but an
  advantage over P/shared is not established. Only one pairwise seed has a
  historical target-test evaluation.
- Source preservation must be measured through the deployed score. Source @1
  averages 0.12776 for P, 0.12607 for utility S and 0.11460 for pairwise S, versus
  0.13350 at initialization. Strong retained prediction heads do not imply
  retained selection utility.
- Shared and dual-head models differ in score scale as well as separation.
  Geometry-aware architecture and set attention remain unvalidated hypotheses,
  not rejected method classes.
- Corrected nested total-label manifests are built; their training experiments
  and new architecture/replay experiments remain pending.

## Current authoritative reports

- [Cross-dataset research summary](SUMMARY_REPORT.md).
- [Detailed E29–E32 results and reproducibility](ADAPTATION_FOLLOWUP_RESULTS.md).
- [Updated adaptation PDF](../reports/adaptation_report.pdf).
- [Proposed v3 research plan](../explore_v3/RESEARCH_PLAN.md).

This is a retrospective within-library adaptation gain, not architectural
superiority or general cross-library transfer. Computational-only; no OptiPrime
hybrid, teacher or feature borrowing is involved.
