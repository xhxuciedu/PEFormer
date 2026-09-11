# Standalone adaptation follow-up: E29–E32

Date: 2026-09-10. Computational research only. OptiPrime is a frozen evaluation
reference: no OptiPrime scores, parameters, features or teachers enter our model.
This plan supplements, rather than rewrites, the historical E25–E28 experiments.

## Questions and order

1. **E29: matched baseline and head-specific audit.** Evaluate the exact starting
   checkpoint and existing P/S/M/shared/Z checkpoints on target validation and
   source fold 0. Report prediction and deployed selection scores separately,
   including both heads' achieved efficiency @1 and pooled Spearman. Cache
   candidate-level predictions with checkpoint and input hashes. Compare P and S
   directly with paired locus-component intervals on target validation. Source
   fold 0 is an audit surface, not a replay or model-selection set. Separately
   audit the already reported 200-group test comparison against the actual start.
   Do not interpret source-head preservation as selector preservation.
2. **E30: replicate the promising pairwise loss.** Keep the original architecture,
   training groups, optimizer, temperature, epoch cap and stopping rule unchanged.
   Add seeds 20260911 and 20260912 to the existing seed 20260910. Compare with P,
   S-utility and shared at matching seeds using validation, not test. Report
   individual seeds and their mean; average outcomes across seeds is not an
   ensemble. A consistent positive validation contrast is a promotion criterion,
   not proof of superiority. Final paired intervals condition on selected
   checkpoints and do not capture the entire model-development process.
3. **E31: repair the label-budget protocol.** Construct nested, complete-component
   training subsets at approximate budgets of 200/1,000/5,000/all groups for three
   subset seeds. Count candidate measurements, components, training groups and
   validation labels separately. Whole components may make nominal caps inexact.
   Never break a component merely to hit an exact number. An actual total-label
   experiment must allocate an inner validation subset within the budget; fixed
   5,561-group validation is explicitly a training-budget experiment. Add tests
   for nesting, complete components, determinism and split isolation before runs.
   Include the unadapted initialization as an eligible validation-selected
   checkpoint (epoch zero before updates), so a low-budget deployment can select
   no adaptation. Report equal-update and equal-epoch comparisons separately;
   they ask different questions when the training budget changes.
4. **E32: controlled improvement experiments, after the audit.** Compare native
   ordinal prediction with direct expected-efficiency regression; compare shared
   and separate heads with comparable decision-logit scales and identity-initialized
   adapters. Tune a small temperature grid on validation only. If deployed-selector
   forgetting is substantial, prioritize source-training replay or an adapter
   anchored to our own frozen model, excluding source fold 0 and target-held-out
   components. Preserve identical target-update counts and report extra replay
   compute. Geometry/segment pooling follows only after feature definitions are
   validated; failure of scalar concatenation is not rejection of segment pooling.

## First execution tranche

Execute E29, both E30 replications, and E31 protocol construction/tests. Do not
launch an open-ended architecture sweep. Reserve idle GPUs only; do not disturb
other running jobs. Keep new runs and predictions in separate follow-up paths;
do not overwrite historical checkpoints or silently regenerate the old report.
Record actual completed work and unresolved steps in a follow-up results file.

## Evaluation and claim limits

- Primary adaptation endpoint: group-mean measured efficiency of the nominated
  candidate, with complete distinct-design candidate sets and a fixed tie rule.
- E25 components remain the target independence and partitioning unit. Source
  endpoints retain their original informative-group eligibility; do not compare
  raw source and target endpoint levels as if they were the same population.
- The existing target test was inspected in E27 and informed this follow-up.
  Any additional test audit is retrospective; keep it separate from validation
  selection. No new claim of untouched independent confirmation is possible here.
- Do not report equivalent methods merely because a difference is nonsignificant.
  Direct arm contrasts and uncertainty are required; practical equivalence needs
  a declared margin and a suitable interval within it.
- Comparisons with released OptiPrime have unequal target-label exposure. A win
  supports practical adapted-model performance, not architectural superiority.
- Subsequent confirmation requires a computational evaluation on an independent
  existing library or a fully declared outer adaptation protocol. No wet-lab work.

## Deliverables

Reproducible evaluation/replication scripts, hashed candidate prediction caches,
direct head/arm comparisons, validated nested budget manifests, and a concise
results report separating completed experiments from proposed follow-ups.
