# Replication gate, after the initial 14 fits

10 September 2026. Promote A, D, E and F to optimizer seeds 20260911 and
20260912 at both total budgets, retaining subset seed 20260910. This is the
16-fit replication tranche in the original plan, not a hyperparameter search.

Reason: at nominal 1,000 groups, E achieves target @1 0.041304 with source-audit
@1 0.133085, versus 0.039302 / 0.131962 for A and 0.041334 / 0.124735 for C.
The trade-off is promising but single-seed and confounded by encoder update
policy relative to C. D is the strongest conventional target point estimate
(0.041665), with a large source cost. F did not improve E in that seed but is
retained to test the proposed preservation component rather than substituting
a new winner. No temperatures, replay weights, steps or checkpoint rules change.

The source audit informs this hypothesis-level assessment, not per-run checkpoint
selection or hyperparameter tuning. Replication remains exploratory on an already
exposed development population. Source validation also remains recorded, and the
differences in source-study composition must accompany any retention interpretation.

No independent study is scored during replication. A good replicated result
would still require matched frozen-head controls and independent validation.
