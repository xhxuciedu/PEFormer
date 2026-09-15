# Evidence inventory and review decisions

Research review: 12 September 2026; method-centered writing revision: 14 September 2026. Research baseline: `b55a14462a4b7e217041ffd763e90edf83f8860e`.
Paths below are relative to the repository root. Local reports are evidence about this research, not published literature. More recent corrected numerical ledgers take precedence over chronological prose. Proposals are not counted as completed results.

## Claim-to-record map

| Paper material | Numerical / implementation record | Interpretation retained |
|---|---|---|
| Initial data reconstruction and retracted pilot | `reports/research_log.md`, `training_data_reconstruction.md`, `dataset_reconstruction_status.md`, `baseline_reproduction_notes.md`, `optiprime_input_specification.md` | Early padded-comparator margins excluded; exact official corpus controls the source claims. |
| Source generations and final ensemble | `reports/round2_results.md`, `round3_results.md`, `round4_results.md`, their final model specifications; `results/round4/final_bootstrap.json` | 0.907874 vs 0.869003; four generation-level evaluations plus later audits, not pristine final test. |
| Source objective/mixer contrasts | `reports/round5_research_log.md`; old paper's factorial section and underlying OOF records | Observational 2×2, earlier baseline cell, not causal randomized attribution. |
| Negative source model searches | `reports/round2_research_log.md` through `round6_research_log.md`, `round6_results.md`, `round7_diagnosis.md`, `round9_research_log.md` | Tested settings and replication failures; do not generalize to all architectures/losses. |
| Weighting, censoring and low-rank controls | `revision/task_1_8_weighting_by_partition.json`, `task_2_round9_wave_results.json`, round-9 log | Within-model weight intervention; no numerical decomposition of cross-model causality. |
| Source within-group/condition robustness | `revision/task_1_1_per_target.json`, `task_1_3_tie_robust.json`, `task_1_4_leakage_free.json` | Protospacer metric differs from fixed-edit choice; dependence-unit correction retained. |
| Calibration and repeatability | `revision/task_1_5_calibration_baselines.json`, `task_1_6_ceiling_uncertainty.json`; `explore_v2/e21_correlation_surfaces.json` | Actual isotonic ties acknowledged; empirical repeatability reference is not an identified Bayes ceiling. |
| Canonical decisions and corrected utility | `explore_v2/canon.py`, `endpoints.py`, `utility_table_numbers.json`, `e16_corrected_endpoints.json` | Distinct candidates, both-strand indel normalization, depth eligibility, tied-max random baseline. |
| E01–E09 context analysis | `explore_v2/e01_decision_manifest.md` through `e09_clean_interaction_retest.md`, `GATE_A_REPORT.md`, `CORRECTIONS.md` | Old quartet population explicitly labeled; mixed-coordinate interaction headline superseded. |
| E10–E12 initially reserved library | `explore_v2/e10_build_reserved_panel.*`, `PREREGISTRATION_reserved_panel.md`, `e11_score_reserved_panel.*`, `e12_context_sensitivity.*` | Initial zero-shot analysis declared before scoring; later reuse loses reserved status. Corrected E16 endpoints lead. |
| E13–E24 source repairs and panel retests | `explore_v2/e13_regroup_training.*`, `e17_exposure_replay.*`, `e18_head_on_panel.*`, `e19_matrix_results.*`, `e22_best_arm_on_panel.*`, `e23_where_the_decision_is_lost.*`, `e24_perfold_recipe.*` | Canonical batching helps without ranking; frozen feature statistics required; no standalone external selection win. Historical comparator blends not included as our standalone method. |
| E25–E28 full-budget adaptation | `explore_v2/e25_adaptation_partition.json`, `e26_adaptation_pilot.py`, `e27_adaptation_test.json`, `adapt.py` | Target-supervised gain on retrospective test against unadapted comparator. Separate-head correlation and utility have different score identities. |
| E29–E32 corrections | `explore_v2/e29_head_audit.json`, `e30_pairwise_replication.json`, `e31_label_budgets.json`, `e32_score_scale_diagnostic.json`, `ADAPTATION_FOLLOWUP_RESULTS.md` | Deployed-head retention replaces unused-head result; pairwise only one historical test seed; validation labels count. |
| V3 initial low-label study | `explore_v3/RESEARCH_REPORT.md`, `screen_results.json`, `replication_results.json`, `control_results.json`, `retention_diagnostic.json`, protocols and `model.py` | 34 fits; one acquisition, three optimizer seeds; fixed-recipe anchor advantage not generalized beyond later controls. |
| V3.1 controls and factorial | `explore_v3_1/controls_results.json`, `factorial_results.json`, `secondary_results.json`, `EXECUTION_PROTOCOL.md` | 20+12 fits; inner-selected configurations; loss scales/sampling not causally isolated. |
| V3.1 replication | `explore_v3_1/replication_results.json`, `REPLICATION_RESULTS.md`, `acquisition_audit.json`, `REPLICATION_DECISION.json` | 48 configurations, 4 exact reuses, 44 new; six-run outcome averages; conditional component intervals. |
| V3.1 verification and runtime | `explore_v3_1/verification.json`, `replication_verification.json`, `RUNTIME_NOTES.md`, `inference_cost.json`, `deployment_*.json` | Arithmetic/deployment checks do not establish independent evidence. Numerical fallback identity is not a biological contrast. |
| Independent data feasibility | `explore_v3/DATA_AUDIT.md`, `explore_v3_1/DATA_FEASIBILITY.md`, `data_feasibility.json`, `alphabet_audit.json`, `core_overlap_audit.json` | No model scoring of acquired independent panels; RNA/DNA compatibility and linked outcomes remain important. |

The earlier narrative manuscripts (`reports/paper/pe_rankformer_paper.tex`, `reports/explore_v2_report.tex`, `reports/adaptation_report.tex`) were used as cross-checks, not as the final authority when inconsistent with corrected records.

## Additional editorial corrections made in this manuscript

1. Removed the claim that paired comparisons cancel optimistic bias from repeated benchmark use.
2. Replaced “ordinal loss optimizes Spearman” and guaranteed decreasing cumulative curves with the implemented independent-threshold objective and its actual limitations.
3. Corrected “isotonic leaves Spearman unchanged” using actual before/after values: source 0.907874→0.907894, external 0.709087→0.708868.
4. Described missing-indel behavior as the implemented two-logit normalized fallback, not an automatically exact marginal likelihood under arbitrary assay denominators.
5. Disclosed that the GBM configuration was selected on the benchmark and that the old internal lockbox participates in shortlist selection.
6. Removed causal inference that FiLM cannot reorder designs, missing context must explain the residual, or independent per-candidate scoring cannot solve selection.
7. Distinguished the source heterogeneous ensemble, external ordinal–S4D ensemble, and adapted single backbone throughout. Recorded population dependence of historical rank averaging.
8. Preserved algorithmic failure/negative results without turning them into formal power bounds or universal absence-of-effect claims.
9. Marked author identity, affiliation, competing interests, funding and release permissions as unconfirmed rather than inheriting earlier placeholders as facts.
10. Reframed the title, abstract, introduction, main results and discussion around the proposed PE-RankFormer method, with rank-based superiority first and selection-efficiency gains second. Promoted existing Kendall's tau-b and positive-outcome Spearman results into the main ranking table; these are not new experiments.
11. Added a full-budget external selection figure and per-seed table from the unchanged E27 ledger, including all three native-ordinal seeds and their existing paired intervals. No confidence interval was invented for the across-seed mean.
12. Moved detailed calibration, training, zero-shot stratification, alternative-head and low-label/retention analyses into the supplement. The zero-shot selection shortfall, target-label asymmetry and retrospective exposure remain explicit in the main paper. The research findings and their scope are unchanged.

## Primary-literature metadata checks

Direct publisher/author records checked for Anzalone (Nature 2019), PRIDICT (Nature Biotechnology 2023), DeepPrime (Cell 2023; PubMed 37119812), PRIDICT2.0/ePRIDICT (Nature Biotechnology **2025 issue**, online 2024), OPED (Nature Machine Intelligence 2023), OptiPrime (online August 2026), S4D (NeurIPS 2022), FiLM (AAAI 2018), and CORAL (Pattern Recognition Letters 2020). URLs/DOIs are in `references.bib`; no related work is claimed to have been reproduced unless explicitly reported above.
