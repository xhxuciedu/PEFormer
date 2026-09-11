# Execution notes and audit corrections

11 September 2026.

1. Controls use GPU 7 for adapted backbones and GPU 0 for frozen heads. GPUs 2
   and 6 were occupied by unrelated work and were not used. Workspace-local
   temporary storage is used because the system `/tmp` volume was nearly full.
2. All four E controls completed before the source factorial began. E's recipe
   was frozen using target inner validation alone in `e_recipe.json`: 100 target
   updates and LR multiplier 1. The 12 source fits run on the released GPU 0
   while other conventional controls finish on GPU 7. This scheduling change
   alters neither experiment count nor selection rule. Full control selection
   must reproduce the same E recipe after all 20 controls complete.
3. The first follow-up external homology audit used DNA-only spacer strings and
   consequently recognized only 398 source cores: source pegRNAs are RNA, not
   DNA strings. `alphabet_audit.json` and
   `data/epridict_input_audit_alphabet_corrected.parquet` supersede those core
   counts and combined eligibility. RNA-to-DNA normalization recognizes 38,814
   source cores; no exact external core matches occur, but two external designs
   are flagged by the combined local-homology rules. The corrected population
   is 140 token-safe K562 designs, 111 alleles, still 15 multi-design groups.
   Previous files remain as an explicit audit trail, not the final eligibility.
4. The legacy pegRNA tokenizer maps RNA U to the N token. All source pegRNA
   fields contain only A/C/G/U and no actual N or T, making this mapping
   information-preserving on the historical corpus. The target adaptation cache
   uses the same convention. Feeding external A/C/G/T pegRNAs directly would
   instead activate the unused T token. This is a deployment-convention issue,
   not evidence that the current matched experiments discarded the fourth base.
   No tokenizer, pretrained checkpoint or training cache was changed mid-study.
5. A stronger core-19 identity sensitivity check finds one source-replay/validation
   core shared with target outer validation, none with target budget pool or
   source audit. The original split used full-spacer and allele components.
   Quantify the affected evaluation groups and report an excluded-group
   sensitivity; do not silently alter the primary population mid-experiment.
6. After the complete 32-fit screen passed artifact verification, the recorded
   promotion gate advanced balanced teacher-margin preservation at weight 1.
   Acquisition replication freezes all four target-inner-selected recipes;
   48 configurations reuse four exact screen fits and add 44 new fits. GPUs
   0, 1, 3 and 7 were idle at replication launch. GPU 4 was separately used
   for a short saved-model inference benchmark after its own idle check.
   No unrelated job was stopped. `M` denotes margin preservation in v3.1,
   distinct from the earlier v2 multi-task arm.
7. Screen analysis/verification records remain snapshots of the initial 32
   fits. New replication analysis uses its explicit 48-job manifest, so adding
   acquisitions cannot silently change the screened hyperparameter selection.
   The original screen scripts intentionally reject an expanded all-runs
   directory. Diagnostic scripts added after training do not modify the
   fingerprinted trainer, helper, cache or execution protocol.
8. Two additional tests cover depth-conditioned top-k evaluation and
   within-group correlation with constant groups and non-contiguous indices.
   The combined suite now has 43 passing tests; the screen report's 41-test
   count records verification at the earlier promotion decision.
9. All 44 additional replication fits completed successfully, bringing v3.1
   to 76 unique fits. Independent replication verification recomputed 1,056
   checkpoint-surface summaries and 432 endpoint evaluations and checked 685
   fingerprints. The analysis separately verified its 680 model/input hashes;
   the verifier also includes five launch-manifest dependencies.
10. For fallback-only procedures, averaging six identical floating-point
    outcomes introduces differences below 1e-17 from the unaveraged baseline
    in the raw statistical ledger. These are numerical roundoff, not biological
    effects. Their tiny interval endpoints and bootstrap p-values have no
    inferential meaning; the deployed procedure and its true contrast with
    initialization are exactly unchanged. They round to zero in the tables.
    No substantive model comparison or manuscript claim relies on these null
    p-values. Historical numerical artifacts are retained transparently.
11. The final manuscript checks cover 44 original prose values, 64 follow-up
    prose values and exact parity of all ten generated replication-table rows.
    Original utility-check expectations were updated to the already corrected
    v2 fixed-allele artifact, not relaxed to accept retired protospacer utility.
    The original reproduction pipeline now regenerates that corrected utility
    table after its legacy table-generation step; the complete original
    training pipeline was not rerun. Follow-up training is documented separately.
