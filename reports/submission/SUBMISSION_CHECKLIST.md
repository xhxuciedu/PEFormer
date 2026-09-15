# Before actual journal submission

## Author-supplied items

- Supply author names, order, affiliations, ORCIDs and corresponding-author contact.
- Confirm contributor roles, funding/compute support, competing interests and acknowledgements.
- Approve final scientific claims, including limitations and corrections. Do not replace negative replicated findings with a favorable single fit.
- Confirm any required disclosure of AI-assisted coding/writing and human responsibility for the paper.

## Data, software and publication access

- Obtain/confirm permission to redistribute the author-supplied harmonized corpus, or state the exact permitted access route. Some source material was not fully reconstructed from public releases.
- Confirm repository accessibility and license. A GitHub URL alone does not establish public access or an approved release license.
- Decide what checkpoints and preprocessing assets can be released, with checksums and a verified inference example.
- Archive the approved code/manuscript state and record the resulting immutable release identifier/DOI. Do not cite a DOI that does not exist.
- Confirm the draft's ethical/data-use declarations against the original data terms and chosen journal requirements.

## Scientific boundaries to retain

- The original source result is 0.9079 vs 0.8690 Spearman, but the source benchmark was revisited across generations and later audits.
- External zero-shot comparison uses a five-fold ordinal–S4D ensemble, not the heterogeneous source ensemble.
- Full-budget adaptation outperforms **unadapted released** OptiPrime on a retrospective library test subset; this is not equal-label adaptation or independent-study confirmation.
- V3.1 has no confirmed low-label OptiPrime top-choice win. Balanced margin preservation improves the tested source-retention trade-off at a target cost.
- V3.1 confidence intervals condition on the fitted acquisition/optimizer runs. The point feasibility rule is not a statistical guarantee.
- Independent ePRIDICT/OPED candidates have not been scored by these models. Their data-feasibility audits are not validation results.
- No higher-label replicated curves, true dual-representation branch, excluded-domain pretraining or independent final-model confirmation is claimed.

## Production checks

- Rebuild with `build.py`, run `verify_submission.py`, and visually inspect the final PDF after metadata edits.
- Keep the per-head distinction in the historical adaptation table: its correlation column uses the prediction output, while utility uses the deployed selector.
- Keep the distinction between means of run outcomes and metrics of averaged scores.
- Adapt bibliography/section/figure conventions only after choosing a journal; no journal-specific page constraints were applied here.
- Submission, author correspondence, commit and push are not performed automatically by this writing task.
