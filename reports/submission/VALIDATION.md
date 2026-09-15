# Validation of the manuscript package

Writing-revision validation completed 14 September 2026.

- Compiled PDF: 35 pages, including the supplementary material and bibliography.
- Final LaTeX build: no warnings, undefined citations/references or overfull boxes.
- `verify_submission.py`: 170 checks passed. The exact PDF fingerprint is recorded in `verification.json`. New checks cover the promoted rank endpoints and three-seed selection figure; moved diagnostic contrasts remain numerically checked in the supplement.
- Existing endpoint/adaptation/protocol tests: 43 passed during the 12 September research review (`explore_v2/test_endpoints.py`, `explore_v2/test_adaptation_followup.py`, `explore_v3/test_protocol.py`, `explore_v3_1/test_followup.py`). These model tests were not rerun for this writing-only revision; training and evaluation implementations are unchanged.
- Extracted the revised source ZIP into a separate temporary directory outside the repository. Rebuilt without private data/checkpoints or repository-relative assets; 131 portable checks passed and reproduced a 35-page document.
- Visually inspected the revised title/abstract, main ranking table, architecture diagram, ranking figure, selection figure and seed-level selection table. The prior review also inspected methods and supplementary-table pages.
- Historical manuscripts under `reports/paper/` and original research artifacts were not modified. No new model fitting, biological-outcome scoring, journal submission, commit or push was performed in this revision.

These checks establish compilation, selected numerical consistency, provenance and package portability. They do not certify every interpretive sentence, remove retrospective evaluation exposure or establish independent-study generalization. Author declarations and data/software release permissions remain in `SUBMISSION_CHECKLIST.md`.
