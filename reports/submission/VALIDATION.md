# Validation of the manuscript package

Completed 12 September 2026.

- Compiled PDF: 33 pages, including the supplementary material and bibliography.
- Final LaTeX build: no warnings, undefined citations/references or overfull boxes.
- `verify_submission.py`: 147 checks passed. The exact PDF fingerprint is recorded in `verification.json`.
- Existing endpoint/adaptation/protocol tests: 43 passed (`explore_v2/test_endpoints.py`, `explore_v2/test_adaptation_followup.py`, `explore_v3/test_protocol.py`, `explore_v3_1/test_followup.py`).
- Extracted the source ZIP into a separate temporary directory outside the repository. Rebuilt without private data/checkpoints or repository-relative assets; 111 portable checks passed and reproduced a 33-page document.
- Visually inspected representative title, result-table, quantitative-figure, equation, architecture and supplementary-table pages.
- Existing manuscripts and historical research artifacts were not modified. No new model fitting, biological-outcome scoring, submission, commit or push was performed.

These checks establish compilation, selected numerical consistency, provenance and package portability. They do not certify every interpretive sentence, remove retrospective evaluation exposure or establish independent-study generalization. Author declarations and data/software release permissions remain in `SUBMISSION_CHECKLIST.md`.
