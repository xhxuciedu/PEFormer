# Complete computational research manuscript

Start with [main.pdf](main.pdf) or edit [main.tex](main.tex). This is a newly integrated paper covering the corrected source benchmark, external-library decisions, historical adaptation, fixed-recipe v3 work and acquisition-replicated v3.1 work. Earlier manuscripts under `reports/paper/` are preserved unchanged.

The scientific draft is complete; this is **not yet an administratively submission-ready or independently confirmed study**. Required author/release decisions are in [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md). No training or new outcome scoring was performed to write this manuscript.

## Contents

- `main.tex`, `sections/`: abstract, introduction, integrated results, discussion, detailed methods, declarations and comprehensive supplement.
- `references.bib`: primary literature; publication metadata checked during this review. Mathis et al. uses its 2025 issue year and notes its 2024 online publication. OptiPrime is cited as the 2026 online article without an invented volume/page assignment.
- `figures/`: four generated quantitative figures, PDF and PNG. The fifth figure is a vector architecture/workflow diagram directly in LaTeX.
- `tables/`: seven automatically generated result tables. Additional design, source-summary and supplementary accounting tables are in the section sources.
- `source_data.json`: exact displayed generated-table values, underlying figure data and SHA-256 aggregate-input provenance.
- `adaptation_source_data.json`: complete control/factorial/replication aggregate ledgers, including conditional intervals and run-level summaries, not private candidate data.
- `replication_summary.csv`: all replicated method/policy/budget aggregates.
- `evidence_inventory.md`: claim-to-artifact map, exposure qualifications and reviewed corrections.
- `verify_submission.py`: numerical/provenance, source-reference and PDF checks. This does not validate scientific independence or every interpretive sentence.
- `submission_sources.zip`: portable source/PDF package produced with `build.py --zip`.

## Build anywhere

Requirements: Python 3 and a TeX installation containing pdfLaTeX, BibTeX, natbib, TikZ, lineno, booktabs, tabularx, longtable, microtype and Latin Modern fonts. No private data, GPU, network, training checkpoints or repository-relative figures are needed to compile the supplied package.

```bash
cd reports/submission
python3 build.py
# Optional portable package:
python3 build.py --zip
```

Upload the contents to a LaTeX editor with `main.tex` as the root document. Line numbers are enabled for review; remove `\linenumbers` to disable them. Figures use percent axes while numerical tables use efficiency fractions, explicitly labeled in their captions.

## Regenerate and verify within this repository

From the repository root:

```bash
.venv/bin/python reports/submission/generate_assets.py
.venv/bin/python reports/submission/build.py
.venv/bin/python reports/submission/verify_submission.py
```

Asset regeneration requires matplotlib and the archived aggregate JSON records, including the locally retained original-source metrics under `results/round4/`; it does not read sequence-bearing caches. Full numerical verification additionally uses historical source/revision ledgers in this repository and PyMuPDF for PDF inspection. In a standalone extracted package, `verify_submission.py --portable` checks packaged tables, references and PDF structure without repository inputs. To package the exact already-verified PDF without rebuilding it, use `build.py --package-only --zip`.

The reviewed research baseline is commit `b55a14462a4b7e217041ffd763e90edf83f8860e`. Manuscript files have not been committed or pushed by this writing task. A final archival release identifier must be set after author approval.
