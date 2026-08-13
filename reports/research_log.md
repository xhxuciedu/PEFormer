# Research log — PE-RankFormer pilot

Chronological record of tasks, findings, decisions, problems and resolutions.
Newest entries appended at the bottom.

---

## 2026-08-12 — Stage 0.1 Repository inventory

**Task.** Inspect the repository before changing anything (spec §1.1).

**Finding.** Working tree clean at commit `c6697e1`. Repository contained only
`claude.md`, `pe_rankformer_pilot_proposal.pdf`, and
`data/41587_2026_3261_MOESM3_ESM.xlsx` (Hsu et al. 2026 Supplementary Tables, 5.3 MB,
sha256 `af55814d…7ff1`). No prior code or environment to preserve.

**Decision.** Moved the workbook to `data/raw/hsu2026/` to match the target layout;
checksum verified unchanged. Created the full directory tree from spec §2.

**Output.** `reports/project_inventory.md`.

---

## 2026-08-12 — Stage 0.2 Environment

**Task.** Create the Python environment and validate the GPU (spec §1.2).

**Problem.** The spec asks for Python 3.11 via `python3 -m venv`, but the host has no 3.11
interpreter — `python3` is a conda 3.10.9 and `/usr/bin/python3` is 3.10.

**Resolution.** Created the environment with `conda create -p ./.venv python=3.11`
(Python 3.11.15). It lives at the expected `.venv/` path and is used as `./.venv/bin/python`.

**Problem.** The host has 8 GPUs; the proposal targets an RTX PRO 6000 Blackwell. Device 6
is the Blackwell card (95 GiB, **sm_120**). sm_120 is not supported by default PyTorch
CUDA 12.6 wheels.

**Resolution.** Installed `torch==2.11.0+cu128` from the cu128 index. Verified on device 6:
`cuda.is_available()=True`, BF16 matmul executes, `torch.cuda.is_bf16_supported()=True`,
fused SDPA runs. All training and profiling must set `CUDA_VISIBLE_DEVICES=6`.

**Output.** `reports/environment.txt`, `requirements.txt`, `pyproject.toml`.

---

## 2026-08-12 — Stage 0.3 Hsu workbook audit

**Task.** Identify the Lib-MMR / Lib-CV sheets and reproduce the 74,769 count (spec §4).

**Finding.** The workbook has exactly 3 sheets: `Supp Table 3 Endo_gRNAs` (283 arrayed
endogenous designs, no efficiencies), `Supp Table 4 LibMMR` (10,000 designs),
`Supp Table 5 LibCV` (10,406 designs). Both library sheets carry 8 measurement columns:
`{HEK293T,HeLa}_{PE2,PE4}_{editing,indel}`.

**Finding — count reproduced exactly.** Nonmissing values in the four `*_editing` columns:

| Sheet | HEK293T_PE2 | HEK293T_PE4 | HeLa_PE2 | HeLa_PE4 | Subtotal |
|---|---:|---:|---:|---:|---:|
| Lib-MMR | 8,906 | 8,872 | 9,387 | 9,395 | 36,560 |
| Lib-CV | 8,930 | 8,874 | 10,207 | 10,198 | 38,209 |
| **Total** | | | | | **74,769** |

74,769 − 74,769 = 0. No filtering beyond "value present" is required to hit the target.

**Finding.** Efficiencies are already fractions in [0,1] (max observed 0.9685), so no
percent→fraction conversion is needed for the Hsu source. This must still be unit-tested
per spec §18 and re-checked for every prior study.

**Note.** The workbook covers only 4 experimental contexts. The 40 contexts of the full
OptiPrime corpus must come from the three prior studies.

**Next step.** Build the long-format Hsu table and `data/processed/hsu2026_74769.parquet`;
in parallel, locate the OptiPrime/DeepPrime/PRIDICT/PRIDICT2.0 repositories.

---

## 2026-08-12 — Stage 0.4 Hsu long-format table built

**Task.** Build `data/processed/hsu2026_74769.parquet` (spec §4).

**Decision — orientation.** Canonical target sequences use the `ps-pam-edit` orientation
(protospacer strand 5'->3'). The workbook's `edit-pam-ps` columns are the exact reverse
complement and were dropped as redundant. `pbs_sequence`/`rtt_sequence` are stored in
**pegRNA** orientation, i.e. `revcomp(PBS)` and `revcomp(edit product + homology arm)`.
This rule reproduces Lib-CV's own `extension` column exactly, and it matches OptiPrime's
convention (`pegrna = spacer + scaffold + rtt + pbs + linker`).

**Decision — derived edit fields.** `edit_type`/`edit_length`/`edit_position` are computed
by stripping the shared prefix and suffix of the WT/edited pair rather than trusting the
`Design category` label. This is algorithmically identical to OptiPrime's `split_edit`.

**Problem.** 28 rows (15 Lib-MMR designs) placed the edit 28-30 nt *upstream* of the nick,
which is mechanistically impossible.

**Resolution.** These targets contain a tandem duplication of the site. The 21 nt "5G"
spacer failed to match the truncated first copy but matched the second copy at offset 30,
so `find_protospacer` was locating the wrong copy. Changed the search to take the
**leftmost** match across the full and G-trimmed spacer. After the fix all edits lie
0-23 nt downstream of the nick and the PAM is recovered for every row.

**Validation.** 74,769 rows exactly; 19,934 designs; 1,146 protospacers; 1,152 targets;
protospacer and PAM resolved for 100% of rows; PAM is NGG for 100%; efficiencies in
[0, 0.968] with none outside [0,1]; `revcomp(pbs_sequence)` occurs in the unedited target
for 100% of rows, independently confirming the orientation convention.

**Output.** `data/processed/hsu2026_74769.parquet`
(sha256 `993d114f…1788`), `reports/hsu_data_validation.md`.

---

## 2026-08-12 — Stage 0.5 OptiPrime loader reverse-engineering

**Task.** Determine exactly which prior datasets OptiPrime trained on (spec §5).

**Problem.** The Nature article redirects to a login wall and bioRxiv is Cloudflare-gated;
neither `curl` nor the fetch tool could retrieve the Methods or Data Availability text.
No Zenodo deposit for OptiPrime exists.

**Resolution.** Worked from the official source repository instead, which the spec
designates as a source of truth: `github.com/alvin-hsu/optiprime-src` @ `475db8a`, cloned
to `external/optiprime/`.

**Finding — corpus assembly.** `RxDataset.load_dir` globs `*.csv` in one directory and
concatenates everything. There is no study-level subsetting logic to reverse-engineer:
the corpus is defined by which files were placed there.

**Finding — only two row filters exist.** `format_pe_df` applies `weight > 0` and drops
null `unedited`/`edited`/`weight`. Since `unedited = 1 - (edited_frac + indel_frac)`, a
missing indel rate also drops the row. There is no filtering by edit size, edit type,
editor, cell type, epegRNA status, length, or duplicates, and **no cross-study
de-duplication at all**.

**Finding — 12 groups, recovered from released weights.** Every `group_factors` dict in
`weights/model_{1..5}/log_rates/*.pkl` lists the same 12 `{lab}_{cell}` groups:
Liu×{HEK293T, HeLa}; Kim×{HEK293T, HeLa, A549, DLD1, HCT116, MDA-MB-231, NIH3T3};
Schwank×{HEK293T, K562, U2OS}. So the three prior studies are DeepPrime (Kim),
PRIDICT and PRIDICT2.0 (Schwank). The `YKim` PAM-variant handler and the `HAP1` cell type
are declared in code but have no learned group factor, so they were **not** in the corpus
that produced the released models.

**Problem — folds are unrecoverable.** `split_preset` reads a preset integer `split`
column carried inside the (unpublished) CSVs. The published protospacer-stratified fold
assignment therefore cannot be recovered.

**Decision.** Generate our own deterministic protospacer-grouped folds with a fixed seed
and evaluate every model, including baselines, on the same held-out rows.

**Problem — upstream context.** OptiPrime requires `full_unedited` to start 4 bp upstream
of the protospacer (`PS20_OFFSET = 4`, `proto30` asserted to be 30 nt). Lib-MMR targets in
the Supplementary workbook start exactly at the protospacer, so 4 nt of upstream context
is missing for those designs. Flagged as a blocker for running official OptiPrime
inference on Hsu rows; to be resolved at the baseline stage.

**Output.** `reports/optiprime_data_loader_reverse_engineering.md`.

**Next step.** Assemble the Kim (DeepPrime) and Schwank (PRIDICT/PRIDICT2.0) sources and
reconcile against the 223,193 target.

---

## 2026-08-12 — Stage 0.6 User supplied the OptiPrime paper PDFs

**Task.** User uploaded `s41587-026-03261-7.pdf` (main text) and
`41587_2026_3261_MOESM1_ESM.pdf` (Supplementary Information). Moved into
`data/raw/hsu2026/` as `hsu2026_main_text.pdf` / `hsu2026_supplementary_info.pdf`.

**Problem.** Both PDFs use subsetted fonts without a ToUnicode map on most pages, so
`pypdf` text extraction returns glyph-index garbage for the Supplementary body pages
(figures/tables). Plain-text extraction only worked for the main text and the
Supplementary Text 1 methods section.

**Resolution.** Rendered the affected pages to PNG with `pymupdf` at 400-600 dpi and read
them visually. Page 11 of the Supplementary Information turned out to contain the single
most valuable figure for this project: "OptiPrime 5-fold cross-validation performance
(all datasets)", a 42-bar chart giving the **exact n** of every dataset partition used to
train OptiPrime, labeled by lab (Liu/Schwank/Kim), cell type, and a
PEmax/epegRNA/MLH1dn/NRCH design flag.

**Finding — exact reconciliation achieved.** Transcribed all 42 values into
`data/manifests/optiprime_context_counts.csv`. They sum to **exactly 297,962**
(Liu 65,594 + Schwank 174,067 + Kim 58,301), with zero adjustment — effectively a
checksum confirming the transcription is correct, since an accidental match across 42
independently-read multi-digit numbers is not plausible. This supersedes the
group-identity-only result from the model-weights analysis (§5 finding) with an exact
per-partition target.

**Finding — 17/18 Kim partitions matched to specific DeepPrime release files** by their
boolean flags (e.g. PEmax=1/epegRNA=1/MLH1dn=1/NRCH=0 uniquely identifies
`DP_variant_293T_PE4max_epegRNA_Opti_220428`). One partition (DLD1 PE2max, n=3,423) is
ambiguous: DeepPrime ships two files with identical flags
(`_220428` and `_221114`), most likely merged into the reported partition.

**Finding — new discrepancy: Liu/Hsu 74,769 (raw) vs. 65,594 (used by OptiPrime).** The
four Liu partitions in this figure sum to 65,594, not the 74,769 nonmissing measurements
we extracted from the Supplementary workbook (and which the main text itself describes as
"74,769 PE efficiencies"). ~9,175 rows with a measured editing value were apparently
excluded before OptiPrime training, by a filter not stated in the text and not present in
the released loader (which only filters `weight > 0` / non-null outcome).

**Decision.** Keep `hsu2026_74769.parquet` at the literal, spec-mandated 74,769 count —
that is explicitly the task target and is independently reproducible from public data.
Document the 65,594-vs-74,769 gap as an open discrepancy rather than reverse-engineer an
unstated filter; flag it wherever Liu-subset results are compared to OptiPrime's own
reported numbers.

**Finding — Data Availability confirmed.** No processed training corpus was deposited
anywhere (SRA has only raw reads: BioProject `PRJNA1314411`, plus reanalyzed
`PRJNA735408`, `PRJNA1055086`, `PRJNA1211588`). The per-partition figure is therefore the
best available ground truth, not a shortcut around a missing but findable dataset.

**Output.** `data/manifests/optiprime_context_counts.csv`,
`reports/dataset_reconstruction_status.md` (rewritten), `reports/optiprime_data_loader_reverse_engineering.md`
(§8 appended).

**Next step.** Assemble Kim/DeepPrime and Schwank/PRIDICT+PRIDICT2.0 row-level tables
against the 42 per-partition targets, not just lab-level totals.
