# Project inventory — initial state

Recorded 2026-08-12, before any modification of existing files.

## Repository state at start

```
$ git status
On branch main
Your branch is up to date with 'origin/main'.
nothing to commit, working tree clean

$ git log --oneline -1
c6697e1 init
```

Files present (the entire repository):

| Path | Size | SHA256 | Description |
|---|---|---|---|
| `claude.md` | 30 KB | — | Task specification for this study |
| `pe_rankformer_pilot_proposal.pdf` | 200 KB | — | Research proposal (9 pages) |
| `data/41587_2026_3261_MOESM3_ESM.xlsx` | 5.3 MB | `af55814d2f8b9d5d5c54751581751678c0cddf678377e40551b2837b73a47ff1` | Hsu et al. 2026 Supplementary Tables |

No source code, no environment, no processed data existed. Nothing was overwritten.
The only change to pre-existing files was **moving** the workbook from `data/` to
`data/raw/hsu2026/` to fit the target layout; content is unchanged (checksum verified).

## Supplementary workbook contents

`data/raw/hsu2026/41587_2026_3261_MOESM3_ESM.xlsx` contains **3 sheets**:

### 1. `Supp Table 3 Endo_gRNAs` — 283 data rows × 15 columns

Arrayed endogenous (e)pegRNA/nsgRNA designs used in the paper's figures. No efficiency
values in this sheet. Columns: `Figure`, `Figure label / description`, `Editor`,
`Editor modality`, `(e)pegRNA modality`, `(e)pegRNA spacer`, `(e)pegRNA scaffold`,
`(e)pegRNA RTT`, `(e)pegRNA PBS`, `(e)pegRNA motif`, `Full (e)pegRNA sequence`,
`nsgRNA modality`, `nsgRNA spacer`, `nsgRNA scaffold`, `Full nsgRNA sequence`.

Relevant to §38 of the task spec (optional external validation — includes the ATP1A3-type
arrayed validation designs), not to the training corpus.

### 2. `Supp Table 4 LibMMR` — 10,000 data rows × 23 columns  → **Lib-MMR**

Library of paired pegRNA–target designs. Design columns:
`ID`, `Name, spacer-target`, `PBS`, `Design category`, `Name`, `Designed barcode`,
`Designed barcode (revcom)`, `Designed 5G pegRNA spacer`, `Homology arm`,
`Designed pegRNA extension (pbs-edit-hom)`, `Designed pegRNA extension (hom-edit-pbs)`,
`Designed target (ps-pam-edit)`, `Designed edited target (ps-pam-edit)`,
`Designed target (edit-pam-ps)`, `Designed edited target (edit-pam-ps)`.

Measurement columns (8 = 4 contexts × {editing, indel}):
`HEK293T_PE2_editing`, `HEK293T_PE2_indel`, `HEK293T_PE4_editing`, `HEK293T_PE4_indel`,
`HeLa_PE2_editing`, `HeLa_PE2_indel`, `HeLa_PE4_editing`, `HeLa_PE4_indel`.

### 3. `Supp Table 5 LibCV` — 10,406 data rows × 22 columns → **Lib-CV**

ClinVar-derived library. Design columns:
`index`, `mutation_name`, `gene`, `spacer`, `pbs_bind`, `edit_product`, `homology_arm`,
`extension`, `barcode`, `barcode_revcom`, `unedited_target`, `edited_target`,
`designed_oligo`, `designed_oligo_revcom`.

Same 8 measurement columns as Lib-MMR.

## Efficiency-measurement count (verified)

The four `*_editing` columns across the two library sheets:

| Sheet | HEK293T_PE2 | HEK293T_PE4 | HeLa_PE2 | HeLa_PE4 | Subtotal |
|---|---:|---:|---:|---:|---:|
| Lib-MMR (10,000 designs) | 8,906 | 8,872 | 9,387 | 9,395 | 36,560 |
| Lib-CV (10,406 designs) | 8,930 | 8,874 | 10,207 | 10,198 | 38,209 |
| **Total** | 17,836 | 17,746 | 19,594 | 19,593 | **74,769** |

This reproduces the expected **74,769** exactly (task spec §4). Values are already
fractions in `[0, 1]` (observed range 0 – 0.9685), not percentages.

Note that the workbook exposes **4 experimental contexts** (HEK293T/HeLa × PE2/PE4), not
the 40 contexts spanning the whole OptiPrime corpus; the remaining contexts must come
from the three prior studies.

## What is NOT in the repository and must be acquired

- Hsu et al. 2026 main text / Methods / Supplementary Methods (for the exact definition of
  the 297,962-row training corpus and refs 54–56).
- OptiPrime source code and its data loader (`external/optiprime/`).
- DeepPrime, PRIDICT, PRIDICT2.0 repositories and their processed data
  (`external/`, `data/raw/{deepprime,pridict,pridict2}/`).
- The 223,193 historical observations (297,962 − 74,769).

## Compute environment discovered

Multi-GPU host `simons-1.ics.uci.edu`, 64 CPU cores, 187 GiB RAM, 2.9 TB free on
`/srv/disk01`. GPUs: 5× RTX 2080 Ti (11 GB), 2× L40 (46 GB), and **GPU 6 = NVIDIA
RTX PRO 6000 Blackwell Max-Q Workstation Edition, 95 GiB, sm_120**, which is the card the
proposal targets. All training therefore pins `CUDA_VISIBLE_DEVICES=6`.

Note: sm_120 requires a CUDA 12.8+ PyTorch build; `torch==2.11.0+cu128` was installed and
verified for BF16 matmul and fused SDPA. See `reports/environment.txt`.

## Addendum — OptiPrime paper PDFs (supplied 2026-08-12, after initial inventory)

The user separately supplied the OptiPrime main-text and Supplementary Information PDFs.
Moved to `data/raw/hsu2026/`:

| Path | SHA256 |
|---|---|
| `hsu2026_main_text.pdf` (`s41587-026-03261-7.pdf`) | `98333b596849a02c0b15a0ccc3c7eea6d585f0056b6373e0c995b4297afcb84c` |
| `hsu2026_supplementary_info.pdf` (`41587_2026_3261_MOESM1_ESM.pdf`) | `7b51945dfebb8312161450c42e2b948e5dafc2471e27df8a76aebb0391e6f6a2` |

Supplementary page 11 contains the exact per-partition dataset sizes used to train
OptiPrime (42 partitions, summing exactly to 297,962) — see
`reports/dataset_reconstruction_status.md` and
`data/manifests/optiprime_context_counts.csv`.

## Deviations from the task spec

1. **Python version**: the spec prefers 3.11, but no system 3.11 interpreter exists
   (only 3.10). The environment at `.venv/` was therefore created with
   `conda create -p ./.venv python=3.11` instead of `python3 -m venv`. It behaves as a
   normal prefix environment: use `./.venv/bin/python`.
