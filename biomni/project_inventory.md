# Project Inventory

## Project: PE-RankFormer
**Goal**: Develop and evaluate a pure-learning model for prime-editing efficiency prediction, benchmarked against OptiPrime, DeepPrime-FT, and PRIDICT2.0.

## Initial State (recorded 2026-08-12)

### User-Uploaded Files
| File | Description |
|------|-------------|
| `claude.md` | Full research task instructions (33,487 chars) |
| `41587_2026_3261_MOESM3_ESM.xlsx` | Hsu et al. supplementary Excel (3 sheets: Endo_gRNAs, LibMMR, LibCV) |
| `s41587-026-03261-7.pdf` | Main Hsu et al. 2026 paper |
| `41587_2026_3261_MOESM1_ESM.pdf` | Supplementary text (model architecture details) |

### Cloned External Repositories (`/workspace/external/`)
| Repo | Source | Contents |
|------|--------|----------|
| `optiprime/` | github.com/alvin-hsu/optiprime-src | OptiPrime source code (JAX/Flax), model weights, example CSVs |
| `deepprime/` | github.com/hkimlab/DeepPrime | DeepPrime model code + weights (no training data) |
| `deepprime_official/` | github.com/yumin-c/DeepPrime | DeepPrime model code + **training data** (main + 19 variants) |
| `pridict/` | github.com/uzh-dqbm-cmi/PRIDICT | PRIDICT v1 model code (no training data in main branch) |
| `pridict2/` | github.com/uzh-dqbm-cmi/PRIDICT2 | PRIDICT2.0 model code + dataset (data_23k_v1.csv) |
| `pridict_supp/` | github.com/uzh-dqbm-cmi/PRIDICT (supplementary_files branch) | PRIDICT v1 supplementary data (editing tables, subscreens) |
| `epridict_supp/` | github.com/Schwank-Lab/epridict (supplementary_files branch) | PRIDICT2.0 supplementary Excel with spacer/PBS/RTT + test splits |

### Project Structure Created
```
data/raw/{hsu2026,deepprime,pridict,pridict2}/
data/interim/
data/processed/
data/manifests/
src/pe_rankformer/{data,models,training,evaluation,utils}/
scripts/{data,train,evaluate}/
configs/
tests/
results/figures/
checkpoints/
logs/
reports/
```

### Environment
- Python 3.11.13 (conda-forge)
- pandas 3.0.5, pyarrow 25.0.1, openpyxl 3.1.5, numpy 2.4.6
- scipy 1.17.1, scikit-learn 1.9.0, matplotlib 3.11.1, statsmodels 0.14.6
- Full details in `reports/environment.txt`

### Key Verified Facts
- Hsu data: 74,769 nonmissing PE efficiency values (LibMMR: 36,560 + LibCV: 38,209)
- Target total: 297,962 = 74,769 (Hsu) + 223,193 (historical refs 54-56)
- Historical sources: PRIDICT v1 (ref 54, Mathis 2023), DeepPrime (ref 55, Yu 2023), PRIDICT2.0 (ref 56, Mathis 2024)
- 40 experimental contexts total (4 Hsu + 36 historical)
