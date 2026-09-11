# V3-01: acquired-data eligibility audit

No model was scored on either acquired panel.

## OPED

The supplement has 30 primary ClinVar designs, 40 tool-comparison designs over 8 alleles,
and 18 top-three designs over 6 alleles. These sheets have no measured-efficiency columns.
The [primary paper](https://www.nature.com/articles/s42256-023-00739-w) points to raw sequencing
under PRJNA882795; linking and processing it is still needed. Do not invent outcomes from design ranks.

## ePRIDICT

Acquired 146 arrayed pegRNA rows and authors' CRISPResso sequence manifests.
Resolved 143 rows to an unambiguous canonical intended allele;
2 ambiguous and 1 unmapped rows remain excluded.
Grouping uses intended edited versus unedited amplicon sequences, not only locus/edit class.

| Population | Rows | Alleles | Depth >=2 | >=5 | >=8 | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| K562_mapped | 143 | 114 | 15 | 0 | 0 | 3 |
| K562_sequence_overlap_excluded | 143 | 114 | 15 | 0 | 0 | 3 |
| HEK293T_mapped | 54 | 54 | 0 | 0 | 0 | 1 |
| HEK293T_sequence_overlap_excluded | 54 | 54 | 0 | 0 | 0 | 1 |

Exact allele and conservative spacer-core overlaps are excluded in the second population.
This does not establish full homology independence. Editor/context and input support still require
validation before inference. Published data: [ePRIDICT supplementary files](https://github.com/Schwank-Lab/epridict/tree/supplementary_files).

## Gate decision

Two independent deep-selection studies are not yet secured. Broad independent confirmation
is **not ready**. Continue only the bounded retrospective Kim development screen and protocol
work; defer an expanded architecture sweep and strong generalization claims.

Hashes, source-context counts, per-design eligibility and unresolved rows are recorded in
`data_inventory.json` and `data/epridict_eligibility.parquet`. No external outcomes were used for tuning.
