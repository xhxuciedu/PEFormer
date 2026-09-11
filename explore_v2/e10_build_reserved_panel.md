# E10 - the reserved panel, built and validated

## Reconstruction

288,730 of 288,793 rows reconstructed (63 dropped: masked block unusable, PAM not NGG, or the RT-PBS footprint running past the 74-mer). Nick at index 21, protospacer at [4, 24].

## Validation

| check | result |
|---|---|
| `AfterRTT_left4` == `WT74[21+RTlen:+4]`, substitution rows | 153,974 / 153,974 |
| PBS and RTT rules reproduce the corpus's stored pegRNAs | 100% of Kim rows |
| reconstructed PBS agrees with the corpus on the overlap | **1.0000** of 1,575 matched rows |
| reconstructed RTT agrees with the corpus on the overlap | **0.9968** |

Overlap match key: identical 40 bp WT window, identical 40 bp edited window, PBSlen, RTlen. Under a looser key that leaves the intended edit free, RTT agreement is 0.77 -- which is the key's fault, not the reconstruction's, since a different edit must template a different RTT.

## Overlap removal

56,578 rows share a protospacer with the training corpus and 32,904 share a canonical allele; both are removed, leaving 219,669 rows.

## The panel

118,187 rows in 30,475 decision groups with two or more alternative designs for one intended allele.

| candidate depth | groups |
|---|---:|
| >= 2 designs | 30,475 |
| >= 3 designs | 18,660 |
| >= 5 designs | 8,775 |
| >= 8 designs | 2,400 |
| >= 12 designs | 541 |

24,668 groups have a non-constant outcome and can discriminate an ordering; 5,807 have every design at exactly zero and contribute zero achieved efficiency to deployment utility rather than being dropped. The best design in a group averages 0.0461 and the spread between best and worst averages 0.0412.

Edit types: {'sub': 58027, 'ins': 34742, 'del': 25418}

## Outputs

- PE-RankFormer input: `explore_v2/reserved_panel_v2.parquet`, SHA-256 `6dc91509073c11e60511581113b4e7be...`
- OptiPrime input: `data/interim/reserved_panel_kim_large/Kim_HEK293T_LibSmall_PE2max_test.csv`, SHA-256 `ba3dadd0237dfec0530aceb0db3b0e63...`

Context assigned to every row, identically for both predictors: {'source_study': 'deepprime', 'cell_type': 'HEK293T', 'pe_type': 'PE2', 'cas9_type': 'PEmax-Cas9', 'cas9_pam': 'SpNGG', 'motif': 'none', 'scaffold_name': 'GC_F+E', 'rt_name': 'PE2-RT', 'group': 'Kim_HEK293T', 'time': 7.0, 'PEmax': 1, 'epegRNA': 0, 'MLH1dn': 0, 'NRCH': 0, 'linker': None, 'weight': 0.1}

No model has been scored on this panel at the time of writing.
