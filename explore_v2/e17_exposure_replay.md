# E17a - the comparisons the ranking loss actually sees

One epoch of the real `GroupedBatchSampler` and `sample_ranking_pairs` replayed under each key, at the training config's own settings ({'min_target_gap': 0.02, 'max_pairs_per_group_per_batch': 4, 'batch_size': 512, 'max_group_take': 8, 'epochs_replayed': 1}). Nothing trained.

> **Confound, adopted from the review.** corpus.group_key is passed to GroupedBatchSampler (train_pilot.py:668) and to sample_ranking_pairs (train_pilot.py:704). E13 changed both channels at once, so E14/E15 measure the combined effect, not the ranking channel alone.

| measured per epoch | current ranking key | canonical decision group | ratio |
|---|---:|---:|---:|
| distinct group keys over training rows | 209,161 | 132,066 |  |
| sampled pair instances | 15,695 | 62,028 | 3.95x |
| unique row pairs visited | 15,695 | 62,028 | 3.95x |
| groups receiving an update | 6,792 | 25,175 |  |
| share of groups updated | 3.2% | 19.1% | |
| pairs spanning two alleles | 3 | 0 | |

**The sharper statement of the defect.** Under the current key, 100.0% of sampled pairs compare two designs of the **same RTT length** — because for one allele the stored window's extent is determined by the RTT, so an identical window pair implies an identical RTT length. Under the canonical key that share is 6.8%. The ranking term was therefore almost never asked to compare the geometry variation that distinguishes alternative pegRNAs.

Absolute RTT-length difference within sampled pairs — current key: {'0': 15695}; canonical: {'0': 4199, '1': 2208, '2': 1576, '3': 7553, '4': 8597, '5': 7344, '6': 1612, '7': 8367}

Target-gap distribution of sampled pairs — current: {'mean': 0.12130168789238753, 'median': 0.0801287293434143, 'p90': 0.2760459065437317}; canonical: {'mean': 0.21160388936751642, 'median': 0.1399644911289215, 'p90': 0.520777201652527}

Two wording corrections follow, and are applied in `e13_regroup_training.md`: 14,277 / 368,307 counts *possible* within-group design pairs, not measured exposure; and `lambda_rank = 0.25` is a coefficient, not a statement that ranking contributes a quarter of the loss or the gradient.
