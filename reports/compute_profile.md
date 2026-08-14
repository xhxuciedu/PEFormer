# Compute profile

GPU: `NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition` (device 6, CUDA_VISIBLE_DEVICES=6)

Model: PE-RankFormer, 26.6M params, d_model=384

Sequence lengths: edit=102, pegRNA=90

Precision: BF16 autocast, AdamW, fused SDPA (PyTorch default backend)

Corpus: 262,508 rows (236,257 assumed train split for epoch-time estimate)


## Batch size scaling (100 timed steps after 20 warm-up)

| batch | step time (s) | examples/sec | peak mem (GB) | steps/epoch | min/epoch |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.0690 | 1856 | 3.70 | 1845 | 2.12 |
| 256 | 0.0781 | 3279 | 6.82 | 922 | 1.20 |
| 512 | 0.1361 | 3763 | 13.11 | 461 | 1.05 |
| 1024 | 0.2812 | 3641 | 25.67 | 230 | 1.08 |

## Recommendation
Best throughput at batch_size=512 (3763 examples/sec, 13.11 GB peak).
Estimated full training time: 0.35h (20 epochs) - 0.52h (30 epochs), single model, single fold.