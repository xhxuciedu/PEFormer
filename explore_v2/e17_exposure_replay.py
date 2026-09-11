"""E17a - what comparisons the ranking loss actually sees, by replaying the real sampler.

`NEXT_STEPS_AFTER_E15.md` section 2 makes two corrections that hold regardless of new
training, and both are adopted here:

* 14,277 / 368,307 is a ratio of *possible* within-group design pairs, not measured training
  exposure. The sampler takes at most `max_group_take=8` members per group per batch and
  `sample_ranking_pairs` keeps at most 4 pairs per group per batch subject to a 0.02 target
  gap, so what the objective sees is a different and smaller thing.
* `lambda_rank = 0.25` is a loss coefficient, not evidence that ranking supplies a quarter
  of the loss or of the gradient.

It also identifies a confound in E13: `corpus.group_key` is passed both to
`GroupedBatchSampler` (train_pilot.py:668) and to `sample_ranking_pairs`
(train_pilot.py:704), so changing it changed batch composition *and* pair eligibility at
once. E13/E14/E15 therefore measure the combined effect; this script measures the two
channels separately so E17's matched matrix can be designed on facts.

One epoch of the real sampler is replayed under each key. Nothing is trained.

Usage: PYTHONPATH=src .venv/bin/python explore_v2/e17_exposure_replay.py
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2common as C  # noqa: E402
from canon import CANON_VERSION  # noqa: E402
from pe_rankformer.data.context import ContextVocab  # noqa: E402
from pe_rankformer.data.dataset import load_featurized  # noqa: E402
from pe_rankformer.training.ranking import GroupedBatchSampler, sample_ranking_pairs  # noqa: E402

KEYS = {"current ranking key": "data/processed/featurized_official.npz",
        "canonical decision group": "data/processed/featurized_official_canongroup.npz"}
MIN_DIFF, MAX_PAIRS, BATCH = 0.02, 4, 512


def replay(npz: str, vocab, seed: int, rtt_len: np.ndarray, allele: np.ndarray) -> dict:
    corpus = load_featurized(str(ROOT / npz), vocab)
    fold = np.asarray(corpus.fold)
    train = np.where(fold >= 2)[0]
    gk = np.asarray(corpus.group_key)[train]
    tgt = torch.from_numpy(np.asarray(corpus.target)[train].astype(np.float32))
    sampler = GroupedBatchSampler(gk, batch_size=BATCH, seed=seed)
    sampler.set_epoch(0)
    gkt = torch.from_numpy(gk.astype(np.int64))

    gen = torch.Generator().manual_seed(seed)
    n_batches = n_sampled = 0
    groups_updated: Counter = Counter()
    unique_pairs: set = set()
    gaps: list = []
    d_rtt: Counter = Counter()
    cross_allele = 0
    for batch in sampler:
        b = np.asarray(batch)
        n_batches += 1
        i, j = sample_ranking_pairs(gkt[b], tgt[b], MIN_DIFF, MAX_PAIRS, gen)
        if i.numel() == 0:
            continue
        # `b` indexes the training subset the sampler was built on; corpus-level lookups
        # (RTT length, allele) must be mapped back through `train` first.
        bi, bj = b[i.numpy()], b[j.numpy()]
        gi, gj = train[bi], train[bj]
        n_sampled += len(gi)
        for x, y in zip(gi, gj):
            unique_pairs.add((min(x, y), max(x, y)))
        groups_updated.update(gk[bi].tolist())
        # `i`, `j` are indices into the batch, not into the training subset: the gap must
        # be read at the mapped positions or it is a different pair's difference entirely.
        gaps.extend((tgt[bi] - tgt[bj]).tolist())
        d_rtt.update(np.abs(rtt_len[gi] - rtt_len[gj]).tolist())
        cross_allele += int((allele[gi] != allele[gj]).sum())
    g = np.asarray(gaps)
    return {"npz": npz, "train_rows": int(len(train)),
            "distinct_group_keys": int(len(np.unique(gk))),
            "batches_per_epoch": n_batches,
            "sampled_pair_instances_per_epoch": int(n_sampled),
            "unique_row_pairs_visited_per_epoch": int(len(unique_pairs)),
            "groups_receiving_an_update_per_epoch": int(len(groups_updated)),
            "share_of_groups_updated": float(len(groups_updated) / len(np.unique(gk))),
            "pairs_spanning_two_alleles": int(cross_allele),
            "target_gap": {"mean": float(g.mean()), "median": float(np.median(g)),
                           "p90": float(np.quantile(g, 0.9))} if g.size else {},
            "abs_rtt_length_difference_within_sampled_pairs":
                {str(k): int(v) for k, v in sorted(d_rtt.items())[:8]},
            "share_of_sampled_pairs_with_equal_rtt_length":
                float(d_rtt.get(0, 0) / max(n_sampled, 1))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260812)   # the training config's seed
    args = ap.parse_args()

    man = pd.read_parquet(C.require_manifest())
    corp = pd.read_parquet(C.CORPUS, columns=["record_id", "rtt"])
    man = man.merge(corp, on="record_id", validate="1:1")
    vocab = ContextVocab.load(str(ROOT / "data/processed/context_vocab_official.json"))
    base = load_featurized(str(ROOT / KEYS["current ranking key"]), vocab)
    order = pd.Series(np.arange(len(base.record_id)),
                      index=np.asarray(base.record_id))
    man = man.assign(row=order.reindex(man.record_id).to_numpy())
    rtt_len = np.zeros(len(base.record_id), dtype=np.int32)
    rtt_len[man.row.to_numpy()] = man.rtt.str.len().to_numpy()
    allele = np.empty(len(base.record_id), dtype=object)
    allele[man.row.to_numpy()] = man.edit_key.to_numpy()

    res = {"provenance": C.provenance([C.CORPUS], args.seed),
           "canon_version": CANON_VERSION,
           "settings": {"min_target_gap": MIN_DIFF, "max_pairs_per_group_per_batch": MAX_PAIRS,
                        "batch_size": BATCH, "max_group_take": 8, "epochs_replayed": 1},
           "confound": ("corpus.group_key is passed to GroupedBatchSampler "
                        "(train_pilot.py:668) and to sample_ranking_pairs "
                        "(train_pilot.py:704). E13 changed both channels at once, so "
                        "E14/E15 measure the combined effect, not the ranking channel alone."),
           "replays": {}}
    for label, npz in KEYS.items():
        res["replays"][label] = replay(npz, vocab, args.seed, rtt_len, allele)
        print(label, {k: v for k, v in res["replays"][label].items()
                      if isinstance(v, (int, float))}, flush=True)
    a, b = res["replays"]["current ranking key"], res["replays"]["canonical decision group"]
    res["ratios"] = {
        "sampled_pair_instances": round(b["sampled_pair_instances_per_epoch"]
                                        / max(a["sampled_pair_instances_per_epoch"], 1), 2),
        "unique_row_pairs": round(b["unique_row_pairs_visited_per_epoch"]
                                  / max(a["unique_row_pairs_visited_per_epoch"], 1), 2),
        "groups_updated": round(b["groups_receiving_an_update_per_epoch"]
                                / max(a["groups_receiving_an_update_per_epoch"], 1), 2)}
    C.write_outputs("e17_exposure_replay", res, render(res))


def render(r: dict) -> str:
    a = r["replays"]["current ranking key"]
    b = r["replays"]["canonical decision group"]
    L = ["# E17a - the comparisons the ranking loss actually sees\n",
         "One epoch of the real `GroupedBatchSampler` and `sample_ranking_pairs` replayed "
         f"under each key, at the training config's own settings ({r['settings']}). Nothing "
         "trained.\n",
         "> **Confound, adopted from the review.** " + r["confound"] + "\n",
         "| measured per epoch | current ranking key | canonical decision group | ratio |",
         "|---|---:|---:|---:|"]
    for key, lab in (("distinct_group_keys", "distinct group keys over training rows"),
                     ("sampled_pair_instances_per_epoch", "sampled pair instances"),
                     ("unique_row_pairs_visited_per_epoch", "unique row pairs visited"),
                     ("groups_receiving_an_update_per_epoch", "groups receiving an update")):
        ratio = ""
        for rk, rv in r["ratios"].items():
            if rk in key:
                ratio = f"{rv}x"
        L.append(f"| {lab} | {a[key]:,} | {b[key]:,} | {ratio} |")
    L.append(f"| share of groups updated | {a['share_of_groups_updated']:.1%} | "
             f"{b['share_of_groups_updated']:.1%} | |")
    L.append(f"| pairs spanning two alleles | {a['pairs_spanning_two_alleles']:,} | "
             f"{b['pairs_spanning_two_alleles']:,} | |")
    L.append(f"\n**The sharper statement of the defect.** Under the current key, "
             f"{a['share_of_sampled_pairs_with_equal_rtt_length']:.1%} of sampled pairs "
             "compare two designs of the **same RTT length** — because for one allele the "
             "stored window's extent is determined by the RTT, so an identical window pair "
             "implies an identical RTT length. Under the canonical key that share is "
             f"{b['share_of_sampled_pairs_with_equal_rtt_length']:.1%}. The ranking term was "
             "therefore almost never asked to compare the geometry variation that "
             "distinguishes alternative pegRNAs.\n")
    L.append("Absolute RTT-length difference within sampled pairs — current key: "
             f"{a['abs_rtt_length_difference_within_sampled_pairs']}; canonical: "
             f"{b['abs_rtt_length_difference_within_sampled_pairs']}\n")
    L.append(f"Target-gap distribution of sampled pairs — current: {a['target_gap']}; "
             f"canonical: {b['target_gap']}\n")
    L.append("Two wording corrections follow, and are applied in `e13_regroup_training.md`: "
             "14,277 / 368,307 counts *possible* within-group design pairs, not measured "
             "exposure; and `lambda_rank = 0.25` is a coefficient, not a statement that "
             "ranking contributes a quarter of the loss or the gradient.\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
