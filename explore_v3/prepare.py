"""Prepare component-isolated source replay and budgeted target representations.

No E25 test is encoded or scored. Target outer validation is evaluation-only.
Source fold0 is kept as an audit, excluded component-wise from replay/validation.
"""
from __future__ import annotations
import argparse
import time
import numpy as np
import pandas as pd
import torch
from common import ROOT, OUT, CACHE, provenance, write_json
import adapt_data as AD
from e25_adaptation_partition import components
from e26_adaptation_pilot import load_base
from pe_rankformer.data.dataset import PEDataset, collate, featurize
from pe_rankformer.data.context import ContextVocab

PART = ROOT / "explore_v2/cache/adaptation_partition_v2.parquet"
BUDGETS = ROOT / "explore_v2/cache/adapt_followup_budget_components.parquet"
SOURCE = ROOT / "explore_v2/cache/corpus_canonical_v2.parquet"
START = ROOT / "checkpoints/r4p2_ordSSM_cv1_1787193463/best.pt"

def source_roles(source, target):
    source = source.copy()
    source["protospacer"] = source.spacer
    source["component"] = components(source)
    blocked = set(source.loc[source.fold == 0, "component"])
    overlap = (source.edit_key.isin(target.edit_key) |
               source.spacer.isin(target.protospacer))
    blocked.update(source.loc[overlap, "component"])
    available = np.sort(source.loc[~source.component.isin(blocked), "component"].unique())
    order = np.random.default_rng(20260910).permutation(available)
    if len(order) < 600:
        raise ValueError(f"Only {len(order)} eligible source components")
    validation = set(order[:500])
    replay = set(order[500:2500])
    source["surface"] = "unused"
    source.loc[source.component.isin(replay), "surface"] = "source_replay"
    source.loc[source.component.isin(validation), "surface"] = "source_val"
    source.loc[source.fold == 0, "surface"] = "source_audit"
    chosen = source.loc[source.surface != "unused"].copy()
    # Identical molecules/context are one candidate; mean observed outcome.
    chosen["y"] = chosen.groupby(["surface", "decision_group", "design_key"]).edited.transform("mean")
    chosen["n_meas"] = chosen.groupby(["surface", "decision_group", "design_key"]).edited.transform("size")
    chosen = chosen.sort_values("record_id").drop_duplicates(["surface", "decision_group", "design_key"])
    chosen["group_id"] = chosen.decision_group
    chosen["component"] = "source_" + chosen.component.astype(str)
    return chosen, {"available_components": len(available), "blocked_components": len(blocked),
                    "source_validation_seen_in_pretraining": True,
                    "source_fold0_used_for_replay_or_selection": False}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; run with GPU access, not silent CPU fallback")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / "prepared.pt"
    if dest.exists() or (CACHE / "prepared.json").exists():
        raise FileExistsError("Preserve existing preparation; inspect before rerunning")
    t0 = time.time()
    target_all = AD.load_candidates()
    manifest = pd.read_parquet(BUDGETS)
    chosen_components = manifest.loc[manifest.nominal_groups == 1000, "component"].unique()
    target = target_all.loc[(target_all.split == "val") |
                            ((target_all.split == "train") & target_all.component.isin(chosen_components))].copy()
    assert "test" not in set(target.split)
    target["surface"] = np.where(target.split == "val", "target_outer_val", "target_budget_pool")
    target["group_id"] = target.edit_key
    target["component_numeric"] = target.component
    target["component"] = "target_" + target.component.astype(str)
    source, source_meta = source_roles(pd.read_parquet(SOURCE), target_all)
    source["component_numeric"] = -1
    frame = pd.concat([target, source], ignore_index=True).sort_values(
        ["surface", "group_id", "design_key"]).reset_index(drop=True)
    frame["edited"] = frame.y
    frame["indel"] = 0.0
    frame["fold"] = frame.fold.fillna(-1).astype(int)
    frame["gid"] = pd.factorize(frame.surface + "|" + frame.group_id, sort=True)[0]
    frame["row"] = np.arange(len(frame))
    cols = ["row", "record_id", "surface", "group_id", "design_key", "edit_key", "component",
            "component_numeric", "gid", "y", "n_meas", "spacer", "source_study", "cell_type", "pe_type"]
    index = frame[cols].copy()
    for col in ("component", "group_id"):
        assert index.groupby(col).surface.nunique().max() == 1
    index.to_parquet(CACHE / "index.parquet", index=False)
    vocab = ContextVocab.load(str(ROOT / "data/processed/context_vocab_official.json"))
    corpus = featurize(frame, vocab)
    ds = PEDataset(corpus)
    # Save the actual tokenized inputs for matched final-block baselines.
    inputs = {"edit_ids": torch.from_numpy(corpus.edit_ids.astype(np.int64)),
              "peg_nuc_ids": torch.from_numpy(corpus.peg_nuc_ids.astype(np.int64)),
              "peg_seg_ids": torch.from_numpy(corpus.peg_seg_ids.astype(np.int64))}
    inputs.update({"ctx_" + k: torch.from_numpy(v.astype(np.int64)) for k, v in corpus.context_ids.items()})
    base = load_base("cuda").eval()
    holder = {}
    hook = base.head.register_forward_pre_hook(lambda module, x: holder.update(h=x[0]))
    hs, qs = [], []
    with torch.no_grad():
        for a in range(0, len(frame), args.batch_size):
            batch = {k: v[a:a+args.batch_size].cuda() for k, v in inputs.items()}
            out = base(batch)
            hs.append(holder["h"].cpu())
            qs.append(base.efficiency_from_output(out).cpu())
            if a % (args.batch_size * 20) == 0:
                print(f"FP32 encoding {a}/{len(frame)}", flush=True)
    hook.remove()
    h, q = torch.cat(hs), torch.cat(qs)
    with torch.no_grad():
        reproduced = base.efficiency_from_output(base.head(h[:1024].cuda())).cpu()
    error = float((reproduced - q[:1024]).abs().max())
    assert error < 1e-6, error
    assert torch.isfinite(h).all() and torch.isfinite(q).all()
    torch.save({"h": h, "q0": q, "inputs": inputs,
                "y": torch.tensor(index.y.to_numpy(), dtype=torch.float32)}, dest)
    summary = {"inputs_sha256": provenance([PART, BUDGETS, SOURCE, START, __file__,
                 ROOT / "explore_v2/adapt_data.py", ROOT / "explore_v2/canon.py"]),
               "outputs_sha256": provenance([dest, CACHE / "index.parquet"]),
               "source": source_meta, "precision": "FP32; TF32 disabled",
               "head_reproduction_max_abs_error": error, "device": torch.cuda.get_device_name(),
               "seconds": time.time()-t0, "surfaces": {}}
    for name, g in index.groupby("surface"):
        depth = g.groupby("group_id").size()
        summary["surfaces"][name] = {"candidates": len(g), "groups": len(depth),
            "components": g.component.nunique(), "depth_ge2": int((depth >= 2).sum()),
            "depth_ge5": int((depth >= 5).sum()), "measurements": int(g.n_meas.sum())}
    write_json(CACHE / "prepared.json", summary)
    print(summary, flush=True)

if __name__ == "__main__":
    main()
