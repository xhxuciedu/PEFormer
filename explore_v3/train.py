"""Bounded total-label v3 screen; inner validation alone selects checkpoints.

Run one arm/seed/budget per invocation. Output paths are immutable.
"""
from __future__ import annotations
import argparse
import copy
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from common import ROOT, OUT, CACHE, provenance, write_json, sha256
from model import Selector, slices, pairwise, utility, preserve
from prepare import BUDGETS, START
import adapt
import adapt_data as AD
from e26_adaptation_pilot import load_base
from e29_summarize_audit import metrics
from pe_rankformer.training.losses import ordinal_loss

def budget_indices(frame, manifest, budget, seed):
    sub = manifest.loc[(manifest.nominal_groups == budget) & (manifest.seed == seed)]
    if sub.empty or sub.component.duplicated().any():
        raise ValueError("Invalid budget manifest")
    result = {}
    for role in ("train", "inner_val"):
        comps = sub.loc[sub.total_budget_role == role, "component"]
        result[role] = np.flatnonzero(((frame.surface == "target_budget_pool") &
                                      frame.component_numeric.isin(comps)).to_numpy())
        if not len(result[role]):
            raise ValueError("Empty budget role")
    assert not set(frame.iloc[result["train"]].component) & set(frame.iloc[result["inner_val"]].component)
    return result

def endpoint_frame(frame, ix, score, prediction):
    f = frame.iloc[ix][["group_id", "design_key", "component", "y"]].rename(columns={"group_id": "edit_key"}).copy()
    f["selection"], f["prediction"] = score, prediction
    return f

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list("ABCDEFG")+["C_frozen", "F_replay"], required=True)
    ap.add_argument("--budget", type=int, choices=[200,1000], required=True)
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--subset-seed", type=int, default=20260910)
    ap.add_argument("--steps", type=int, default=100)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for declared screen")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    srng = np.random.default_rng(args.seed + 100)
    tag = f"{args.arm}_b{args.budget}_sub{args.subset_seed}_s{args.seed}"
    run = OUT / "runs" / tag
    run.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    frame = pd.read_parquet(CACHE / "index.parquet")
    prepared_meta = json.loads((CACHE / "prepared.json").read_text())
    for p, digest in prepared_meta["outputs_sha256"].items():
        if sha256(ROOT / p) != digest:
            raise ValueError(f"Preparation fingerprint changed: {p}")
    data = torch.load(CACHE / "prepared.pt", map_location="cpu", weights_only=False)
    roles = budget_indices(frame, pd.read_parquet(BUDGETS), args.budget, args.subset_seed)
    ix_train, ix_val = roles["train"], roles["inner_val"]
    gid = frame.gid.to_numpy()
    y, q0, h = data["y"].cuda(), data["q0"].cuda(), data["h"].cuda()
    qmu, qsd = q0[ix_train].mean().detach(), q0[ix_train].std().clamp_min(1e-6).detach()
    ymu, ysd = y[ix_train].mean().detach(), y[ix_train].std().clamp_min(1e-6).detach()
    q = (q0-qmu)/qsd
    frozen = args.arm in ("E", "F", "G", "C_frozen", "F_replay")
    residual = args.arm in ("E", "F", "G", "F_replay")
    base = None if frozen else load_base("cuda")
    holder = {}
    opened = []
    if base is not None:
        opened = adapt.unfreeze(base, "final_block")
        base.head.register_forward_pre_hook(lambda module, x: holder.update(h=x[0]))
    selector = None if args.arm == "A" else Selector(h.shape[1], residual=residual).cuda()
    params = []
    if base is not None:
        params.append({"params": [p for p in base.parameters() if p.requires_grad], "lr": 3e-5})
    if selector is not None:
        params.append({"params": selector.parameters(), "lr": .001})
    optimizer = torch.optim.AdamW(params, weight_decay=.01)
    trainable = sum(p.numel() for group in params for p in group["params"])
    thresholds = None if base is None else torch.tensor(base.config.ordinal_thresholds, device="cuda")
    source_ix = np.flatnonzero((frame.surface == "source_replay").to_numpy())
    source_batches = AD.group_batches(gid[source_ix], 256, srng)
    source_batches = [source_ix[b] for b in source_batches]

    def forward(ix):
        if base is None:
            pred, features, out = q0[ix], h[ix], None
        else:
            batch = {k: v[ix].cuda() for k, v in data["inputs"].items()}
            out = base(batch)
            pred, features = base.efficiency_from_output(out), holder["h"]
        s = ((pred-qmu)/qsd if selector is None else selector(features, q[ix]))
        predicted = s*ysd+ymu if args.arm in ("B", "D") else pred
        return s, predicted, out

    def evaluate(ix, original=False, source=False, save=None):
        if base is not None:
            base.eval()
        if selector is not None:
            selector.eval()
        ss, pp = [], []
        with torch.no_grad():
            for a in range(0, len(ix), 512):
                b = ix[a:a+512]
                s, p = (q[b], q0[b]) if original else forward(b)[:2]
                ss.append(s.cpu().numpy())
                pp.append(p.cpu().numpy())
        f = endpoint_frame(frame, ix, np.concatenate(ss), np.concatenate(pp))
        m, _ = metrics(f, source=source)
        if save:
            f.to_parquet(save, index=False)
        return m

    # Utility's fixed scale depends on target training outcomes only.
    train_frame = frame.iloc[ix_train]
    spans = train_frame.groupby("gid").y.agg(lambda v: v.max()-v.min())
    utility_scale = max(float(spans[spans > 0].mean()), 1e-6)
    hist = [{"step": 0, "inner": evaluate(ix_val, original=True)}]
    best_value = hist[0]["inner"]["selection"]["achieved_at_1"]
    best_step, best_state = 0, None
    prov = {"args": vars(args), "inputs_sha256": provenance([START, BUDGETS, CACHE / "prepared.json",
            OUT / "train.py", OUT / "model.py", OUT / "EXECUTION_PROTOCOL.md"]),
            "precision": "FP32, TF32 disabled", "torch": torch.__version__,
            "device": torch.cuda.get_device_name(), "opened": opened,
            "trainable_parameters": trainable, "q_mean": float(qmu), "q_sd": float(qsd),
            "y_mean": float(ymu), "y_sd": float(ysd), "utility_scale": utility_scale,
            "label_budget": {role: {"groups": frame.iloc[ix].gid.nunique(),
                "candidates": len(ix), "measurements": int(frame.iloc[ix].n_meas.sum()),
                "components": frame.iloc[ix].component.nunique()} for role,ix in roles.items()},
            "checkpoint_selection": "budget inner validation only, epoch-zero original deployment eligible"}
    write_json(run / "provenance.json", prov)
    print(tag, "starting", prov["label_budget"], "parameters", trainable, flush=True)
    batches, cursor, source_cursor = [], 0, 0
    total_source_rows = 0
    target_rows = 0
    for step in range(1, args.steps+1):
        if cursor == len(batches):
            batches = [ix_train[b] for b in AD.group_batches(gid[ix_train], 256, rng)]
            cursor = 0
        ix = batches[cursor]
        cursor += 1
        if base is not None:
            base.train()
        if selector is not None:
            selector.train()
        optimizer.zero_grad(set_to_none=True)
        s, pred, out = forward(ix)
        groups = slices(gid[ix])
        if args.arm == "A":
            loss = ordinal_loss(out[:, :len(thresholds)], y[ix], thresholds)
        elif args.arm == "B":
            loss = F.huber_loss(s, (y[ix]-ymu)/ysd, delta=1.)
        elif args.arm == "G":
            loss = utility(s, y[ix], groups, utility_scale)
        else:
            loss = pairwise(s, y[ix], groups)
            if args.arm == "D":
                loss = loss + F.huber_loss(s, (y[ix]-ymu)/ysd, delta=1.)
        target_loss = float(loss.detach())
        retention_loss = 0.
        if args.arm in ("F", "G", "F_replay"):
            if source_cursor == len(source_batches):
                srng.shuffle(source_batches)
                source_cursor = 0
            sx = source_batches[source_cursor]
            source_cursor += 1
            ss, _, _ = forward(sx)
            ret = (pairwise(ss, y[sx], slices(gid[sx])) if args.arm == "F_replay"
                   else preserve(ss, q[sx], slices(gid[sx])))
            loss = loss + ret
            retention_loss = float(ret.detach())
            total_source_rows += len(sx)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_([p for group in params for p in group["params"]], 1.)
        optimizer.step()
        target_rows += len(ix)
        if step % 10 == 0 or step == args.steps:
            inner = evaluate(ix_val)
            value = inner["selection"]["achieved_at_1"]
            hist.append({"step": step, "inner": inner, "target_loss": target_loss,
                         "preservation_loss": retention_loss, "gradient_norm": float(grad)})
            if value > best_value + 1e-12:
                best_value, best_step = value, step
                best_state = {"base": None if base is None else {k:v.cpu().clone() for k,v in base.state_dict().items()},
                              "selector": None if selector is None else {k:v.cpu().clone() for k,v in selector.state_dict().items()}}
            print(tag, "step", step, "inner@1", round(value, 6), "best", best_step, flush=True)
    training_seconds = time.time()-t0
    if best_state is not None:
        if base is not None:
            base.load_state_dict(best_state["base"])
        if selector is not None:
            selector.load_state_dict(best_state["selector"])
    torch.save({"state": best_state, "best_step": best_step, "provenance": prov}, run / "best.pt")
    results = {}
    for surface in ("target_outer_val", "source_val", "source_audit"):
        ix = np.flatnonzero((frame.surface == surface).to_numpy())
        results[surface] = evaluate(ix, original=best_step == 0, source=surface.startswith("source"),
                                    save=run / f"{surface}.parquet")
        results[surface+"_start"] = evaluate(ix, original=True, source=surface.startswith("source"))
        print(tag, surface, results[surface]["selection"], flush=True)
    write_json(run / "results.json", {"arm": args.arm, "budget": args.budget, "seed": args.seed,
        "subset_seed": args.subset_seed, "best_step": best_step, "history": hist, "surfaces": results,
        "target_updates": args.steps, "target_rows_processed": target_rows,
        "replay_rows_processed": total_source_rows, "training_seconds": training_seconds,
        "total_seconds": time.time()-t0, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "outputs_sha256": provenance([run / "best.pt"] + list(run.glob("*.parquet")))})

if __name__ == "__main__":
    main()
