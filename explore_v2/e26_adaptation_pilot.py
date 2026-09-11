"""E26 - the P/S/M/Z adaptation pilot (ADAPTATION_SELECTION_PLAN.md §8).

Four arms differing only in what is updated and what is optimised, from one predetermined
starting checkpoint, on the E25 training components, selected on the E25 validation
components, with the test components untouched:

| arm | updated                                   | objective                        |
|-----|-------------------------------------------|----------------------------------|
| P   | head + pooling/cross-attention + last block | the model's native ordinal loss |
| S   | same, plus a new selection head           | smooth expected-regret           |
| M   | same, both heads                          | ordinal + selection              |
| Z   | selection head only, encoder frozen       | smooth expected-regret           |

Two internal controls: `G` optimises the same selection loss on candidate geometry alone
(if it matches an adapted encoder, the encoder adds nothing a few scalars do not), and
`shared` gives M's extra capacity to a single score used for both losses (so an M-over-shared
gain is attributable to separating the outputs, not to the parameters).

The starting checkpoint is `r4p2_ordSSM_cv1`, chosen by fold index. E24 showed arm A does not
reproduce the published model, so initialising from the published family matters; choosing by
index rather than by any panel metric keeps the choice independent of the evaluation surface.

OptiPrime appears only as a frozen evaluation reference. Its scores enter no input, loss,
teacher or ensemble.

Usage: PYTHONPATH=src CUDA_VISIBLE_DEVICES=n .venv/bin/python explore_v2/e26_adaptation_pilot.py --arm P
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2common as C  # noqa: E402
import adapt  # noqa: E402
import adapt_data as AD  # noqa: E402
from canon import CANON_VERSION  # noqa: E402
from pe_rankformer.models.pe_rankformer import PERankFormer, PERankFormerConfig  # noqa: E402
from pe_rankformer.training.losses import ordinal_loss  # noqa: E402

START_CKPT = "r4p2_ordSSM_cv1"
ARMS = {
    "P": dict(mode="P", stage="final_block", w_pred=1.0, w_sel=0.0, geom=False),
    "S": dict(mode="S", stage="final_block", w_pred=0.0, w_sel=1.0, geom=False),
    "M": dict(mode="M", stage="final_block", w_pred=1.0, w_sel=1.0, geom=False),
    "Z": dict(mode="Z", stage=None, w_pred=0.0, w_sel=1.0, geom=False),
    "shared": dict(mode="shared", stage="final_block", w_pred=1.0, w_sel=1.0, geom=False),
    "G": dict(mode="G", stage=None, w_pred=0.0, w_sel=1.0, geom=True),
    # Plan §7A: the same selector with candidate geometry made explicit alongside the
    # pooled representation. Its control is arm S, which is identical but pooled-only.
    "Sgeom": dict(mode="S", stage="final_block", w_pred=0.0, w_sel=1.0, geom=True),
    "Mgeom": dict(mode="M", stage="final_block", w_pred=1.0, w_sel=1.0, geom=True),
}
OUTDIR = C.CACHE / "adapt_runs"


def achieved_at_1(score: np.ndarray, y: np.ndarray, gid: np.ndarray,
                  informative_only: bool = False) -> float:
    """Mean measured efficiency of each group's top-scored candidate."""
    o = np.lexsort((-score, gid))
    g = gid[o]
    first = o[np.flatnonzero(np.r_[True, g[1:] != g[:-1]])]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    ymax = np.maximum.reduceat(y[o], starts)
    ymin = np.minimum.reduceat(y[o], starts)
    keep = ymax > ymin if informative_only else np.ones(len(starts), bool)
    return float(y[first][keep].mean())


def load_base(device: str) -> PERankFormer:
    d = sorted((ROOT / "checkpoints").glob(f"{START_CKPT}_*"))[-1]
    ck = torch.load(d / "best.pt", map_location=device, weights_only=False)
    m = PERankFormer(PERankFormerConfig(**ck["model_config"])).to(device)
    m.load_state_dict(ck["model_state_dict"])
    return m


@torch.no_grad()
def evaluate(model, batcher, idx, batches, device, geom_only=False):
    model.eval()
    S, P, Y, G = [], [], [], []
    for b in batches:
        batch, geom, y, gid = batcher.make(b, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if geom_only:
                s = model(geom)
                pred = s
            else:
                out, s = model(batch, geom)
                pred = model.base.efficiency_from_output(out)
                s = model.selection_score(out, s)
        S.append(s.float().cpu().numpy())
        P.append(pred.float().cpu().numpy())
        Y.append(y.cpu().numpy())
        G.append(gid)
    S, P, Y, G = (np.concatenate(x) for x in (S, P, Y, G))
    return {"achieved_at_1": achieved_at_1(S, Y, G),
            "achieved_at_1_informative": achieved_at_1(S, Y, G, True),
            "spearman": C.spearman(P, Y), "n": int(len(Y))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--max-rows", type=int, default=512)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-base", type=float, default=3e-4)
    ap.add_argument("--sel-loss", default="utility", choices=list(adapt.SELECTION_LOSSES))
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--train-groups", type=int, default=0,
                    help="learning-curve budget: cap on training decision groups (0 = all)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = ARMS[args.arm]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    c = AD.load_candidates()
    corpus = AD.featurize_candidates(c)
    train_mask = (c.split == "train").to_numpy()
    geom = AD.geometry_matrix(c, train_mask)
    gid_codes = pd.factorize(c.edit_key, sort=False)[0]
    batcher = AD.Batcher(corpus, geom, c.y.to_numpy(), gid_codes)

    idx = {s: np.flatnonzero((c.split == s).to_numpy()) for s in ("train", "val", "test")}
    if args.train_groups:
        # whole groups only, so a budget never supplies a partial candidate list
        gs = pd.unique(gid_codes[idx["train"]])
        keep = set(rng.choice(gs, size=min(args.train_groups, len(gs)), replace=False))
        idx["train"] = idx["train"][np.isin(gid_codes[idx["train"]], list(keep))]
    val_batches = AD.group_batches(gid_codes[idx["val"]], 1024, rng, shuffle=False)
    val_batches = [idx["val"][b] for b in val_batches]

    # ---- model -----------------------------------------------------------------------
    geom_only = cfg["mode"] == "G"
    if geom_only:
        model = adapt.GeometryOnly(geom.shape[1]).to(device)
        base = None
        params = [{"params": model.parameters(), "lr": args.lr_head}]
    else:
        base = load_base(device)
        model = adapt.Adapted(base, cfg["mode"],
                              geom_dim=geom.shape[1] if cfg["geom"] else 0).to(device)
        if cfg["stage"] is None:
            for p in base.parameters():
                p.requires_grad_(False)
            opened = []
        else:
            opened = adapt.unfreeze(base, cfg["stage"])
        lr_scale = (next(s.lr_scale for s in adapt.STAGES if s.name == cfg["stage"])
                    if cfg["stage"] else 0.0)
        params = []
        bp = [p for p in base.parameters() if p.requires_grad]
        if bp:
            params.append({"params": bp, "lr": args.lr_base * lr_scale})
        for extra in (model.sel, model.adapter):
            if extra is not None:
                params.append({"params": extra.parameters(), "lr": args.lr_head})
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    thresholds = (torch.tensor(base.config.ordinal_thresholds, device=device)
                  if base is not None else None)
    k_ord = len(base.config.ordinal_thresholds) if base is not None else 0
    sel_fn = adapt.SELECTION_LOSSES[args.sel_loss]

    # ---- fixed loss scales, measured once on training batches before any update -------
    scales = {"pred": 1.0, "sel": 1.0}
    if not geom_only:
        model.eval()
        probe = AD.group_batches(gid_codes[idx["train"]], args.max_rows, rng)[:12]
        pv, sv = [], []
        with torch.no_grad():
            for b in probe:
                batch, g, y, gg = batcher.make(idx["train"][b], device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out, s = model(batch, g)
                if cfg["w_pred"]:
                    pv.append(float(ordinal_loss(out[:, :k_ord].float(), y, thresholds)))
                if cfg["w_sel"]:
                    sc = s if s is not None else model.base.efficiency_from_output(out)
                    sv.append(float(sel_fn(sc.float(), y, adapt._group_slices(gg),
                                           T=args.temp)))
        if pv:
            scales["pred"] = max(float(np.mean(pv)), 1e-8)
        if sv:
            scales["sel"] = max(float(np.mean(sv)), 1e-8)

    # ---- train -------------------------------------------------------------------------
    name = f"{args.arm}{('_' + args.tag) if args.tag else ''}_s{args.seed}"
    hist, best, bad, best_state = [], None, 0, None
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        if base is not None and cfg["stage"] is None:
            base.eval()  # a frozen feature extractor should not be dropping units
        tb = AD.group_batches(gid_codes[idx["train"]], args.max_rows, rng)
        tot = 0.0
        for b in tb:
            batch, g, y, gg = batcher.make(idx["train"][b], device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if geom_only:
                    s, out = model(g), None
                else:
                    out, s = model(batch, g)
            loss = torch.zeros((), device=device)
            if cfg["w_pred"] and out is not None:
                loss = loss + cfg["w_pred"] * ordinal_loss(
                    out[:, :k_ord].float(), y, thresholds) / scales["pred"]
            if cfg["w_sel"]:
                # `shared` has no second output: both objectives act on the one score,
                # which is the whole point of the control.
                sc = s if s is not None else model.base.efficiency_from_output(out)
                loss = loss + cfg["w_sel"] * sel_fn(
                    sc.float(), y, adapt._group_slices(gg), T=args.temp) / scales["sel"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for grp in params for p in grp["params"]], 1.0)
            opt.step()
            tot += float(loss.detach())
        v = evaluate(model, batcher, idx["val"], val_batches, device, geom_only)
        v["epoch"] = ep
        v["train_loss"] = tot / max(len(tb), 1)
        hist.append(v)
        print(f"{name} ep{ep:02d} loss={v['train_loss']:.4f} "
              f"val@1={v['achieved_at_1']:.5f} val_rho={v['spearman']:.4f}", flush=True)
        # selection on validation achieved @1, as the plan specifies
        if best is None or v["achieved_at_1"] > best["achieved_at_1"] + 1e-9:
            best, bad = v, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"{name} early stop at epoch {ep}", flush=True)
                break

    OUTDIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "arm": args.arm, "cfg": cfg, "args": vars(args)},
               OUTDIR / f"{name}.pt")
    (OUTDIR / f"{name}.json").write_text(json.dumps(
        {"name": name, "arm": args.arm, "cfg": cfg, "args": vars(args),
         "start_checkpoint": START_CKPT, "loss_scales": scales,
         "unfrozen": opened if not geom_only else ["geometry-only"],
         "trainable_params": int(adapt.trainable(model)),
         "train_groups": int(len(pd.unique(gid_codes[idx["train"]]))),
         "train_candidates": int(len(idx["train"])),
         "best_val": best, "history": hist,
         "wall_seconds": round(time.time() - t0, 1)}, indent=1))
    print(f"{name} DONE best val@1={best['achieved_at_1']:.5f} "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
