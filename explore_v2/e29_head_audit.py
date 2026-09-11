"""Matched-start and both-head audit; preserves E25–E28 artifacts.

Historical test predictions are retrospective audits, never a selection surface.
New follow-up models are evaluated on validation/source only by default.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import _v2common as C
import adapt
import adapt_data as AD
from e26_adaptation_pilot import ARMS, START_CKPT, load_base
from e30_replicate_pairwise import sha256
from pe_rankformer.data.context import ContextVocab
from pe_rankformer.data.dataset import PEDataset, collate, load_featurized

CACHE = C.CACHE / "adapt_followup_predictions"
DEFAULT_MODELS = ["start_checkpoint", "P_s20260910", "P_s20260911", "P_s20260912",
                  "S_s20260910", "S_s20260911", "S_s20260912", "M_s20260910",
                  "shared_s20260910", "Z_s20260910", "P_n200_s20260910",
                  "S_pairwise_s20260910"]


@torch.no_grad()
def score_both(model, dataset, rows, device, batch_size):
    model.eval()
    pred, sel = [], []
    for offset in range(0, len(rows), batch_size):
        ix = rows[offset:offset + batch_size]
        batch = {k: v.to(device) for k, v in collate([dataset[int(i)] for i in ix]).items()}
        amp = (torch.autocast("cuda", dtype=torch.bfloat16)
               if device == "cuda" and torch.cuda.is_bf16_supported() else nullcontext())
        with amp:
            out, s = model(batch)
            p = model.base.efficiency_from_output(out)
            selected = model.selection_score(out, s)
        pred.append(p.float().cpu().numpy())
        sel.append(selected.float().cpu().numpy())
    return np.concatenate(pred), np.concatenate(sel)


def metadata(name):
    if name == "start_checkpoint":
        path = sorted((C.ROOT / "checkpoints").glob(f"{START_CKPT}_*"))[-1] / "best.pt"
        return {"arm": "P", "cfg": ARMS["P"], "name": name}, path
    paths = [C.CACHE / folder / f"{name}.json"
             for folder in ("adapt_runs", "adapt_followup_runs")]
    found = [p for p in paths if p.exists()]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one checkpoint for {name}: {found}")
    return json.loads(found[0].read_text()), found[0].with_suffix(".pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--target-splits", nargs="+", choices=["val", "test"], default=["val"])
    parser.add_argument("--skip-source", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CACHE.mkdir(parents=True, exist_ok=True)
    c = AD.load_candidates()
    target_ds = PEDataset(AD.featurize_candidates(c))
    surfaces = {}
    for split in args.target_splits:
        ix = np.flatnonzero(c.split.to_numpy() == split)
        frame = c.iloc[ix][["edit_key", "design_key", "component", "y"]].reset_index(drop=True)
        surfaces[f"target_{split}"] = (target_ds, ix, frame)
    input_paths = [C.CACHE / "adaptation_partition_v2.parquet", C.OUT / "reserved_panel_v2.parquet"]
    if not args.skip_source:
        vocab = ContextVocab.load(str(C.ROOT / "data/processed/context_vocab_official.json"))
        source_path = C.ROOT / "data/processed/featurized_official.npz"
        corpus = load_featurized(str(source_path), vocab)
        ix = np.flatnonzero(np.asarray(corpus.fold) == 0)
        manifest = pd.read_parquet(C.require_manifest(),
                                  columns=["record_id", "decision_group", "design_key", "spacer", "edited"])
        frame = pd.DataFrame({"record_id": np.asarray(corpus.record_id)[ix]}).merge(
            manifest, on="record_id", validate="1:1", how="left")
        if frame.isna().any().any():
            raise ValueError("Incomplete source manifest join")
        frame = frame.rename(columns={"decision_group": "edit_key", "edited": "y", "spacer": "component"})
        surfaces["source_fold0"] = (PEDataset(corpus), ix, frame)
        input_paths.extend([source_path, C.require_manifest()])
    input_paths.extend([Path(__file__), C.ROOT / "explore_v2/adapt.py",
                        C.ROOT / "explore_v2/adapt_data.py"])
    inputs = {str(p.relative_to(C.ROOT)): sha256(p) for p in input_paths}
    for name in args.models:
        meta, path = metadata(name)
        cfg = meta["cfg"]
        if cfg["geom"] or cfg["mode"] == "G":
            raise ValueError("Geometry needs a separately validated source feature pipeline")
        provenance = {"model": name, "checkpoint": str(path.relative_to(C.ROOT)),
                      "checkpoint_sha256": sha256(path), "inputs_sha256": inputs,
                      "torch": torch.__version__, "batch_size": args.batch_size,
                      "precision": "bf16" if device == "cuda" and torch.cuda.is_bf16_supported() else "fp32"}
        pending = []
        for surface in surfaces:
            output = CACHE / f"{name}__{surface}.parquet"
            sidecar = output.with_suffix(".json")
            if output.exists() or sidecar.exists():
                if not (output.exists() and sidecar.exists()):
                    raise RuntimeError(f"Incomplete cache: {output}")
                old = json.loads(sidecar.read_text())
                if old != dict(provenance, surface=surface):
                    raise RuntimeError(f"Provenance mismatch: {output}; preserve and investigate")
                print(f"cached {name} {surface}", flush=True)
            else:
                pending.append((surface, output, sidecar))
        if not pending:
            continue
        base = load_base(device)
        model = adapt.Adapted(base, cfg["mode"], geom_dim=0).to(device)
        if name != "start_checkpoint":
            model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["state_dict"])
        for surface, output, sidecar in pending:
            ds, ix, frame = surfaces[surface]
            p, s = score_both(model, ds, ix, device, args.batch_size)
            if not (np.isfinite(p).all() and np.isfinite(s).all()):
                raise ValueError("Non-finite predictions")
            frame.assign(prediction=p, selection=s).to_parquet(output, index=False)
            sidecar.write_text(json.dumps(dict(provenance, surface=surface), indent=2))
            print(f"scored {name} {surface}: {len(frame)} rows", flush=True)
        del model, base
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
