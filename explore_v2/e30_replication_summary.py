"""Validation-only replication summary. Never reads target-test results."""
from __future__ import annotations

import json

import numpy as np

import _v2common as C


def main():
    models = {}
    pending = []
    for seed in (20260910, 20260911, 20260912):
        for arm in ("P", "S", "shared", "S_pairwise"):
            name = f"{arm}_s{seed}"
            paths = [C.CACHE / directory / f"{name}.json"
                     for directory in ("adapt_runs", "adapt_followup_runs")]
            found = [p for p in paths if p.exists()]
            if not found:
                pending.append(name)
                continue
            if len(found) != 1:
                raise ValueError(f"Ambiguous model: {name}")
            meta = json.loads(found[0].read_text())
            if not found[0].with_suffix(".pt").exists():
                raise ValueError(f"Metadata without checkpoint: {name}")
            models[name] = {"arm": arm, "seed": seed,
                            "validation_achieved_at_1": meta["best_val"]["achieved_at_1"],
                            "best_epoch": meta["best_val"]["epoch"],
                            "epochs_run": len(meta["history"]), "wall_seconds": meta["wall_seconds"],
                            "train_groups": meta["train_groups"], "args": meta["args"]}
    contrasts = {}
    for reference in ("P", "S", "shared"):
        differences, seeds = [], []
        for seed in (20260910, 20260911, 20260912):
            a, b = f"S_pairwise_s{seed}", f"{reference}_s{seed}"
            if a in models and b in models:
                differences.append(models[a]["validation_achieved_at_1"] -
                                   models[b]["validation_achieved_at_1"])
                seeds.append(seed)
        contrasts[f"pairwise_minus_{reference}"] = {
            "seeds": seeds, "per_seed": differences,
            "mean": float(np.mean(differences)) if differences else None,
            "three_seed_replication_complete": len(seeds) == 3,
            "all_positive": bool(differences and all(x > 0 for x in differences))}
    result = {"surface": "E25 validation; no test results read",
              "note": "Selected-epoch validation values are descriptive, not independent confirmation. "
                      "Three-seed averages are not ensembles.",
              "pending": pending, "models": models, "contrasts": contrasts}
    lines = ["# E30: pairwise-loss replication", "", result["surface"], "", result["note"], "",
             "Pending checkpoints: " + (", ".join(pending) if pending else "none"), "",
             "| Model | Validation @1 | Best epoch (zero-indexed) | Epochs run | Training seconds |",
             "|---|---:|---:|---:|---:|"]
    for name, m in models.items():
        lines.append(f"| {name} | {m['validation_achieved_at_1']:.5f} | {m['best_epoch']} | "
                     f"{m['epochs_run']} | {m['wall_seconds']:.1f} |")
    lines += ["", "| Contrast | Seeds | Per-seed differences | Mean difference |",
              "|---|---|---|---:|"]
    for name, v in contrasts.items():
        delta = ', '.join(f"{x:+.5f}" for x in v['per_seed'])
        mean = f"{v['mean']:+.5f}" if v['mean'] is not None else "pending"
        lines.append(f"| {name} | {v['seeds']} | {delta} | {mean} |")
    C.write_outputs("e30_pairwise_replication", result, "\n".join(lines))


if __name__ == "__main__":
    main()
