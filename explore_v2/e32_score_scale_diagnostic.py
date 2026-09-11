"""Label-free diagnostic of the selection-loss score-scale control.

This does not retrain a model or claim an improvement. Temperature changes alone
cannot change deterministic argmax choices; they change the training surrogate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import softmax

import _v2common as C
from e29_head_audit import CACHE


def summarize_scores(frame, temperature):
    ranges, maxima, entropy = [], [], []
    for _, group in frame.groupby("edit_key", sort=False):
        s = group.selection.to_numpy(dtype=float)
        if len(s) < 2:
            continue
        p = softmax(s / temperature)
        ranges.append(float(np.ptp(s)))
        maxima.append(float(p.max()))
        entropy.append(float(-(p * np.log(np.maximum(p, 1e-300))).sum() / np.log(len(s))))
    return {"groups": len(ranges), "median_score_range": float(np.median(ranges)),
            "p90_score_range": float(np.quantile(ranges, .9)),
            "mean_max_probability": float(np.mean(maxima)),
            "mean_normalized_entropy": float(np.mean(entropy))}


def main():
    results = {}
    for name in ("M_s20260910", "shared_s20260910", "S_s20260910", "S_pairwise_s20260910"):
        path = CACHE / f"{name}__target_val.parquet"
        if not path.exists() or not path.with_suffix(".json").exists():
            continue
        # Outcome labels are intentionally not loaded for this diagnostic.
        frame = pd.read_parquet(path, columns=["edit_key", "selection"])
        results[name] = {str(t): summarize_scores(frame, t) for t in (1.0, .1, .01)}
    result = {"note": "Label-free score-scale diagnostic, not a temperature-tuned performance result. "
                      "Post-hoc positive temperature scaling cannot change argmax selection.",
              "bounded_score_eight_candidate_max_probability_at_T1": float(np.e / (np.e + 7)),
              "models": results}
    lines = ["# E32 preliminary diagnostic: selection-score scale", "", result["note"], "",
             "For eight candidates with scores in [0,1], T=1 bounds the largest softmax",
             "probability by e/(e+7) = 0.2797. Unbounded heads do not share this constraint.", "",
             "| Model | Temperature | Median within-group range | Mean max probability | Normalized entropy |",
             "|---|---:|---:|---:|---:|"]
    for name, temperatures in results.items():
        for temperature, v in temperatures.items():
            lines.append(f"| {name} | {temperature} | {v['median_score_range']:.4f} | "
                         f"{v['mean_max_probability']:.4f} | {v['mean_normalized_entropy']:.4f} |")
    lines += ["", "A clean training control needs comparable logit parameterization or a declared",
              "validation-selected temperature protocol. These measurements do not establish",
              "which architecture would win under that control. No E32 training was run."]
    C.write_outputs("e32_score_scale_diagnostic", result, "\n".join(lines))


if __name__ == "__main__":
    main()
