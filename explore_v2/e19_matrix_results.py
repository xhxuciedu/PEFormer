"""E19 - the decision-aligned training matrix, and the checkpoint-selector comparison.

Reads every run in the matrix, reports the two selection criteria side by side from the
training histories, and scores each arm's chosen checkpoints on fold 0's canonical decision
groups. Arms, per `MANUSCRIPT_IMPROVEMENT_PLAN.md`:

  A  original batching, original ranking key      matched historical control (3 seeds)
  D  canonical batching, canonical ranking key     the full repair (3 seeds)
  B  canonical batching, ORIGINAL ranking key      is the gain batch composition?
  C  canonical both, 1 pair per group              allocation under a throttled budget
  E  canonical batching, ranking disabled          does ranking add anything at all?
  F  canonical both + feature branch               explicit design geometry
  G  canonical both, 16 pairs per group            more exposure

Two things are read off the same trajectories, with no extra compute: whether the repair
replicates across seeds, and whether selecting the checkpoint by mean achieved efficiency @1
over canonical validation decisions beats selecting it by pooled validation Spearman. Because
both selections come from one run, the comparison is not confounded by different stopping
times.

Evaluation surface for the final numbers is fold 0, which no arm trains on or stops on. It has
546 informative decision groups, so it resolves roughly +/-0.002 -- adequate for the
seed-replication question, not for the small component effects, which is why E18 used the
panel. Panel scoring of the winning arm is a separate, disclosed step.

Usage: PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 .venv/bin/python explore_v2/e19_matrix_results.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2common as C  # noqa: E402
import endpoints as E  # noqa: E402
from canon import CANON_VERSION  # noqa: E402
from pe_rankformer.data.context import ContextVocab  # noqa: E402
from pe_rankformer.data.dataset import PEDataset, collate, load_featurized  # noqa: E402
from pe_rankformer.data.family_c_features import attach_family_c_features  # noqa: E402
from pe_rankformer.models.pe_rankformer import PERankFormer, PERankFormerConfig  # noqa: E402

ARMS = {
    "m_A_s1": ("A", "original batching, original ranking key", 20260812),
    "m_A_s2": ("A", "original batching, original ranking key", 20260813),
    "m_A_s3": ("A", "original batching, original ranking key", 20260814),
    "m_D_s1": ("D", "canonical batching, canonical ranking key", 20260812),
    "m_D_s2": ("D", "canonical batching, canonical ranking key", 20260813),
    "m_D_s3": ("D", "canonical batching, canonical ranking key", 20260814),
    "m_B_batch": ("B", "canonical batching, original ranking key", 20260812),
    "m_C_p1": ("C", "canonical both, 1 pair per group", 20260812),
    "m_E_norank": ("E", "canonical batching, ranking disabled", 20260812),
    "m_F_feat": ("F", "canonical both, feature branch", 20260812),
    "m_G_p16": ("G", "canonical both, 16 pairs per group", 20260812),
    # follow-up wave: combine what the first matrix showed actually works
    "m_F_s2": ("F", "canonical both, feature branch", 20260813),
    "m_F_s3": ("F", "canonical both, feature branch", 20260814),
    "m_E_s2": ("E", "canonical batching, ranking disabled", 20260813),
    "m_E_s3": ("E", "canonical batching, ranking disabled", 20260814),
    "m_H_s1": ("H", "canonical batching, feature branch, ranking disabled", 20260812),
    "m_H_s2": ("H", "canonical batching, feature branch, ranking disabled", 20260813),
    "m_H_s3": ("H", "canonical batching, feature branch, ranking disabled", 20260814),
}


@torch.no_grad()
def score(ckpt: Path, corpus, corpus_feat, rows: np.ndarray, device: str,
          bs: int) -> np.ndarray:
    """Score one checkpoint, using the feature-attached corpus if it has a feature branch."""
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = PERankFormerConfig(**ck["model_config"])
    m = PERankFormer(cfg).to(device).eval()
    m.load_state_dict(ck["model_state_dict"])
    ds = PEDataset(corpus_feat if cfg.n_features else corpus)
    out = []
    for s in range(0, len(rows), bs):
        b = {k: v.to(device) for k, v in collate([ds[i] for i in rows[s:s + bs]]).items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            o = m(b)
        out.append(m.efficiency_from_output(o).float().cpu().numpy())
    del m
    torch.cuda.empty_cache()
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "results/runs")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab = ContextVocab.load(str(ROOT / "data/processed/context_vocab_official.json"))
    # Always score through the ORIGINAL featurization: the two npz files differ only in
    # group_key, which inference never reads, but reading one file keeps this unambiguous.
    corpus = load_featurized(str(ROOT / "data/processed/featurized_official.npz"), vocab)
    # The feature-branch arm reads `features` from the batch, so a feature-attached copy is
    # prepared once. Standardisation is fitted on the training folds, never on fold 0.
    corpus_feat = attach_family_c_features(
        load_featurized(str(ROOT / "data/processed/featurized_official.npz"), vocab),
        str(ROOT / "data/processed/family_c_features.parquet"),
        np.where(np.asarray(corpus.fold) >= 2)[0])
    fold = np.asarray(corpus.fold)
    rows = np.where(fold == 0)[0]
    man = pd.read_parquet(C.require_manifest())
    base = (pd.DataFrame({"record_id": np.asarray(corpus.record_id)[rows]})
            .merge(man, on="record_id", validate="1:1")
            .rename(columns={"edited": "edited_frac", "spacer": "protospacer"}))

    found, cols = {}, {}
    for run, (arm, desc, seed) in ARMS.items():
        dirs = sorted(args.runs_dir.glob(f"{run}_*"))
        if not dirs:
            continue
        rd = dirs[-1]
        info_f = rd / "run_info.json"
        hist_f = rd / "training_history.csv"
        if not info_f.exists():
            print(f"{run}: still training", flush=True)
            continue
        info = json.loads(info_f.read_text())
        hist = pd.read_csv(hist_f)
        ck_dir = ROOT / "checkpoints" / rd.name
        rec = {"run": run, "arm": arm, "description": desc, "seed": seed,
               "epochs_run": int(info["total_epochs_run"]),
               "best_val_spearman": info["best_val_spearman"],
               "best_epoch_by_spearman": info["best_epoch"],
               "best_val_decision": info.get("best_val_decision_achieved_at_1"),
               "best_epoch_by_decision": info.get("best_decision_epoch"),
               "max_pairs_per_group": info.get("max_pairs_per_group"),
               "min_pair_diff": info.get("min_pair_diff"),
               "rank_group_npz": info.get("rank_group_npz"),
               "mean_train_pairs_per_epoch": float(hist.train_pairs_per_epoch.mean()),
               "selectors_agree": info["best_epoch"] == info.get("best_decision_epoch")}
        for sel, fn in (("spearman", "best.pt"), ("decision", "best_decision.pt")):
            ck = ck_dir / fn
            if ck.exists():
                col = f"{run}__{sel}"
                cols[col] = score(ck, corpus, corpus_feat, rows, device, args.batch_size)
                rec[f"checkpoint_{sel}"] = str(ck.relative_to(ROOT))
        found[run] = rec
        print(f"scored {run} ({arm})", flush=True)

    if not cols:
        raise SystemExit("no completed runs yet")
    for k, v in cols.items():
        base[k] = v
    # Group by decision_group (allele x context), not by allele: fold 0 carries the same
    # allele in up to fourteen conditions, and pooling them would rank across contexts.
    cand = E.aggregate_candidates(base, group="decision_group", scores=tuple(cols))
    nd = cand.groupby("decision_group", observed=True).design_key.transform("nunique")
    cand = cand[nd >= 2]
    t = E.score_population(cand, list(cols), ks=(1,), group="decision_group")
    ti = t[t.informative]

    for run, rec in found.items():
        for sel in ("spearman", "decision"):
            col = f"{run}__{sel}"
            if col in cols:
                rec[f"fold0_achieved_{sel}"] = float(ti[f"{col}_at_1"].mean())
                rec[f"fold0_hit_{sel}"] = float(ti[f"{col}_hit"].mean())
                rec[f"fold0_pooled_rho_{sel}"] = C.spearman(base[col].to_numpy(),
                                                            base.edited_frac.to_numpy())

    res = {"provenance": C.provenance([C.CORPUS], args.seed),
           "canon_version": CANON_VERSION,
           "surface": {"fold": 0, "informative_decision_groups": int(len(ti)),
                       "protospacer_clusters": int(ti.site.nunique()),
                       "note": "no arm trains on fold 0 or early-stops on it"},
           "runs": list(found.values())}

    # arm-level replication, and every paired contrast that the matrix is for
    df = pd.DataFrame(found.values())
    res["by_arm"] = []
    for arm, g in df.groupby("arm"):
        res["by_arm"].append({
            "arm": arm, "description": g.description.iloc[0], "n_seeds": int(len(g)),
            "mean_fold0_achieved": float(g.fold0_achieved_spearman.mean()),
            "sd_fold0_achieved": float(g.fold0_achieved_spearman.std()) if len(g) > 1 else None,
            "per_seed": [round(x, 5) for x in g.fold0_achieved_spearman.tolist()],
            "mean_pooled_rho": float(g.fold0_pooled_rho_spearman.mean()),
            "mean_train_pairs_per_epoch": float(g.mean_train_pairs_per_epoch.mean())})
    res["paired"] = {}
    for a, b in (("D", "A"), ("B", "A"), ("D", "B"), ("D", "C"), ("D", "E"), ("F", "D"),
                 ("G", "D"), ("E", "A"), ("F", "A"), ("H", "A"), ("H", "F"), ("H", "E"),
                 ("F", "E")):
        ga = df[df.arm == a]
        gb = df[df.arm == b]
        if ga.empty or gb.empty:
            continue
        # seed-paired where both arms have the seed, otherwise arm means
        common = sorted(set(ga.seed) & set(gb.seed))
        if common:
            d = [float(ga[ga.seed == s].fold0_achieved_spearman.iloc[0]
                       - gb[gb.seed == s].fold0_achieved_spearman.iloc[0]) for s in common]
            res["paired"][f"{a}_minus_{b}"] = {
                "seeds": common, "per_seed": [round(x, 5) for x in d],
                "mean": float(np.mean(d)),
                "same_sign": bool(len(set(np.sign(d))) == 1)}
    # the selector comparison, pooled over every completed run
    both = df[df.fold0_achieved_decision.notna() & df.fold0_achieved_spearman.notna()] \
        if "fold0_achieved_decision" in df else pd.DataFrame()
    if len(both):
        d = (both.fold0_achieved_decision - both.fold0_achieved_spearman).to_numpy()
        res["selector_comparison"] = {
            "runs": int(len(both)),
            "runs_where_selectors_disagree": int((~both.selectors_agree).sum()),
            "mean_gain_from_decision_selection": float(d.mean()),
            "per_run": {r: round(v, 5) for r, v in zip(both.run, d)},
            "mean_pooled_rho_cost": float((both.fold0_pooled_rho_decision
                                           - both.fold0_pooled_rho_spearman).mean())}
    C.write_outputs("e19_matrix_results", res, render(res))


def render(r: dict) -> str:
    L = ["# E19 - the decision-aligned training matrix\n",
         f"Fold 0, {r['surface']['informative_decision_groups']:,} informative decision groups "
         f"over {r['surface']['protospacer_clusters']:,} protospacer clusters; "
         f"{r['surface']['note']}.\n",
         "## Arms\n",
         "| arm | what it changes | seeds | mean achieved @1 | per seed | pooled rho | pairs/epoch |",
         "|---|---|---:|---:|---|---:|---:|"]
    for a in sorted(r["by_arm"], key=lambda x: x["arm"]):
        L.append(f"| {a['arm']} | {a['description']} | {a['n_seeds']} | "
                 f"{a['mean_fold0_achieved']:.5f} | {a['per_seed']} | "
                 f"{a['mean_pooled_rho']:.4f} | {a['mean_train_pairs_per_epoch']:,.0f} |")
    if r.get("paired"):
        L.append("\n## Seed-paired contrasts\n")
        L.append("| contrast | seeds | per seed | mean | same sign |\n|---|---:|---|---:|---|")
        for k, v in r["paired"].items():
            L.append(f"| {k.replace('_minus_', ' − ')} | {len(v['seeds'])} | {v['per_seed']} | "
                     f"{v['mean']:+.5f} | {v['same_sign']} |")
    sc = r.get("selector_comparison")
    if sc:
        L.append("\n## Checkpoint selection: pooled Spearman versus the decision\n")
        L.append(f"Both selections come from the same {sc['runs']} trajectories, so the "
                 "comparison is not confounded by different stopping times. The selectors "
                 f"chose different epochs in {sc['runs_where_selectors_disagree']} of "
                 f"{sc['runs']} runs.\n")
        L.append(f"- mean gain in fold-0 achieved efficiency from selecting on the decision: "
                 f"**{sc['mean_gain_from_decision_selection']:+.5f}**\n"
                 f"- mean pooled-rho cost of doing so: {sc['mean_pooled_rho_cost']:+.4f}\n")
        L.append(f"Per run: {sc['per_run']}\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
