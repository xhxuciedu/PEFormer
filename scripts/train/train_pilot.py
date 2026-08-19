"""Train PE-RankFormer on the locked protospacer-disjoint split (task spec §22, §26-27).

Usage:
    python scripts/train/train_pilot.py --run-name model_a --lambda-rank 0.25
    python scripts/train/train_pilot.py --run-name model_b_norank --lambda-rank 0.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pe_rankformer.data.context import ContextVocab  # noqa: E402
from pe_rankformer.data.dataset import PEDataset, collate, load_featurized  # noqa: E402
from pe_rankformer.evaluation.metrics import global_metrics  # noqa: E402
from pe_rankformer.models.pe_rankformer import PERankFormer, PERankFormerConfig  # noqa: E402
from pe_rankformer.training.losses import LossWeights, total_loss  # noqa: E402
from pe_rankformer.training.ranking import GroupedBatchSampler, sample_ranking_pairs  # noqa: E402
from pe_rankformer.training.schedule import warmup_cosine_schedule  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_pilot")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def evaluate(model, corpus, indices, device, batch_size=1024):
    model.eval()
    ds = PEDataset(corpus)
    preds, targets = [], []
    for start in range(0, len(indices), batch_size):
        idx = indices[start : start + batch_size]
        batch = collate([ds[i] for i in idx])
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(batch)
        preds.append(model.efficiency_from_output(out).float().cpu().numpy())
        targets.append(batch["target"].float().cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return global_metrics(targets, preds), preds, targets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--lambda-rank", type=float, default=None, help="override loss.lambda_rank")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument(
        "--no-context", action="store_true", help="Model C ablation: disable FiLM context conditioning"
    )
    ap.add_argument("--patience", type=int, default=None, help="override early-stop patience")
    ap.add_argument(
        "--simplex-head", action="store_true",
        help="3-way outcome distribution head over (unedited, edited, indel)",
    )
    ap.add_argument(
        "--ordinal-head", action="store_true",
        help="round-4 CORAL-style ordinal head: K-1 cumulative-threshold predictions at "
             "quantiles of the training efficiency distribution (metric-matched to Spearman)",
    )
    ap.add_argument("--ordinal-bins", type=int, default=20, help="K for --ordinal-head")
    ap.add_argument(
        "--bag-frac", type=float, default=1.0,
        help="round-4: train on this fraction of training PROTOSPACERS (bagging). "
             "Subsampling is by protospacer, not by row, because rows sharing a "
             "protospacer are strongly correlated and row-level bagging would leave "
             "every protospacer represented and barely decorrelate anything.",
    )
    ap.add_argument(
        "--regression-space", choices=["raw", "logit"], default=None,
        help="regression loss space (task spec §19 comparison)",
    )
    ap.add_argument(
        "--context-strategy", choices=["late", "layerwise"], default=None,
        help="round-2 Family A: FiLM once after pooling (late) vs at every block (layerwise)",
    )
    ap.add_argument(
        "--feature-branch", action="store_true",
        help="round-2 Family C: add the continuous-feature MLP branch (§9)",
    )
    ap.add_argument(
        "--features-path", type=str, default="data/processed/family_c_features.parquet",
        help="parquet from scripts/data/compute_family_c_features.py",
    )
    ap.add_argument(
        "--beta-corr", type=float, default=None,
        help="round-2 §13: weight on the batch Pearson-correlation loss (0 disables it)",
    )
    ap.add_argument(
        "--moe-experts", type=int, default=None,
        help="round-2 Family B (§8): number of context-gated experts in the head (0 disables)",
    )
    ap.add_argument(
        "--dev-folds-file", type=str, default=None,
        help="round-3 §5: parquet from build_round3_dev_folds.py, overrides "
             "val_fold/train_folds with a Liu+Kim-matched split when set",
    )
    ap.add_argument(
        "--dev-fold-col", type=str, default=None,
        help="column in --dev-folds-file to use, e.g. round3_dev_fold_0 (values 'train'/'val')",
    )
    ap.add_argument(
        "--init-from", type=str, default=None,
        help="round-3 §8: initialise weights from this checkpoint (domain-adaptive fine-tuning)",
    )
    ap.add_argument(
        "--train-sources", nargs="+", default=None,
        choices=["hsu2026", "deepprime", "pridict_pridict2"],
        help="round-3 §8-10: restrict TRAINING rows to these source studies "
             "(validation is unaffected). Liu=hsu2026, Kim=deepprime, Schwank=pridict_pridict2",
    )
    ap.add_argument(
        "--schwank-replay-frac", type=float, default=None,
        help="round-3 §9: keep this fraction of the Schwank training rows when "
             "--train-sources restricts to Liu+Kim (e.g. 0.10 for 10%% replay)",
    )
    ap.add_argument(
        "--val-sources", nargs="+", default=None,
        choices=["hsu2026", "deepprime", "pridict_pridict2"],
        help="round-3 §7: restrict VALIDATION rows to these sources, so checkpoint "
             "selection/early-stopping targets the Liu+Kim benchmark rather than the "
             "Schwank-dominated official fold",
    )
    ap.add_argument(
        "--sequence-mixer", choices=["attention", "ssm"], default=None,
        help="round-4 §8: intra-sequence mixing -- Transformer attention or bidirectional SSM",
    )
    ap.add_argument("--lr", type=float, default=None, help="override optim.lr (fine-tuning uses a small LR)")
    args = ap.parse_args()
    if args.dev_folds_file and not args.dev_fold_col:
        ap.error("--dev-folds-file requires --dev-fold-col")

    cfg = yaml.safe_load(args.config.read_text())
    if args.lambda_rank is not None:
        cfg["loss"]["lambda_rank"] = args.lambda_rank
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.max_epochs is not None:
        cfg["train"]["max_epochs"] = args.max_epochs
    if args.no_context:
        cfg["model"]["use_context"] = False
    if args.patience is not None:
        cfg["train"]["early_stop_patience"] = args.patience
    if args.regression_space is not None:
        cfg["loss"]["regression_space"] = args.regression_space
    if args.simplex_head:
        cfg["model"]["outcome_head"] = "simplex"
        cfg["loss"]["outcome_head"] = "simplex"
    if args.ordinal_head:
        if args.simplex_head:
            raise SystemExit("--ordinal-head and --simplex-head are mutually exclusive")
        cfg["model"]["outcome_head"] = "ordinal"
        cfg["loss"]["outcome_head"] = "ordinal"
    if args.context_strategy is not None:
        cfg["model"]["context_strategy"] = args.context_strategy
    if args.beta_corr is not None:
        cfg["loss"]["beta_corr"] = args.beta_corr
    if args.moe_experts is not None:
        cfg["model"]["moe_experts"] = args.moe_experts
    if args.sequence_mixer is not None:
        cfg["model"]["sequence_mixer"] = args.sequence_mixer
    if args.lr is not None:
        cfg["optim"]["lr"] = args.lr

    set_seed(cfg["seed"])
    device = torch.device("cuda")
    logger.info("device: %s", torch.cuda.get_device_name(0))

    vocab = ContextVocab.load(cfg["data"]["vocab"])
    corpus_path = Path(cfg["data"]["corpus"])
    corpus = load_featurized(str(corpus_path), vocab)
    dataset_hash = file_sha256(corpus_path)
    logger.info("corpus: %d rows, hash=%s", len(corpus), dataset_hash[:12])

    fold = corpus.fold
    test_idx = np.where(fold == cfg["data"]["test_fold"])[0]

    if args.dev_folds_file:
        # Round-3 §5: Liu+Kim-matched development split, replacing the official
        # (Schwank-heavy) val_fold/train_folds for model *selection* purposes. The
        # held-out test_idx above is untouched either way.
        dev = pd.read_parquet(args.dev_folds_file, columns=["record_id", args.dev_fold_col])
        record_id_to_pos = {rid: i for i, rid in enumerate(corpus.record_id)}
        dev_pos = dev["record_id"].map(record_id_to_pos)
        assert dev_pos.notna().all(), "dev-folds-file has record_ids not present in this corpus"
        dev_pos = dev_pos.to_numpy(dtype=np.int64)
        is_val = (dev[args.dev_fold_col] == "val").to_numpy()
        val_idx = dev_pos[is_val]
        train_idx = dev_pos[~is_val]
        logger.info("using round-3 dev fold: %s (col=%s)", args.dev_folds_file, args.dev_fold_col)
    else:
        val_idx = np.where(fold == cfg["data"]["val_fold"])[0]
        train_idx = np.where(np.isin(fold, cfg["data"]["train_folds"]))[0]
    pos_to_source = None
    if args.train_sources is not None or args.val_sources is not None:
        src = pd.read_parquet(cfg["data"].get("source_df", "data/processed/optiprime_official_318471.parquet"),
                              columns=["record_id", "source_study"])
        pos_to_source = src.set_index("record_id").source_study.reindex(corpus.record_id).to_numpy()

    if args.val_sources is not None:
        # Round-3 §7: the selection target is Liu+Kim Spearman, not the Schwank-heavy
        # official fold. Filtering validation makes early stopping / best-checkpoint
        # selection optimise the benchmark we actually care about.
        val_source = pos_to_source[val_idx]
        n_before = len(val_idx)
        val_idx = val_idx[np.isin(val_source, args.val_sources)]
        logger.info("val-source filter %s: %d -> %d rows", args.val_sources, n_before, len(val_idx))

    if args.train_sources is not None:
        # Round-3 §8-10: restrict TRAINING rows by source study.
        train_source = pos_to_source[train_idx]
        keep = np.isin(train_source, args.train_sources)

        if args.schwank_replay_frac:
            # §9: keep a random subset of otherwise-excluded Schwank rows, so late-stage
            # optimization is Liu+Kim-dominated without fully discarding Schwank signal.
            excluded_schwank = np.where((train_source == "pridict_pridict2") & ~keep)[0]
            rng = np.random.default_rng(cfg["seed"])
            n_keep = int(round(args.schwank_replay_frac * len(excluded_schwank)))
            keep[rng.choice(excluded_schwank, size=n_keep, replace=False)] = True
            logger.info("Schwank replay: added %d of %d excluded Schwank rows (%.0f%%)",
                        n_keep, len(excluded_schwank), 100 * args.schwank_replay_frac)

        n_before = len(train_idx)
        train_idx = train_idx[keep]
        logger.info(
            "train-source filter %s: %d -> %d rows (%s)",
            args.train_sources, n_before, len(train_idx),
            {s: int((train_source[keep] == s).sum()) for s in np.unique(train_source[keep])},
        )

    if args.bag_frac < 1.0:
        # Classical bagging, the one decorrelation mechanism that does not depend on a
        # new architecture: each member sees a different protospacer sample, so members
        # are wrong about different loci. Seeded by the run seed, so seed variants give
        # genuinely different bags rather than the same bag twice.
        bag_src = pd.read_parquet(
            cfg["data"].get("source_df", "data/processed/optiprime_official_318471.parquet"),
            columns=["record_id", "spacer"],
        )
        pos_to_spacer = bag_src.set_index("record_id").spacer.reindex(corpus.record_id).to_numpy()
        train_spacers = np.unique(pos_to_spacer[train_idx])
        rng = np.random.default_rng(cfg["seed"])
        keep_spacers = rng.choice(
            train_spacers, size=int(round(args.bag_frac * len(train_spacers))), replace=False
        )
        n_before = len(train_idx)
        train_idx = train_idx[np.isin(pos_to_spacer[train_idx], keep_spacers)]
        logger.info(
            "bagging: kept %d/%d training protospacers (frac=%.2f), rows %d -> %d",
            len(keep_spacers), len(train_spacers), args.bag_frac, n_before, len(train_idx),
        )

    logger.info("train=%d val=%d test=%d (test fold locked, not touched)", len(train_idx), len(val_idx), len(test_idx))

    if args.feature_branch:
        from pe_rankformer.data.family_c_features import FEATURE_COLS, attach_family_c_features

        corpus = attach_family_c_features(corpus, args.features_path, train_idx)
        cfg["model"]["n_features"] = len(FEATURE_COLS)
        logger.info(
            "attached %d Family-C features from %s (normalized on %d training rows)",
            len(FEATURE_COLS), args.features_path, len(train_idx),
        )

    if cfg["model"].get("outcome_head") == "ordinal":
        # Thresholds are quantiles of the TRAINING targets only -- deriving them from the
        # full corpus would leak the validation/test target distribution into the model's
        # output parameterisation. Duplicates are dropped so the thresholds stay strictly
        # increasing (efficiency has heavy mass at 0, so low quantiles can coincide).
        qs = np.linspace(0.0, 1.0, args.ordinal_bins + 1)[1:-1]
        thr = np.unique(np.quantile(corpus.target[train_idx], qs)).astype(float)
        cfg["model"]["ordinal_thresholds"] = tuple(thr.tolist())
        logger.info(
            "ordinal head: %d thresholds from %d training targets (range %.4f - %.4f)",
            len(thr), len(train_idx), thr[0], thr[-1],
        )

    model_cfg = PERankFormerConfig(
        context_fields=vocab.fields, context_vocab_sizes=vocab.sizes(), **cfg["model"]
    )
    model = PERankFormer(model_cfg).to(device)
    logger.info("model: %.1fM params", model.num_parameters() / 1e6)

    if args.init_from:
        # Round-3 §8: domain-adaptive fine-tuning. Strict load -- an architecture
        # mismatch between the pretrained checkpoint and this run's config should be a
        # hard error, not silently-random weights for the mismatched submodules.
        init_ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(init_ckpt["model_state_dict"])
        logger.info(
            "initialised from %s (its best epoch %s, val_spearman %.4f)",
            args.init_from, init_ckpt.get("epoch"),
            init_ckpt.get("config", {}).get("_best_val_spearman", float("nan")),
        )

    weights = LossWeights(
        lambda_rank=cfg["loss"]["lambda_rank"],
        huber_beta=cfg["loss"]["huber_beta"],
        min_pair_diff=cfg["loss"]["min_pair_diff"],
        regression_space=cfg["loss"].get("regression_space", "raw"),
        outcome_head=cfg["loss"].get("outcome_head", "scalar"),
        ordinal_thresholds=(
            torch.tensor(model_cfg.ordinal_thresholds, dtype=torch.float32)
            if model_cfg.outcome_head == "ordinal" else None
        ),
        beta_corr=cfg["loss"].get("beta_corr", 0.0),
    )
    max_pairs_per_group = cfg["loss"]["max_pairs_per_group"]

    batch_size = cfg["train"]["batch_size"]
    sampler = GroupedBatchSampler(corpus.group_key[train_idx], batch_size=batch_size, seed=cfg["seed"])
    steps_per_epoch = len(sampler)
    total_steps = steps_per_epoch * cfg["train"]["max_epochs"]

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])
    sched = warmup_cosine_schedule(opt, total_steps, warmup_frac=cfg["optim"]["warmup_frac"])

    run_id = f"{args.run_name}_{int(time.time())}"
    run_dir = Path("results/runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints") / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.dump(cfg))

    history_rows = []
    best_val_spearman = -1.0
    best_epoch = -1
    patience_counter = 0
    ds = PEDataset(corpus)

    start_time = time.time()
    for epoch in range(cfg["train"]["max_epochs"]):
        sampler.set_epoch(epoch)
        model.train()
        epoch_t0 = time.time()
        running = {"loss": 0.0, "reg": 0.0, "rank": 0.0, "corr": 0.0, "n_pairs": 0}
        n_batches = 0
        for local_batch in sampler:
            global_idx = train_idx[local_batch]
            batch = collate([ds[i] for i in global_idx])
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(batch)
                rank_score = model.ranking_score(out)
                pi, pj = sample_ranking_pairs(
                    batch["group_key"], batch["target"],
                    min_diff=cfg["loss"]["min_pair_diff"], max_pairs_per_group=max_pairs_per_group,
                )
                loss, parts = total_loss(
                    out, batch["target"], pi, pj, weights,
                    rank_score=rank_score, target_indel=batch.get("target_indel"),
                )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["grad_clip"])
            opt.step()
            sched.step()

            for k in ("loss", "reg", "rank", "corr"):
                running[k] += parts[k]
            running["n_pairs"] += pi.numel()
            n_batches += 1

        epoch_time = time.time() - epoch_t0
        val_metrics, _, _ = evaluate(model, corpus, val_idx, device)
        lr_now = sched.get_last_lr()[0]

        row = {
            "epoch": epoch,
            "train_loss": running["loss"] / n_batches,
            "train_reg_loss": running["reg"] / n_batches,
            "train_rank_loss": running["rank"] / n_batches,
            "train_pairs_per_epoch": running["n_pairs"],
            "val_pearson": val_metrics.pearson,
            "val_spearman": val_metrics.spearman,
            "val_mae": val_metrics.mae,
            "val_rmse": val_metrics.rmse,
            "lr": lr_now,
            "epoch_time_s": epoch_time,
        }
        history_rows.append(row)
        logger.info(
            "epoch %2d/%d  train_loss=%.4f (reg=%.4f rank=%.4f)  val_spearman=%.4f val_pearson=%.4f "
            "val_mae=%.4f  time=%.1fs",
            epoch, cfg["train"]["max_epochs"] - 1, row["train_loss"], row["train_reg_loss"],
            row["train_rank_loss"], val_metrics.spearman, val_metrics.pearson, val_metrics.mae, epoch_time,
        )

        pd.DataFrame(history_rows).to_csv(run_dir / "training_history.csv", index=False)

        improved = val_metrics.spearman > best_val_spearman
        if improved:
            best_val_spearman = val_metrics.spearman
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "config": cfg, "model_config": model_cfg.__dict__, "epoch": epoch},
                ckpt_dir / "best.pt",
            )
        else:
            patience_counter += 1

        if epoch >= cfg["train"]["early_stop_min_warmup_epochs"] and patience_counter >= cfg["train"]["early_stop_patience"]:
            logger.info("early stopping at epoch %d (best epoch %d, val_spearman=%.4f)", epoch, best_epoch, best_val_spearman)
            break

    total_time = time.time() - start_time
    torch.save(
        {"model_state_dict": model.state_dict(), "config": cfg, "model_config": model_cfg.__dict__, "epoch": epoch},
        ckpt_dir / "final.pt",
    )

    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    run_info = {
        "run_id": run_id,
        "run_name": args.run_name,
        "lambda_rank": cfg["loss"]["lambda_rank"],
        "seed": cfg["seed"],
        "git_commit": git_commit,
        "dataset_hash": dataset_hash,
        "gpu": torch.cuda.get_device_name(0),
        "n_params": model.num_parameters(),
        "best_epoch": best_epoch,
        "best_val_spearman": best_val_spearman,
        "total_epochs_run": epoch + 1,
        "total_train_time_s": total_time,
        "checkpoint_best": str(ckpt_dir / "best.pt"),
        "checkpoint_final": str(ckpt_dir / "final.pt"),
        "dev_folds_file": args.dev_folds_file,
        "dev_fold_col": args.dev_fold_col,
        "init_from": args.init_from,
        "train_sources": args.train_sources,
        "val_sources": args.val_sources,
        "schwank_replay_frac": args.schwank_replay_frac,
        "lr": cfg["optim"]["lr"],
        "n_train_rows": int(len(train_idx)),
        "n_val_rows": int(len(val_idx)),
    }
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    logger.info("done: %s", json.dumps(run_info, indent=2))


if __name__ == "__main__":
    main()
