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
from pe_rankformer.training.losses import LossWeights, shift_loss, total_loss  # noqa: E402
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
        "--source-weights", nargs="+", default=None, metavar="STUDY=W",
        help="round-6 Lead 1: per-source loss weights, e.g. deepprime=2.5 hsu2026=2.0 "
             "pridict_pridict2=0.3. Corrects the train/eval mismatch (58.4%% of training "
             "rows are Schwank, which is 0%% of the evaluation set).",
    )
    ap.add_argument(
        "--lambda-shift", type=float, default=0.0,
        help="round-6 §9: weight on the same-design cross-context rank-shift loss. "
             "Differencing one design across two contexts cancels the universal "
             "design-quality term, so this supervises the interaction directly.",
    )
    ap.add_argument(
        "--shift-pairs-per-step", type=int, default=128,
        help="number of design-context pairs drawn per optimiser step for the shift loss",
    )
    ap.add_argument(
        "--shift-shuffle-control", action="store_true",
        help="round-6 §23 mechanism-free control: identical computation and gradient "
             "magnitude, but the rank-shift targets are permuted, so any gain cannot be "
             "attributed to real interaction signal",
    )
    ap.add_argument(
        "--ctx-primary", action="store_true",
        help="round-7: train the PRIMARY ordinal head on context-normalised quantiles "
             "F_c(y) instead of global ones. Removes the cross-condition mean shift as a "
             "shortcut, forcing the sequence model to learn within-condition ranking; "
             "inference maps the predicted quantile back through each condition's "
             "training CDF, which supplies location exactly rather than approximately.",
    )
    ap.add_argument(
        "--source-conditional-head", action="store_true",
        help="round-8 B1: per-source final projection over a shared trunk (batch-effect "
             "model: shared biology, per-assay readout)",
    )
    ap.add_argument(
        "--tie-source-heads", action="store_true",
        help="round-8 B1 control: same parameter count, all sources scored by head 0, so "
             "a gain must come from differentiation rather than capacity",
    )
    ap.add_argument(
        "--coral-head", action="store_true",
        help="round-6 Lead 2a: rank-consistent ordinal head (shared weights, ordered biases)",
    )
    ap.add_argument(
        "--hurdle-head", action="store_true",
        help="round-6 Lead 3a: zero-inflation two-part head, P(y>0) gate + conditional ordinal",
    )
    ap.add_argument("--mono-penalty", type=float, default=0.0,
                    help="round-6 Lead 2c: penalty on non-monotone cumulative probabilities")
    ap.add_argument("--ema-decay", type=float, default=0.0,
                    help="round-6 Lead 4d: evaluate an EMA of weights (0 disables)")
    ap.add_argument("--ssm-state-dim", type=int, default=None)
    ap.add_argument("--n-edit-layers", type=int, default=None)
    ap.add_argument("--n-peg-layers", type=int, default=None)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--ffn-dim", type=int, default=None)
    ap.add_argument("--moe-experts-r6", type=int, default=None, dest="moe_r6",
                    help="round-6: MoE experts (alias avoiding the round-2 flag's default)")
    ap.add_argument(
        "--quantile-head", action="store_true",
        help="round-5 §13: conditional quantile regression with pinball loss",
    )
    ap.add_argument("--quantile-levels", type=float, nargs="+",
                    default=[0.1, 0.25, 0.5, 0.75, 0.9])
    ap.add_argument(
        "--aux-simplex-weight", type=float, default=0.0,
        help="round-5 §6 dual-head: weight on an auxiliary simplex head alongside ordinal",
    )
    ap.add_argument(
        "--aux-ordinal-bins", type=int, nargs="+", default=None,
        help="round-5 §7 multi-resolution: extra ordinal head resolutions, e.g. 8 50",
    )
    ap.add_argument("--aux-ordinal-weight", type=float, default=0.0)
    ap.add_argument(
        "--aux-context-weight", type=float, default=0.0,
        help="round-5 §8: weight on an auxiliary ordinal head over context-normalised quantiles",
    )
    ap.add_argument(
        "--context-quantile-fields", nargs="+", default=["source_study", "cell_type", "pe_type"],
        help="grouping used to define the context-normalised target for --aux-context-weight",
    )
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
        "--val-fold", type=int, default=None, choices=[1, 2, 3, 4, 5],
        help="round-4: override the config's official val_fold, setting train_folds to the "
             "other four. Fold 0 is the locked held-out set and is never selectable here.",
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
        "--sequence-mixer",
        choices=["attention", "ssm", "hybrid_alt", "hybrid_par", "selective", "selective_frozen"],
        default=None,
        help="round-4 §8: intra-sequence mixing -- Transformer attention or bidirectional SSM",
    )
    ap.add_argument("--lr", type=float, default=None, help="override optim.lr (fine-tuning uses a small LR)")
    args = ap.parse_args()
    if args.dev_folds_file and not args.dev_fold_col:
        ap.error("--dev-folds-file requires --dev-fold-col")
    if args.val_fold is not None and args.dev_folds_file:
        # The dev-folds path replaces val_fold/train_folds wholesale, so honouring both
        # would silently ignore one of them.
        ap.error("--val-fold and --dev-folds-file are mutually exclusive")

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

    if args.val_fold is not None:
        cfg["data"]["val_fold"] = args.val_fold
        cfg["data"]["train_folds"] = [f for f in (1, 2, 3, 4, 5) if f != args.val_fold]
        logger.info("official split override: val_fold=%d train_folds=%s",
                    args.val_fold, cfg["data"]["train_folds"])

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

    if args.source_weights:
        # Weights apply to TRAINING rows only. Validation stays unweighted so the
        # selection metric remains the plain Liu+Kim Spearman we actually report --
        # otherwise a weight sweep would move the target it is being judged against.
        wmap = {}
        for spec in args.source_weights:
            k_, v_ = spec.split("=")
            wmap[k_] = float(v_)
        srcw = pd.read_parquet(
            cfg["data"].get("source_df", "data/processed/optiprime_official_318471.parquet"),
            columns=["record_id", "source_study"],
        ).set_index("record_id").source_study.reindex(corpus.record_id).to_numpy()
        w = np.array([wmap.get(x, 1.0) for x in srcw], dtype=np.float32)
        corpus.sample_weight = w
        eff = {s_: float(w[(srcw == s_)][0]) for s_ in np.unique(srcw)}
        tr_src = srcw[train_idx]
        share = {s_: float((w[train_idx] * (tr_src == s_)).sum() / w[train_idx].sum())
                 for s_ in np.unique(tr_src)}
        logger.info("source weights %s -> effective training gradient share %s",
                    eff, {k_: round(v_, 3) for k_, v_ in share.items()})

    shift_pairs = None
    if args.lambda_shift > 0:
        pr = pd.read_parquet("data/processed/round6_context_pairs.parquet",
                             columns=["record_a", "record_b", "delta_rank"])
        pos = {r: i for i, r in enumerate(corpus.record_id)}
        tr_set = set(train_idx.tolist())
        ia = pr.record_a.map(pos).to_numpy()
        ib = pr.record_b.map(pos).to_numpy()
        ok = np.array([not (np.isnan(x) or np.isnan(y)) for x, y in zip(ia, ib)])
        ia, ib = ia[ok].astype(np.int64), ib[ok].astype(np.int64)
        dr = pr.delta_rank.to_numpy()[ok].astype(np.float32)
        # BOTH rows of a pair must be training rows, or the loss would read a label
        # from a validation row and leak it into training.
        keep = np.array([a in tr_set and b in tr_set for a, b in zip(ia, ib)])
        shift_pairs = (ia[keep], ib[keep], dr[keep])
        logger.info("shift loss: %d usable train-only pairs of %d total (lambda=%.3f%s)",
                    keep.sum(), len(pr), args.lambda_shift,
                    ", SHUFFLED CONTROL" if args.shift_shuffle_control else "")

    logger.info("train=%d val=%d test=%d (test fold locked, not touched)", len(train_idx), len(val_idx), len(test_idx))

    if args.feature_branch:
        from pe_rankformer.data.family_c_features import FEATURE_COLS, attach_family_c_features

        corpus = attach_family_c_features(corpus, args.features_path, train_idx)
        cfg["model"]["n_features"] = len(FEATURE_COLS)
        logger.info(
            "attached %d Family-C features from %s (normalized on %d training rows)",
            len(FEATURE_COLS), args.features_path, len(train_idx),
        )

    if args.source_conditional_head:
        cfg["model"]["source_conditional_head"] = 4  # vocab size for source_study
        cfg["model"]["tie_source_heads"] = args.tie_source_heads
    if args.coral_head:
        cfg["model"]["outcome_head"] = "coral"
        cfg["loss"]["outcome_head"] = "coral"
    if args.hurdle_head:
        cfg["model"]["outcome_head"] = "hurdle"
        cfg["loss"]["outcome_head"] = "hurdle"
    if args.mono_penalty > 0:
        cfg["loss"]["mono_penalty"] = args.mono_penalty
    for k_, v_ in (("ssm_state_dim", args.ssm_state_dim), ("n_edit_layers", args.n_edit_layers),
                   ("n_peg_layers", args.n_peg_layers), ("d_model", args.d_model),
                   ("ffn_dim", args.ffn_dim), ("moe_experts", args.moe_r6)):
        if v_ is not None:
            cfg["model"][k_] = v_
    if args.quantile_head:
        cfg["model"]["outcome_head"] = "quantile"
        cfg["loss"]["outcome_head"] = "quantile"
        cfg["model"]["quantile_levels"] = tuple(args.quantile_levels)
        cfg["model"].pop("ordinal_thresholds", None)
    if args.aux_simplex_weight > 0:
        cfg["model"]["aux_simplex_weight"] = args.aux_simplex_weight
    if args.aux_ordinal_bins:
        cfg["model"]["aux_ordinal_weight"] = args.aux_ordinal_weight
    if args.aux_context_weight > 0:
        cfg["model"]["aux_context_ordinal"] = True
        cfg["model"]["aux_context_weight"] = args.aux_context_weight

    if args.init_from and cfg["model"].get("outcome_head") in ("ordinal", "coral", "hurdle"):
        # Fine-tuning must INHERIT the checkpoint's thresholds rather than recompute
        # them. Recomputing on a data subset changes the output parameterisation --
        # training on Liu+Kim alone yields 16 distinct quantiles where the full mix
        # yields 18 -- which both breaks the strict load and silently redefines what
        # the head predicts. For domain adaptation only the data should change.
        _ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        _thr = _ck.get("model_config", {}).get("ordinal_thresholds")
        if _thr:
            cfg["model"]["ordinal_thresholds"] = tuple(_thr)
            logger.info("inherited %d ordinal thresholds from %s (not recomputed)",
                        len(_thr), args.init_from)
        del _ck

    if (cfg["model"].get("outcome_head") in ("ordinal", "coral", "hurdle")
            and not args.ctx_primary and not cfg["model"].get("ordinal_thresholds")):
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
        if args.aux_ordinal_bins:
            aux = []
            for k in args.aux_ordinal_bins:
                qa = np.linspace(0.0, 1.0, k + 1)[1:-1]
                ta = np.unique(np.quantile(corpus.target[train_idx], qa)).astype(float)
                aux.append(tuple(ta.tolist()))
                logger.info("  aux ordinal resolution K=%d -> %d distinct thresholds", k, len(ta))
            cfg["model"]["ordinal_thresholds_aux"] = tuple(aux)

    if args.ctx_primary:
        args.aux_context_weight = max(args.aux_context_weight, 1e-9)  # trigger computation
    if args.aux_context_weight > 0:
        # Context-normalised target: each row's rank within its own experimental
        # context, mapped to [0,1]. Fitted on TRAINING rows only -- the empirical CDF is
        # a statistic of the data, so building it from all rows would leak the
        # validation target distribution into the training signal. Validation rows are
        # mapped through the training CDF by interpolation; contexts unseen in training
        # fall back to the global training CDF.
        ctx_df = pd.read_parquet(
            cfg["data"].get("source_df", "data/processed/optiprime_official_318471.parquet"),
            columns=["record_id"] + args.context_quantile_fields,
        ).set_index("record_id").reindex(corpus.record_id)
        key = ctx_df[args.context_quantile_fields].astype(str).agg("|".join, axis=1).to_numpy()
        y_all = corpus.target
        ctx_q = np.empty(len(corpus), dtype=np.float32)
        train_mask = np.zeros(len(corpus), dtype=bool)
        train_mask[train_idx] = True
        global_ref = np.sort(y_all[train_idx])
        n_small = 0
        for k_ in np.unique(key):
            in_k = key == k_
            ref = np.sort(y_all[in_k & train_mask])
            if len(ref) < 50:
                ref = global_ref
                n_small += 1
            ctx_q[in_k] = np.searchsorted(ref, y_all[in_k], side="right") / max(len(ref), 1)
        corpus.target_ctx_q = np.clip(ctx_q, 0.0, 1.0)
        logger.info(
            "context-normalised targets over %d groups (%s); %d groups too small, using global CDF",
            len(np.unique(key)), "+".join(args.context_quantile_fields), n_small,
        )
        if args.ctx_primary:
            # Swap the supervised target itself. Thresholds become the uniform grid on
            # [0,1] because the target is now a quantile, not an efficiency. The
            # per-condition training CDFs are stored so evaluation can invert the map;
            # without them the predicted quantiles are not comparable across conditions
            # and pooled Spearman would be meaningless.
            corpus.target_global = corpus.target.copy()
            corpus.target = corpus.target_ctx_q.astype(np.float32)
            cfg["model"]["ordinal_thresholds"] = tuple(
                np.linspace(0.0, 1.0, args.ordinal_bins + 1)[1:-1].tolist()
            )
            cdfs = {}
            for k_ in np.unique(key):
                ref = np.sort(y_all[(key == k_) & train_mask])
                cdfs[str(k_)] = (ref if len(ref) >= 50 else global_ref).tolist()
            cfg["ctx_primary"] = {"fields": args.context_quantile_fields, "cdfs": cdfs}
            logger.info("ctx-primary: target replaced by within-condition quantile; "
                        "stored %d condition CDFs for inference", len(cdfs))

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
        mono_penalty=cfg["loss"].get("mono_penalty", 0.0),
        ordinal_thresholds=(
            torch.tensor(model_cfg.ordinal_thresholds, dtype=torch.float32)
            if model_cfg.outcome_head in ("ordinal", "coral", "hurdle") else None
        ),
        head_segments=model.head_segments,
        quantile_levels=(
            torch.tensor(model_cfg.quantile_levels, dtype=torch.float32)
            if model_cfg.outcome_head == "quantile" else None
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
                    target_ctx_q=batch.get("target_ctx_q"),
                    sample_weight=batch.get("sample_weight"),
                )

            # --- round-6 §9: same-design cross-context rank-shift loss --------------
            # Computed on an extra pair batch rather than opportunistically within the
            # main batch, because same-design cross-context pairs almost never co-occur
            # under ordinary sampling -- the signal would be seen a handful of times per
            # epoch and contribute nothing.
            if shift_pairs is not None:
                ia_all, ib_all, dr_all = shift_pairs
                sel = torch.randint(0, len(ia_all), (args.shift_pairs_per_step,)).numpy()
                ds_ = PEDataset(corpus)
                ba = collate([ds_[i] for i in ia_all[sel]])
                bb = collate([ds_[i] for i in ib_all[sel]])
                ba = {k: v.to(device, non_blocking=True) for k, v in ba.items()}
                bb = {k: v.to(device, non_blocking=True) for k, v in bb.items()}
                tgt = torch.tensor(dr_all[sel], device=device)
                if args.shift_shuffle_control:
                    # Same computation, same gradient scale, no real interaction signal.
                    tgt = tgt[torch.randperm(len(tgt), device=device)]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    sa = model.ranking_score(model(ba))
                    sb = model.ranking_score(model(bb))
                l_shift = shift_loss(sa.float(), sb.float(), tgt)
                loss = loss + args.lambda_shift * l_shift
                parts["shift"] = l_shift.item()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["grad_clip"])
            opt.step()
            sched.step()

            for k in ("loss", "reg", "rank", "corr"):
                running[k] += parts[k]
            if "shift" in parts:
                running["shift"] = running.get("shift", 0.0) + parts["shift"]
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
