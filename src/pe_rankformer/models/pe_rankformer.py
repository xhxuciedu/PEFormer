"""PE-RankFormer: context-conditioned relational Transformer for PE efficiency.

Five components (task spec §12, §15-18):
  1. Edit encoder: self-attention over paired WT/edited tokens.
  2. pegRNA encoder: self-attention over segment-aware spacer/PBS/RTT tokens.
  3. Bidirectional cross-attention between the two streams.
  4. FiLM context conditioning from experimental-context categorical embeddings.
  5. Attention-pooled prediction head emitting one pre-sigmoid score.

The model emits a raw (pre-sigmoid) score `s`. `torch.sigmoid(s)` is the predicted
efficiency used for the regression loss; `s` itself is used directly in the pairwise
ranking loss (task spec §20), matching the proposal's `si - sj` formulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.dataset import EDIT_MAX_LEN, PEG_MAX_LEN
from ..data.tokenizer import EDIT_PAD_ID, EDIT_VOCAB_SIZE, NUC_PAD_ID, NUC_VOCAB_SIZE, N_SEGMENTS


@dataclass
class PERankFormerConfig:
    d_model: int = 384
    n_heads: int = 6
    ffn_dim: int = 1536
    dropout: float = 0.10
    n_edit_layers: int = 6
    n_peg_layers: int = 4
    n_cross_blocks: int = 2
    edit_seq_len: int = EDIT_MAX_LEN + 2
    peg_seq_len: int = PEG_MAX_LEN
    context_fields: tuple[str, ...] = ()
    context_vocab_sizes: dict[str, int] = field(default_factory=dict)
    context_embed_dim: int = 32
    use_context: bool = True  # False = Model C ablation (no FiLM conditioning)
    outcome_head: str = "scalar"  # "scalar" | "simplex" (3-way outcome distribution)
    # | "ordinal" (round-4: CORAL-style cumulative-threshold head). The evaluation metric
    # is Spearman, which depends only on order, so a head that directly estimates *which
    # quantile* a row falls into is metric-matched in a way neither regression nor the
    # simplex head is. It predicts K-1 independent cumulative indicators P(y > t_k) at
    # fixed target quantiles t_k; the score is their mean, an estimate of the normalised
    # rank. Chosen for round 4 mainly because its loss geometry (K-1 binary decisions on
    # the target's marginal distribution) differs fundamentally from the simplex head's
    # 3-way cross-entropy over outcome proportions, and error decorrelation -- not
    # standalone accuracy -- is what the round-3 analysis showed the ensemble rewards.
    ordinal_thresholds: tuple[float, ...] = ()  # required iff outcome_head == "ordinal";
    # fixed quantiles of the *training* efficiency distribution, stored in the checkpoint
    # so evaluation reproduces the exact same score mapping.

    # --- round-5 auxiliary heads (spec §6-§8) ---------------------------------------
    # All three round-5 supervision experiments share one mechanism: extra output
    # segments concatenated onto the primary head's logits, each with its own loss and
    # weight. The *primary* head alone produces the ranking score, so an auxiliary head
    # can only help by shaping the shared representation -- it never directly moves the
    # prediction. That is what makes these multitask experiments rather than ensembling
    # in disguise.
    aux_simplex_weight: float = 0.0  # §6 dual-head: + 3 logits trained as the simplex
    # outcome distribution, alongside the ordinal primary.
    ordinal_thresholds_aux: tuple[tuple[float, ...], ...] = ()  # §7 multi-resolution:
    # additional ordinal segments at *other* threshold resolutions (e.g. K=7 and K=43
    # beside the primary K=18).
    aux_ordinal_weight: float = 0.0  # weight on each auxiliary ordinal segment.
    aux_context_ordinal: bool = False  # §8: one extra ordinal segment supervised on
    # *context-normalised* quantiles F_c(y) rather than global ones.
    aux_context_weight: float = 0.0
    context_strategy: str = "late"  # "late" (single FiLM after pooling) | "layerwise" (round-2
    # Family A: FiLM applied at every edit/pegRNA encoder layer and every cross-attention
    # block, so context modulates sequence representation throughout the network rather
    # than only at the end -- see claude_code_round2_pe_rankformer_model_search.md §7).
    n_features: int = 0  # round-2 Family C (§9): 0 disables the feature branch; else the
    # number of continuous input features (missingness indicators are handled internally
    # and do not need to be counted here).
    feature_hidden_dim: int = 64
    moe_experts: int = 0  # round-2 Family B (§8): 0 disables MoE; else K in {4, 8},
    # replacing the head's hidden expansion with K context-gated experts.
    sequence_mixer: str = "attention"  # round-4 §8: "attention" (Transformer self-
    # attention, all prior rounds) | "ssm" (bidirectional diagonal state-space mixing).
    # Only the intra-sequence mixing changes; cross-attention, pooling, context
    # conditioning and head are identical, so a comparison isolates inductive bias.
    ssm_state_dim: int = 64

    def __post_init__(self) -> None:
        if self.context_strategy == "layerwise" and not self.use_context:
            raise ValueError("context_strategy='layerwise' requires use_context=True")
        if self.moe_experts > 0 and not self.use_context:
            raise ValueError("moe_experts>0 requires use_context=True (the gate is context-conditioned)")
        if self.sequence_mixer not in ("attention", "ssm"):
            raise ValueError(f"unknown sequence_mixer: {self.sequence_mixer!r}")
        if self.outcome_head not in ("scalar", "simplex", "ordinal"):
            raise ValueError(f"unknown outcome_head: {self.outcome_head!r}")
        if self.outcome_head == "ordinal":
            t = self.ordinal_thresholds
            if len(t) < 2:
                raise ValueError("outcome_head='ordinal' requires >=2 ordinal_thresholds")
            if any(b <= a for a, b in zip(t, t[1:])):
                # Non-increasing thresholds would make the cumulative targets incoherent
                # (P(y>t_k) must be non-increasing in k) and silently corrupt training.
                raise ValueError(f"ordinal_thresholds must be strictly increasing, got {t}")
        elif self.ordinal_thresholds:
            raise ValueError("ordinal_thresholds set but outcome_head is not 'ordinal'")
        aux_on = (self.aux_simplex_weight > 0 or self.ordinal_thresholds_aux
                  or self.aux_context_ordinal)
        if aux_on and self.outcome_head != "ordinal":
            # The auxiliary machinery assumes an ordinal primary (that is the round-5
            # baseline). Failing loudly beats silently training a head nothing reads.
            raise ValueError("round-5 auxiliary heads require outcome_head='ordinal'")
        if self.ordinal_thresholds_aux and self.aux_ordinal_weight <= 0:
            raise ValueError("ordinal_thresholds_aux set but aux_ordinal_weight is 0")
        if self.aux_context_ordinal and self.aux_context_weight <= 0:
            raise ValueError("aux_context_ordinal set but aux_context_weight is 0")
        for t in self.ordinal_thresholds_aux:
            if len(t) < 2 or any(b <= a for a, b in zip(t, t[1:])):
                raise ValueError(f"auxiliary thresholds must be strictly increasing, got {t}")
        if self.sequence_mixer == "ssm" and self.context_strategy == "layerwise":
            # The layerwise path interleaves FiLM between individually-exposed encoder
            # layers; the SSM stack is not currently decomposed that way. Fail loudly
            # rather than silently ignoring one of the two settings.
            raise ValueError(
                "sequence_mixer='ssm' with context_strategy='layerwise' is not implemented; "
                "use context_strategy='late' for SSM runs"
            )


def _encoder_layer(d_model: int, n_heads: int, ffn_dim: int, dropout: float) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=ffn_dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


def _encoder_stack(
    d_model: int, n_heads: int, ffn_dim: int, dropout: float, n_layers: int,
    mixer: str = "attention", ssm_state_dim: int = 64,
) -> nn.Module:
    if mixer == "ssm":
        from .ssm import BiS4DStack

        return BiS4DStack(d_model, ffn_dim, dropout, n_layers, ssm_state_dim)
    layer = _encoder_layer(d_model, n_heads, ffn_dim, dropout)
    # norm_first (pre-LN) encoder layers can't use PyTorch's nested-tensor fast path;
    # disabling it explicitly avoids a benign warning at every model construction.
    return nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)


class CrossAttentionBlock(nn.Module):
    """One bidirectional cross-attention block: edit<->pegRNA, each with its own FFN."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.edit_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.peg_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.edit_ln1 = nn.LayerNorm(d_model)
        self.peg_ln1 = nn.LayerNorm(d_model)
        self.edit_ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model)
        )
        self.peg_ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model)
        )
        self.edit_ln2 = nn.LayerNorm(d_model)
        self.peg_ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, edit_h: torch.Tensor, peg_h: torch.Tensor,
        edit_pad_mask: torch.Tensor, peg_pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Edit positions attend to pegRNA positions, and vice versa (pre-LN residual).
        e_q = self.edit_ln1(edit_h)
        p_q = self.peg_ln1(peg_h)
        e_attn_out, _ = self.edit_attn(e_q, p_q, p_q, key_padding_mask=peg_pad_mask, need_weights=False)
        p_attn_out, _ = self.peg_attn(p_q, e_q, e_q, key_padding_mask=edit_pad_mask, need_weights=False)
        edit_h = edit_h + self.dropout(e_attn_out)
        peg_h = peg_h + self.dropout(p_attn_out)

        edit_h = edit_h + self.dropout(self.edit_ffn(self.edit_ln2(edit_h)))
        peg_h = peg_h + self.dropout(self.peg_ffn(self.peg_ln2(peg_h)))
        return edit_h, peg_h


class AttentionPool(nn.Module):
    """Single learned query attends over the sequence, respecting padding."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        b = h.size(0)
        q = self.query.expand(b, -1, -1)
        out, _ = self.attn(q, self.ln(h), self.ln(h), key_padding_mask=pad_mask, need_weights=False)
        return out.squeeze(1)


class ContextEncoder(nn.Module):
    """Per-field categorical embeddings, concatenated into one context vector."""

    def __init__(self, fields: tuple[str, ...], vocab_sizes: dict[str, int], embed_dim: int):
        super().__init__()
        self.fields = fields
        self.embeds = nn.ModuleDict(
            {f_: nn.Embedding(vocab_sizes[f_], embed_dim) for f_ in fields}
        )
        self.out_dim = embed_dim * len(fields)

    def forward(self, ctx: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [self.embeds[f_](ctx[f"ctx_{f_}"]) for f_ in self.fields]
        return torch.cat(parts, dim=-1)


class FiLM(nn.Module):
    """context_vector -> (gamma, beta) applied to a representation.

    Handles both a pooled representation `h` of shape (B, feature_dim) (round-1 "late"
    strategy) and a per-token representation of shape (B, L, feature_dim) (round-2
    "layerwise" strategy, where gamma/beta are broadcast over the sequence dim).
    """

    def __init__(self, context_dim: int, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * feature_dim),
        )
        self.feature_dim = feature_dim

    def forward(self, h: torch.Tensor, context_vector: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(context_vector).chunk(2, dim=-1)
        if h.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return (1 + gamma) * h + beta


class FeatureBranch(nn.Module):
    """Round-2 Family C (§9): a small MLP over externally-computed continuous features
    (PBS/RTT length & GC, Tm, MFE, RuleSet3 activity, edit geometry -- see
    scripts/data/compute_family_c_features.py), concatenated onto the pooled sequence
    representation. Missingness is tracked, not silently imputed away: the caller
    passes already-imputed values (train-set mean) plus a parallel 0/1 mask, and the
    mask is concatenated into the MLP input so the model can learn to discount
    imputed entries rather than treat them as observed zeros.
    """

    def __init__(self, n_features: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(n_features)
        self.net = nn.Sequential(
            nn.Linear(2 * n_features, hidden_dim),  # features ++ missingness mask
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_dim = hidden_dim

    def forward(self, features: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm(features)
        x = torch.cat([x, missing_mask], dim=-1)
        return self.net(x)


class MoEHead(nn.Module):
    """Round-2 Family B (§8): context-gated mixture of experts, replacing the head's
    single hidden-expansion FFN. `K` experts each map the pooled representation to a
    hidden vector; a context+representation-conditioned softmax gate combines them:

        g(c, h) = softmax(W[c; h]),   h_out = sum_k g_k * f_k(h)

    Applied only in the head (the model's single "final FFN block" in the round-2
    spec's terms), keeping the sequence encoders and cross-attention fully shared.
    Gate probabilities from the most recent forward pass are cached on
    `self.last_gate_probs` (B, K) for utilization/entropy tracking -- not returned
    from `forward`, so callers that just want `h_out` are unaffected.
    """

    def __init__(self, pooled_dim: int, hidden_dim: int, context_dim: int, n_experts: int, dropout: float):
        super().__init__()
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(pooled_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
             for _ in range(n_experts)]
        )
        self.gate = nn.Linear(context_dim + pooled_dim, n_experts)
        self.n_experts = n_experts
        self.last_gate_probs: torch.Tensor | None = None

    def forward(self, h: torch.Tensor, context_vector: torch.Tensor) -> torch.Tensor:
        gate_in = torch.cat([context_vector, h], dim=-1)
        g = torch.softmax(self.gate(gate_in), dim=-1)  # (B, K)
        self.last_gate_probs = g.detach()
        expert_out = torch.stack([e(h) for e in self.experts], dim=1)  # (B, K, hidden)
        return (g.unsqueeze(-1) * expert_out).sum(dim=1)  # (B, hidden)


def moe_load_balance_loss(gate_probs: torch.Tensor) -> torch.Tensor:
    """Encourages uniform expert utilization: squared coefficient of variation of the
    batch-mean gate probability per expert. Zero when every expert gets equal traffic.
    Not wired into training by default -- round-2 §8 says to use it only "if expert
    collapse occurs"; call this explicitly and add its weighted value to the loss if
    `MoEHead.last_gate_probs` shows collapse (see also `expert_utilization_stats`)."""
    usage = gate_probs.mean(dim=0)  # (K,)
    return (usage.var(unbiased=False) / (usage.mean() ** 2 + 1e-8))


def expert_utilization_stats(gate_probs: torch.Tensor) -> dict[str, float]:
    """Per-expert mean utilization and the entropy of the mean gate distribution
    (log(K) = maximally uniform; 0 = total collapse onto one expert). Round-2 §8:
    "Track expert utilization; entropy of gating; expert-by-context preferences" --
    diagnostic only, not used to steer training automatically."""
    usage = gate_probs.mean(dim=0)
    entropy = -(usage.clamp_min(1e-12) * usage.clamp_min(1e-12).log()).sum()
    return {"entropy": entropy.item(), **{f"expert_{i}_usage": u.item() for i, u in enumerate(usage)}}


class PERankFormer(nn.Module):
    def __init__(self, config: PERankFormerConfig):
        super().__init__()
        self.config = config
        d = config.d_model

        self.edit_token_embed = nn.Embedding(EDIT_VOCAB_SIZE, d, padding_idx=EDIT_PAD_ID)
        self.edit_pos_embed = nn.Embedding(config.edit_seq_len, d)

        self.peg_nuc_embed = nn.Embedding(NUC_VOCAB_SIZE, d, padding_idx=NUC_PAD_ID)
        self.peg_seg_embed = nn.Embedding(N_SEGMENTS, d)
        self.peg_pos_embed = nn.Embedding(config.peg_seq_len, d)

        self.cross_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(d, config.n_heads, config.ffn_dim, config.dropout)
                for _ in range(config.n_cross_blocks)
            ]
        )

        self.edit_pool = AttentionPool(d, config.n_heads, config.dropout)
        self.peg_pool = AttentionPool(d, config.n_heads, config.dropout)

        pooled_dim = 2 * d
        if config.use_context:
            self.context_encoder = ContextEncoder(
                config.context_fields, config.context_vocab_sizes, config.context_embed_dim
            )
        else:
            self.context_encoder = None

        self.layerwise = config.context_strategy == "layerwise"
        if self.layerwise:
            # Family A (round-2 §7): FiLM at every block instead of once after pooling.
            # nn.TransformerEncoder is a black-box container that runs all its layers
            # internally, so per-layer conditioning needs the individual encoder layers
            # exposed directly rather than wrapped.
            ctx_dim = self.context_encoder.out_dim
            self.edit_layers = nn.ModuleList(
                [_encoder_layer(d, config.n_heads, config.ffn_dim, config.dropout) for _ in range(config.n_edit_layers)]
            )
            self.edit_films = nn.ModuleList([FiLM(ctx_dim, d) for _ in range(config.n_edit_layers)])
            self.peg_layers = nn.ModuleList(
                [_encoder_layer(d, config.n_heads, config.ffn_dim, config.dropout) for _ in range(config.n_peg_layers)]
            )
            self.peg_films = nn.ModuleList([FiLM(ctx_dim, d) for _ in range(config.n_peg_layers)])
            self.cross_edit_films = nn.ModuleList([FiLM(ctx_dim, d) for _ in range(config.n_cross_blocks)])
            self.cross_peg_films = nn.ModuleList([FiLM(ctx_dim, d) for _ in range(config.n_cross_blocks)])
            self.film = None
        else:
            self.edit_encoder = _encoder_stack(
                d, config.n_heads, config.ffn_dim, config.dropout, config.n_edit_layers,
                config.sequence_mixer, config.ssm_state_dim,
            )
            self.peg_encoder = _encoder_stack(
                d, config.n_heads, config.ffn_dim, config.dropout, config.n_peg_layers,
                config.sequence_mixer, config.ssm_state_dim,
            )
            self.film = FiLM(self.context_encoder.out_dim, pooled_dim) if config.use_context else None

        if config.n_features > 0:
            self.feature_branch = FeatureBranch(config.n_features, config.feature_hidden_dim, config.dropout)
            pooled_dim += self.feature_branch.out_dim
        else:
            self.feature_branch = None

        # "simplex": every edited locus ends in exactly one of {unedited, correctly
        # edited, indel}, and the measured values are the proportions of each. Predicting
        # a 3-way distribution respects that constraint by construction and supervises on
        # the indel signal the scalar head throws away -- without importing any
        # biochemical reaction graph (task spec §11 forbids that, but the mutual
        # exclusivity of measured outcomes is a property of the assay, not a mechanism).
        if config.outcome_head == "simplex":
            n_out = 3
        elif config.outcome_head == "ordinal":
            n_out = len(config.ordinal_thresholds)
        else:
            n_out = 1

        # Round-5: auxiliary segments appended after the primary. `head_segments` is the
        # single source of truth for the layout and is consumed by the loss, so the model
        # and the objective cannot disagree about which logits mean what.
        self.head_segments: list[dict] = [
            {"kind": config.outcome_head, "start": 0, "end": n_out, "weight": 1.0}
        ]
        if config.aux_simplex_weight > 0:
            self.head_segments.append({"kind": "simplex", "start": n_out, "end": n_out + 3,
                                       "weight": config.aux_simplex_weight})
            n_out += 3
        for t in config.ordinal_thresholds_aux:
            self.head_segments.append({"kind": "ordinal", "start": n_out, "end": n_out + len(t),
                                       "weight": config.aux_ordinal_weight, "thresholds": t})
            n_out += len(t)
        if config.aux_context_ordinal:
            k = len(config.ordinal_thresholds)
            self.head_segments.append({"kind": "ordinal_ctx", "start": n_out, "end": n_out + k,
                                       "weight": config.aux_context_weight})
            n_out += k
        if config.moe_experts > 0:
            # Disaggregated head, only used for MoE (round-2 Family B, §8) -- kept as a
            # separate code path rather than the default so every pre-round-2 checkpoint
            # (whose state_dict keys are `head.0/1/4.*`) keeps loading unmodified below.
            self.head = None
            self.head_norm = nn.LayerNorm(pooled_dim)
            self.head_hidden = MoEHead(
                pooled_dim, d, self.context_encoder.out_dim, config.moe_experts, config.dropout
            )
            self.head_out = nn.Linear(d, n_out)
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(pooled_dim),
                nn.Linear(pooled_dim, d),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(d, n_out),
            )
            self.head_norm = self.head_hidden = self.head_out = None

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns the raw (pre-sigmoid) score, shape (batch,)."""
        d = self.config.d_model
        edit_ids = batch["edit_ids"]
        peg_ids = batch["peg_nuc_ids"]
        peg_seg = batch["peg_seg_ids"]

        edit_pad_mask = edit_ids == EDIT_PAD_ID
        peg_pad_mask = peg_ids == NUC_PAD_ID

        context_vector = self.context_encoder(batch) if self.config.use_context else None

        edit_pos = torch.arange(edit_ids.size(1), device=edit_ids.device)
        edit_h = self.edit_token_embed(edit_ids) + self.edit_pos_embed(edit_pos)[None]

        peg_pos = torch.arange(peg_ids.size(1), device=peg_ids.device)
        peg_h = (
            self.peg_nuc_embed(peg_ids) + self.peg_seg_embed(peg_seg) + self.peg_pos_embed(peg_pos)[None]
        )

        if self.layerwise:
            # Family A (round-2 §7): h'_l = (1+gamma_l(c)) * h_l + beta_l(c); h_{l+1} = block_l(h'_l).
            for film, layer in zip(self.edit_films, self.edit_layers):
                edit_h = film(edit_h, context_vector)
                edit_h = layer(edit_h, src_key_padding_mask=edit_pad_mask)
            for film, layer in zip(self.peg_films, self.peg_layers):
                peg_h = film(peg_h, context_vector)
                peg_h = layer(peg_h, src_key_padding_mask=peg_pad_mask)
            for i, block in enumerate(self.cross_blocks):
                edit_h = self.cross_edit_films[i](edit_h, context_vector)
                peg_h = self.cross_peg_films[i](peg_h, context_vector)
                edit_h, peg_h = block(edit_h, peg_h, edit_pad_mask, peg_pad_mask)
        else:
            edit_h = self.edit_encoder(edit_h, src_key_padding_mask=edit_pad_mask)
            peg_h = self.peg_encoder(peg_h, src_key_padding_mask=peg_pad_mask)
            for block in self.cross_blocks:
                edit_h, peg_h = block(edit_h, peg_h, edit_pad_mask, peg_pad_mask)

        edit_pooled = self.edit_pool(edit_h, edit_pad_mask)
        peg_pooled = self.peg_pool(peg_h, peg_pad_mask)
        pooled = torch.cat([edit_pooled, peg_pooled], dim=-1)

        if self.film is not None:
            pooled = self.film(pooled, context_vector)

        if self.feature_branch is not None:
            z_f = self.feature_branch(batch["features"], batch["features_missing"])
            pooled = torch.cat([pooled, z_f], dim=-1)

        if self.config.moe_experts > 0:
            pooled = self.head_norm(pooled)
            hidden = self.head_hidden(pooled, context_vector)
            out = self.head_out(hidden)
        else:
            out = self.head(pooled)
        if self.config.outcome_head in ("simplex", "ordinal"):
            return out  # simplex: (B,3); ordinal: (B,K-1) [+ auxiliary segments appended]
        return out.squeeze(-1)  # (B,) raw score

    def ranking_score(self, out: torch.Tensor) -> torch.Tensor:
        """Monotone-in-efficiency scalar used by the pairwise ranking loss.

        For the simplex head this is the log-odds of a correct edit against no edit,
        which is the natural ordering quantity and is invariant to the indel logit.
        """
        if self.config.outcome_head == "simplex":
            return out[:, 1] - out[:, 0]
        if self.config.outcome_head == "ordinal":
            # Mean of the cumulative probabilities: monotone in the predicted quantile,
            # and already the quantity the metric cares about. Sliced to the PRIMARY
            # segment so auxiliary heads never leak into the ranking score.
            k = len(self.config.ordinal_thresholds)
            return torch.sigmoid(out[..., :k]).mean(dim=-1)
        return out

    def efficiency_from_output(self, out: torch.Tensor) -> torch.Tensor:
        if self.config.outcome_head == "simplex":
            return torch.softmax(out, dim=-1)[:, 1]
        if self.config.outcome_head == "ordinal":
            # (1/(K-1)) * sum_k P(y > t_k) in [0,1]: an estimate of the row's normalised
            # rank, not of its efficiency. Rank-correlation metrics are unaffected, but
            # absolute-error metrics against raw efficiency are meaningless for this head.
            k = len(self.config.ordinal_thresholds)
            return torch.sigmoid(out[..., :k]).mean(dim=-1)
        return torch.sigmoid(out)

    def predict_efficiency(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.efficiency_from_output(self.forward(batch))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
