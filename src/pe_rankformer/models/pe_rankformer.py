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


def _encoder_stack(d_model: int, n_heads: int, ffn_dim: int, dropout: float, n_layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=ffn_dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
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
    """context_vector -> (gamma, beta) applied to a pooled representation."""

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
        return (1 + gamma) * h + beta


class PERankFormer(nn.Module):
    def __init__(self, config: PERankFormerConfig):
        super().__init__()
        self.config = config
        d = config.d_model

        self.edit_token_embed = nn.Embedding(EDIT_VOCAB_SIZE, d, padding_idx=EDIT_PAD_ID)
        self.edit_pos_embed = nn.Embedding(config.edit_seq_len, d)
        self.edit_encoder = _encoder_stack(d, config.n_heads, config.ffn_dim, config.dropout, config.n_edit_layers)

        self.peg_nuc_embed = nn.Embedding(NUC_VOCAB_SIZE, d, padding_idx=NUC_PAD_ID)
        self.peg_seg_embed = nn.Embedding(N_SEGMENTS, d)
        self.peg_pos_embed = nn.Embedding(config.peg_seq_len, d)
        self.peg_encoder = _encoder_stack(d, config.n_heads, config.ffn_dim, config.dropout, config.n_peg_layers)

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
            self.film = FiLM(self.context_encoder.out_dim, pooled_dim)
        else:
            self.context_encoder = None
            self.film = None

        # "simplex": every edited locus ends in exactly one of {unedited, correctly
        # edited, indel}, and the measured values are the proportions of each. Predicting
        # a 3-way distribution respects that constraint by construction and supervises on
        # the indel signal the scalar head throws away -- without importing any
        # biochemical reaction graph (task spec §11 forbids that, but the mutual
        # exclusivity of measured outcomes is a property of the assay, not a mechanism).
        n_out = 3 if config.outcome_head == "simplex" else 1
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d, n_out),
        )

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

        edit_pos = torch.arange(edit_ids.size(1), device=edit_ids.device)
        edit_h = self.edit_token_embed(edit_ids) + self.edit_pos_embed(edit_pos)[None]
        edit_h = self.edit_encoder(edit_h, src_key_padding_mask=edit_pad_mask)

        peg_pos = torch.arange(peg_ids.size(1), device=peg_ids.device)
        peg_h = (
            self.peg_nuc_embed(peg_ids) + self.peg_seg_embed(peg_seg) + self.peg_pos_embed(peg_pos)[None]
        )
        peg_h = self.peg_encoder(peg_h, src_key_padding_mask=peg_pad_mask)

        for block in self.cross_blocks:
            edit_h, peg_h = block(edit_h, peg_h, edit_pad_mask, peg_pad_mask)

        edit_pooled = self.edit_pool(edit_h, edit_pad_mask)
        peg_pooled = self.peg_pool(peg_h, peg_pad_mask)
        pooled = torch.cat([edit_pooled, peg_pooled], dim=-1)

        if self.config.use_context:
            context_vector = self.context_encoder(batch)
            pooled = self.film(pooled, context_vector)

        out = self.head(pooled)
        if self.config.outcome_head == "simplex":
            return out  # (B, 3) logits over [unedited, edited, indel]
        return out.squeeze(-1)  # (B,) raw score

    def ranking_score(self, out: torch.Tensor) -> torch.Tensor:
        """Monotone-in-efficiency scalar used by the pairwise ranking loss.

        For the simplex head this is the log-odds of a correct edit against no edit,
        which is the natural ordering quantity and is invariant to the indel logit.
        """
        if self.config.outcome_head == "simplex":
            return out[:, 1] - out[:, 0]
        return out

    def efficiency_from_output(self, out: torch.Tensor) -> torch.Tensor:
        if self.config.outcome_head == "simplex":
            return torch.softmax(out, dim=-1)[:, 1]
        return torch.sigmoid(out)

    def predict_efficiency(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.efficiency_from_output(self.forward(batch))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
