"""Adaptation harness for ADAPTATION_SELECTION_PLAN.md: heads, losses and the training loop.

The plan asks for four arms that differ in *what is updated* and *what is optimised*, sharing
everything else. This module supplies the pieces so that the arms can be built from one code
path and therefore differ only where intended:

* `Adapted` wraps a published PE-RankFormer without modifying it. The shared representation
  `pooled` is captured with a forward pre-hook on `base.head`, so no model source changes and
  every existing checkpoint still loads unmodified.
* `selection_utility_loss` is the plan's smooth expected-regret surrogate; `listnet_loss` is
  its listwise alternative; both are averaged over groups, not pairs.
* `unfreeze` implements the staged schedule: head only, then pooling/cross-attention and the
  final sequence block, then everything.

Nothing here imports OptiPrime or any of its outputs. It is an evaluation comparator in this
phase, not a source of features, targets or teachers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- model ----


class Adapted(nn.Module):
    """A published checkpoint plus an optional selection head.

    `mode` selects the arm:
      "P"  prediction only, the pretrained head fine-tuned (ordinary adaptation)
      "S"  selection only, a new scalar head trained with a group loss
      "M"  both heads, multitask
      "Z"  selection only, encoder frozen
      "shared"  M's capacity without M's separation: one extra head whose single score is
                used for BOTH losses, so an M-over-shared gain is attributable to having
                two outputs rather than to the extra parameters
    """

    def __init__(self, base: nn.Module, mode: str, geom_dim: int = 0,
                 hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.base, self.mode, self.geom_dim = base, mode, geom_dim
        pooled_dim = base.head[0].normalized_shape[0]
        self.pooled_dim = pooled_dim
        self._pooled: torch.Tensor | None = None
        base.head.register_forward_pre_hook(self._capture)
        need_sel = mode in ("S", "M", "Z")
        self.sel = (nn.Sequential(
            nn.LayerNorm(pooled_dim + geom_dim),
            nn.Linear(pooled_dim + geom_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1)) if need_sel else None)
        # The capacity-matched control for M. M's claim is that having two OUTPUTS helps;
        # the control must therefore add the same parameters WITHOUT adding an output. It
        # spends them on a residual adapter in the shared trunk, so a single score carries
        # both objectives. Hidden width is halved because the adapter projects back to
        # pooled_dim, which keeps the parameter counts close; both are reported.
        self.adapter = (nn.Sequential(
            nn.LayerNorm(pooled_dim), nn.Linear(pooled_dim, max(hidden // 2, 8)), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(max(hidden // 2, 8), pooled_dim))
            if mode == "shared" else None)

    def _capture(self, _module, args):
        pooled = args[0]
        if self.adapter is not None:
            pooled = pooled + self.adapter(pooled)
        self._pooled = pooled
        return (pooled,)

    def forward(self, batch: dict, geom: torch.Tensor | None = None):
        out = self.base(batch)
        pooled = self._pooled
        s = None
        if self.sel is not None:
            z = pooled if self.geom_dim == 0 else torch.cat([pooled, geom], dim=-1)
            s = self.sel(z).squeeze(-1)
        return out, s

    def selection_score(self, out: torch.Tensor, s: torch.Tensor | None) -> torch.Tensor:
        """What the arm actually nominates with at deployment.

        `shared` deliberately nominates with the prediction score: that is what having a
        single output means.
        """
        if self.mode in ("S", "Z", "M") and s is not None:
            return s
        return self.base.efficiency_from_output(out)


class GeometryOnly(nn.Module):
    """Control: the same selection loss on candidate geometry alone, no sequence model.

    If this matches an adapted encoder, the encoder is contributing nothing that a handful
    of scalars do not already carry.
    """

    def __init__(self, geom_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(geom_dim), nn.Linear(geom_dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1))

    def forward(self, geom: torch.Tensor) -> torch.Tensor:
        return self.net(geom).squeeze(-1)


# -------------------------------------------------------------------------- losses ----


def _group_slices(gid: np.ndarray) -> list[slice]:
    """Contiguous slices per group. Rows must already be sorted by group."""
    starts = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1]])
    ends = np.r_[starts[1:], len(gid)]
    return [slice(int(a), int(b)) for a, b in zip(starts, ends)]


def selection_utility_loss(s: torch.Tensor, y: torch.Tensor, slices, T: float = 1.0,
                           scale: float = 1.0) -> torch.Tensor:
    """The plan's smooth expected-regret surrogate, averaged over groups.

        p = softmax(s / T);  L(g) = max_i y_i - sum_i p_i y_i

    Errors are weighted by how much efficiency they cost, and a group whose candidates all
    measure the same receives exactly zero gradient rather than an arbitrary target. `scale`
    is one fixed training-derived constant, not a per-group range normalisation, so that
    high-consequence decisions keep their extra weight.
    """
    tot, n = s.new_zeros(()), 0
    for sl in slices:
        yy = y[sl]
        if yy.numel() < 2:
            continue
        m = yy.max()
        if m <= yy.min():
            continue  # constant group: no decision to make, no gradient
        p = torch.softmax(s[sl] / T, dim=0)
        tot = tot + (m - (p * yy).sum())
        n += 1
    return tot / max(n, 1) / scale


def listnet_loss(s: torch.Tensor, y: torch.Tensor, slices, T: float = 1.0,
                 T_y: float = 0.02) -> torch.Tensor:
    """Listwise cross-entropy against a soft target built from measured efficiency.

    Constant-outcome groups are skipped: their uniform target carries no ordering
    information and would act as undeclared regularisation.
    """
    tot, n = s.new_zeros(()), 0
    for sl in slices:
        yy = y[sl]
        if yy.numel() < 2 or yy.max() <= yy.min():
            continue
        q = torch.softmax(yy / T_y, dim=0)
        logp = torch.log_softmax(s[sl] / T, dim=0)
        tot = tot - (q * logp).sum()
        n += 1
    return tot / max(n, 1)


def utility_pairwise_loss(s: torch.Tensor, y: torch.Tensor, slices, T: float = 1.0,
                          cap: float = 0.05, max_pairs: int = 8,
                          gen: torch.Generator | None = None) -> torch.Tensor:
    """Utility-weighted pairwise logistic loss, normalised by sampled pairs within a group.

    The gap weight is capped so a handful of very large differences cannot dominate, and no
    minimum-gap cutoff is imposed: the plan asks that the old 0.02 threshold not be carried
    over untested. `T` scales the score difference, matching the other two losses' signature
    so the objective can be swapped without changing anything else.
    """
    tot, n = s.new_zeros(()), 0
    for sl in slices:
        yy, ss = y[sl], s[sl]
        k = yy.numel()
        if k < 2 or yy.max() <= yy.min():
            continue
        i, j = torch.triu_indices(k, k, offset=1, device=s.device)
        if i.numel() > max_pairs:
            sel = torch.randperm(i.numel(), device=s.device, generator=gen)[:max_pairs]
            i, j = i[sel], j[sel]
        gap = (yy[i] - yy[j]).clamp(-cap, cap)
        w = gap.abs()
        if w.sum() <= 0:
            continue
        tgt = (gap > 0).to(s.dtype)
        per = F.binary_cross_entropy_with_logits((ss[i] - ss[j]) / T, tgt, reduction="none")
        tot = tot + (w * per).sum() / w.sum()
        n += 1
    return tot / max(n, 1)


SELECTION_LOSSES = {"utility": selection_utility_loss, "listnet": listnet_loss,
                    "pairwise": utility_pairwise_loss}


# ---------------------------------------------------------------------- unfreezing ----


@dataclass
class Stage:
    name: str
    modules: tuple[str, ...]
    lr_scale: float


STAGES = (
    Stage("head", ("head",), 1.0),
    Stage("readout", ("head", "edit_pool", "peg_pool", "film", "cross_blocks"), 0.3),
    Stage("final_block", ("head", "edit_pool", "peg_pool", "film", "cross_blocks",
                          "edit_encoder.-1", "peg_encoder.-1"), 0.1),
    Stage("all", ("*",), 0.05),
)


def unfreeze(base: nn.Module, stage: str) -> list[str]:
    """Freeze everything, then re-enable the named stage. Returns the unfrozen prefixes."""
    for p in base.parameters():
        p.requires_grad_(False)
    spec = next(s for s in STAGES if s.name == stage)
    if spec.modules == ("*",):
        for p in base.parameters():
            p.requires_grad_(True)
        return ["*"]
    opened = []
    for name in spec.modules:
        if name.endswith(".-1"):
            target = _last_block(getattr(base, name[:-3], None))
        else:
            target = getattr(base, name, None)
        if target is None:
            continue
        for p in target.parameters():
            p.requires_grad_(True)
        opened.append(name)
    return opened


def _last_block(stack) -> nn.Module | None:
    """The final sequence block of an encoder stack.

    The encoder is an `nn.Sequential`, an `nn.TransformerEncoder` or a `BiS4DStack`
    depending on the mixer; only the last exposes its blocks as `.layers` rather than by
    indexing, so unfreezing "the final block" has to handle all three.
    """
    if stack is None:
        return None
    inner = getattr(stack, "layers", stack)
    try:
        return inner[-1] if len(inner) else None
    except TypeError:
        return None


def trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
