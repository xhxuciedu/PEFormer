"""Bidirectional diagonal state-space (S4D-style) sequence mixer.

Round-4 experiment B (spec §8): the round-3 ensemble's gain came from error
decorrelation, and every member so far mixes sequence information with attention.
A state-space model mixes it with a *learned convolution* whose kernel is generated
by a linear recurrence -- a materially different inductive bias (smooth, distance-
parameterised, translation-equivariant) rather than the same model renamed.

Implemented in plain PyTorch: `mamba_ssm` is not installed, and for the sequence
lengths here (102 edit tokens, 90 pegRNA tokens) a custom scan kernel buys nothing.
The diagonal-SSM kernel is materialised in closed form and applied by FFT
convolution, which is exact and fast at this length.

Diagonal SSM (S4D): with state matrix A diagonal, the convolution kernel is
    K[l] = 2 * Re( sum_n C_n * (exp(dt*A_n) - 1)/A_n * exp(dt*A_n)^l )
so the whole kernel is available in closed form without unrolling the recurrence.
Bidirectionality uses two independent kernels (forward and time-reversed), summed --
necessary here because PE efficiency depends on context on both sides of the edit.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class S4DKernel(nn.Module):
    """Generates a length-L convolution kernel per channel from diagonal SSM params."""

    def __init__(self, d_model: int, n_state: int = 64, dt_min: float = 1e-3, dt_max: float = 1e-1):
        super().__init__()
        # Per-channel timescale, log-spaced init: gives channels different receptive fields.
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # S4D-Lin initialisation: real part decays, imaginary part spreads frequencies.
        half = n_state // 2
        log_A_real = torch.log(0.5 * torch.ones(d_model, half))
        A_imag = math.pi * torch.arange(half).repeat(d_model, 1)
        self.log_A_real = nn.Parameter(log_A_real)
        self.A_imag = nn.Parameter(A_imag)
        self.C = nn.Parameter(torch.randn(d_model, half, 2) * 0.5**0.5)

    def forward(self, L: int) -> torch.Tensor:
        dt = torch.exp(self.log_dt).unsqueeze(-1)  # (H, 1)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H, N/2)
        C = torch.view_as_complex(self.C)  # (H, N/2)

        dtA = A * dt  # (H, N/2)
        # C_tilde folds the zero-order-hold discretisation into C.
        C_tilde = C * (torch.exp(dtA) - 1.0) / A
        power = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H, N/2, L)
        return 2.0 * torch.einsum("hn,hnl->hl", C_tilde, torch.exp(power)).real


class BiS4DLayer(nn.Module):
    """Pre-LN residual block: bidirectional SSM mix + gated pointwise FFN.

    Mirrors the structure of the Transformer encoder layer it replaces (same residual
    topology, same pre-LN placement, same FFN width) so that a PE-SSM vs. Transformer
    comparison isolates the sequence-mixing mechanism rather than confounding it with
    block-level design differences.
    """

    def __init__(self, d_model: int, ffn_dim: int, dropout: float, n_state: int = 64):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.kernel_fwd = S4DKernel(d_model, n_state)
        self.kernel_bwd = S4DKernel(d_model, n_state)
        self.D = nn.Parameter(torch.randn(d_model))  # skip (feedthrough) term
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def _conv(self, u: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """FFT convolution of (B, H, L) signal with (H, L) kernel, zero-padded to 2L."""
        L = u.size(-1)
        n = 2 * L
        u_f = torch.fft.rfft(u.float(), n=n)
        k_f = torch.fft.rfft(k.float(), n=n)
        return torch.fft.irfft(u_f * k_f.unsqueeze(0), n=n)[..., :L]

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, L, H); pad_mask: (B, L) True where padded.
        residual = x
        h = self.norm1(x)
        if pad_mask is not None:
            # Zero padded positions before convolving: the conv is linear, so zeros
            # contribute nothing to any output, which keeps padding from leaking in.
            h = h.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        u = h.transpose(1, 2)  # (B, H, L)
        L = u.size(-1)
        y = self._conv(u, self.kernel_fwd(L))
        y = y + self._conv(u.flip(-1), self.kernel_bwd(L)).flip(-1)
        y = y + u * self.D.unsqueeze(-1)
        y = y.transpose(1, 2).to(x.dtype)  # (B, L, H)

        y = self.out_proj(y) * F.silu(self.gate(h))
        x = residual + self.dropout(y)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return x


class BiS4DStack(nn.Module):
    """Drop-in replacement for an nn.TransformerEncoder stack."""

    def __init__(self, d_model: int, ffn_dim: int, dropout: float, n_layers: int, n_state: int = 64):
        super().__init__()
        self.layers = nn.ModuleList(
            [BiS4DLayer(d_model, ffn_dim, dropout, n_state) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
        return x


class HybridStack(nn.Module):
    """Interleave SSM and Transformer blocks in one encoder (round-5 spec §12).

    Motivated by the round-5 factorial: architecture was the larger effect
    (+0.0192 SSM over Transformer, vs +0.0092 for the ordinal objective), and the
    two architectures' *errors* are among the least correlated pairs measured
    (residual corr 0.6453 for the Transformer/simplex vs SSM/ordinal diagonal). If
    convolutional and attentional mixing capture genuinely different structure, a
    model containing both may capture more than either -- rather than only an
    ensemble of the two doing so.

    Two designs, because they test different things:

    ``alternating`` -- blocks run in sequence (SSM, attention, SSM, ...). Cheap, and
    lets each mechanism refine the other's output. Depth is preserved, so parameter
    count sits between the two pure stacks.

    ``parallel`` -- both mixers see the same input and are combined by a learned
    per-channel gate, ``h = g * h_ssm + (1 - g) * h_attn``. More expensive, but the
    gate is *interpretable*: after training it reports how much each channel relies
    on convolutional versus attentional mixing, which is a direct readout of whether
    the hybrid is genuinely using both or collapsing onto one.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float,
                 n_layers: int, n_state: int = 64, mode: str = "alternating"):
        super().__init__()
        if mode not in ("alternating", "parallel"):
            raise ValueError(f"unknown hybrid mode: {mode!r}")
        self.mode = mode
        self.n_layers = n_layers

        def _attn():
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)

        if mode == "alternating":
            # Even layers mix with the SSM, odd layers with attention.
            self.blocks = nn.ModuleList(
                [BiS4DLayer(d_model, ffn_dim, dropout, n_state) if i % 2 == 0 else _attn()
                 for i in range(n_layers)]
            )
        else:
            self.ssm_blocks = nn.ModuleList(
                [BiS4DLayer(d_model, ffn_dim, dropout, n_state) for _ in range(n_layers)]
            )
            self.attn_blocks = nn.ModuleList([_attn() for _ in range(n_layers)])
            # Gate starts at zero => sigmoid 0.5 => an even mix, so neither mechanism is
            # privileged at initialisation and any preference has to be learned.
            self.gates = nn.ParameterList(
                [nn.Parameter(torch.zeros(d_model)) for _ in range(n_layers)]
            )

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode == "alternating":
            for i, blk in enumerate(self.blocks):
                x = (blk(x, src_key_padding_mask) if i % 2 == 0
                     else blk(x, src_key_padding_mask=src_key_padding_mask))
            return x

        for ssm, attn, g in zip(self.ssm_blocks, self.attn_blocks, self.gates):
            h_ssm = ssm(x, src_key_padding_mask)
            h_attn = attn(x, src_key_padding_mask=src_key_padding_mask)
            gate = torch.sigmoid(g)
            x = gate * h_ssm + (1.0 - gate) * h_attn
            if src_key_padding_mask is not None:
                # The attention branch leaves padded positions non-zero; re-zero them so
                # the two branches agree and padding cannot reach the pooling step.
                x = x.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
        return x

    def gate_summary(self) -> list[float]:
        """Mean SSM weight per layer -- 0.5 means an even mix, >0.5 favours the SSM."""
        if self.mode != "parallel":
            return []
        with torch.no_grad():
            return [float(torch.sigmoid(g).mean()) for g in self.gates]


class SelectiveScan(nn.Module):
    """Input-dependent (Mamba/S6-style) diagonal SSM scan, one direction.

    The distinction from `S4DKernel` is the whole point of this module. S4D is
    **linear time-invariant**: its kernel is a function of the sequence length alone,
    so the same convolution is applied to every input. It can encode "positions eight
    apart interact" but never "retain *this* position because of what it contains".

    Here dt, B and C are produced by linear projections of the token representation,
    so the recurrence becomes content-dependent:

        h_t = exp(dt_t * A) h_{t-1} + dt_t * B_t * x_t
        y_t = C_t . h_t + D * x_t

    Why this should matter for prime editing specifically: the edit position moves from
    design to design, and the PBS/RTT must match a particular location on the target.
    An LTI kernel has to average over every position the edit could occupy; a selective
    one can gate on the edit itself.

    Because the recurrence is no longer LTI it cannot be evaluated by FFT convolution;
    it needs a scan. At L <= 102 a sequential scan in plain PyTorch is affordable, and
    keeps the implementation transparent.

    `freeze_selection` supports the round-8 control: the projections exist (so the
    parameter count matches) but are held at initialisation and excluded from
    gradients, making the scan effectively input-independent. That isolates selectivity
    from the capacity the extra projections add.
    """

    def __init__(self, d_model: int, n_state: int = 16, dt_rank: int | None = None,
                 freeze_selection: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_state = n_state
        dt_rank = dt_rank or max(1, d_model // 16)
        self.dt_rank = dt_rank

        # Real diagonal A, initialised as the standard S4D-Real spectrum. Kept as
        # log(-A) so A stays strictly negative and the recurrence stays stable.
        A = torch.arange(1, n_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.log_neg_A = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))

        # The selection projections: these are what make the scan content-dependent.
        self.x_proj = nn.Linear(d_model, dt_rank + 2 * n_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)
        # Bias init so softplus(dt) starts in a moderate range, as in the S6 reference.
        with torch.no_grad():
            self.dt_proj.bias.uniform_(math.log(1e-3), math.log(1e-1))
            self.dt_proj.bias.copy_(self.dt_proj.bias.exp().clamp(min=1e-4).log())

        self.freeze_selection = freeze_selection
        if freeze_selection:
            for p in (self.x_proj.weight, self.dt_proj.weight, self.dt_proj.bias):
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, H) -> (B, L, H). Scans left to right."""
        B, L, H = x.shape
        N = self.n_state
        A = -torch.exp(self.log_neg_A)                       # (H, N), negative
        proj = self.x_proj(x)                                # (B, L, dt_rank + 2N)
        dt, Bm, Cm = torch.split(proj, [self.dt_rank, N, N], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                    # (B, L, H), positive

        # Discretisation is done inside the loop rather than materialised up front.
        # Precomputing dA and dBx as (B, L, H, N) needs ~94GB at batch 512 for this
        # model; computing (B, H, N) slices per step keeps the same maths at a fraction
        # of the peak memory, at the cost of a Python-level loop over <=102 steps.
        h = x.new_zeros(B, H, N)
        ys = []
        for t in range(L):
            dt_t = dt[:, t].unsqueeze(-1)                    # (B, H, 1)
            dA_t = torch.exp(dt_t * A)                       # (B, H, N)
            dBx_t = dt_t * Bm[:, t].unsqueeze(1) * x[:, t].unsqueeze(-1)
            h = dA_t * h + dBx_t
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))   # (B, H)
        y = torch.stack(ys, dim=1)                           # (B, L, H)
        return y + x * self.D


class BiSelectiveLayer(nn.Module):
    """Bidirectional selective-SSM block, drop-in for `BiS4DLayer`.

    Deliberately mirrors BiS4DLayer's residual topology, normalisation placement, gate
    and FFN width, so an S4D-vs-selective comparison isolates the mixing mechanism
    rather than confounding it with block design -- the same discipline used when the
    SSM was first compared against the Transformer.
    """

    def __init__(self, d_model: int, ffn_dim: int, dropout: float, n_state: int = 16,
                 freeze_selection: bool = False, checkpoint_scan: bool = True):
        super().__init__()
        # The scan keeps four (B, H, N) tensors per timestep for autograd, across up to
        # 102 steps, two directions and ten layers -- about 52GB at batch 256, which
        # OOMs a 96GB card. Gradient checkpointing recomputes the scan during backward
        # instead of storing it: exact same gradients, roughly 30% more compute, and it
        # is what makes this experiment runnable at a sensible batch size at all.
        self.checkpoint_scan = checkpoint_scan
        self.norm1 = nn.LayerNorm(d_model)
        self.fwd = SelectiveScan(d_model, n_state, freeze_selection=freeze_selection)
        self.bwd = SelectiveScan(d_model, n_state, freeze_selection=freeze_selection)
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        h = self.norm1(x)
        if pad_mask is not None:
            # Zero padded positions before scanning. Unlike the convolutional case this
            # is not sufficient on its own -- a left-to-right scan carries state past a
            # pad -- so outputs are re-masked below and padded positions are excluded
            # from pooling downstream.
            h = h.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        if self.checkpoint_scan and self.training:
            from torch.utils.checkpoint import checkpoint

            y = (checkpoint(self.fwd, h, use_reentrant=False)
                 + checkpoint(self.bwd, h.flip(1), use_reentrant=False).flip(1))
        else:
            y = self.fwd(h) + self.bwd(h.flip(1)).flip(1)
        y = self.out_proj(y) * F.silu(self.gate(h))
        x = residual + self.dropout(y)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return x


class BiSelectiveStack(nn.Module):
    """Drop-in replacement for an nn.TransformerEncoder / BiS4DStack."""

    def __init__(self, d_model: int, ffn_dim: int, dropout: float, n_layers: int,
                 n_state: int = 16, freeze_selection: bool = False,
                 checkpoint_scan: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            BiSelectiveLayer(d_model, ffn_dim, dropout, n_state, freeze_selection,
                             checkpoint_scan)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
        return x
