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
