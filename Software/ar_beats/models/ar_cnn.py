"""
models/ar_cnn.py
----------------
Spatial autoregressive CNN over a 2D token grid.

Architecture:
  - Input:  E ∈ R^{B × H_p × W_p × D}
  - 6 masked convolutional layers, 3×3 kernels, dilation d=8
  - Each layer predicts both μ and log σ² (diagonal Gaussian)
  - Output: μ, σ² ∈ R^{B × H_p × W_p × D}
  - Loss:   NLL = 0.5 * sum_d [(E_d - μ_d)² / σ²_d + log σ²_d]
  - Score:  A_{i,j} = NLL (per token) → A_clip = max_{i,j} A_{i,j}

Masked convolution enforces the AR constraint:
  - Raster-scan (row-major) ordering
  - First layer: center pixel MASKED (no access to E_{i,j} itself)
  - Later layers: center pixel UNMASKED (safe — no direct input access)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------
#  Masked Conv2d
# ---------------------------------------------------------------

class MaskedConv2d(nn.Conv2d):
    """
    Conv2d with a causal mask for raster-scan AR ordering.

    mask_center=True  → first layer  (mask out E_{i,j})
    mask_center=False → later layers (allow center, already masked)
    """

    def __init__(self, *args, mask_center: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_center = mask_center
        self.register_buffer("mask", self._build_mask())

    def _build_mask(self) -> torch.Tensor:
        kH, kW = self.kernel_size
        mask = torch.ones(kH, kW)

        center_h = kH // 2
        center_w = kW // 2

        # Zero out everything AFTER (i,j) in raster-scan order
        # i.e. rows below the center, and right of center in the same row
        mask[center_h + 1:, :] = 0.0
        mask[center_h, center_w + 1:] = 0.0
        if self.mask_center:
            mask[center_h, center_w] = 0.0

        # Expand to (out_channels, in_channels, kH, kW)
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = mask.expand(self.weight.shape)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply mask without modifying weights in-place.
        # In-place modification interacts badly with the optimizer's
        # momentum buffers for the zeroed positions.
        return F.conv2d(
            x,
            self.weight * self.mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


# ---------------------------------------------------------------
#  Residual block with masked dilated conv
# ---------------------------------------------------------------

class ARResBlock(nn.Module):
    """
    One residual AR layer:
        MaskedConv2d (dilated) → LayerNorm → GELU → 1×1 proj

    Uses a 1×1 projection on the residual path if dims differ.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        dilation: int,
        mask_center: bool = False,
    ):
        super().__init__()
        self.conv = MaskedConv2d(
            in_channels,
            hidden_channels,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,  # same spatial size
            mask_center=mask_center,
        )
        self.norm = nn.LayerNorm(hidden_channels)
        self.act = nn.GELU()
        self.proj = (
            nn.Conv2d(in_channels, hidden_channels, 1)
            if in_channels != hidden_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        residual = self.proj(x)
        out = self.conv(x)
        # LayerNorm over channel dim: permute to (B, H, W, C)
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)  # back to (B, C, H, W)
        out = self.act(out)
        return out + residual


# ---------------------------------------------------------------
#  AR CNN
# ---------------------------------------------------------------

class ARCNN(nn.Module):
    """
    6-layer masked dilated CNN over BEATs token grid.

    Args:
        embed_dim (int): D — BEATs embedding dimension (768).
        hidden_dim (int): Internal channel width (default 512).
        n_layers (int): Number of AR residual layers (default 6).
        dilation (int): Dilation factor (default 8).

    Input:   E  — (B, H_p, W_p, D)
    Output:  mu — (B, H_p, W_p, D)
             log_var — (B, H_p, W_p, D)   [log σ²]
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 512,
        n_layers: int = 6,
        dilation: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Input projection: D → hidden_dim
        # First layer uses mask_center=True
        self.input_proj = MaskedConv2d(
            embed_dim, hidden_dim,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
            mask_center=True,
        )
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_act = nn.GELU()

        # Subsequent layers: mask_center=False
        self.layers = nn.ModuleList([
            ARResBlock(
                in_channels=hidden_dim,
                hidden_channels=hidden_dim,
                dilation=dilation,
                mask_center=False,
            )
            for _ in range(n_layers - 1)
        ])

        # Output heads: hidden_dim → D for both μ and log σ²
        self.head_mu = nn.Conv2d(hidden_dim, embed_dim, kernel_size=1)
        self.head_log_var = nn.Conv2d(hidden_dim, embed_dim, kernel_size=1)

    def forward(self, E: torch.Tensor):
        """
        Args:
            E: (B, H_p, W_p, D)

        Returns:
            mu:      (B, H_p, W_p, D)
            log_var: (B, H_p, W_p, D)  — log σ²
        """
        # Permute to (B, D, H_p, W_p) for conv layers
        x = E.permute(0, 3, 1, 2).contiguous()

        # Input layer (mask_center=True)
        x = self.input_proj(x)
        x = x.permute(0, 2, 3, 1)
        x = self.input_norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.input_act(x)

        # Residual layers
        for layer in self.layers:
            x = layer(x)

        # Output heads
        mu = self.head_mu(x).permute(0, 2, 3, 1)           # (B, H_p, W_p, D)
        log_var = self.head_log_var(x).permute(0, 2, 3, 1)  # (B, H_p, W_p, D)

        # Clamp log_var for numerical stability: σ² ∈ [e^-10, e^10]
        log_var = torch.clamp(log_var, min=-3.0, max=10.0)

        return mu, log_var

    def nll(self, E: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token NLL under the diagonal Gaussian.

        A_{i,j} = 0.5 * sum_d [ (E_d - μ_d)² / σ²_d + log σ²_d ]

        Args:
            E: (B, H_p, W_p, D)

        Returns:
            scores: (B, H_p, W_p) — per-token NLL
        """
        mu, log_var = self.forward(E)
        var = torch.exp(log_var)
        nll = 0.5 * ((E - mu) ** 2 / var + log_var)  # (B, H_p, W_p, D)
        return nll.sum(dim=-1)                         # (B, H_p, W_p)

    def clip_score(self, E: torch.Tensor) -> torch.Tensor:
        """
        Clip-level anomaly score via max-pooling over token grid.

        Args:
            E: (B, H_p, W_p, D)

        Returns:
            scores: (B,) — one scalar per clip
        """
        token_scores = self.nll(E)                    # (B, H_p, W_p)
        B = token_scores.shape[0]
        return token_scores.view(B, -1).max(dim=-1).values  # (B,)


# ---------------------------------------------------------------
#  NLL loss for training
# ---------------------------------------------------------------

def nll_loss(E: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Mean NLL over all tokens and dimensions in the batch.

    L(θ) = mean_{b,i,j,d} [ 0.5 * ((E_d - μ_d)² / σ²_d + log σ²_d) ]

    Args:
        E:       (B, H_p, W_p, D) — target embeddings
        mu:      (B, H_p, W_p, D) — predicted means
        log_var: (B, H_p, W_p, D) — predicted log variances

    Returns:
        Scalar loss.
    """
    var = torch.exp(log_var)
    loss = 0.5 * ((E - mu) ** 2 / var + log_var)
    return loss.mean()


# ---------------------------------------------------------------
#  Receptive field utility
# ---------------------------------------------------------------

def compute_receptive_field(n_layers: int, dilation: int, kernel_size: int = 3) -> dict:
    """
    Compute the theoretical receptive field of the AR CNN.

    Per-layer span: d*(k-1)+1
    Total RF after L layers: L*d*(k-1)+1

    Returns dict with both token count and time/freq in ms/Hz.
    """
    per_layer = dilation * (kernel_size - 1) + 1
    total_tokens = n_layers * dilation * (kernel_size - 1) + 1
    time_ms = total_tokens * 100  # each token = 100ms (stride 10 × 10ms hop)
    return {
        "per_layer_tokens": per_layer,
        "total_tokens": total_tokens,
        "time_ms": time_ms,
        "time_s": time_ms / 1000,
    }


if __name__ == "__main__":
    # Quick sanity check
    rf = compute_receptive_field(n_layers=6, dilation=8)
    print(f"Receptive field: {rf['total_tokens']} tokens = {rf['time_s']:.1f}s")

    model = ARCNN(embed_dim=768, hidden_dim=512, n_layers=6, dilation=8)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"AR CNN parameters: {total_params:,}")

    # Forward pass check — correct grid for 5s BEATs clip: 31×8
    # H_p = floor(498 mel frames / 16 patch) = 31
    # W_p = 128 mel bins / 16 patch = 8
    B, H_p, W_p, D = 2, 31, 8, 768
    E = torch.randn(B, H_p, W_p, D)
    scores = model.clip_score(E)
    print(f"Clip scores shape: {scores.shape}")   # should be (2,)
    print(f"Clip scores: {scores}")