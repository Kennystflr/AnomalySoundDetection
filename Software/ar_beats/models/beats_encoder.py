"""
models/beats_encoder.py
-----------------------
Frozen BEATs wrapper that converts raw audio waveforms into a 2D
token grid E ∈ R^{B × H_p × W_p × D}.

VERIFIED against official BEATs source (microsoft/unilm/beats/BEATs.py):

  INPUT:
    extract_features(source, padding_mask)
      source:       (B, T)   raw waveform at 16000 Hz
      padding_mask: (B, T)   bool, False = valid sample (no padding)

  INTERNAL PREPROCESSING (BEATs handles this — do NOT pre-compute):
    ta_kaldi.fbank(waveform, num_mel_bins=128,
                   sample_frequency=16000,
                   frame_length=25, frame_shift=10)
    → (T_mel, 128) mel filterbank, normalized with fixed
      fbank_mean=15.41663, fbank_std=6.55582

  PATCH EMBEDDING:
    nn.Conv2d(1, embed_dim,
              kernel_size=input_patch_size,
              stride=input_patch_size)   ← NON-OVERLAPPING (stride = patch size)

  OUTPUT:
    extract_features(...)[0] → (B, N, D)   where N = H_p * W_p

CORRECT GRID for a 5s clip at 16000 Hz:
  waveform:  80000 samples
  mel frames: floor((80000 - 400) / 160) + 1 = 498   (frame_length=25ms=400,
                                                         frame_shift=10ms=160)
  patch_size: 16 (non-overlapping, stride=16)
  H_p = floor(498 / 16) = 31   (time)
  W_p = 128 / 16 = 8            (frequency)
  D   = 768
  N   = 31 × 8 = 248

  Each token covers: 16 × 10ms = 160ms in time, 16 mel bins in frequency.

Verified via SpeechBrain example:
  audio (4, 10000) → 61 mel frames → 3×8 = 24 tokens
  outputs.shape == torch.Size([4, 24, 768])  ✓

Download weights:
    https://aka.ms/beats/BEATs_iter3_plus_AS2M.pt
    → place at: checkpoints/BEATs_iter3_plus_AS2M.pt

Install BEATs source:
    git clone https://github.com/microsoft/unilm
    export PYTHONPATH=$PYTHONPATH:$(pwd)/unilm/beats
"""

import torch
import torch.nn as nn
import numpy as np


class BEATsEncoder(nn.Module):
    """
    Wraps BEATs to extract a 2D token grid from raw waveforms.
    Weights are frozen. Optional token-level mean subtraction applied
    after extraction for normalization.

    Args:
        model_path (str):  Path to BEATs checkpoint (.pt).
        device (str):      'cuda' or 'cpu'.
        token_mean:        np.ndarray of shape (D,) or None.
                           Subtracted from all token embeddings when set.
                           Compute with data.dataset.compute_token_mean().
    """

    def __init__(self, model_path: str, device: str = "cuda",
                 token_mean=None):
        super().__init__()
        self.device = device
        self.token_mean_tensor = None

        if token_mean is not None:
            self.token_mean_tensor = torch.tensor(
                token_mean, dtype=torch.float32, device=device
            )

        try:
            from BEATs import BEATs, BEATsConfig
        except ImportError:
            raise ImportError(
                "BEATs source not found.\n"
                "  git clone https://github.com/microsoft/unilm\n"
                "  export PYTHONPATH=$PYTHONPATH:$(pwd)/unilm/beats\n"
                "  wget https://aka.ms/beats/BEATs_iter3_plus_AS2M.pt "
                "-O checkpoints/BEATs_iter3_plus_AS2M.pt"
            )

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        cfg = BEATsConfig(checkpoint["cfg"])
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

        self.embed_dim = cfg.encoder_embed_dim  # 768

        # Read patch size from checkpoint config (non-overlapping: stride=patch)
        # input_patch_size defaults to -1 in BEATsConfig but is overridden
        # by the checkpoint cfg dict. Typically 16 for all released checkpoints.
        self.patch_size = cfg.input_patch_size
        if self.patch_size <= 0:
            self.patch_size = 16  # safe fallback
            print(f"WARNING: input_patch_size={cfg.input_patch_size} in checkpoint "
                  f"cfg, falling back to 16.")

        # Fixed mel grid dimensions for 5s clips
        # W_p is fixed regardless of clip length (128 mel bins / patch_size)
        self.W_p = 128 // self.patch_size  # = 8

        print(f"BEATs loaded | embed_dim={self.embed_dim} | "
              f"patch_size={self.patch_size} | W_p(freq)={self.W_p}")

    def set_token_mean(self, token_mean: np.ndarray):
        """Apply after computing training-set token mean."""
        self.token_mean_tensor = torch.tensor(
            token_mean, dtype=torch.float32, device=self.device
        )

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T) — raw audio at 16000 Hz, float32.
                      For 5s clips: T = 80000.

        Returns:
            tokens: (B, H_p, W_p, D)
                    For 5s clips: (B, 31, 8, 768)
        """
        waveform = waveform.to(self.device)
        B, T = waveform.shape

        # Padding mask: False = valid, shape must match waveform (B, T)
        # (verified from official example: torch.zeros(1, 10000).bool())
        padding_mask = torch.zeros(B, T, dtype=torch.bool, device=self.device)

        # Returns tuple; [0] is the token sequence (B, N, D)
        tokens = self.model.extract_features(
            waveform, padding_mask=padding_mask
        )[0]

        # Reshape (B, N, D) → (B, H_p, W_p, D)
        N = tokens.shape[1]
        H_p = N // self.W_p
        tokens = tokens.view(B, H_p, self.W_p, self.embed_dim)

        # Token-level normalization
        if self.token_mean_tensor is not None:
            tokens = tokens - self.token_mean_tensor

        return tokens  # (B, H_p, W_p, D)