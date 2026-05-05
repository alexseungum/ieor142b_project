"""
model.py
CNN + Transformer encoder for DDR chart generation.

Architecture:
  1. Local CNN   : extracts rhythmic features from mel context windows
  2. Positional encoding
  3. Transformer encoder : attends over the full sequence
  4. Difficulty embedding : injected as a learned offset (like a class token bias)
  5. Step head   : binary logit — is there any arrow at this timestep?
  6. Arrow head  : 4-way multi-label logits — which arrows are active?
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

N_MELS        = 80
CONTEXT_LEN   = 15   # context*2+1 frames per timestep (context=7)
N_DIFFICULTIES = 5   # 0..4


# ─────────────────────────────────────────────
# POSITIONAL ENCODING
# ─────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────
# LOCAL CNN ENCODER
# ─────────────────────────────────────────────

class LocalCNNEncoder(nn.Module):
    """
    Processes a context window of mel frames for each timestep.
    Input:  (B*T, CONTEXT_LEN, N_MELS)  — treat context as sequence, mels as channels
    Output: (B*T, d_model)
    """
    def __init__(self, d_model: int = 256, context_len: int = CONTEXT_LEN, n_mels: int = N_MELS):
        super().__init__()
        self.net = nn.Sequential(
            # (B*T, 1, context_len, n_mels)
            nn.Conv2d(1, 32, kernel_size=(3, 8), padding=(1, 4)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1, 4)),                   # -> (B*T, 32, ctx, n_mels/4)

            nn.Conv2d(32, 64, kernel_size=(3, 8), padding=(1, 4)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((1, 4)),                   # -> (B*T, 64, ctx, n_mels/16)

            nn.Conv2d(64, 128, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),            # -> (B*T, 128, 1, 1)
        )
        self.proj = nn.Linear(128, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, context_len, n_mels)
        x = x.unsqueeze(1)                          # (B*T, 1, context_len, n_mels)
        x = self.net(x)                             # (B*T, 128, 1, 1)
        x = x.flatten(1)                            # (B*T, 128)
        return self.proj(x)                         # (B*T, d_model)


# ─────────────────────────────────────────────
# DIFFICULTY CONDITIONING
# ─────────────────────────────────────────────

class DifficultyEmbedding(nn.Module):
    """
    Learned embedding for difficulty level (0-4).
    Added to every position in the sequence as a global bias.
    """
    def __init__(self, n_difficulties: int = N_DIFFICULTIES, d_model: int = 256):
        super().__init__()
        self.emb = nn.Embedding(n_difficulties, d_model)

    def forward(self, diff: torch.Tensor, T: int) -> torch.Tensor:
        # diff: (B,)
        e = self.emb(diff)          # (B, d_model)
        return e.unsqueeze(1).expand(-1, T, -1)  # (B, T, d_model)


# ─────────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────────

class DDRTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        context_len: int = CONTEXT_LEN,
        n_mels: int = N_MELS,
    ):
        super().__init__()

        self.d_model = d_model

        # 1. Local CNN to encode each timestep's context window
        self.cnn = LocalCNNEncoder(d_model, context_len, n_mels)

        # 2. Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        # 3. Difficulty conditioning
        self.diff_emb = DifficultyEmbedding(N_DIFFICULTIES, d_model)

        # 4. Transformer encoder (BERT-style: full bidirectional attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,    # (B, T, d_model)
            norm_first=True,     # Pre-LN: more stable training (improvement over original)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # 5. Output heads
        self.step_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),    # binary: step or no step
        )
        self.arrow_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 4),    # multi-label: which arrows
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,       # (B, T, context_len, n_mels)
        diff: torch.Tensor,    # (B,)  difficulty level
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            step_logits  : (B, T, 1)  — logit for step placement
            arrow_logits : (B, T, 4)  — logits for each arrow direction
        """
        B, T, C, M = x.shape

        # CNN: process all (B*T) windows independently
        x_flat = x.reshape(B * T, C, M)
        feat = self.cnn(x_flat)              # (B*T, d_model)
        feat = feat.reshape(B, T, self.d_model)  # (B, T, d_model)

        # Add positional encoding
        feat = self.pos_enc(feat)            # (B, T, d_model)

        # Add difficulty embedding
        diff_bias = self.diff_emb(diff, T)   # (B, T, d_model)
        feat = feat + diff_bias

        # Transformer encoder: full self-attention over sequence
        out = self.transformer(feat)         # (B, T, d_model)

        # Heads
        step_logits  = self.step_head(out)   # (B, T, 1)
        arrow_logits = self.arrow_head(out)  # (B, T, 4)

        return step_logits, arrow_logits


# ─────────────────────────────────────────────
# LOSS WITH LABEL SMOOTHING
# ─────────────────────────────────────────────

class DDRLoss(nn.Module):
    """
    Combined loss:
      - Step placement: BCE with label smoothing + positive weight (steps are rare)
      - Arrow selection: BCE with label smoothing (multi-label)
    """
    def __init__(
        self,
        step_pos_weight: float = 5.0,  # upweight positive steps (class imbalance)
        label_smoothing: float = 0.1,
        arrow_weight: float = 1.0,
        step_weight: float = 1.0,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.arrow_weight = arrow_weight
        self.step_weight = step_weight
        self.register_buffer('pos_weight', torch.tensor([step_pos_weight]))

    def smooth(self, target: torch.Tensor) -> torch.Tensor:
        """Apply label smoothing: shift labels away from 0/1."""
        eps = self.label_smoothing
        return target * (1 - eps) + eps * 0.5

    def forward(
        self,
        step_logits:  torch.Tensor,   # (B, T, 1)
        arrow_logits: torch.Tensor,   # (B, T, 4)
        y:            torch.Tensor,   # (B, T, 4) ground truth
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Step target: 1 if any arrow is active
        step_target = (y.sum(-1, keepdim=True) > 0).float()  # (B, T, 1)
        step_target_smooth = self.smooth(step_target)

        step_loss = F.binary_cross_entropy_with_logits(
            step_logits,
            step_target_smooth,
            pos_weight=self.pos_weight.to(step_logits.device),
        )

        # Arrow loss (only on timesteps where a step occurs in ground truth)
        mask = step_target.squeeze(-1).bool()   # (B, T)
        if mask.any():
            arrow_logits_masked = arrow_logits[mask]   # (N_active, 4)
            y_masked = self.smooth(y[mask])            # (N_active, 4)
            arrow_loss = F.binary_cross_entropy_with_logits(arrow_logits_masked, y_masked)
        else:
            arrow_loss = torch.tensor(0.0, device=step_logits.device)

        total = self.step_weight * step_loss + self.arrow_weight * arrow_loss
        return total, step_loss, arrow_loss


# ─────────────────────────────────────────────
# INFERENCE / GENERATION
# ─────────────────────────────────────────────

@torch.no_grad()
def generate_chart(
    model: DDRTransformer,
    X: torch.Tensor,              # (1, T, context_len, n_mels)
    difficulty: int = 2,
    step_threshold: float = 0.5,  # tuning knob: lower = more steps (easier feel)
    device: str = 'cpu',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a chart for a single song.
    Returns:
        step_mask   : (T,) bool — where steps occur
        arrow_preds : (T, 4) int — arrow combination at each active step
    """
    model.eval()
    model.to(device)
    X = X.to(device)
    diff = torch.tensor([difficulty], dtype=torch.long, device=device)

    step_logits, arrow_logits = model(X, diff)
    step_probs  = torch.sigmoid(step_logits).squeeze(-1).squeeze(0).cpu().numpy()  # (T,)
    arrow_probs = torch.sigmoid(arrow_logits).squeeze(0).cpu().numpy()             # (T, 4)

    step_mask   = step_probs > step_threshold
    arrow_preds = (arrow_probs > 0.5).astype(int)
    arrow_preds[~step_mask] = 0  # zero out arrows where no step predicted

    return step_mask, arrow_preds