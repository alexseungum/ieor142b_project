"""
model.py
CNN + Transformer encoder-decoder for DDR chart generation.

Architecture:
  Encoder:
    1. Local CNN   : extracts rhythmic features from mel context windows
    2. Positional encoding
    3. Transformer encoder : attends over the full sequence
    4. Difficulty + subdivision embeddings
  Decoder:
    5. Arrow embedding : project previous arrow combo into d_model (teacher-forced)
    6. Causal self-attention with 16-step lookback window
    7. Cross-attention over encoder audio features
    8. Step head + Arrow head : jointly predict step placement and directions
"""

import math
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import N_MELS, CONTEXT_LEN, N_DIFFICULTIES, N_SUBDIV_TYPES


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


class SubdivisionEmbedding(nn.Module):
    """
    Learned embedding for subdivision type (0=4th, 1=8th, 2=12th, 3=16th).
    Added per-timestep so the model knows whether each slot is a triplet or not.
    """
    def __init__(self, n_types: int = N_SUBDIV_TYPES, d_model: int = 256):
        super().__init__()
        self.emb = nn.Embedding(n_types, d_model)

    def forward(self, subdiv_types: torch.Tensor) -> torch.Tensor:
        # subdiv_types: (B, T)
        return self.emb(subdiv_types)  # (B, T, d_model)


# ─────────────────────────────────────────────
# DECODER COMPONENTS
# ─────────────────────────────────────────────

class ArrowEmbedding(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.proj = nn.Linear(4, d_model)
        self.start_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, arrows: torch.Tensor) -> torch.Tensor:
        # arrows: (B, T, 4) — shift right by 1, prepend start token
        embedded = self.proj(arrows)  # (B, T, d_model)
        start = self.start_token.expand(arrows.size(0), -1, -1)
        return torch.cat([start, embedded[:, :-1, :]], dim=1)  # (B, T, d_model)


def make_windowed_causal_mask(T: int, window: int = 16, device: str = 'cpu') -> torch.Tensor:
    causal  = torch.triu(torch.ones(T, T), diagonal=1).bool()
    lookback = torch.tril(torch.ones(T, T), diagonal=-(window + 1)).bool()
    return (causal | lookback).to(device)


class ArrowDecoderLayer(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 4, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.norm3   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor,
                self_attn_mask: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, attn_mask=self_attn_mask)
        x = residual + self.dropout(x)

        residual = x
        x = self.norm2(x)
        x, _ = self.cross_attn(x, encoder_out, encoder_out)
        x = residual + self.dropout(x)

        residual = x
        x = self.norm3(x)
        x = self.ff(x)
        x = residual + self.dropout(x)
        return x


class ArrowDecoder(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 512, dropout: float = 0.1, window: int = 16):
        super().__init__()
        self.window = window
        self.arrow_embedding = ArrowEmbedding(d_model)
        self.layers = nn.ModuleList([
            ArrowDecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.step_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.arrow_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 4)
        )

    def forward(self, arrows_gt: torch.Tensor,
                encoder_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # arrows_gt: (B, T, 4)  encoder_out: (B, T, d_model)
        B, T, _ = encoder_out.shape
        x = self.arrow_embedding(arrows_gt)
        mask = make_windowed_causal_mask(T, self.window, encoder_out.device)
        for layer in self.layers:
            x = layer(x, encoder_out, mask)
        x = self.norm(x)
        return self.step_head(x), self.arrow_head(x)  # (B,T,1), (B,T,4)


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
        decoder_layers: int = 2,
        decoder_heads: int = 4,
        decoder_window: int = 16,
    ):
        super().__init__()

        self.d_model = d_model

        # 1. Local CNN to encode each timestep's context window
        self.cnn = LocalCNNEncoder(d_model, context_len, n_mels)

        # 2. Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        # 3. Difficulty conditioning
        self.diff_emb   = DifficultyEmbedding(N_DIFFICULTIES, d_model)
        self.subdiv_emb = SubdivisionEmbedding(N_SUBDIV_TYPES, d_model)

        # 4. Transformer encoder (BERT-style: full bidirectional attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # 5. Autoregressive decoder — owns step and arrow prediction
        self.decoder = ArrowDecoder(
            d_model=d_model,
            n_heads=decoder_heads,
            n_layers=decoder_layers,
            d_ff=dim_feedforward,
            dropout=dropout,
            window=decoder_window,
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self,
        x: torch.Tensor,            # (B, T, context_len, n_mels)
        diff: torch.Tensor,         # (B,)
        subdiv_types: torch.Tensor, # (B, T)
    ) -> torch.Tensor:
        """Run CNN + transformer encoder; return (B, T, d_model)."""
        B, T, C, M = x.shape
        feat = self.cnn(x.reshape(B * T, C, M)).reshape(B, T, self.d_model)
        feat = self.pos_enc(feat)
        feat = feat + self.diff_emb(diff, T)
        feat = feat + self.subdiv_emb(subdiv_types)
        return self.transformer(feat)

    def forward(
        self,
        x: torch.Tensor,            # (B, T, context_len, n_mels)
        diff: torch.Tensor,         # (B,)
        subdiv_types: torch.Tensor, # (B, T)
        arrows_gt: torch.Tensor,    # (B, T, 4) ground-truth arrows for teacher forcing
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Teacher-forced forward pass.
        Returns:
            step_logits  : (B, T, 1)
            arrow_logits : (B, T, 4)
        """
        encoder_out = self.encode(x, diff, subdiv_types)
        return self.decoder(arrows_gt, encoder_out)


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
    X: torch.Tensor,                    # (1, T, context_len, n_mels)
    subdiv_types: torch.Tensor,         # (1, T) subdivision types
    difficulty: int = 2,
    step_threshold: float = 0.5,
    device: str = 'cpu',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Autoregressively generate a chart for a single song.
    Returns:
        step_mask   : (T,) bool — where steps occur
        arrow_preds : (T, 4) int — arrow combination at each active step
    """
    model.eval()
    model.to(device)
    X = X.to(device)
    subdiv_types = subdiv_types.to(device)
    diff = torch.tensor([difficulty], dtype=torch.long, device=device)

    encoder_out = model.encode(X, diff, subdiv_types)  # (1, T, d_model)
    B, T, _ = encoder_out.shape
    arrows   = torch.zeros(B, T, 4, device=device)
    step_mask = np.zeros(T, dtype=bool)

    for t in range(T):
        step_logits, arrow_logits = model.decoder(arrows, encoder_out)
        has_step = (torch.sigmoid(step_logits[:, t, 0]) > step_threshold).item()
        step_mask[t] = has_step
        if has_step:
            arrow_prob = torch.sigmoid(arrow_logits[:, t, :])  # (B, 4)
            predicted = (arrow_prob > step_threshold).float()
            if predicted.sum() == 0:
                predicted[0, arrow_prob[0].argmax()] = 1.0
            arrows[:, t, :] = predicted

    arrow_preds = arrows.squeeze(0).long().cpu().numpy()  # (T, 4)
    arrow_preds[~step_mask] = 0
    return step_mask, arrow_preds