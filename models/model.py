"""
model.py
CNN + Transformer encoder-decoder for DDR chart generation.

Architecture:
  Encoder (bidirectional, full chunk at once):
    1. Local CNN              : 2D conv over mel context window (15 frames × 80 mels) → d_model=512
    2. Sinusoidal positional encoding
    3. Difficulty embedding + subdivision embedding (4th/8th/12th/16th note type)
    4. Arrow history embedding : right-shifted GT arrows via Linear(4→512), 10% token dropout
    5. 4-layer bidirectional transformer (d_model=512, 8 heads, d_ff=1024)
    6. Step head              : Linear(512→64)→ReLU→Linear(64→1) → step placement logits

  Decoder (autoregressive, causal):
    7. StepContextEmbedding   : right-shifted arrows + delta since last step + beat phase → d_model=512
                                10% token dropout
    8. 2-layer causal transformer with full-sequence causal self-attention + cross-attention to encoder
    9. Arrow head             : Linear(512→64)→ReLU→Linear(64→16) → 16-class combo logits

  Loss:
    - Step : BCE with label smoothing + dynamic pos_weight
    - Arrow: 16-class cross-entropy with per-batch inverse-frequency class weights

  Inference:
    Overlapping chunks (size=768, stride=384). Encoder runs once per chunk with accumulated
    arrow history. Decoder generates arrows autoregressively token by token.
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
from config import N_MELS, CONTEXT_LEN, N_DIFFICULTIES, N_SUBDIV_TYPES, N_VALID_PER_MEASURE, SEQ_LEN


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

def compute_delta_subdiv(arrows: torch.Tensor, max_delta: int = 48) -> torch.Tensor:
    """For each timestep t, compute valid-position distance since the last step."""
    B, T, _ = arrows.shape
    has_step = (arrows.sum(-1) > 0)
    t_idx = torch.arange(T, device=arrows.device)
    step_idx = torch.where(
        has_step,
        t_idx.unsqueeze(0).expand(B, -1),
        torch.full((B, T), -1, device=arrows.device, dtype=torch.long),
    )
    last_step_pos = torch.cummax(step_idx, dim=1).values       # (B, T)
    delta = (t_idx.unsqueeze(0) - last_step_pos).clamp(0, max_delta)
    return delta.long()


class StepContextEmbedding(nn.Module):
    """
    Replaces ArrowEmbedding. Embeds three signals:
      - arrow_proj : the arrow combination at t-1 (right-shifted)
      - delta_emb  : valid-position distance since last step (right-shifted)
      - phase_emb  : which of 24 valid positions within the measure (NOT shifted)
    beat_phase gives each decoder position a unique cross-attention query even
    when the entire arrow history is zeros, fixing the all-zeros collapse at inference.
    """
    def __init__(self, d_model: int = 256, max_delta: int = 48):
        super().__init__()
        self.arrow_proj  = nn.Linear(4, d_model)
        self.delta_emb   = nn.Embedding(max_delta + 1, d_model)
        self.phase_emb   = nn.Embedding(N_VALID_PER_MEASURE, d_model)
        self.start_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.max_delta   = max_delta

    def forward(self, arrows: torch.Tensor) -> torch.Tensor:
        # arrows: (B, T, 4) unshifted
        B, T, _ = arrows.shape

        # Arrow history + rhythmic spacing — right-shifted
        delta        = compute_delta_subdiv(arrows, self.max_delta)       # (B, T)
        hist         = self.arrow_proj(arrows) + self.delta_emb(delta)    # (B, T, d_model)
        start        = self.start_token.expand(B, -1, -1)
        hist_shifted = torch.cat([start, hist[:, :-1, :]], dim=1)         # (B, T, d_model)

        # Beat phase at current position — NOT shifted
        phase     = torch.arange(T, device=arrows.device) % N_VALID_PER_MEASURE
        phase_emb = self.phase_emb(phase.unsqueeze(0).expand(B, -1))      # (B, T, d_model)

        return hist_shifted + phase_emb


def make_windowed_causal_mask(T: int, window: int = None, device: str = 'cpu') -> torch.Tensor:
    causal = torch.triu(torch.ones(T, T), diagonal=1).bool()
    if window is None:
        return causal.to(device)
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
                 d_ff: int = 512, dropout: float = 0.1, window: int = None,
                 token_dropout: float = 0.1):
        super().__init__()
        self.window = window
        self.token_dropout = token_dropout
        self.arrow_embedding = StepContextEmbedding(d_model)
        self.layers = nn.ModuleList([
            ArrowDecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.arrow_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 16)
        )

    def forward(self, arrows_gt: torch.Tensor,
                encoder_out: torch.Tensor) -> torch.Tensor:
        # arrows_gt: (B, T, 4)  encoder_out: (B, T, d_model)
        B, T, _ = encoder_out.shape
        if self.training and self.token_dropout > 0:
            # with prob token_dropout, wipe the entire arrow history for the sequence
            drop = (torch.rand(B, 1, 1, device=arrows_gt.device) < self.token_dropout).float()
            arrows_gt = arrows_gt * (1.0 - drop)
        x = self.arrow_embedding(arrows_gt)
        mask = make_windowed_causal_mask(T, self.window, encoder_out.device)
        for layer in self.layers:
            x = layer(x, encoder_out, mask)
        x = self.norm(x)
        return self.arrow_head(x)  # (B,T,4)


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
        decoder_window: int = None,
        token_dropout: float = 0.1,
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

        self.arrow_hist_emb = nn.Linear(4, d_model)

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

        # 5. Step prediction head — on encoder output (bidirectional audio context)
        self.step_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        # 6. Autoregressive decoder — arrow prediction only
        self.decoder = ArrowDecoder(
            d_model=d_model,
            n_heads=decoder_heads,
            n_layers=decoder_layers,
            d_ff=dim_feedforward,
            dropout=dropout,
            window=decoder_window,
            token_dropout=token_dropout,
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self,
        x: torch.Tensor,                        # (B, T, context_len, n_mels)
        diff: torch.Tensor,                     # (B,)
        subdiv_types: torch.Tensor,             # (B, T)
        arrows_shifted: torch.Tensor = None,    # (B, T, 4) right-shifted arrow history
    ) -> torch.Tensor:
        """Run CNN + transformer encoder; return (B, T, d_model)."""
        B, T, C, M = x.shape
        feat = self.cnn(x.reshape(B * T, C, M)).reshape(B, T, self.d_model)
        feat = self.pos_enc(feat)
        feat = feat + self.diff_emb(diff, T)
        feat = feat + self.subdiv_emb(subdiv_types)
        if arrows_shifted is not None:
            if self.training:
                drop = (torch.rand(arrows_shifted.shape[0], 1, 1, device=arrows_shifted.device) < 0.1).float()
                arrows_shifted = arrows_shifted * (1.0 - drop)
            feat = feat + self.arrow_hist_emb(arrows_shifted)
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
        # Right-shift arrows so encoder at position t sees arrow from t-1
        B, T, _ = arrows_gt.shape
        start = torch.zeros(B, 1, 4, device=arrows_gt.device)
        arrows_shifted = torch.cat([start, arrows_gt[:, :-1, :]], dim=1)

        encoder_out  = self.encode(x, diff, subdiv_types, arrows_shifted)
        step_logits  = self.step_head(encoder_out)          # (B, T, 1)
        arrow_logits = self.decoder(arrows_gt, encoder_out) # (B, T, 4)
        return step_logits, arrow_logits


# ─────────────────────────────────────────────
# LOSS WITH LABEL SMOOTHING
# ─────────────────────────────────────────────

class DDRLoss(nn.Module):
    """
    Combined loss:
      - Step placement: BCE with label smoothing + positive weight (steps are rare)
      - Arrow selection: 16-class cross-entropy over all arrow combinations
    """
    def __init__(
        self,
        label_smoothing: float = 0.1,
        arrow_weight: float = 1.0,
        step_weight: float = 1.0,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.arrow_weight = arrow_weight
        self.step_weight = step_weight

    def smooth(self, target: torch.Tensor) -> torch.Tensor:
        eps = self.label_smoothing
        return target * (1 - eps) + eps * 0.5

    def forward(
        self,
        step_logits:  torch.Tensor,   # (B, T, 1)
        arrow_logits: torch.Tensor,   # (B, T, 16)
        y:            torch.Tensor,   # (B, T, 4) ground truth
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Step target: 1 if any arrow is active
        step_target = (y.sum(-1, keepdim=True) > 0).float()  # (B, T, 1)

        density    = step_target.mean().clamp(0.01, 0.99)
        pos_weight = ((1 - density) / density).to(step_logits.device)

        step_loss = F.binary_cross_entropy_with_logits(
            step_logits,
            self.smooth(step_target),
            pos_weight=pos_weight,
        )

        # Arrow loss: 16-class cross-entropy only at active step positions
        mask = step_target.squeeze(-1).bool()   # (B, T)
        if mask.any():
            bits = torch.tensor([8, 4, 2, 1], device=y.device, dtype=torch.float32)
            arrow_targets = (y[mask] * bits).sum(-1).long()    # (N_active,) in [1,15]
            # Inverse-frequency class weights so rare combos (jumps, hands) aren't ignored
            counts = torch.bincount(arrow_targets, minlength=16).float().clamp(min=1)
            class_weights = (1.0 / counts)
            class_weights = class_weights / class_weights.sum() * 16  # keep scale stable
            arrow_loss = F.cross_entropy(arrow_logits[mask], arrow_targets, weight=class_weights)
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
    temperature: float = 1.0,           # >1 = more diverse arrows, <1 = sharper/greedier
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Autoregressively generate a chart using 50%-overlapping encoder chunks (STRIDE=SEQ_LEN//2).
    Each chunk's encoder sees SEQ_LEN timesteps; only the first STRIDE positions are emitted
    as output. Arrow history is carried across chunks to warm up the decoder.
    Returns:
        step_mask   : (T_out,) bool  — where steps occur
        arrow_preds : (T_out, 4) int — arrow combination at each active step
        step_probs  : (T_out,) float — raw step probability at every timestep
    """
    model.eval()
    model.to(device)
    X = X.to(device)
    subdiv_types = subdiv_types.to(device)
    diff = torch.tensor([difficulty], dtype=torch.long, device=device)

    T      = X.shape[1]
    STRIDE = SEQ_LEN // 2    # 384
    n_chunks = T // STRIDE
    T_out    = n_chunks * STRIDE

    step_mask   = np.zeros(T_out, dtype=bool)
    step_probs  = np.zeros(T_out, dtype=np.float32)
    arrows_np   = np.zeros((T_out, 4), dtype=np.float32)
    arrow_preds = np.zeros((T_out, 4), dtype=np.int64)

    for chunk_idx in range(n_chunks):
        if chunk_idx == 0:
            enc_start, enc_end = 0, SEQ_LEN
        else:
            enc_start = (chunk_idx - 1) * STRIDE
            enc_end   = enc_start + SEQ_LEN

        if enc_end > T:
            break

        X_c  = X[:, enc_start:enc_end, :, :]
        st_c = subdiv_types[:, enc_start:enc_end]

        arrows = torch.zeros(1, SEQ_LEN, 4, device=device)

        if chunk_idx == 0:
            gen_start_pos = 0
            gen_end_pos   = STRIDE
        else:
            prev_s = (chunk_idx - 1) * STRIDE
            prev_e = chunk_idx * STRIDE
            arrows[0, :STRIDE, :] = torch.from_numpy(arrows_np[prev_s:prev_e]).to(device)
            gen_start_pos = STRIDE
            gen_end_pos   = SEQ_LEN

        # Re-encode with arrow history so encoder matches training conditioning
        start_tok = torch.zeros(1, 1, 4, device=device)
        arrows_shifted = torch.cat([start_tok, arrows[:, :-1, :]], dim=1)
        encoder_out = model.encode(X_c, diff, st_c, arrows_shifted)
        step_logits_chunk = model.step_head(encoder_out)

        for t in range(gen_start_pos, gen_end_pos):
            al       = model.decoder(arrows, encoder_out)
            prob     = torch.sigmoid(step_logits_chunk[:, t, 0]).item()
            has_step = torch.bernoulli(torch.tensor(prob)).bool().item()
            global_t = t if chunk_idx == 0 else (chunk_idx - 1) * STRIDE + t
            step_probs[global_t]  = prob
            step_mask[global_t]   = has_step
            if has_step:
                logits    = al[0, t, :] / temperature           # (16,)
                probs     = torch.softmax(logits, dim=-1)
                combo_idx = torch.multinomial(probs, 1).item()
                if combo_idx == 0:                              # all-zero invalid at step
                    combo_idx = probs[1:].argmax().item() + 1
                bits      = torch.tensor([8, 4, 2, 1], device=device, dtype=torch.long)
                predicted = ((combo_idx & bits) > 0).float()
                arrows[0, t, :] = predicted
                arrows_np[global_t]   = predicted.cpu().numpy()
                arrow_preds[global_t] = predicted.long().cpu().numpy()

    return step_mask, arrow_preds, step_probs