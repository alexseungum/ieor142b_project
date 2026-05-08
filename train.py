"""
train.py
Training script with curriculum learning.

Usage:
    python train.py --data_root data/ddc --epochs_per_stage 20 --batch_size 32

For Colab, use the provided notebook instead (notebooks/train_colab.ipynb).
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_curriculum_loaders
from models.model import DDRTransformer, DDRLoss
from config import (
    D_MODEL, NHEAD, N_LAYERS, D_FF, DROPOUT,
    BATCH_SIZE, LR, WEIGHT_DECAY, EPOCHS_PER_STAGE,
    PATIENCE, POS_WEIGHT, LABEL_SMOOTHING, ARROW_WEIGHT, NUM_WORKERS, CURRICULUM_START,
)


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def compute_f1(step_logits, y, threshold=0.5):
    """Compute step placement F1 score."""
    preds = (torch.sigmoid(step_logits.squeeze(-1)) > threshold).float()
    targets = (y.sum(-1) > 0).float()

    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return f1, precision, recall


def compute_arrow_acc(arrow_logits, y, threshold=0.5):
    """
    Arrow accuracy at timesteps where a step actually occurs (ground truth).
    - exact_match: fraction of active timesteps where all 4 arrows are correct
    """
    mask = (y.sum(-1) > 0)          # (B, T) — only score where steps exist
    if not mask.any():
        return 0.0
    preds   = (torch.sigmoid(arrow_logits[mask]) > threshold)  # (N, 4)
    targets = y[mask].bool()                                    # (N, 4)
    return (preds == targets).all(dim=-1).float().mean().item()


# ─────────────────────────────────────────────
# TRAIN / EVAL LOOPS
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, scaler=None, epoch=1, epochs_per_stage=30):
    model.train()
    total_loss = step_loss_sum = arrow_loss_sum = 0.0
    f1_sum = arrow_exact_sum = 0.0
    n = 0

    # Scheduled sampling: linear ramp 0% → 50% over epochs_per_stage
    ss_ratio = 0.5 * (epoch - 1) / max(epochs_per_stage - 1, 1)

    for X, y, subdiv_types, diff in loader:
        X, y, subdiv_types, diff = X.to(device), y.to(device), subdiv_types.to(device), diff.to(device)
        optimizer.zero_grad()

        # Build mixed arrow input for scheduled sampling
        if ss_ratio > 0:
            with torch.no_grad():
                _, arrow_logits_tf = model(X, diff, subdiv_types, y)
            arrow_preds = (torch.sigmoid(arrow_logits_tf) > 0.5).float()
            # Per-timestep coin flip: True = use model prediction, False = use GT
            use_pred = (torch.rand(y.shape[0], y.shape[1], 1, device=device) < ss_ratio)
            arrows_in = torch.where(use_pred, arrow_preds, y)
        else:
            arrows_in = y

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                step_logits, arrow_logits = model(X, diff, subdiv_types, arrows_in)
                loss, sl, al = criterion(step_logits, arrow_logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            step_logits, arrow_logits = model(X, diff, subdiv_types, arrows_in)
            loss, sl, al = criterion(step_logits, arrow_logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        f1, _, _ = compute_f1(step_logits.detach(), y.detach())
        exact = compute_arrow_acc(arrow_logits.detach(), y.detach())
        total_loss      += loss.item()
        step_loss_sum   += sl.item()
        arrow_loss_sum  += al.item()
        f1_sum          += f1
        arrow_exact_sum += exact
        n += 1

    return {
        'loss':        total_loss / n,
        'step_loss':   step_loss_sum / n,
        'arrow_loss':  arrow_loss_sum / n,
        'f1':          f1_sum / n,
        'arrow_exact': arrow_exact_sum / n,
    }


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, ss_ratio=0.0):
    model.eval()
    total_loss = step_loss_sum = arrow_loss_sum = 0.0
    f1_sum = arrow_exact_sum = 0.0
    n = 0

    for X, y, subdiv_types, diff in loader:
        X, y, subdiv_types, diff = X.to(device), y.to(device), subdiv_types.to(device), diff.to(device)

        if ss_ratio > 0:
            _, arrow_logits_gt = model(X, diff, subdiv_types, y)
            arrow_preds = (torch.sigmoid(arrow_logits_gt) > 0.5).float()
            use_pred = (torch.rand(y.shape[0], y.shape[1], 1, device=device) < ss_ratio)
            arrows_in = torch.where(use_pred, arrow_preds, y)
        else:
            arrows_in = y

        step_logits, arrow_logits = model(X, diff, subdiv_types, arrows_in)
        loss, sl, al = criterion(step_logits, arrow_logits, y)

        f1, _, _ = compute_f1(step_logits, y)
        exact = compute_arrow_acc(arrow_logits, y)
        total_loss      += loss.item()
        step_loss_sum   += sl.item()
        arrow_loss_sum  += al.item()
        f1_sum          += f1
        arrow_exact_sum += exact
        n += 1

    return {
        'loss':        total_loss / n,
        'step_loss':   step_loss_sum / n,
        'arrow_loss':  arrow_loss_sum / n,
        'f1':          f1_sum / n,
        'arrow_exact': arrow_exact_sum / n,
    }


# ─────────────────────────────────────────────
# MAIN TRAINING LOOP WITH CURRICULUM
# ─────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Use AMP only on CUDA
    use_amp = (device.type == 'cuda')
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None

    model = DDRTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.n_layers,
        dim_feedforward=args.d_ff,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    criterion = DDRLoss(
        step_pos_weight=args.pos_weight,
        label_smoothing=args.label_smoothing,
        arrow_weight=args.arrow_weight,
    )

    # ── Curriculum learning: stages 0 → 4 ──────────────────────────────────
    # Stage k: train on charts with difficulty <= k
    # We start from the easiest (beginner) and progressively add harder charts

    all_train_history = []
    all_val_history   = []

    for stage in range(args.curriculum_start, 5):
        print(f"\n{'='*60}")
        print(f"CURRICULUM STAGE {stage}  (difficulty <= {stage})")
        print(f"{'='*60}")

        train_loader, val_loader = get_curriculum_loaders(
            data_root=args.data_root,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            curriculum_stage=stage,
        )

        # Skip stage if dataset has no samples at this difficulty
        # (e.g. pack has no Beginner charts — start from Easy instead)
        if train_loader is None or val_loader is None:
            print(f"  No samples at difficulty <= {stage}, skipping stage.")
            continue

        # Fresh optimizer + scheduler per stage (warm restart)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs_per_stage, eta_min=1e-6)

        patience_counter = 0
        stage_best_score = 0.0
        use_arrow_metric = False  # flips to True once step F1 crosses threshold

        for epoch in range(1, args.epochs_per_stage + 1):
            ss_ratio    = 0.5 * (epoch - 1) / max(args.epochs_per_stage - 1, 1)
            train_stats = train_epoch(model, train_loader, optimizer, criterion, device, scaler, epoch=epoch, epochs_per_stage=args.epochs_per_stage)
            val_stats   = eval_epoch(model, val_loader, criterion, device, ss_ratio=ss_ratio)
            scheduler.step()

            # Switch model selection metric once step placement is good enough
            if not use_arrow_metric and val_stats['f1'] >= args.arrow_metric_threshold:
                use_arrow_metric = True
                stage_best_score = 0.0  # reset so arrow_exact can start fresh
                patience_counter = 0
                print(f"  [metric switch] val step_F1 >= {args.arrow_metric_threshold} — now optimizing arrow_exact")

            score        = val_stats['arrow_exact'] if use_arrow_metric else val_stats['f1']
            metric_label = 'arr_exact' if use_arrow_metric else 'F1'

            print(
                f"  Stage {stage} | Epoch {epoch:3d}/{args.epochs_per_stage} | "
                f"Train loss {train_stats['loss']:.4f} (step {train_stats['step_loss']:.4f}, arrow {train_stats['arrow_loss']:.4f}) "
                f"step_F1 {train_stats['f1']:.4f} arrow_exact {train_stats['arrow_exact']:.4f} | "
                f"Val loss {val_stats['loss']:.4f} step_F1 {val_stats['f1']:.4f} arrow_exact {val_stats['arrow_exact']:.4f}"
            )

            all_train_history.append({'stage': stage, 'epoch': epoch, **train_stats})
            all_val_history.append({'stage': stage, 'epoch': epoch, **val_stats})

            import json as _json
            with open('logs/train_history.json', 'w') as _f:
                _json.dump({'train': all_train_history, 'val': all_val_history}, _f)

            ckpt = {
                'stage': stage,
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_f1': val_stats['f1'],
                'val_arrow_exact': val_stats['arrow_exact'],
                'args': vars(args),
            }

            # Always save per-stage best (epoch 1 always saves since stage_best_score=0)
            if score > stage_best_score:
                stage_best_score = score
                patience_counter = 0
                torch.save(ckpt, os.path.join(args.checkpoint_dir, f'stage{stage}_best.pt'))
                # best_model.pt always reflects the latest stage's best
                torch.save(ckpt, os.path.join(args.checkpoint_dir, 'best_model.pt'))
                print(f"    ✓ Saved stage {stage} best ({metric_label}={score:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
                    break

    print(f"\nTraining complete.")
    print(f"Checkpoints saved to {args.checkpoint_dir}/")


# ─────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train DDR Chart Generator')
    p.add_argument('--data_root',        type=str,   default='data/ddc',       help='Path to unpacked DDC dataset')
    p.add_argument('--cache_dir',        type=str,   default='data/cache',     help='Where to cache processed samples')
    p.add_argument('--checkpoint_dir',   type=str,   default='checkpoints',    help='Where to save model checkpoints')
    p.add_argument('--epochs_per_stage', type=int,   default=EPOCHS_PER_STAGE, help='Training epochs per curriculum stage')
    p.add_argument('--batch_size',       type=int,   default=BATCH_SIZE)
    p.add_argument('--lr',               type=float, default=LR)
    p.add_argument('--weight_decay',     type=float, default=WEIGHT_DECAY)
    p.add_argument('--d_model',          type=int,   default=D_MODEL)
    p.add_argument('--nhead',            type=int,   default=NHEAD)
    p.add_argument('--n_layers',         type=int,   default=N_LAYERS)
    p.add_argument('--d_ff',             type=int,   default=D_FF)
    p.add_argument('--dropout',          type=float, default=DROPOUT)
    p.add_argument('--pos_weight',             type=float, default=POS_WEIGHT,       help='Positive class weight for step BCE loss')
    p.add_argument('--arrow_weight',           type=float, default=ARROW_WEIGHT,     help='Scale arrow loss relative to step loss')
    p.add_argument('--arrow_metric_threshold', type=float, default=0.85,             help='Val step F1 at which to switch early stopping to arrow_exact')
    p.add_argument('--label_smoothing',  type=float, default=LABEL_SMOOTHING)
    p.add_argument('--patience',         type=int,   default=PATIENCE,         help='Early stopping patience within each stage')
    p.add_argument('--num_workers',      type=int,   default=NUM_WORKERS)
    p.add_argument('--curriculum_start', type=int,   default=CURRICULUM_START, help='Start at this difficulty stage (0=beginner)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)