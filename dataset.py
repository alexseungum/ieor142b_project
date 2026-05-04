"""
dataset.py
PyTorch Dataset for DDR chart generation with curriculum learning support.
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from utils.data_utils import build_sample, difficulty_to_int

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class DDRDataset(Dataset):
    """
    Each item: a single song's full sequence, chunked into fixed-length windows
    for batching. Returns:
        x      : (SEQ_LEN, context*2+1, N_MELS)  float32
        y      : (SEQ_LEN, 4)                     float32
        diff   : ()                               int64  (scalar)
    """
    SEQ_LEN = 256   # timesteps per training chunk

    def __init__(
        self,
        data_root: str,
        cache_path: Optional[str] = None,
        max_difficulty: int = 4,   # for curriculum: only include charts <= this level
        split: str = 'train',
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        self.seq_len = self.SEQ_LEN
        self.max_difficulty = max_difficulty

        if cache_path and os.path.exists(cache_path):
            print(f"Loading dataset cache from {cache_path}")
            with open(cache_path, 'rb') as f:
                all_samples = pickle.load(f)
        else:
            all_samples = self._build_from_root(data_root)
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump(all_samples, f)
                print(f"Saved dataset cache to {cache_path}")

        # Train/val split by song (not by chunk) to avoid leakage
        rng = np.random.default_rng(seed)
        n = len(all_samples)
        idx = rng.permutation(n)
        n_val = max(1, int(n * val_fraction))
        if split == 'val':
            selected = [all_samples[i] for i in idx[:n_val]]
        else:
            selected = [all_samples[i] for i in idx[n_val:]]

        # Filter by difficulty for curriculum learning
        selected = [s for s in selected if s['difficulty'] <= max_difficulty]
        print(f"[{split}] {len(selected)} songs after difficulty filter (<={max_difficulty})")

        # Chunk into fixed-length windows
        self.chunks = []  # list of (X_chunk, y_chunk, difficulty)
        for s in selected:
            X, y, d = s['X'], s['y'], s['difficulty']
            T = X.shape[0]
            for start in range(0, T - self.seq_len + 1, self.seq_len // 2):  # 50% overlap
                end = start + self.seq_len
                self.chunks.append((
                    X[start:end],
                    y[start:end],
                    d,
                ))
        print(f"[{split}] {len(self.chunks)} chunks total")

    def _build_from_root(self, data_root: str) -> List[dict]:
        """
        Walk data_root, find all pairs of (audio, .sm) files.
        Expects structure: data_root/<song_name>/<song_name>.sm + audio file
        """
        samples = []
        root = Path(data_root)
        audio_exts = {'.ogg', '.mp3', '.wav'}

        song_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        print(f"Found {len(song_dirs)} song directories")

        for song_dir in song_dirs:
            sm_files = list(song_dir.glob('*.sm'))
            audio_files = [f for f in song_dir.iterdir() if f.suffix.lower() in audio_exts]

            if not sm_files or not audio_files:
                continue

            sm_path = str(sm_files[0])
            audio_path = str(audio_files[0])

            # Build one sample per difficulty level present
            for diff_str in ['beginner', 'easy', 'medium', 'hard', 'challenge']:
                sample = build_sample(audio_path, sm_path, difficulty_filter=diff_str)
                if sample is not None:
                    samples.append(sample)

        print(f"Built {len(samples)} song-difficulty samples")
        return samples

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        X, y, d = self.chunks[idx]
        return (
            torch.from_numpy(X),
            torch.from_numpy(y),
            torch.tensor(d, dtype=torch.long),
        )


def get_curriculum_loaders(
    data_root: str,
    cache_dir: str = 'data/cache',
    batch_size: int = 32,
    num_workers: int = 2,
    curriculum_stage: int = 4,   # 0=beginner only, 4=all difficulties
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader) for a given curriculum stage.
    Stage 0: only beginner charts
    Stage k: charts with difficulty <= k
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_train = os.path.join(cache_dir, f'train_diff{curriculum_stage}.pkl')
    cache_val   = os.path.join(cache_dir, f'val_diff{curriculum_stage}.pkl')

    train_ds = DDRDataset(
        data_root, cache_path=cache_train,
        max_difficulty=curriculum_stage, split='train'
    )
    val_ds = DDRDataset(
        data_root, cache_path=cache_val,
        max_difficulty=curriculum_stage, split='val'
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
