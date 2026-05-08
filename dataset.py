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

from config import SEQ_LEN as _SEQ_LEN
from utils.data_utils import build_sample, difficulty_to_int


def _process_song(args):
    """Top-level worker function for multiprocessing — processes one song across all difficulties."""
    audio_path, sm_path = args
    samples = []
    for diff_str in ['beginner', 'easy', 'medium', 'hard', 'challenge']:
        try:
            sample = build_sample(audio_path, sm_path, difficulty_filter=diff_str)
            if sample is not None:
                samples.append(sample)
        except Exception:
            pass
    return samples

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
    SEQ_LEN = _SEQ_LEN  # timesteps per training chunk (set in config.py)

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
            # Use a shared base cache (all difficulties, all songs) so audio
            # processing only happens once across all curriculum stages
            base_cache = cache_path.replace(
                os.path.basename(cache_path),
                f'base_{split}.pkl'
            ) if cache_path else None

            if base_cache and os.path.exists(base_cache):
                print(f"Loading base cache from {base_cache}")
                with open(base_cache, 'rb') as f:
                    all_samples = pickle.load(f)
            else:
                all_samples = self._build_from_root(data_root)
                if base_cache:
                    os.makedirs(os.path.dirname(base_cache), exist_ok=True)
                    with open(base_cache, 'wb') as f:
                        pickle.dump(all_samples, f)
                    print(f"Saved base cache to {base_cache}")

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
        self.chunks = []  # list of (X_chunk, y_chunk, subdiv_types_chunk, difficulty)
        for s in selected:
            X, y, st, d = s['X'], s['y'], s['subdiv_types'], s['difficulty']
            T = X.shape[0]
            for start in range(0, T - self.seq_len + 1, self.seq_len // 2):  # 50% overlap
                end = start + self.seq_len
                self.chunks.append((
                    X[start:end],
                    y[start:end],
                    st[start:end],
                    d,
                ))
        print(f"[{split}] {len(self.chunks)} chunks total")

    def _build_from_root(self, data_root: str) -> List[dict]:
        """
        Walk data_root recursively, find all folders containing both a .sm and audio file.
        Processes songs in parallel using multiprocessing for faster cache building.
        """
        from multiprocessing import Pool, cpu_count
        root = Path(data_root)
        audio_exts = {'.ogg', '.mp3', '.wav'}

        # Find every folder that has at least one .sm or .ssc file
        all_sm  = list(root.rglob('*.sm'))
        all_ssc = list(root.rglob('*.ssc'))
        song_dirs = sorted({f.parent for f in all_sm + all_ssc})
        print(f"Found {len(song_dirs)} song directories ({len(all_sm)} .sm, {len(all_ssc)} .ssc)")

        # Build list of (audio_path, chart_path) pairs
        # Prefer .sm over .ssc if both exist in the same folder
        pairs = []
        for song_dir in song_dirs:
            sm_files    = list(song_dir.glob('*.sm'))
            ssc_files   = list(song_dir.glob('*.ssc'))
            audio_files = [f for f in song_dir.iterdir() if f.suffix.lower() in audio_exts]
            chart_files = sm_files if sm_files else ssc_files
            if chart_files and audio_files:
                pairs.append((str(audio_files[0]), str(chart_files[0])))

        # Process in parallel — use min(8, cpu_count) workers
        n_workers = min(8, cpu_count())
        print(f"Processing {len(pairs)} songs with {n_workers} workers...")

        samples = []
        with Pool(n_workers) as pool:
            results = pool.map(_process_song, pairs)
        for song_samples in results:
            samples.extend(song_samples)

        print(f"Built {len(samples)} song-difficulty samples")
        return samples

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        X, y, st, d = self.chunks[idx]
        return (
            torch.from_numpy(X),
            torch.from_numpy(y),
            torch.from_numpy(st),
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
    ) if len(train_ds) > 0 else None
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    ) if len(val_ds) > 0 else None
    return train_loader, val_loader