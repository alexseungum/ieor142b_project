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

from config import SEQ_LEN as _SEQ_LEN, CONTEXT_FRAMES, N_MELS, VALID_SUBDIV_POSITIONS, SUBDIVISION
from utils.data_utils import build_sample, difficulty_to_int, get_subdiv_type

# Repeating 24-position subdiv pattern for reconstructing old caches
_SUBDIV_PATTERN = np.array(
    [get_subdiv_type(p, SUBDIVISION) for p in VALID_SUBDIV_POSITIONS], dtype=np.int64
)


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
        # Each chunk stores (beat_frames_slice, y_slice, st_slice, mel, difficulty)
        # mel is shared across all chunks of the same song — windows extracted in __getitem__
        self.chunks = []
        for s in selected:
            y, d = s['y'], s['difficulty']
            T = y.shape[0]
            if 'subdiv_types' in s:
                st = s['subdiv_types']
            else:
                reps = (T + len(_SUBDIV_PATTERN) - 1) // len(_SUBDIV_PATTERN)
                st = np.tile(_SUBDIV_PATTERN, reps)[:T]

            if 'beat_frames' in s:
                bf  = s['beat_frames']
                mel = s['mel']          # (N_MELS, T_frames) float16
            else:
                # very old cache with pre-computed X — keep using it directly
                X = s['X']
                for start in range(0, T - self.seq_len + 1, self.seq_len // 2):
                    end = start + self.seq_len
                    self.chunks.append((X[start:end], None, y[start:end], st[start:end], d))
                continue

            for start in range(0, T - self.seq_len + 1, self.seq_len // 2):
                end = start + self.seq_len
                self.chunks.append((bf[start:end], mel, y[start:end], st[start:end], d))
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


def build_cache(data_root: str, cache_dir: str, n_workers: int = 2) -> int:
    """
    Build base_train.pkl and base_val.pkl in cache_dir if they don't already exist.
    Returns number of samples cached (0 if already existed).
    """
    os.makedirs(cache_dir, exist_ok=True)
    base_train = os.path.join(cache_dir, 'base_train.pkl')
    base_val   = os.path.join(cache_dir, 'base_val.pkl')

    if os.path.exists(base_train) and os.path.exists(base_val):
        with open(base_train, 'rb') as f:
            n = len(pickle.load(f))
        print(f"Cache already exists — skipping build.")
        print(f"  base_train.pkl : {os.path.getsize(base_train)/1e9:.2f} GB  ({n} samples)")
        print(f"  base_val.pkl   : {os.path.getsize(base_val)/1e9:.2f} GB")
        return 0

    root       = Path(data_root)
    audio_exts = {'.ogg', '.mp3', '.wav'}
    all_sm     = list(root.rglob('*.sm'))
    all_ssc    = list(root.rglob('*.ssc'))
    song_dirs  = sorted({f.parent for f in all_sm + all_ssc})

    pairs = []
    for song_dir in song_dirs:
        sm_f  = list(song_dir.glob('*.sm'))
        ssc_f = list(song_dir.glob('*.ssc'))
        audio = [f for f in song_dir.iterdir() if f.suffix.lower() in audio_exts]
        chart = sm_f if sm_f else ssc_f
        if chart and audio:
            pairs.append((str(audio[0]), str(chart[0])))

    print(f"Building cache for {len(pairs)} songs with {n_workers} workers...")
    print("This takes ~10-20 min on first run.")

    from multiprocessing import Pool
    samples = []
    failed  = 0
    with Pool(n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_process_song, pairs)):
            samples.extend(result)
            if not result:
                failed += 1
            if (i + 1) % 10 == 0 or (i + 1) == len(pairs):
                print(f"  [{i+1}/{len(pairs)}] {len(samples)} samples built"
                      + (f"  ({failed} skipped)" if failed else ""))

    if failed:
        print(f"\n{failed} songs skipped (corrupt audio/chart or no dance-single chart)")

    print("Saving to Drive...")
    with open(base_train, 'wb') as f:
        pickle.dump(samples, f)
    with open(base_val, 'wb') as f:
        pickle.dump(samples, f)
    print(f"Saved: {os.path.getsize(base_train)/1e9:.2f} GB per file  ({len(samples)} samples)")
    return len(samples)

    def __getitem__(self, idx):
        bf, mel, y, st, d = self.chunks[idx]
        if mel is not None:
            # vectorized window extraction: pad once, index all at once
            ctx = CONTEXT_FRAMES
            mel_f32 = np.pad(mel.astype(np.float32), ((0, 0), (ctx, ctx)))
            col_idx = bf[:, None] + np.arange(-ctx, ctx + 1)[None, :]  # (T, 15)
            X = mel_f32[:, col_idx].transpose(1, 2, 0)                  # (T, 15, N_MELS)
        else:
            X = bf  # old-cache path: bf holds pre-computed X
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