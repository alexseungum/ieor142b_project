"""
data_utils.py
Utilities for parsing .sm files and extracting audio features.
"""

import os
import re
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Audio processing
import librosa
import librosa.display

# ─────────────────────────────────────────────
# SM FILE PARSING
# ─────────────────────────────────────────────

ARROW_COLS = 4  # DDR uses 4 columns: L R U D

def parse_sm_file(sm_path: str) -> Dict:
    """
    Parse a StepMania .sm file.
    Returns dict with metadata and list of charts per difficulty.
    Each chart is a list of measures, each measure a list of beat-rows (strings of '0'/'1').
    """
    with open(sm_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    def get_tag(tag):
        match = re.search(rf'#{tag}:([^;]*);', content, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ''

    bpm_str = get_tag('BPMS')
    offset_str = get_tag('OFFSET')
    title = get_tag('TITLE')

    # Parse BPMs: "beat=bpm,beat=bpm,..."
    bpms = []
    for part in bpm_str.split(','):
        part = part.strip()
        if '=' in part:
            beat, bpm = part.split('=')
            bpms.append((float(beat), float(bpm)))

    offset = float(offset_str) if offset_str else 0.0

    # Parse all NOTES sections
    notes_blocks = re.findall(
        r'#NOTES:\s*(.*?)\s*;',
        content, re.DOTALL | re.IGNORECASE
    )

    charts = []
    for block in notes_blocks:
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        # First 5 lines: chart_type, desc, difficulty, meter, radar
        if len(lines) < 6:
            continue
        chart_type = lines[0].rstrip(':').strip()
        difficulty = lines[2].rstrip(':').strip()
        meter = int(lines[3].rstrip(':').strip()) if lines[3].rstrip(':').strip().isdigit() else 0

        # Remaining lines are the note data
        note_lines = lines[5:]
        measures = []
        current_measure = []
        for line in note_lines:
            if line.startswith('//'):
                continue
            if line == ',':
                measures.append(current_measure)
                current_measure = []
            elif line == ';' or line == '':
                if current_measure:
                    measures.append(current_measure)
                break
            else:
                # Keep only the first 4 chars (LDUR columns)
                row = line[:ARROW_COLS].ljust(ARROW_COLS, '0')
                current_measure.append(row)

        charts.append({
            'chart_type': chart_type,
            'difficulty': difficulty,
            'meter': meter,
            'measures': measures,
        })

    return {
        'title': title,
        'bpms': bpms,
        'offset': offset,
        'charts': charts,
    }


def difficulty_to_int(difficulty_str: str) -> int:
    """Map difficulty string to integer 0-4."""
    mapping = {
        'beginner': 0,
        'easy': 1,
        'medium': 2,
        'hard': 3,
        'challenge': 4,
        'edit': 4,
    }
    return mapping.get(difficulty_str.lower(), 2)


def measures_to_timestep_labels(measures: List[List[str]], subdivision: int = 16) -> np.ndarray:
    """
    Convert measure/row representation to a flat array of shape (T, 4).
    Each row is a binary vector indicating which arrows are active.
    Resamples each measure to `subdivision` rows (standard: 16th notes = 16).
    """
    rows = []
    for measure in measures:
        n = len(measure)
        # Resample to subdivision rows using nearest-neighbor
        indices = np.round(np.linspace(0, n - 1, subdivision)).astype(int)
        for idx in indices:
            row_str = measure[idx] if idx < len(measure) else '0000'
            # Convert '1','2','4' -> active arrow; '0' -> inactive
            vec = np.array([0 if c == '0' else 1 for c in row_str[:4]], dtype=np.float32)
            rows.append(vec)
    return np.array(rows)  # (T, 4)


# ─────────────────────────────────────────────
# AUDIO FEATURE EXTRACTION
# ─────────────────────────────────────────────

SR = 22050          # sample rate
HOP_LENGTH = 512    # hop size for STFT
N_MELS = 80         # mel bands
N_FFT = 2048

def load_audio(audio_path: str, sr: int = SR) -> Tuple[np.ndarray, int]:
    y, sr_ = librosa.load(audio_path, sr=sr, mono=True)
    return y, sr_


def extract_mel_spectrogram(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Returns mel spectrogram of shape (N_MELS, T_frames).
    Log-scaled and normalized to zero mean unit variance.
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_S = librosa.power_to_db(S, ref=np.max)
    # Normalize
    log_S = (log_S - log_S.mean()) / (log_S.std() + 1e-8)
    return log_S  # (N_MELS, T_frames)


def frames_to_beats(bpms: List[Tuple[float, float]], offset: float,
                    n_frames: int, sr: int = SR, hop_length: int = HOP_LENGTH,
                    subdivision: int = 16) -> np.ndarray:
    """
    Map each audio frame index to the nearest beat-subdivision index.
    Returns array of shape (n_frames,) with beat-subdivision indices.
    """
    frame_times = librosa.frames_to_time(
        np.arange(n_frames), sr=sr, hop_length=hop_length
    )
    # Build a time -> beat map using BPM changes
    beat_times = []
    current_time = -offset
    current_beat = 0.0
    for i, (beat, bpm) in enumerate(bpms):
        next_beat = bpms[i + 1][0] if i + 1 < len(bpms) else None
        spb = 60.0 / bpm  # seconds per beat
        if next_beat is not None:
            end_time = current_time + (next_beat - beat) * spb
            t = current_time
            b = current_beat
            while b < next_beat:
                beat_times.append((t, b))
                t += spb / subdivision
                b += 1.0 / subdivision
            current_time = end_time
            current_beat = next_beat
        else:
            t = current_time
            b = current_beat
            max_beats = current_beat + (frame_times[-1] - current_time) / spb + 8
            while b < max_beats:
                beat_times.append((t, b))
                t += spb / subdivision
                b += 1.0 / subdivision

    beat_times = np.array(beat_times)  # (N_beats, 2): [time, beat_idx]
    # For each frame time, find nearest beat subdivision
    frame_beat_idx = np.searchsorted(beat_times[:, 0], frame_times, side='left')
    frame_beat_idx = np.clip(frame_beat_idx, 0, len(beat_times) - 1)
    return frame_beat_idx


# ─────────────────────────────────────────────
# DATASET BUILDER
# ─────────────────────────────────────────────

CONTEXT_FRAMES = 7   # frames of context on each side of current frame

def build_sample(
    audio_path: str,
    sm_path: str,
    difficulty_filter: Optional[str] = None,
    subdivision: int = 16,
    context: int = CONTEXT_FRAMES,
) -> Optional[Dict]:
    """
    Build (X, y, difficulty_level) arrays for one song.
    X: (T, context*2+1, N_MELS) — mel context windows per timestep
    y: (T, 4) — binary arrow labels
    difficulty: int scalar
    """
    try:
        sm_data = parse_sm_file(sm_path)
        y_audio, sr = load_audio(audio_path)
        mel = extract_mel_spectrogram(y_audio, sr)  # (N_MELS, T_frames)
    except Exception as e:
        print(f"  [skip] {sm_path}: {e}")
        return None

    # Pick chart
    chart = None
    for c in sm_data['charts']:
        if c['chart_type'].lower() in ('dance-single', 'dance single'):
            if difficulty_filter is None or c['difficulty'].lower() == difficulty_filter.lower():
                chart = c
                break
    if chart is None:
        return None

    labels = measures_to_timestep_labels(chart['measures'], subdivision)  # (T_beats, 4)
    T_beats = len(labels)
    T_frames = mel.shape[1]

    # Build context windows: for each beat-subdivision step, take ±context frames
    # We first downsample mel to T_beats frames by uniform sampling
    frame_indices = np.round(np.linspace(0, T_frames - 1, T_beats)).astype(int)

    X_list = []
    for fi in frame_indices:
        lo = max(0, fi - context)
        hi = min(T_frames - 1, fi + context)
        window = mel[:, lo:hi + 1]  # (N_MELS, window_len)
        # Pad if at edges
        pad_l = context - (fi - lo)
        pad_r = context - (hi - fi)
        if pad_l > 0:
            window = np.concatenate([np.zeros((N_MELS, pad_l)), window], axis=1)
        if pad_r > 0:
            window = np.concatenate([window, np.zeros((N_MELS, pad_r))], axis=1)
        X_list.append(window.T)  # (window_len, N_MELS)

    X = np.stack(X_list, axis=0)  # (T_beats, context*2+1, N_MELS)
    diff_int = difficulty_to_int(chart['difficulty'])

    return {
        'X': X.astype(np.float32),
        'y': labels.astype(np.float32),
        'difficulty': diff_int,
        'title': sm_data['title'],
    }
