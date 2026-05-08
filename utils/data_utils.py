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

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SR, HOP_LENGTH, N_FFT, N_MELS, CONTEXT_FRAMES, SUBDIVISION, VALID_SUBDIV_POSITIONS

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

    bpm_str    = get_tag('BPMS')
    offset_str = get_tag('OFFSET')
    title      = get_tag('TITLE')

    # Parse BPMs: "beat=bpm,beat=bpm,..."
    bpms = []
    for part in bpm_str.split(','):
        part = part.strip()
        if '=' in part:
            try:
                beat, bpm = part.split('=', 1)
                bpms.append((float(beat.strip()), float(bpm.strip())))
            except ValueError:
                continue

    offset = 0.0
    try:
        offset = float(offset_str) if offset_str else 0.0
    except ValueError:
        pass

    # Parse NOTES blocks directly from raw content so we don't lose the ';' boundary.
    # Each block starts at '#NOTES:' and ends at the next ';' that is NOT inside note rows.
    # We use a raw split approach: find every '#NOTES:' and read until ';'
    charts = []
    for m in re.finditer(r'#NOTES:', content, re.IGNORECASE):
        start = m.end()
        end   = content.find(';', start)
        if end == -1:
            end = len(content)
        block = content[start:end]

        # Split into raw lines, keep blank lines as delimiters (they separate measures in some packs)
        raw_lines = block.split('\n')

        # The first 5 non-empty lines (stripped) are the header fields:
        #   chart_type, description, difficulty, meter, radar
        header = []
        header_indices = []
        for i, line in enumerate(raw_lines):
            stripped = line.strip()
            if stripped and not stripped.startswith('//'):
                header.append(stripped)
                header_indices.append(i)
            if len(header) == 5:
                break

        if len(header) < 5:
            continue

        chart_type = header[0].rstrip(':').strip()
        difficulty = header[2].rstrip(':').strip()
        meter_str  = header[3].rstrip(':').strip()
        meter      = int(meter_str) if meter_str.lstrip('-').isdigit() else 0

        # Everything after the 5th header line is note data
        note_start = header_indices[4] + 1
        note_lines = raw_lines[note_start:]

        measures        = []
        current_measure = []
        for line in note_lines:
            stripped = line.strip()

            # Skip comments
            if stripped.startswith('//'):
                continue

            # Comma = end of measure
            if stripped == ',':
                if current_measure:
                    measures.append(current_measure)
                current_measure = []
                continue

            # Skip empty lines
            if not stripped:
                continue

            # Valid note row: must be 4+ chars of 0/1/2/3/4/M/F/K
            # Reject lines that look like header artifacts
            if len(stripped) >= 4 and re.match(r'^[0-9MFKLmfkl]{4}', stripped):
                row = stripped[:ARROW_COLS].ljust(ARROW_COLS, '0')
                current_measure.append(row)

        # Don't forget the last measure (no trailing comma in some files)
        if current_measure:
            measures.append(current_measure)

        if not measures:
            continue

        charts.append({
            'chart_type': chart_type,
            'difficulty': difficulty,
            'meter':      meter,
            'measures':   measures,
        })

    return {
        'title':  title,
        'bpms':   bpms,
        'offset': offset,
        'charts': charts,
    }


def parse_ssc_file(ssc_path: str) -> Dict:
    """
    Parse a StepMania .ssc file.
    Returns the same dict format as parse_sm_file so the rest of the pipeline is identical.
    .ssc differs from .sm in that each chart is wrapped in a #NOTEDATA: block with
    individual tags (#STEPSTYPE, #DIFFICULTY, #METER, #NOTES) instead of one combined block.
    """
    with open(ssc_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    def get_tag(tag):
        match = re.search(rf'#{tag}:([^;]*);', content, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ''

    title      = get_tag('TITLE')
    bpm_str    = get_tag('BPMS')
    offset_str = get_tag('OFFSET')

    bpms = []
    for part in bpm_str.split(','):
        part = part.strip()
        if '=' in part:
            try:
                beat, bpm = part.split('=', 1)
                bpms.append((float(beat.strip()), float(bpm.strip())))
            except ValueError:
                continue

    offset = 0.0
    try:
        offset = float(offset_str) if offset_str else 0.0
    except ValueError:
        pass

    # Each chart lives in a #NOTEDATA: ... ; block
    charts = []
    for nd_match in re.finditer(r'#NOTEDATA\s*:', content, re.IGNORECASE):
        # Find the extent of this NOTEDATA block (ends at the ; after the #NOTES: data)
        block_start = nd_match.end()
        # Find next #NOTEDATA or end of file
        next_nd = re.search(r'#NOTEDATA\s*:', content[block_start:], re.IGNORECASE)
        block_end = block_start + next_nd.start() if next_nd else len(content)
        block = content[block_start:block_end]

        def get_block_tag(tag):
            m = re.search(rf'#{tag}\s*:([^;]*);', block, re.DOTALL | re.IGNORECASE)
            return m.group(1).strip() if m else ''

        chart_type = get_block_tag('STEPSTYPE')
        difficulty = get_block_tag('DIFFICULTY')
        meter_str  = get_block_tag('METER')
        meter      = int(meter_str) if meter_str.lstrip('-').isdigit() else 0

        if chart_type.lower() not in ('dance-single', 'dance single'):
            continue

        # Extract note rows from #NOTES: tag in this block
        notes_match = re.search(r'#NOTES\s*:([^;]*);', block, re.DOTALL | re.IGNORECASE)
        if not notes_match:
            continue
        note_content = notes_match.group(1)

        measures = []
        current_measure = []
        for line in note_content.split('\n'):
            stripped = line.strip()
            if stripped.startswith('//'):
                continue
            if stripped == ',':
                if current_measure:
                    measures.append(current_measure)
                current_measure = []
                continue
            if not stripped:
                continue
            if len(stripped) >= 4 and re.match(r'^[0-9MFKLmfkl]{4}', stripped):
                row = stripped[:ARROW_COLS].ljust(ARROW_COLS, '0')
                current_measure.append(row)

        if current_measure:
            measures.append(current_measure)
        if not measures:
            continue

        charts.append({
            'chart_type': chart_type,
            'difficulty': difficulty,
            'meter':      meter,
            'measures':   measures,
        })

    return {
        'title':  title,
        'bpms':   bpms,
        'offset': offset,
        'charts': charts,
    }


def parse_chart_file(path: str) -> Dict:
    """Parse either a .sm or .ssc file, dispatching based on extension."""
    if path.lower().endswith('.ssc'):
        return parse_ssc_file(path)
    return parse_sm_file(path)


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


def get_subdiv_type(pos: int, subdivision: int = SUBDIVISION) -> int:
    """Return subdivision type: 0=4th, 1=8th, 2=12th(triplet), 3=16th."""
    if pos % (subdivision // 4)  == 0: return 0
    if pos % (subdivision // 8)  == 0: return 1
    if pos % (subdivision // 12) == 0: return 2
    return 3


def measures_to_timestep_labels(measures: List[List[str]], subdivision: int = SUBDIVISION):
    """
    Convert measure/row representation to arrays of shape (T, 4) and (T,).
    Only emits the 24 valid positions per measure (divisible by 3 or 4 at subdivision=48).

    Each SM row sits at exact position row_idx * subdivision / n within the measure.
    For standard measure sizes (n = 4, 8, 12, 16, 48, 192, ...) this is always an
    integer that falls exactly on one of the 24 valid positions — no rounding.
    Notes that don't land on a valid position (non-standard subdivisions) are skipped.
    """
    valid_pos_lookup  = {p: i for i, p in enumerate(VALID_SUBDIV_POSITIONS)}
    N_VALID           = len(VALID_SUBDIV_POSITIONS)
    subdiv_types_row  = [get_subdiv_type(p, subdivision) for p in VALID_SUBDIV_POSITIONS]

    all_labels = []
    for measure in measures:
        n = len(measure)
        measure_labels = np.zeros((N_VALID, 4), dtype=np.float32)
        for row_idx, row_str in enumerate(measure):
            pos_float = row_idx * subdivision / n
            pos_int   = int(pos_float)
            if pos_float != pos_int or pos_int not in valid_pos_lookup:
                continue
            vi = valid_pos_lookup[pos_int]
            for col, c in enumerate(row_str[:4]):
                if c != '0':
                    measure_labels[vi, col] = 1.0
        all_labels.append(measure_labels)

    if not all_labels:
        return np.zeros((0, 4), dtype=np.float32), np.array([], dtype=np.int64)
    labels = np.vstack(all_labels)
    types  = np.tile(subdiv_types_row, len(all_labels)).astype(np.int64)
    return labels, types  # (T, 4), (T,)


# ─────────────────────────────────────────────
# AUDIO FEATURE EXTRACTION
# ─────────────────────────────────────────────

# Audio constants imported from config.py

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
                    subdivision: int = SUBDIVISION) -> np.ndarray:
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

# CONTEXT_FRAMES imported from config.py

def build_sample(
    audio_path: str,
    sm_path: str,
    difficulty_filter: Optional[str] = None,
    subdivision: int = SUBDIVISION,
    context: int = CONTEXT_FRAMES,
) -> Optional[Dict]:
    """
    Build (X, y, difficulty_level) arrays for one song.
    X: (T, context*2+1, N_MELS) — mel context windows per timestep
    y: (T, 4) — binary arrow labels
    difficulty: int scalar
    Accepts both .sm and .ssc chart files.
    """
    try:
        sm_data = parse_chart_file(sm_path)
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

    labels, subdiv_types = measures_to_timestep_labels(chart['measures'], subdivision)
    T_frames = mel.shape[1]

    # Compute accurate frame index for each valid position in each measure.
    # time = offset + (measure * subdivision + pos) * sec_per_one_subdivision_slot
    bpm    = sm_data['bpms'][0][1] if sm_data['bpms'] else 120.0
    offset = sm_data['offset']
    sec_per_slot = (60.0 / bpm) / (subdivision / 4)  # duration of one subdivision slot

    frame_indices = []
    for m_idx in range(len(chart['measures'])):
        for pos in VALID_SUBDIV_POSITIONS:
            t  = offset + (m_idx * subdivision + pos) * sec_per_slot
            fi = int(round(t * SR / HOP_LENGTH))
            fi = max(0, min(T_frames - 1, fi))
            frame_indices.append(fi)
    frame_indices = np.array(frame_indices)

    diff_int = difficulty_to_int(chart['difficulty'])

    return {
        'mel':          mel.astype(np.float16),         # (N_MELS, T_frames) — windowed on-the-fly
        'beat_frames':  frame_indices.astype(np.int32), # (T,) frame index per timestep
        'y':            labels.astype(np.float32),
        'subdiv_types': subdiv_types,
        'difficulty':   diff_int,
        'title':        sm_data['title'],
    }