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
                if c in ('1', '2', '4'):  # tap, hold start, roll start — skip hold/roll end ('3')
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


# ─────────────────────────────────────────────
# DATASET SCANNING & DIAGNOSTICS
# ─────────────────────────────────────────────

def quick_difficulties(chart_path: str) -> List[str]:
    """Fast regex scan for dance-single difficulty tags without full parsing."""
    try:
        content = open(chart_path, 'r', encoding='utf-8', errors='ignore').read()
        is_ssc  = chart_path.endswith('.ssc')
        diffs   = []
        if is_ssc:
            types = re.findall(r'#STEPSTYPE\s*:([^;]+);', content, re.IGNORECASE)
            diffd = re.findall(r'#DIFFICULTY\s*:([^;]+);', content, re.IGNORECASE)
            for t, d in zip(types, diffd):
                if 'single' in t.lower():
                    diffs.append(d.strip().lower())
        else:
            for m in re.finditer(r'#NOTES\s*:(.*?)(?=\n[^,\n]|\Z)', content,
                                  re.DOTALL | re.IGNORECASE):
                block = m.group(1)
                lines = [l.strip().rstrip(':') for l in block.split('\n')
                         if l.strip() and not l.strip().startswith('//')]
                if len(lines) >= 3 and 'single' in lines[0].lower():
                    diffs.append(lines[2].lower())
        return diffs
    except Exception:
        return []


def scan_song_dirs(data_root: str) -> Tuple[List, List, Dict]:
    """
    Recursively find all song directories with a chart + audio file.
    Returns (usable_pairs, no_audio_dirs, pack_stats).
      usable_pairs  : list of (audio_path, chart_path, fmt, pack_name)
      no_audio_dirs : list of song dir paths missing audio
      pack_stats    : dict[pack] -> {songs, sm, ssc, no_audio, difficulties}
    """
    from collections import defaultdict
    root       = Path(data_root)
    audio_exts = {'.ogg', '.mp3', '.wav'}
    all_sm     = list(root.rglob('*.sm'))
    all_ssc    = list(root.rglob('*.ssc'))
    song_dirs  = sorted({f.parent for f in all_sm + all_ssc})

    usable_pairs  = []
    no_audio_dirs = []
    pack_stats    = defaultdict(lambda: {
        'songs': 0, 'sm': 0, 'ssc': 0, 'no_audio': 0,
        'difficulties': defaultdict(int),
    })

    for song_dir in song_dirs:
        pack        = song_dir.parent.name
        sm_files    = list(song_dir.glob('*.sm'))
        ssc_files   = list(song_dir.glob('*.ssc'))
        audio_files = [f for f in song_dir.iterdir() if f.suffix.lower() in audio_exts]
        chart_files = sm_files if sm_files else ssc_files
        fmt         = 'sm' if sm_files else 'ssc'

        pack_stats[pack]['songs'] += 1
        pack_stats[pack][fmt]     += 1

        if not audio_files or not chart_files:
            pack_stats[pack]['no_audio'] += 1
            no_audio_dirs.append(str(song_dir))
            continue

        chart_path = str(chart_files[0])
        usable_pairs.append((str(audio_files[0]), chart_path, fmt, pack))
        for diff in quick_difficulties(chart_path):
            pack_stats[pack]['difficulties'][diff] += 1

    return usable_pairs, no_audio_dirs, dict(pack_stats)


def find_audio_for_title(data_root: str, title: str,
                         pack_name: str = '') -> Tuple[Optional[str], float, float]:
    """
    Search data_root for audio matching the given song title.
    Returns (audio_path_or_None, bpm, offset).
    If pack_name is given, that pack subdirectory is searched first.
    Phase 1: exact title match in .sm/.ssc. Phase 2: fuzzy folder name match.
    """
    audio_exts = {'.mp3', '.ogg', '.wav'}
    root       = Path(data_root)

    # Search the specified pack first, then fall back to all packs
    pack_dir = root / pack_name if pack_name else None
    search_roots = ([pack_dir] if pack_dir and pack_dir.exists() else []) + [root]

    def _norm(s): return s.lower().strip()

    bpm, offset = 120.0, 0.0

    seen = set()
    chart_candidates = []
    for sr in search_roots:
        for p in list(sr.rglob('*.[sS][mM]')) + list(sr.rglob('*.[sS][sS][cC]')):
            if p not in seen:
                seen.add(p)
                chart_candidates.append(p)

    for chart_path in chart_candidates:
        try:
            text = chart_path.read_text(errors='ignore')
            for line in text.splitlines():
                if line.upper().startswith('#TITLE:'):
                    sm_title = line.split(':', 1)[1].rstrip(';').strip()
                    if _norm(sm_title) == _norm(title):
                        for tag_line in text.splitlines():
                            tu = tag_line.upper().lstrip()
                            if tu.startswith('#BPMS:'):
                                try:
                                    bpm = float(tag_line.split(':', 1)[1].split('=')[1]
                                                .rstrip(';').split(',')[0].strip())
                                except Exception:
                                    pass
                            elif tu.startswith('#OFFSET:'):
                                try:
                                    offset = float(tag_line.split(':', 1)[1].rstrip(';').strip())
                                except Exception:
                                    pass
                        for f in chart_path.parent.iterdir():
                            if f.suffix.lower() in audio_exts:
                                return str(f), bpm, offset
                        break
        except Exception:
            pass

    # Fuzzy fallback: match words in folder name
    words = [w for w in _norm(title).split() if len(w) > 3]
    if words:
        for sr in search_roots:
            for song_dir in sr.rglob('*'):
                if not song_dir.is_dir():
                    continue
                if any(w in _norm(song_dir.name) for w in words):
                    for f in song_dir.iterdir():
                        if f.suffix.lower() in audio_exts:
                            return str(f), bpm, offset

    return None, bpm, offset


def compute_step_metrics(step_mask: np.ndarray, y_np: np.ndarray, start: int = 0) -> Dict:
    """Compute step placement F1, precision, recall from position start onwards."""
    gt   = y_np[start:].sum(-1) > 0
    pred = step_mask[start:]
    tp   = int((gt & pred).sum())
    fp   = int((~gt & pred).sum())
    fn   = int((gt & ~pred).sum())
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    return {'f1': f1, 'precision': prec, 'recall': rec, 'tp': tp, 'fp': fp, 'fn': fn}


def compute_arrow_metrics(step_mask: np.ndarray, arrow_preds: np.ndarray,
                          y_np: np.ndarray) -> Dict:
    """
    Compute arrow quality metrics at true-positive step positions (where both
    the model and GT agree there is a step). Since step placement is shared
    between ablation variants, this isolates arrow prediction quality.

    Returns:
      n_tp          : number of true-positive steps evaluated
      arrow_exact   : fraction where all 4 arrows exactly match GT
      per_dir_acc   : (4,) per-direction (L/D/U/R) accuracy
      pred_combo_dist / gt_combo_dist : fraction of steps that are
                      single / bracket / triple / quad
    """
    gt_mask = y_np.sum(-1) > 0
    tp_mask = step_mask & gt_mask
    n_tp    = int(tp_mask.sum())

    if n_tp == 0:
        return {'n_tp': 0, 'arrow_exact': float('nan'),
                'per_dir_acc': np.full(4, float('nan')),
                'pred_combo_dist': {}, 'gt_combo_dist': {}}

    pred_tp = arrow_preds[tp_mask].astype(float)   # (N, 4)
    gt_tp   = y_np[tp_mask].astype(float)           # (N, 4)

    arrow_exact  = float((pred_tp == gt_tp).all(-1).mean())
    per_dir_acc  = (pred_tp == gt_tp).mean(0)        # (4,)

    def _combo_dist(arr):
        n = arr.sum(-1)
        total = max(len(arr), 1)
        return {
            'single':  float((n == 1).sum() / total),
            'bracket': float((n == 2).sum() / total),
            'triple':  float((n == 3).sum() / total),
            'quad':    float((n == 4).sum() / total),
        }

    return {
        'n_tp':            n_tp,
        'arrow_exact':     arrow_exact,
        'per_dir_acc':     per_dir_acc,
        'pred_combo_dist': _combo_dist(pred_tp),
        'gt_combo_dist':   _combo_dist(gt_tp),
    }