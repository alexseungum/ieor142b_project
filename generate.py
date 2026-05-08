"""
generate.py
Generate a DDR chart from any audio file using a trained model.

Usage:
    python generate.py --audio my_song.mp3 --checkpoint checkpoints/best_model.pt \
                       --difficulty 2 --threshold 0.5 --output my_song_chart
"""

import argparse
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.data_utils import (
    load_audio, extract_mel_spectrogram, N_MELS, CONTEXT_FRAMES
)
from config import SUBDIVISION, VALID_SUBDIV_POSITIONS
from utils.data_utils import get_subdiv_type
from utils.sm_writer import write_sm_file
from models.model import DDRTransformer, generate_chart
from visualizer import build_chart_json, build_html


CONTEXT = CONTEXT_FRAMES


def audio_to_model_input(audio_path: str, bpm: float, subdivision: int = SUBDIVISION,
                          context: int = CONTEXT):
    """
    Load audio and convert to model input tensor.
    Only generates timesteps at valid subdivision positions (24 per measure).
    Returns: X (1, T, context*2+1, N_MELS), subdiv_types (1, T)
    """
    y, sr = load_audio(audio_path)
    mel = extract_mel_spectrogram(y, sr)  # (N_MELS, T_frames)

    T_frames = mel.shape[1]
    duration_sec = len(y) / sr
    n_measures = int(duration_sec * (bpm / 60) / 4)  # 4 beats per measure
    sec_per_slot = (60.0 / bpm) / (subdivision / 4)

    X_list = []
    subdiv_types_list = []
    for m_idx in range(n_measures):
        for pos in VALID_SUBDIV_POSITIONS:
            t  = m_idx * subdivision * sec_per_slot + pos * sec_per_slot
            fi = int(round(t * sr / HOP_LENGTH))
            fi = max(0, min(T_frames - 1, fi))
            lo = max(0, fi - context)
            hi = min(T_frames - 1, fi + context)
            window = mel[:, lo:hi + 1]
            pad_l = context - (fi - lo)
            pad_r = context - (hi - fi)
            if pad_l > 0:
                window = np.concatenate([np.zeros((N_MELS, pad_l)), window], axis=1)
            if pad_r > 0:
                window = np.concatenate([window, np.zeros((N_MELS, pad_r))], axis=1)
            X_list.append(window.T)
            subdiv_types_list.append(get_subdiv_type(pos, subdivision))

    X = np.stack(X_list, axis=0).astype(np.float32)
    subdiv_types = np.array(subdiv_types_list, dtype=np.int64)
    return (
        torch.from_numpy(X).unsqueeze(0),           # (1, T, context*2+1, N_MELS)
        torch.from_numpy(subdiv_types).unsqueeze(0) # (1, T)
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--audio',      type=str, required=True,  help='Input audio file (.mp3/.ogg/.wav)')
    p.add_argument('--checkpoint', type=str, required=True,  help='Path to trained model checkpoint')
    p.add_argument('--difficulty', type=int, default=2,      help='Difficulty level 0-4')
    p.add_argument('--threshold',  type=float, default=0.5,  help='Step placement threshold (lower=more steps)')
    p.add_argument('--output',     type=str, default='output_chart', help='Output directory/prefix')
    p.add_argument('--bpm',        type=float, default=None, help='Override BPM (auto-detected if not set)')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_args = ckpt.get('args', {})

    model = DDRTransformer(
        d_model=model_args.get('d_model', 256),
        nhead=model_args.get('nhead', 8),
        num_encoder_layers=model_args.get('n_layers', 4),
        dim_feedforward=model_args.get('d_ff', 1024),
        dropout=0.0,  # no dropout at inference
    )
    state_dict = ckpt['model_state']
    # pos_enc.pe is a fixed sinusoidal buffer — remove it from the checkpoint
    # so the model uses its own freshly computed version at the new max_len
    state_dict.pop('pos_enc.pe', None)
    model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded (val F1 at save: {ckpt.get('val_f1', 'N/A')})")

    # Detect BPM before building model input so T_beats is exact
    if args.bpm is None:
        import librosa
        y, sr = load_audio(args.audio)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo)
        print(f"  Auto-detected BPM: {bpm:.1f}")
    else:
        bpm = args.bpm
        print(f"  Using BPM: {bpm:.1f}")

    # Process audio
    print(f"Processing audio: {args.audio}")
    X, subdiv_types = audio_to_model_input(args.audio, bpm=bpm)
    print(f"  Input shape: {X.shape}")

    # Generate
    print(f"Generating chart (difficulty={args.difficulty}, threshold={args.threshold})...")
    step_mask, arrow_preds = generate_chart(
        model, X, subdiv_types,
        difficulty=args.difficulty,
        step_threshold=args.threshold,
        device=device,
    )
    n_steps = step_mask.sum()
    print(f"  Generated {n_steps} step events out of {len(step_mask)} timesteps ({100*n_steps/len(step_mask):.1f}% density)")

    # Write .sm file
    Path(args.output).mkdir(parents=True, exist_ok=True)
    audio_name = Path(args.audio).name
    sm_path = f"{args.output}/chart.sm"
    write_sm_file(
        output_path=sm_path,
        title=Path(args.audio).stem,
        artist='AI Generated',
        audio_filename=audio_name,
        bpm=bpm,
        offset=0.0,
        step_mask=step_mask,
        arrow_preds=arrow_preds,
        difficulty=args.difficulty,
    )

    # Copy audio to output dir
    import shutil
    shutil.copy(args.audio, f"{args.output}/{audio_name}")

    # Generate visualizer HTML directly from arrays (bypass sm round-trip)
    print("Generating visualizer...")
    try:
        import json as _json
        import base64
        DIFFICULTY_NAMES = {0: 'Beginner', 1: 'Easy', 2: 'Medium', 3: 'Hard', 4: 'Challenge'}
        events = []
        for t_idx in range(len(step_mask)):
            if step_mask[t_idx]:
                arrows = [int(arrow_preds[t_idx, i]) for i in range(4)]
                # If arrow head predicted nothing, default to the most common pattern
                # (left+right alternating) so the chart is at least visible
                if not any(arrows):
                    arrows[t_idx % 4] = 1
                events.append({'t': t_idx, 'arrows': arrows})

        chart_data = {
            'title': Path(args.audio).stem,
            'bpm': bpm,
            'offset': 0.0,
            'difficulty': DIFFICULTY_NAMES.get(args.difficulty, 'Medium'),
            'meter': args.difficulty * 3 + 3,
            'subdivision': SUBDIVISION,
            'total_steps': int(step_mask.sum()),
            'total_timesteps': len(step_mask),
            'events': events,
        }

        # Embed audio as base64 so the HTML is fully self-contained
        suffix = Path(args.audio).suffix.lower()
        mime_map = {'.mp3': 'audio/mpeg', '.ogg': 'audio/ogg', '.wav': 'audio/wav'}
        audio_mime = mime_map.get(suffix, 'audio/mpeg')
        with open(args.audio, 'rb') as af:
            audio_b64 = base64.b64encode(af.read()).decode('utf-8')
        audio_data_uri = f"data:{audio_mime};base64,{audio_b64}"

        print(f"  Visualizer: {chart_data['total_steps']} steps, {len(events)} events")
        html = build_html(chart_data, audio_data_uri=audio_data_uri)
        viz_path = f"{args.output}/visualizer.html"
        with open(viz_path, 'w') as f:
            f.write(html)
        print(f"  Visualizer: {viz_path}")
    except Exception as e:
        print(f"  Visualizer failed: {e}")

    print(f"\nDone! Output in: {args.output}/")
    print(f"  chart.sm      → load in StepMania")
    print(f"  visualizer.html → open in browser")


if __name__ == '__main__':
    main()