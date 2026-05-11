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
from config import HOP_LENGTH, SEQ_LEN, N_VALID_PER_MEASURE
from config import SUBDIVISION, VALID_SUBDIV_POSITIONS
from utils.data_utils import get_subdiv_type
from utils.sm_writer import write_sm_file
from models.model import DDRTransformer, generate_chart, load_model
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


def build_chart_data(step_mask, arrow_preds, bpm: float, title: str,
                     difficulty: int, offset: float = 0.0) -> dict:
    """Build the chart_data dict consumed by build_html from generation outputs."""
    DIFF_NAMES   = {0: 'Beginner', 1: 'Easy', 2: 'Medium', 3: 'Hard', 4: 'Challenge'}
    sec_per_meas = 4 * 60.0 / bpm
    events = []
    for t_idx in range(len(step_mask)):
        if step_mask[t_idx]:
            arrows = [int(arrow_preds[t_idx, i]) for i in range(4)]
            if not any(arrows):
                arrows[t_idx % 4] = 1
            measure_idx = t_idx // N_VALID_PER_MEASURE
            pos   = int(VALID_SUBDIV_POSITIONS[t_idx % N_VALID_PER_MEASURE])
            t_sec = offset + measure_idx * sec_per_meas + pos / SUBDIVISION * sec_per_meas
            events.append({'t': t_idx, 'pos': pos, 't_sec': round(t_sec, 6), 'arrows': arrows})
    total_measures = len(step_mask) // N_VALID_PER_MEASURE
    return {
        'title':           title,
        'bpm':             bpm,
        'offset':          offset,
        'difficulty':      DIFF_NAMES.get(difficulty, 'Medium'),
        'meter':           difficulty * 3 + 3,
        'subdivision':     SUBDIVISION,
        'total_steps':     int(step_mask.sum()),
        'total_timesteps': len(step_mask),
        'total_duration':  round(offset + total_measures * sec_per_meas, 3),
        'events':          events,
    }


@torch.no_grad()
def generate_seeded(model, song: dict, device, temperature: float = 1.2,
                    threshold: float = 0.5, subdiv_scales: list = None,
                    n_seed: int = 16):
    """
    Generate from a cached song dict (mel/beat_frames/y/subdiv_types).
    Seeds decoder with first n_seed GT steps, then generates autoregressively.
    Returns (step_mask, arrow_preds, step_probs, seed_cutoff).
    """
    import torch as _torch
    if subdiv_scales is None:
        subdiv_scales = [1.0, 0.60, 0.50, 0.45]
    subdiv_thresh = [threshold * s for s in subdiv_scales]

    model.eval()
    model.to(device)

    mel  = song['mel']
    bf   = song['beat_frames']
    y_np = song['y'].astype(np.float32)
    st   = song['subdiv_types']
    diff = _torch.tensor([song['difficulty']], dtype=_torch.long, device=device)

    STRIDE   = SEQ_LEN // 2
    n_chunks = len(bf) // STRIDE
    T_total  = n_chunks * STRIDE

    step_mask   = np.zeros(T_total, dtype=bool)
    step_probs  = np.zeros(T_total, dtype=np.float32)
    arrows_np   = np.zeros((T_total, 4), dtype=np.float32)
    arrow_preds = np.zeros((T_total, 4), dtype=np.int64)
    seed_cutoff = 0

    ctx     = CONTEXT
    mel_f32 = np.pad(mel.astype(np.float32), ((0, 0), (ctx, ctx)))

    for chunk_idx in range(n_chunks):
        if chunk_idx == 0:
            enc_start, enc_end = 0, SEQ_LEN
        else:
            enc_start = (chunk_idx - 1) * STRIDE
            enc_end   = enc_start + SEQ_LEN
        if enc_end > len(bf):
            break

        bf_c = bf[enc_start:enc_end]
        y_c  = y_np[enc_start:enc_end]
        st_c = st[enc_start:enc_end]

        col_idx = bf_c[:, None] + np.arange(-ctx, ctx + 1)[None, :]
        X_np    = mel_f32[:, col_idx].transpose(1, 2, 0)
        X_dev   = _torch.from_numpy(X_np).unsqueeze(0).to(device)
        st_dev  = _torch.from_numpy(st_c).unsqueeze(0).to(device)
        arrows  = _torch.zeros(1, SEQ_LEN, 4, device=device)

        if chunk_idx == 0:
            gt_step_pos = np.where(y_c.sum(-1) > 0)[0]
            seed_pos    = [p for p in gt_step_pos if p < STRIDE][:n_seed]
            seed_cutoff = int(seed_pos[-1]) + 1 if (n_seed > 0 and len(seed_pos) >= n_seed) else 0
            for p in seed_pos:
                arrows[0, p, :] = _torch.from_numpy(y_c[p]).to(device)
                step_mask[p]    = True
                arrows_np[p]    = y_c[p]
            gen_start, gen_end = seed_cutoff, STRIDE
        else:
            prev_s = (chunk_idx - 1) * STRIDE
            prev_e = chunk_idx * STRIDE
            arrows[0, :STRIDE, :] = _torch.from_numpy(arrows_np[prev_s:prev_e]).to(device)
            gen_start, gen_end = STRIDE, SEQ_LEN

        encoder_out       = model.encode(X_dev, diff, st_dev)
        step_logits_chunk = model.step_head(encoder_out)

        for t in range(gen_start, gen_end):
            al       = model.decoder(arrows, encoder_out)
            prob     = _torch.sigmoid(step_logits_chunk[0, t, 0]).item()
            stype    = int(st_c[t])
            global_t = t if chunk_idx == 0 else (chunk_idx - 1) * STRIDE + t
            step_probs[global_t] = prob
            if prob > subdiv_thresh[stype]:
                step_mask[global_t] = True
                logits    = al[0, t, :] / temperature
                probs_16  = _torch.softmax(logits, dim=-1)
                combo_idx = _torch.multinomial(probs_16, 1).item()
                if combo_idx == 0:
                    combo_idx = probs_16[1:].argmax().item() + 1
                bits = _torch.tensor([8, 4, 2, 1], device=device, dtype=_torch.long)
                pred = ((combo_idx & bits) > 0).float()
                arrows[0, t, :] = pred
                arrows_np[global_t]   = pred.cpu().numpy()
                arrow_preds[global_t] = pred.long().cpu().numpy()

    arrow_preds[~step_mask] = 0
    return step_mask, arrow_preds, step_probs, seed_cutoff


@torch.no_grad()
def generate_no_ar(model, song: dict, device, temperature: float = 1.2,
                   threshold: float = 0.5, subdiv_scales: list = None):
    """
    Ablation: decoder runs once per chunk on all-zeros arrow history instead of
    token-by-token. Removes autoregressive feedback while keeping the decoder's
    cross-attention to the encoder. Step placement is identical to the full model.
    Returns (step_mask, arrow_preds, step_probs).
    """
    import torch as _torch
    if subdiv_scales is None:
        subdiv_scales = [1.0, 0.60, 0.50, 0.45]
    subdiv_thresh = [threshold * s for s in subdiv_scales]

    model.eval()
    model.to(device)

    mel  = song['mel']
    bf   = song['beat_frames']
    st   = song['subdiv_types']
    diff = _torch.tensor([song['difficulty']], dtype=_torch.long, device=device)

    STRIDE   = SEQ_LEN // 2
    n_chunks = len(bf) // STRIDE
    T_total  = n_chunks * STRIDE

    step_mask   = np.zeros(T_total, dtype=bool)
    step_probs  = np.zeros(T_total, dtype=np.float32)
    arrow_preds = np.zeros((T_total, 4), dtype=np.int64)

    ctx     = CONTEXT
    mel_f32 = np.pad(mel.astype(np.float32), ((0, 0), (ctx, ctx)))

    for chunk_idx in range(n_chunks):
        if chunk_idx == 0:
            enc_start, enc_end = 0, SEQ_LEN
        else:
            enc_start = (chunk_idx - 1) * STRIDE
            enc_end   = enc_start + SEQ_LEN
        if enc_end > len(bf):
            break

        bf_c      = bf[enc_start:enc_end]
        st_c      = st[enc_start:enc_end]
        gen_start = 0      if chunk_idx == 0 else STRIDE
        gen_end   = STRIDE if chunk_idx == 0 else SEQ_LEN

        col_idx = bf_c[:, None] + np.arange(-ctx, ctx + 1)[None, :]
        X_np    = mel_f32[:, col_idx].transpose(1, 2, 0)
        X_dev   = _torch.from_numpy(X_np).unsqueeze(0).to(device)
        st_dev  = _torch.from_numpy(st_c).unsqueeze(0).to(device)

        encoder_out       = model.encode(X_dev, diff, st_dev)
        step_logits_chunk = model.step_head(encoder_out)
        # Single decoder forward pass — zeros arrow history, no token-by-token loop
        al = model.decoder(_torch.zeros(1, SEQ_LEN, 4, device=device), encoder_out)

        for t in range(gen_start, gen_end):
            prob     = _torch.sigmoid(step_logits_chunk[0, t, 0]).item()
            stype    = int(st_c[t])
            global_t = t if chunk_idx == 0 else (chunk_idx - 1) * STRIDE + t
            step_probs[global_t] = prob
            if prob > subdiv_thresh[stype]:
                step_mask[global_t] = True
                logits    = al[0, t, :] / temperature
                probs_16  = _torch.softmax(logits, dim=-1)
                combo_idx = _torch.multinomial(probs_16, 1).item()
                if combo_idx == 0:
                    combo_idx = probs_16[1:].argmax().item() + 1
                bits = _torch.tensor([8, 4, 2, 1], device=device, dtype=_torch.long)
                arrow_preds[global_t] = ((combo_idx & bits) > 0).long().cpu().numpy()

    arrow_preds[~step_mask] = 0
    return step_mask, arrow_preds, step_probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--audio',      type=str, required=True,  help='Input audio file (.mp3/.ogg/.wav)')
    p.add_argument('--checkpoint', type=str, required=True,  help='Path to trained model checkpoint')
    p.add_argument('--difficulty', type=int, default=2,      help='Difficulty level 0-4')
    p.add_argument('--temperature',type=float, default=1.0,  help='Arrow sampling temperature (>1=more diverse)')
    p.add_argument('--threshold',  type=float, default=0.5,  help='Step placement probability threshold (applies to 4th notes)')
    p.add_argument('--subdiv_scales', type=float, nargs=4, default=[1.0, 0.60, 0.50, 0.45],
                   metavar=('S4TH', 'S8TH', 'S12TH', 'S16TH'),
                   help='Threshold multipliers per subdivision type [4th 8th 12th 16th] (default: 1.0 0.60 0.50 0.45)')
    p.add_argument('--output',     type=str, default='output_chart', help='Output directory/prefix')
    p.add_argument('--bpm',        type=float, required=True, help='Song BPM (check the .sm/.ssc file or a BPM detector)')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model, ckpt = load_model(args.checkpoint, device=device)
    print(f"  Loaded (val F1 at save: {ckpt.get('val_f1', 'N/A')})")

    bpm = args.bpm
    print(f"  Using BPM: {bpm:.1f}")

    # Process audio
    print(f"Processing audio: {args.audio}")
    X, subdiv_types = audio_to_model_input(args.audio, bpm=bpm)
    print(f"  Input shape: {X.shape}")

    # Generate
    print(f"Generating chart (difficulty={args.difficulty}, temperature={args.temperature})...")
    step_mask, arrow_preds, step_probs = generate_chart(
        model, X, subdiv_types,
        difficulty=args.difficulty,
        temperature=args.temperature,
        threshold=args.threshold,
        subdiv_scales=args.subdiv_scales,
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

    # Save predictions for notebook plotting
    np.savez(f"{args.output}/predictions.npz",
             step_mask=step_mask, arrow_preds=arrow_preds,
             step_probs=step_probs, bpm=np.array(bpm))

    # Copy audio to output dir
    import shutil
    shutil.copy(args.audio, f"{args.output}/{audio_name}")

    # Visualizer
    print("Generating visualizer...")
    try:
        import base64
        chart_data = build_chart_data(step_mask, arrow_preds, bpm=bpm,
                                      title=Path(args.audio).stem,
                                      difficulty=args.difficulty, offset=0.0)
        suffix    = Path(args.audio).suffix.lower()
        mime_map  = {'.mp3': 'audio/mpeg', '.ogg': 'audio/ogg', '.wav': 'audio/wav'}
        with open(args.audio, 'rb') as af:
            audio_b64 = base64.b64encode(af.read()).decode('utf-8')
        audio_data_uri = f"data:{mime_map.get(suffix, 'audio/mpeg')};base64,{audio_b64}"
        print(f"  Visualizer: {chart_data['total_steps']} steps, {len(chart_data['events'])} events")
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