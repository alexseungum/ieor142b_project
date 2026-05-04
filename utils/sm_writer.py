"""
sm_writer.py
Convert model output (step_mask, arrow_preds) back to a .sm file.
"""

import numpy as np
from typing import List, Tuple


DIFFICULTY_NAMES = {0: 'Beginner', 1: 'Easy', 2: 'Medium', 3: 'Hard', 4: 'Challenge'}
ARROW_CHARS = ['L', 'D', 'U', 'R']   # StepMania column order: Left Down Up Right


def arrow_vec_to_row(vec: np.ndarray) -> str:
    """Convert binary [L, D, U, R] vector to 4-char StepMania row string."""
    return ''.join('1' if v else '0' for v in vec)


def timesteps_to_measures(
    step_mask: np.ndarray,    # (T,) bool
    arrow_preds: np.ndarray,  # (T, 4) int
    subdivision: int = 16,    # rows per measure
) -> List[List[str]]:
    """
    Convert flat timestep arrays back into measure/row format.
    """
    T = len(step_mask)
    n_measures = (T + subdivision - 1) // subdivision
    measures = []

    for m in range(n_measures):
        measure_rows = []
        for row in range(subdivision):
            idx = m * subdivision + row
            if idx < T and step_mask[idx]:
                row_str = arrow_vec_to_row(arrow_preds[idx])
            else:
                row_str = '0000'
            measure_rows.append(row_str)
        measures.append(measure_rows)

    return measures


def write_sm_file(
    output_path: str,
    title: str,
    artist: str,
    audio_filename: str,
    bpm: float,
    offset: float,
    step_mask: np.ndarray,
    arrow_preds: np.ndarray,
    difficulty: int = 2,
    subdivision: int = 16,
):
    """
    Write a complete .sm file from model output.
    """
    measures = timesteps_to_measures(step_mask, arrow_preds, subdivision)
    diff_name = DIFFICULTY_NAMES.get(difficulty, 'Medium')
    meter = difficulty * 3 + 3   # rough meter estimate

    lines = [
        f'#TITLE:{title};',
        f'#ARTIST:{artist};',
        f'#MUSIC:{audio_filename};',
        f'#OFFSET:{offset:.3f};',
        f'#BPMS:0.000={bpm:.3f};',
        f'#STOPS:;',
        '',
        '#NOTES:',
        '     dance-single:',
        '     AI Generated:',
        f'     {diff_name}:',
        f'     {meter}:',
        '     0.000,0.000,0.000,0.000,0.000:',
    ]

    for i, measure in enumerate(measures):
        for row in measure:
            lines.append(row)
        if i < len(measures) - 1:
            lines.append(',')

    lines.append(';')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Written: {output_path}  ({len(measures)} measures, {step_mask.sum()} steps)")
