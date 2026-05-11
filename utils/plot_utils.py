"""
plot_utils.py
Reusable plotting helpers for training diagnostics and chart visualization.
"""
import numpy as np
import matplotlib.pyplot as plt

RHYTHM_COLORS = ['#ff4040', '#4080ff', '#aa44ff', '#44dd88']
RHYTHM_NAMES  = ['1/4', '1/8', '1/12', '1/16']
ARROW_NAMES   = ['L', 'D', 'U', 'R']
STAGE_COLORS  = {0: '#3498db', 1: '#2ecc71', 2: '#f39c12', 3: '#e74c3c', 4: '#9b59b6'}
STAGE_NAMES   = {0: 'Beginner', 1: 'Easy', 2: 'Medium', 3: 'Hard', 4: 'Challenge'}


def plot_training_curves(history: dict, save_path: str = None):
    train_h = history['train']
    val_h   = history['val']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for metric, ax, title in [
        ('loss',      axes[0], 'Total Loss'),
        ('step_loss', axes[1], 'Step Placement Loss'),
        ('f1',        axes[2], 'Step F1 Score'),
    ]:
        for stage in range(5):
            t_vals = [x[metric] for x in train_h if x['stage'] == stage]
            v_vals = [x[metric] for x in val_h   if x['stage'] == stage]
            if not t_vals:
                continue
            color = STAGE_COLORS[stage]
            ax.plot(t_vals, color=color, alpha=0.9, label=f'{STAGE_NAMES[stage]} train')
            ax.plot(v_vals, color=color, alpha=0.5, linestyle='--', label=f'{STAGE_NAMES[stage]} val')
        ax.set_title(title)
        ax.set_xlabel('Epoch within stage')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    plt.suptitle('Training curves by curriculum stage', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_generation(step_probs, y_np, arrow_preds, subdiv_arr, title,
                    subdiv_thresh, seed_cutoff=0, save_path=None):
    """Three-panel plot: step probs, GT arrows, predicted arrows."""
    T = len(step_probs)
    fig, axes = plt.subplots(3, 1, figsize=(16, 8))

    ax = axes[0]
    if seed_cutoff > 0:
        ax.axvspan(0, seed_cutoff, alpha=0.12, color='green', label=f'Seed ({seed_cutoff} steps)')
    ax.plot(step_probs, lw=0.6, color='#2980b9', label='Step prob')
    for stype, (color, name) in enumerate(zip(RHYTHM_COLORS, RHYTHM_NAMES)):
        ax.axhline(subdiv_thresh[stype], color=color, linestyle='--', linewidth=0.8,
                   label=f'{name} thresh={subdiv_thresh[stype]:.2f}')
    ax.set_ylabel('Step probability')
    ax.set_title(f"Step probs — '{title}'")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    for ax, arr, label, shade in [
        (axes[1], y_np,        'Ground truth',           False),
        (axes[2], arrow_preds, 'Predicted (green=seed)', True),
    ]:
        if shade and seed_cutoff > 0:
            ax.axvspan(0, seed_cutoff, alpha=0.12, color='green')
        for i, n in enumerate(ARROW_NAMES):
            ts = np.where(arr[:, i] > 0)[0]
            c  = [RHYTHM_COLORS[int(subdiv_arr[t_])] for t_ in ts]
            ax.scatter(ts, np.full_like(ts, i), c=c, s=4)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(ARROW_NAMES)
        ax.invert_yaxis()
        ax.set_xlim(0, T)
        ax.grid(True, alpha=0.2)
        ax.set_title(label)
        for rc, rn in zip(RHYTHM_COLORS, RHYTHM_NAMES):
            ax.scatter([], [], c=rc, s=20, label=rn)
        ax.legend(fontsize=8, title='Rhythm')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_chart_preview(step_probs, arrow_preds, subdiv_arr, subdiv_thresh,
                       title='', save_path=None):
    """Two-panel plot: step probs + arrow scatter. Used after generate.py CLI output."""
    T = len(step_probs)
    step_mask = arrow_preds.sum(-1) > 0
    fig, axes = plt.subplots(2, 1, figsize=(18, 6))

    ax = axes[0]
    ax.plot(step_probs, lw=0.6, color='#2980b9', label='Step prob')
    for stype, (color, name) in enumerate(zip(RHYTHM_COLORS, RHYTHM_NAMES)):
        ax.axhline(subdiv_thresh[stype], color=color, linestyle='--', linewidth=0.8,
                   label=f'{name} thresh={subdiv_thresh[stype]:.2f}')
    ax.set_ylabel('Step probability')
    ax.set_title(f"Step probabilities — {step_mask.sum()} steps  ({100*step_mask.mean():.1f}% density)")
    ax.set_xlim(0, T)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    for i, n in enumerate(ARROW_NAMES):
        ts = np.where(arrow_preds[:, i] > 0)[0]
        c  = [RHYTHM_COLORS[int(subdiv_arr[t_])] for t_ in ts]
        ax.scatter(ts, np.full_like(ts, i), c=c, s=6, linewidths=0)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(ARROW_NAMES)
    ax.invert_yaxis()
    ax.set_xlim(0, T)
    ax.set_xlabel('Timestep')
    ax.set_title('Generated arrows')
    for rc, rn in zip(RHYTHM_COLORS, RHYTHM_NAMES):
        ax.scatter([], [], c=rc, s=20, label=rn)
    ax.legend(loc='upper right', fontsize=9, title='Rhythm')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_prob_sanity(step_probs, arrow_probs, save_path=None):
    """Step prob trace + per-direction arrow prob histograms."""
    ARROW_FULL   = ['Left', 'Down', 'Up', 'Right']
    ARROW_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    ax.plot(step_probs, linewidth=0.5, color='#2980b9', alpha=0.8)
    ax.axhline(0.5, color='red', linestyle='--', linewidth=0.8, label='threshold=0.5')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Step probability')
    ax.set_title('Step placement probability across song')
    ax.legend()
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    bins = np.linspace(0, 1, 25)
    for i, (name, color) in enumerate(zip(ARROW_FULL, ARROW_COLORS)):
        ax.hist(arrow_probs[:, i], bins=bins, alpha=0.5, color=color, label=name)
    ax.set_xlabel('Arrow probability')
    ax.set_ylabel('Count')
    ax.set_title('Arrow probability distribution by direction')
    ax.legend()
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

    print(f"Step prob — mean: {step_probs.mean():.3f}, median: {np.median(step_probs):.3f}, "
          f">0.5: {(step_probs > 0.5).mean() * 100:.1f}%")
    for i, name in enumerate(ARROW_FULL):
        print(f"  {name}: mean prob {arrow_probs[:, i].mean():.3f}")


def plot_threshold_sweep(thresholds, densities, default_thresh=0.5, save_path=None):
    plt.figure(figsize=(7, 4))
    plt.plot(thresholds, densities, 'o-', color='#e74c3c')
    plt.axvline(default_thresh, color='gray', linestyle='--', alpha=0.5, label='default threshold')
    plt.xlabel('Step threshold')
    plt.ylabel('Fraction of timesteps with a step')
    plt.title('Difficulty knob: threshold vs. chart density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
