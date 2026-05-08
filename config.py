# config.py
# Central config — edit this file to change any hyperparameter.
# All other files import from here.

# ── Audio processing ──────────────────────────────────────────
SR            = 22050   # sample rate
HOP_LENGTH    = 512     # STFT hop size (~23ms per frame)
N_FFT         = 2048
N_MELS        = 80      # mel frequency bins
CONTEXT_FRAMES = 7      # frames of audio context on each side of a timestep
CONTEXT_LEN   = CONTEXT_FRAMES * 2 + 1   # total context window = 15 frames

# ── Dataset ───────────────────────────────────────────────────
SUBDIVISION   = 48      # subdivisions per measure: LCM(4,8,12,16) covers 4th/8th/12th/16th notes

# Only positions divisible by 3 (16th) or 4 (12th) can ever have notes.
# 24 out of 48 positions per measure — skip the always-empty ones.
VALID_SUBDIV_POSITIONS = sorted(set(
    i for i in range(SUBDIVISION)
    if i % (SUBDIVISION // 16) == 0 or i % (SUBDIVISION // 12) == 0
))
N_VALID_PER_MEASURE = len(VALID_SUBDIV_POSITIONS)  # = 24

SEQ_LEN       = N_VALID_PER_MEASURE * 32   # 24 valid positions × 32 measures = 768 timesteps
N_DIFFICULTIES = 5      # difficulty levels: 0=beginner .. 4=challenge
N_SUBDIV_TYPES = 4      # 0=4th, 1=8th, 2=12th(triplet), 3=16th

# ── Model architecture ────────────────────────────────────────
D_MODEL       = 256
NHEAD         = 8
N_LAYERS      = 4
D_FF          = 1024
DROPOUT       = 0.1

# ── Training ──────────────────────────────────────────────────
BATCH_SIZE         = 32     # A100 40GB — bump to 64 if no OOM
LR                 = 3e-4
WEIGHT_DECAY       = 1e-4
EPOCHS_PER_STAGE   = 30
PATIENCE           = 10
POS_WEIGHT         = 5.0    # upweight positive steps (class imbalance)
LABEL_SMOOTHING    = 0.1
NUM_WORKERS        = 2
CURRICULUM_START   = 0      # 0=start from beginner, 4=all difficulties from start
