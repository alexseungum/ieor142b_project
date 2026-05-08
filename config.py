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
SEQ_LEN       = 1536    # timesteps per training chunk (48 subdivisions × 32 measures)
                        # stride = SEQ_LEN // 2 = 768 = 16 measures overlap
N_DIFFICULTIES = 5      # difficulty levels: 0=beginner .. 4=challenge

# ── Model architecture ────────────────────────────────────────
D_MODEL       = 256
NHEAD         = 8
N_LAYERS      = 4
D_FF          = 1024
DROPOUT       = 0.1

# ── Training ──────────────────────────────────────────────────
BATCH_SIZE         = 8      # reduced to handle SEQ_LEN=1536 on L4 GPU
LR                 = 3e-4
WEIGHT_DECAY       = 1e-4
EPOCHS_PER_STAGE   = 30
PATIENCE           = 10
POS_WEIGHT         = 5.0    # upweight positive steps (class imbalance)
LABEL_SMOOTHING    = 0.1
NUM_WORKERS        = 2
CURRICULUM_START   = 0      # 0=start from beginner, 4=all difficulties from start
