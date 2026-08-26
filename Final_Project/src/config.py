"""Shared configuration for the brain tumour classifier.

Every notebook imports from here so that a hyperparameter exists in exactly
one place. Nothing in this module trains anything or touches the disk beyond
creating the outputs directory.
"""
from pathlib import Path

import torch

# ── PATHS ───────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "brain_tumour"
TRAIN_DIR, TEST_DIR = DATA_DIR / "Training", DATA_DIR / "Testing"
OUTPUTS  = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
CACHE_DIR = DATA_DIR.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CKPT_PATH = OUTPUTS / "best_model.pth"

# ── DATA ────────────────────────────────────────────────
IMG_SIZE   = 128      # working resolution; ablated against 192 in notebook 04
CACHE_SIZE = 224      # images are decoded once at this size, then resized down
VAL_FRAC   = 0.15     # carved out of Training/; Testing/ is never touched
BATCH_SIZE = 32

# ── TRAINING ────────────────────────────────────────────
EPOCHS    = 60
LR        = 1e-3
WD        = 1e-4
DROPOUT   = 0.4
PATIENCE  = 8         # 5 fires spuriously on real (noisy) validation loss
SEED      = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Notebook 04 compares configurations against a shared baseline, so two runs
# with the same seed must produce the same numbers. torch.manual_seed alone is
# not enough on CUDA — cuDNN picks non-deterministic algorithms unless told not to.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ── PLOT STYLE (matches Days 1-8) ───────────────────────
FACE    = '#F8F8F6'
PALETTE = {"train": '#378ADD', "val": '#D85A30',
           "good":  '#1D9E75', "accent": '#9B59B6'}
