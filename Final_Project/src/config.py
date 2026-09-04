"""Shared configuration. Every hyperparameter appears here exactly once.

Nothing here trains or touches the network at import time: an earlier version
resolved the dataset through Kaggle in the module body, so importing anything
required internet.
"""
from pathlib import Path

import torch

# ── PATHS ───────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
OUTPUTS   = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
CKPT_PATH = OUTPUTS / "best_model.pth"
MANIFEST_PATH = OUTPUTS / "run_manifest.json"

# One source: 3,064 T1ce slices, 233 patients, 512x512, each with a patient ID
# and a tumour mask. No healthy class -- Cheng ships none, and borrowing one
# from a second collection is not survivable: five pixel statistics separate
# tumour source from healthy source at 0.9717 (Br35H) and 0.9243 (IXI) against
# 0.5000 chance, with standardisation and matched JPEG reaching only a ~0.92
# floor. That shortcut sank the previous dataset, where glioma scored 0.974 at
# 512px and 0.258 elsewhere. So the task is tumour *typing*, not detection.
DATASET   = "figshare-cheng-2017"
CHENG_DIR = ROOT / "data" / "raw" / "cheng"

CLASSES = ["glioma", "meningioma", "pituitary"]


# ── DATA ────────────────────────────────────────────────
IMG_SIZE   = 128    # working resolution; 192 tested and did not pay for itself
CACHE_SIZE = 224    # decoded once here so resolution can be ablated later
VAL_FRAC   = 0.15   # of the non-test pool
TEST_FRAC  = 0.20   # held out first, patient-wise, never touched again
BATCH_SIZE = 32

# Both duplicate conditions must hold: cosine alone flags ~1 image in 10,
# because different patients' slices genuinely resemble each other at 56x56.
DUP_COSINE = 0.995
DUP_L1     = 0.15

# Imbalance is 2.01:1 (glioma 1426, meningioma 708, pituitary 930), worse than
# the previous dataset's 1.57:1 where no correction won -- so it is re-measured
# rather than inherited. Exactly one correction may be active at a time.
# "undersample" is the only one that discards data; it exists for comparison.
# The TEST set is never balanced: macro-averaged metrics on all of it beat
# discarding held-out data to tidy up accuracy.
BALANCE = "weights"       # "weights" | "sampler" | "undersample" | None

# ── TRAINING ────────────────────────────────────────────
EPOCHS   = 80
LR       = 1e-3
WD       = 1e-4
DROPOUT  = 0.4
SEED     = 42

# A runaway guard, not a selection mechanism: the checkpoint already keeps the
# best-validation-loss epoch, so stopping early can only forfeit improvement.
# Patience 8 previously cut a 60-epoch cosine schedule at epoch 20, before the
# low-LR phase where the best epoch actually appears.
PATIENCE = 25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ablations compare against a shared baseline, so equal seeds must give equal
# numbers. torch.manual_seed is not enough on CUDA: cuDNN picks
# non-deterministic algorithms unless told not to.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ── PLOT STYLE ──────────────────────────────────────────
FACE    = '#F8F8F6'
PALETTE = {"train": '#378ADD', "val": '#D85A30',
           "good":  '#1D9E75', "accent": '#9B59B6'}
