"""Shared configuration. Every hyperparameter exists here exactly once.

Nothing in this module trains anything or reaches the network at import time.
That second point is deliberate: an earlier version resolved the dataset by
calling Kaggle inside the module body, which meant importing *anything* required
internet, and the whole pipeline failed offline even though every byte of data
was already cached on disk.
"""
from pathlib import Path

import torch

# ── PATHS ───────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
OUTPUTS   = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
CACHE_DIR = ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = OUTPUTS / "best_model.pth"
MANIFEST_PATH = OUTPUTS / "run_manifest.json"

KAGGLE_DATASET = "masoudnickparvar/brain-tumor-mri-dataset"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def dataset_root():
    """Path to the extracted dataset, preferring the local cache.

    kagglehub stores downloads under ~/.cache/kagglehub/datasets/<owner>/<name>/
    versions/<n>. Checking there first keeps the project usable offline and makes
    imports free; the network is touched only when the data is genuinely absent.
    """
    versions = (Path.home() / ".cache" / "kagglehub" / "datasets"
                ).joinpath(*KAGGLE_DATASET.split("/")) / "versions"
    if versions.is_dir():
        found = [p for p in versions.iterdir() if (p / "Training").is_dir()]
        if found:
            return max(found, key=lambda p: int(p.name) if p.name.isdigit() else -1)
    import kagglehub
    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


_root = dataset_root()
TRAIN_DIR, TEST_DIR = _root / "Training", _root / "Testing"

# ── DATA ────────────────────────────────────────────────
IMG_SIZE   = 128    # working resolution; 192 was tested and did not pay for itself
CACHE_SIZE = 224    # decoded once at this size so resolution can be ablated later
VAL_FRAC   = 0.15   # carved out of Training/; Testing/ is never touched
BATCH_SIZE = 32

# Duplicate detection. Both conditions must hold: a cosine threshold on its own
# flags roughly one image in ten, because MRI slices of different patients
# genuinely resemble each other at thumbnail resolution.
DUP_COSINE = 0.995
DUP_L1     = 0.15

# A second, looser threshold for the leakage that duplicate detection cannot
# reach. This dataset ships no patient IDs, so two different slices of one
# patient are not duplicates and no pixel test identifies them as such -- but
# they are far more similar than unrelated scans. Measured on the deduplicated
# test set: 5.3% of held-out images sit within 0.98 cosine of a same-class
# training image, against 0.0% for the nearest *different*-class image, which is
# the different-patient null. STRICT_LEAK drops those too. It is off by default
# because the threshold is a judgement rather than a measurement, and turning it
# on costs test images. It is ON: inspection of the surviving pairs showed
# unmistakable same-patient adjacent slices -- identical ventricles and skull
# outline, shifted a few millimetres -- and 100% of pairs above 0.95 share a
# class against a 77% baseline for unrelated scans. A smaller honest test set is
# worth more than a larger contaminated one.
STRICT_LEAK        = True
STRICT_LEAK_COSINE = 0.95

# Redundant copies inside Training/ are dropped too. The grouped split already
# stopped them straddling the train/val boundary, but one image counted twice
# still silently doubles its weight in the loss.
DEDUPE_TRAIN = True

# Class balancing. Deduplication left the classes uneven -- notumor lost most,
# because Br35H spread each subject's slices across both shipped folders, so
# honest cleaning removed more of it. Training is 1.57:1 and test 4.26:1.
#
# "weights" scales the loss by inverse class frequency. "sampler" draws minority
# images more often per epoch instead. "undersample" truncates every class to
# the smallest, which is the only option that discards data -- 28% of training
# and 66% of the test set -- and is provided for comparison rather than use.
#
# The TEST set is never balanced. Throwing away two thirds of the held-out data
# to make accuracy look tidy is worse than reporting macro-averaged metrics on
# all of it, which is what evaluation does.
BALANCE = "weights"       # "weights" | "sampler" | "undersample" | None

# ── TRAINING ────────────────────────────────────────────
EPOCHS   = 60
LR       = 1e-3
WD       = 1e-4
DROPOUT  = 0.4
SEED     = 42

# Early stopping cannot improve model selection here — the checkpoint already
# keeps the best-validation-loss epoch, so stopping early can only forfeit
# improvement, never gain it. It is a runaway guard and should be generous.
# A patience of 8 previously halted a 60-epoch cosine schedule at epoch 20,
# before the low-learning-rate phase where the best epoch actually appears.
PATIENCE = 25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The ablation compares configurations against a shared baseline, so two runs
# with the same seed must produce the same numbers. torch.manual_seed alone is
# not enough on CUDA — cuDNN picks non-deterministic algorithms unless told not to.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ── PLOT STYLE ──────────────────────────────────────────
FACE    = '#F8F8F6'
PALETTE = {"train": '#378ADD', "val": '#D85A30',
           "good":  '#1D9E75', "accent": '#9B59B6'}
