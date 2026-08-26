# Brain Tumour MRI Classification — a CNN built from scratch

A four-class brain tumour classifier (glioma / meningioma / notumor / pituitary)
trained on 7,200 real MRI scans. Every layer is hand-built in PyTorch — **no
transfer learning, no pretrained weights**. The point of the project is that the
architecture was derived rather than downloaded.

The headline result is 94.4% test accuracy. The more interesting result is *why
that number is misleading*, which is documented below and in notebook 03.

---

## Results

| Metric | Value |
|---|---|
| Validation accuracy | 0.9833 (epoch 56 of 60) |
| **Test accuracy** | **0.9439** (1413 / 1497) |
| Macro F1 | 0.9418 |
| Macro AUC (one-vs-rest) | 0.9866 |
| Bootstrap 95% CI on accuracy | [0.9325, 0.9539] |
| Worst class recall | glioma, 0.8300 |
| Training time | 10.0 min (RTX 3050 Laptop, 4 GB) |

### The number that actually describes the model

The test set splits into scans where a dataset artefact gives the answer away
and scans where it does not:

| Subset | n | Accuracy |
|---|---|---|
| `notumor` at non-512×512 | 397 | 1.000 |
| All 512×512 scans | 899 | ~0.98 |
| **Tumour scans at non-512×512** | **201** | **0.677** |

On the slice where the shortcut is unavailable, accuracy is **0.68**. The
headline 0.9439 is a weighted average dominated by the easy 87% of the test set.
Both numbers belong in any honest write-up.

---

## The shortcut

This dataset is a merge of three collections. The three tumour classes come from
figshare/SARTAJ and are overwhelmingly 512×512; `notumor` comes from Br35H and
almost never is — 1383 of 1400 training `notumor` images are some other size,
while training glioma is 1400/1400 at 512×512.

So **"not 512×512 ⇒ notumor" is a flawless rule on the training data**, and the
model learned it. Resizing everything to 128px does not remove the signal,
because a 512→128 downscale and a 225→128 downscale leave different sharpness
and JPEG-artefact statistics.

Evidence, all in `notebooks/03_evaluation.ipynb`:

- Test glioma accuracy: **0.975** at 512×512 vs **0.211** at other sizes
- Tumour→`notumor` error rate: 0.001 at 512×512 vs 0.164 at other sizes
- Grad-CAM attends to the skull margin, not the lesion
- The most confident errors are large, obvious tumours called `notumor` at 1.00 confidence

Validation is blind to all of this, because the validation split is carved from
`Training/` and inherits the same correlation. Only the held-out `Testing/` split
and a size-stratified breakdown expose it.

**A mitigation was tried and failed.** `RandomResample` (random downscale/upscale
round-trip, destroying sharpness cues) left the glioma resolution gap at 0.727 vs
0.706 baseline — no improvement. The mechanism was never blur; it is a missing
region of the training distribution. Augmentation cannot invent data that isn't
there. This negative result is reported rather than hidden.

---

## Architecture

```
ConvBlock = Conv2d(bias=False) -> BatchNorm2d -> ReLU     [Kaiming init]

1 -> 32 -> pool -> 64 -> pool -> 128 -> 128 -> pool -> 256 -> 256
                                                        -> GlobalAvgPool
                                                        -> 256 -> 128 -> 64 -> 4
```

| | params | receptive field @128px |
|---|---|---|
| default | 429,732 | 38 px |
| `deep=True` (used) | 1,167,780 | 62 px |

The capacity limit here is **receptive field, not parameter count**. The plain
stack sees only 38px of a 128px scan at its deepest layer — too little context to
judge a lesion's margin or its position relative to the midline. Adding a second
conv at stages 3 and 4 raises it to 62px. This was predicted in notebook 02
before any evaluation, and the ablation confirmed it: **+0.0250 validation
accuracy**, against a 3-seed noise floor of 0.0024.

### Final training recipe

Selected by the notebook 04 ablation, on validation only — the test set was
scored once, at the end.

- 128×128 grayscale, cropped to content, batch 32
- Adam, lr 1e-3, weight decay 1e-4, `CosineAnnealingLR(T_max=60, eta_min=1e-6)` stepped per epoch
- 60 epochs, early stopping patience 8, checkpoint on best validation loss
- `deep=True`, dropout 0.0, label smoothing 0.1
- Augmentation: `RandomAffine(degrees=12, translate=0.06, scale=(0.92,1.08), fill=0)` + `ColorJitter(brightness=0.15, contrast=0.15)`

Two deliberate omissions:

- **No `RandomHorizontalFlip`.** Axial and coronal slices are grossly symmetric so
  a flip would be harmless there — but this dataset mixes acquisition planes, and
  a mirrored *sagittal* slice is anatomically impossible.
- **No class weighting.** The classes are exactly balanced at 1400 each. Verified
  unnecessary rather than silently dropped.

---

## Layout

```
Final_Project/
  src/
    config.py     paths, hyperparameters, cuDNN determinism, plot palette
    data.py       crop_to_content, build_cache, CachedDataset, transforms, splits
    model.py      ConvBlock, BrainTumourCNN, MLPHead, BrainTumourNet, receptive_field
    engine.py     train_one_epoch, evaluate, EarlyStopper, fit, run_experiment
    metrics.py    predict, confusion, per_class_report, roc_ovr, bootstrap_ci
    viz.py        curves, confusion, ROC, feature maps, Grad-CAM
  notebooks/
    01_data.ipynb        cache build, split, normalisation stats, sample grids
    02_train.ipynb       receptive-field calculation, 60-epoch run, checkpoint
    03_evaluation.ipynb  confusion, ROC, bootstrap CIs, Grad-CAM, shortcut audit
    04_ablation.ipynb    9 one-factor-at-a-time runs + full-data confirmation
  outputs/               figures, best_model.pth, history.npy

Model_Foundations_For_References/
  1_CNN.ipynb ... 8_Regularisation.ipynb
```

The reference notebooks are learning artefacts trained on 1,000 **synthetic**
images that are trivially separable. Their 100% test accuracy and *negative*
generalisation gap demonstrate that the machinery works, not that the model does.
All real results live in `Final_Project/`.

Notebooks are thin drivers — they import from `src/` and do orchestration and
plotting only. There is exactly one model definition in the repository.

---

## Data pipeline

Source: Kaggle `masoudnickparvar/brain-tumor-mri-dataset` v2 (7,200 images,
1400 train + 400 test per class), via `kagglehub`. Not committed — see `.gitignore`.

- `Training/` splits 85:15 stratified into **4760 train / 840 val**;
  `Testing/` is held out untouched as **1497 test**.
- The 103 pre-augmented meningioma duplicates (`-aug-` in the filename) are
  **dropped from `Testing/`** — scoring on a rotated copy of a training image
  would inflate the headline. The 100 in `Training/` are kept.
- `crop_to_content` runs first. ~45% of a raw slice is black border, and it is
  not a *consistent* 45%, so resizing without cropping scales the brain by an
  arbitrary amount. It also makes `RandomAffine(fill=0)` correct.
- Grayscale conversion is mandatory, not cosmetic: the dataset genuinely mixes
  `L` and `RGB` files, and some RGB ones carry real coloured annotation marks.
- Normalisation stats (**MEAN 0.2250, STD 0.1901**) are computed on the training
  split only, after cropping and resizing.
- Each split is decoded **once** into an in-RAM uint8 array at 224px. On Windows
  `num_workers=0`, so JPEG decode — not the network — is the bottleneck. This
  takes epochs from ~35s to under 10s, which is what makes the ablation
  affordable. Caching at 224 keeps the 128-vs-192 resolution ablation honest.

---

## Verification

Each notebook ends with hard assertions rather than eyeballed plots:

- train / val / test index sets are disjoint; `Training` and `Testing`
  `class_to_idx` agree
- `len(test_ds) == 1497` after the `-aug-` drop
- cache MD5s are byte-identical across re-runs
- 40 images can be overfit to ~100% train accuracy — if not, the model or loss is broken
- shuffled-label control sits at 25%
- the checkpoint reloads and reproduces test accuracy exactly

All four notebooks run top to bottom in a fresh kernel with zero errors.

---

## Limitations

1. **Source-signature shortcut** — demonstrated, not suspected. See above. This
   is the dominant caveat and it caps how much the headline number means.
2. **No patient IDs.** Multiple slices come from the same patient, so a random
   split puts near-duplicate slices in both train and test. This is the standard
   critique of this dataset in the literature and inflates every published number
   on it, including this one.
3. **Pre-augmented duplicates** exist in the source data; removed from test,
   retained in training.
4. **Single held-out test set, single seed** for the final model. The bootstrap CI
   captures sampling noise in the test set, not training-run variance.
5. **Confidence is not calibrated.** Label smoothing at 0.1 caps max softmax
   around 0.95, so absolute probability thresholds are not meaningful; quantile
   referral is used instead.
6. **Ablation ran on a 1200-image subset for 25 epochs.** Relative ordering is the
   claim; absolute numbers from notebook 04 are not comparable to notebook 02's.

Published from-scratch CNNs on this exact dataset report 94–98%. This project
lands at the bottom of that band — which, given the caveats above, is the honest
end of it.

---

## Running it

```bash
pip install torch torchvision numpy scikit-learn matplotlib pillow kagglehub nbformat
```

Download the dataset to `Final_Project/data/brain_tumour/` (`Training/` and
`Testing/` subfolders), then run the notebooks in order 01 → 04. Notebook 02
writes `outputs/best_model.pth`; notebooks 03 and 04 read it.

On Windows, `sys.stdout.reconfigure(encoding='utf-8')` is needed before running
headless — the default console encoding crashes on the box-drawing characters.
