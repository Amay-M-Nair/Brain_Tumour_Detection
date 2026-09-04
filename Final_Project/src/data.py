"""Data pipeline: cache in RAM, detect duplicates, transform.

Images are decoded once into an in-RAM uint8 array rather than re-read every
epoch, since num_workers must be 0 on Windows and the ablation re-reads the
training set dozens of times.

Duplicate detection is part of the pipeline, not a one-off investigation: the
previous dataset had 293 test images pixel-identical to training images under
different filenames. Cheng has none, but that is a measured result, and the
measurement only exists because the check still runs.

build_cache, drop_augmented and grouped_split serve the old ImageFolder layout
and are unused under Cheng; splits.py builds from .mat files instead.
"""
import random

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .config import (BATCH_SIZE, CACHE_SIZE, DUP_COSINE, DUP_L1, IMG_SIZE,
                     SEED, VAL_FRAC)


# ── loading ─────────────────────────────────────────────

def crop_to_content(img, thresh=10):
    """Trim the black border around a scan.

    The empty fraction varies per image, so resizing without cropping rescales
    the brain by an arbitrary per-image factor. Also makes RandomAffine's
    fill=0 correct, since everything outside the brain really is black.
    """
    arr = np.asarray(img)
    mask = arr > thresh
    if not mask.any():
        return img
    rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    return img.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))


def drop_augmented(samples):
    """Remove author-generated '-aug-' copies. Unused under Cheng."""
    return [(p, y) for p, y in samples if '-aug-' not in str(p).lower()]


def build_cache(root, size=CACHE_SIZE, keep_augmented=False):
    """Decode an ImageFolder split into a (N, size, size) uint8 array.

    Unused under Cheng. Returns paths so evaluation can stratify by file size.
    """
    base = ImageFolder(str(root))
    samples = base.samples if keep_augmented else drop_augmented(base.samples)

    cache  = np.zeros((len(samples), size, size), dtype=np.uint8)
    labels = np.zeros(len(samples), dtype=np.int64)
    for i, (path, label) in enumerate(samples):
        img = Image.open(path).convert("L")
        img = crop_to_content(img).resize((size, size), Image.BILINEAR)
        cache[i], labels[i] = np.asarray(img), label
    return cache, labels, base.classes, [p for p, _ in samples]


class CachedDataset(Dataset):
    """Serves cached images, transforming per access.

    Per access rather than baked in, which is the point of augmentation: the
    model must see a different version of each image every epoch.
    """

    def __init__(self, cache, labels, transform, indices=None):
        self.cache     = cache
        self.labels    = labels
        self.transform = transform
        self.indices   = np.arange(len(labels)) if indices is None else np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        return self.transform(Image.fromarray(self.cache[j], mode="L")), int(self.labels[j])


# ── duplicate detection ─────────────────────────────────

def fingerprints(cache, grid=56):
    """Contrast-invariant thumbnails, for recognising a scan already seen.

    Averaging to 56x56 discards the JPEG noise that stops two saves of one slice
    being byte-identical; z-scoring discards brightness and contrast shifts. What
    survives is anatomy, so cosine near 1 means the same scan, not a similar one.
    """
    n, size = cache.shape[0], cache.shape[1]
    block = size // grid
    x = cache[:, :grid * block, :grid * block]
    x = x.reshape(n, grid, block, grid, block).mean(axis=(2, 4)).reshape(n, -1)
    x = x.astype(np.float32)
    x -= x.mean(1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def contrast_l1(a, b):
    """Mean absolute difference between two batches, contrast matched.

    Cosine alone is not sufficient: different patients' slices genuinely
    resemble each other at thumbnail resolution, and a cosine-only threshold
    flags ~1 image in 10 as a repeat when it is not. Differencing at full cache
    resolution separates a repeat (L1 near 0) from a lookalike.
    """
    a = a.astype(np.float32).reshape(len(a), -1)
    b = b.astype(np.float32).reshape(len(b), -1)
    a = (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-6)
    b = (b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-6)
    return np.abs(a - b).mean(1)


def find_duplicates(query, reference, cos_thresh=DUP_COSINE, l1_thresh=DUP_L1,
                    chunk=512):
    """Which images in `query` already appear in `reference`.

    Returns (is_duplicate, match, cosine). Both thresholds must be met.
    """
    q, r = fingerprints(query), fingerprints(reference)
    match = np.zeros(len(q), dtype=np.int64)
    cos   = np.zeros(len(q), dtype=np.float32)
    for i in range(0, len(q), chunk):
        sims = q[i:i + chunk] @ r.T
        match[i:i + chunk] = sims.argmax(1)
        cos[i:i + chunk]   = sims.max(1)
    close = cos >= cos_thresh
    is_dup = np.zeros(len(q), dtype=bool)
    if close.any():
        idx = np.where(close)[0]
        is_dup[idx] = contrast_l1(query[idx], reference[match[idx]]) <= l1_thresh
    return is_dup, match, cos


def duplicate_groups(cache, cos_thresh=DUP_COSINE, l1_thresh=DUP_L1, chunk=512):
    """Cluster images so repeats of one scan share an id.

    Unique images get their own id, so the result feeds a grouped split
    directly. Connected components rather than pairs, because a scan can appear
    three times and all three must move together.
    """
    f = fingerprints(cache)
    rows, cols = [], []
    for i in range(0, len(f), chunk):
        sims = f[i:i + chunk] @ f.T
        a, b = np.where(sims >= cos_thresh)
        a = a + i
        keep = a < b                       # upper triangle only, no self-pairs
        rows.append(a[keep]); cols.append(b[keep])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    if len(rows):
        real = contrast_l1(cache[rows], cache[cols]) <= l1_thresh
        rows, cols = rows[real], cols[real]
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(f), len(f)))
    _, groups = connected_components(graph, directed=False)
    return groups


def grouped_split(labels, groups, val_frac=VAL_FRAC, seed=SEED):
    """Train/validation indices that never split a group.

    Unused under Cheng; splits.py splits by patient instead.
    """
    n_splits = max(2, round(1 / val_frac))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return next(splitter.split(np.zeros(len(labels)), labels, groups))


# ── transforms ──────────────────────────────────────────

class RandomBlur:
    """Blur by a random radius, to break dependence on sharpness.

    Measured previously, a 2px blur took a comparable model from 0.977 to 0.560.
    Real scans vary in sharpness with slice thickness, motion and reconstruction
    kernel.
    """

    def __init__(self, max_radius=1.5, p=0.5):
        self.max_radius, self.p = max_radius, p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        return img.filter(ImageFilter.GaussianBlur(random.uniform(0.2, self.max_radius)))


class RandomNoise:
    """Additive Gaussian noise, after ToTensor and before Normalize.

    Sigma reaches 0.06 on a 0-1 scale (~15 grey levels), the level at which an
    un-augmented model previously collapsed to 0.579.
    """

    def __init__(self, max_sigma=0.06, p=0.5):
        self.max_sigma, self.p = max_sigma, p

    def __call__(self, tensor):
        if random.random() > self.p:
            return tensor
        return (tensor + torch.randn_like(tensor) * random.uniform(0.0, self.max_sigma)
                ).clamp(0.0, 1.0)


def make_transforms(mean, std, augment=False, img_size=IMG_SIZE, jitter=True,
                    corrupt=False):
    """Build the train or eval pipeline.

    No horizontal flip: this dataset mixes acquisition planes, and a mirrored
    sagittal slice is anatomically impossible.

    ColorJitter stays, mildly. MRI intensity is not calibrated in absolute units
    the way CT Hounsfield numbers are, so jitter reproduces real acquisition
    variance rather than inventing a distortion.

    corrupt adds blur and noise, kept separate from the geometric augmentation
    because it answers brittleness to acquisition quality rather than
    positioning. Neither removes a source-signature shortcut: degrading the
    whole image degrades every class equally.
    """
    steps = [transforms.Resize((img_size, img_size))]
    if augment:
        if corrupt:
            steps.append(RandomBlur())
        steps.append(transforms.RandomAffine(
            degrees=12, translate=(0.06, 0.06), scale=(0.92, 1.08), fill=0))
        if jitter:
            steps.append(transforms.ColorJitter(brightness=0.15, contrast=0.15))
    steps.append(transforms.ToTensor())
    if augment and corrupt:
        steps.append(RandomNoise())
    steps.append(transforms.Normalize([mean], [std]))
    return transforms.Compose(steps)


def compute_stats(cache, indices, img_size=IMG_SIZE):
    """Normalisation statistics from the TRAINING split only.

    These constants are learned from data, so they leak like weights do:
    computing them over everything lets held-out information reach training.
    Computed after resizing, so they describe the tensors the model receives.
    """
    tf = transforms.Compose([transforms.Resize((img_size, img_size)),
                             transforms.ToTensor()])
    total = sq = n = 0.0
    for j in indices:
        x = tf(Image.fromarray(cache[j], mode="L"))
        total += x.sum().item()
        sq    += (x ** 2).sum().item()
        n     += x.numel()
    mean = total / n
    return mean, float(np.sqrt(sq / n - mean ** 2))


def make_loaders(train_ds, val_ds, test_ds=None, batch_size=BATCH_SIZE):
    """Wrap datasets in loaders.

    drop_last only for training, where a trailing batch of one leaves
    BatchNorm's variance undefined. False elsewhere or evaluation silently
    discards images.
    """
    loaders = [DataLoader(train_ds, batch_size, shuffle=True,
                          num_workers=0, drop_last=True),
               DataLoader(val_ds, batch_size, shuffle=False, num_workers=0)]
    if test_ds is not None:
        loaders.append(DataLoader(test_ds, batch_size, shuffle=False, num_workers=0))
    return tuple(loaders)


# ── class balance ───────────────────────────────────────

def class_weights(labels, n_classes=None):
    """Inverse-frequency weights for the loss, normalised to mean 1.

    Corrects the training signal without discarding an image. Cheng is 2.01:1.
    """
    counts = np.bincount(labels, minlength=n_classes or int(labels.max()) + 1)
    n_classes = len(counts)
    weights = len(labels) / (n_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def balanced_sampler(labels, indices=None, seed=SEED):
    """Sample minority classes more often, so each epoch is balanced.

    An alternative to weighting, not a complement -- both applies the correction
    twice. This changes which images are seen; weights change how much each
    counts. Sampling with replacement duplicates within training only.
    """
    from torch.utils.data import WeightedRandomSampler

    labels = labels if indices is None else labels[indices]
    counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    per_sample = (1.0 / np.maximum(counts, 1))[labels]
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.tensor(per_sample, dtype=torch.double),
                                 num_samples=len(labels), replacement=True,
                                 generator=g)


def undersample(labels, indices, seed=SEED):
    """Truncate every class to the smallest.

    The only option that discards data -- 31% of training here -- included so
    the ablation can measure whether that cost buys anything.
    """
    rng = np.random.default_rng(seed)
    lab = labels[indices]
    smallest = np.bincount(lab, minlength=int(lab.max()) + 1)
    smallest = smallest[smallest > 0].min()
    picked = [rng.choice(indices[lab == c], smallest, replace=False)
              for c in np.unique(lab)]
    return np.sort(np.concatenate(picked))
