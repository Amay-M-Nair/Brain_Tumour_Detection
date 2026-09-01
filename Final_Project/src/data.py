"""Data pipeline: decode once, detect duplicates, split without leaking.

Two halves. The first is ordinary loading — decode each split once into an
in-RAM uint8 array rather than re-reading JPEGs every epoch, since num_workers
must be 0 on Windows and the ablation re-reads the training set dozens of times.

The second half exists because this dataset's shipped train/test split is not
actually a split. Measured on it: 293 of its test images are pixel-identical to
training images under different filenames, and Training/ holds 464 redundant
copies internally. Duplicate detection is therefore not an investigation that
was run once, it is part of the pipeline, and the notebooks assert its result.
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

    Roughly a third to a half of a raw slice is empty background, and it is not
    a consistent fraction — these scans come from different fields of view, so
    the brain occupies a different share of each image. Resizing without
    cropping therefore rescales the brain by an arbitrary per-image factor, and
    the network has to spend capacity ignoring a nuisance variable that can
    simply be removed.

    It has a second use: RandomAffine fills rotated-in corners with a constant,
    and fill=0 is only correct if everything outside the brain really is black.
    """
    arr = np.asarray(img)
    mask = arr > thresh
    if not mask.any():
        return img
    rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    return img.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))


def drop_augmented(samples):
    """Remove the copies the dataset author generated, from whichever split.

    Only meningioma was padded this way — 100 files in Training, 103 in Testing,
    all with '-aug-' in the name. Removing them from Testing alone is the
    obvious half and not enough: the copies and their originals both sit in
    Training, so a split that assigns images independently puts a rotated
    duplicate in validation and its source in train.
    """
    return [(p, y) for p, y in samples if '-aug-' not in str(p).lower()]


def build_cache(root, size=CACHE_SIZE, keep_augmented=False):
    """Decode a split once into a (N, size, size) uint8 array.

    Grayscale conversion cannot be skipped: this dataset mixes 'L' and 'RGB'
    files, and a few of the RGB ones carry real colour annotation marks rather
    than being grey stored three times.

    Paths come back alongside the pixels because the filename is not decoration
    — evaluation stratifies accuracy by native file size, which is impossible if
    the cache forgets which file each row came from.
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
    """Serves images from the in-RAM cache, applying transforms per access.

    Transforms run per access rather than being baked into the cache — that is
    the entire point of augmentation, the model must see a different version of
    each image every epoch.
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

    Averaging down to a 56x56 grid discards the JPEG noise that stops two saves
    of one slice from being byte-identical, and z-scoring each thumbnail
    discards the brightness and contrast differences a re-encode leaves behind.
    What survives is anatomy, so a cosine near 1 means the same scan rather than
    merely a similar one.
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

    The cosine alone is not sufficient. MRI slices of *different* patients
    genuinely resemble each other at thumbnail resolution, and a cosine-only
    threshold flags roughly one image in ten as a repeat when it is not one.
    Standardising each image and differencing at full cache resolution separates
    a repeated scan (L1 near 0) from a lookalike.
    """
    a = a.astype(np.float32).reshape(len(a), -1)
    b = b.astype(np.float32).reshape(len(b), -1)
    a = (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-6)
    b = (b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-6)
    return np.abs(a - b).mean(1)


def find_duplicates(query, reference, cos_thresh=DUP_COSINE, l1_thresh=DUP_L1,
                    chunk=512):
    """Which images in `query` already appear in `reference`.

    Returns (is_duplicate, match, cosine): a flag per query image, the index of
    its nearest reference image, and how close that match was. Both thresholds
    must be met, for the reason contrast_l1 gives.
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
    """Cluster a split's own images so repeats of one scan share an id.

    Unique images get an id to themselves, so the result can be handed straight
    to grouped_split. Clusters are connected components rather than pairs,
    because a scan can appear three times and all three copies must move
    together.
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
    """Train/validation indices that never split a duplicate cluster.

    A plain stratified split assigns images independently, so one copy of a
    repeated scan lands in train and the other in validation, and the model is
    then asked to generalise to an image it has memorised. Splitting whole
    groups makes that impossible. It costs exact control over the split size —
    the fold count is the nearest integer to 1/val_frac — which is a fair price
    for a validation set that is genuinely held out.
    """
    n_splits = max(2, round(1 / val_frac))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return next(splitter.split(np.zeros(len(labels)), labels, groups))


# ── transforms ──────────────────────────────────────────

class RandomBlur:
    """Blur by a random radius, to break dependence on sharpness.

    This dataset is uniformly sharp within each source collection, and a model
    trained only on that learns to rely on it — measured previously, a 2px blur
    took a comparable model from 0.977 to 0.560. Real scans vary in sharpness
    with slice thickness, motion and reconstruction kernel.
    """

    def __init__(self, max_radius=1.5, p=0.5):
        self.max_radius, self.p = max_radius, p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        return img.filter(ImageFilter.GaussianBlur(random.uniform(0.2, self.max_radius)))


class RandomNoise:
    """Additive Gaussian noise, applied after ToTensor and before Normalize.

    Sigma reaches 0.06 on a 0-1 scale, about 15 grey levels of 255 — chosen to
    be the level at which an un-augmented model previously collapsed to 0.579,
    so the augmentation covers the regime that actually fails.
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

    No RandomHorizontalFlip. Axial and coronal slices are grossly symmetric so a
    flip would be harmless there, but this dataset mixes acquisition planes and
    a mirrored *sagittal* slice is anatomically impossible.

    ColorJitter stays, mildly. MRI intensity is not calibrated in absolute units
    the way CT Hounsfield numbers are — scanner, sequence and windowing all
    shift brightness and contrast — so jittering reproduces real acquisition
    variance rather than inventing a distortion that never happens.

    corrupt adds blur and noise, kept separate from the geometric augmentation
    because they answer a different objection: affine transforms address
    positioning, blur and noise address brittleness to acquisition quality. What
    neither can do is remove the source-signature shortcut — degrading the whole
    image leaves every class degraded equally.
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

    These two constants are learned from data, which makes them as capable of
    leaking as model weights: computing them over the whole dataset lets
    information about the held-out scans reach the training pipeline. Computed
    after cropping and resizing so they describe the tensors the model receives.
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

    drop_last is True only for training, where a trailing batch of one makes
    BatchNorm's variance undefined. On validation and test it must be False or
    evaluation silently discards images.
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

    Deduplication left this dataset uneven -- notumor lost the most, because
    Br35H spread each subject's slices across both shipped folders, so removing
    same-patient bleed removed more of that class than of any other. Weighting
    the loss corrects the training signal without discarding a single image,
    which matters when the imbalance is a consequence of cleaning rather than a
    property of the disease.
    """
    counts = np.bincount(labels, minlength=n_classes or int(labels.max()) + 1)
    n_classes = len(counts)
    weights = len(labels) / (n_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def balanced_sampler(labels, indices=None, seed=SEED):
    """Sample minority classes more often, so each epoch is class-balanced.

    An alternative to weighting rather than a complement -- doing both applies
    the correction twice. This one changes which images the model sees; weights
    change how much each one counts. Sampling with replacement means a minority
    image can appear several times in one epoch, which is a form of duplication,
    but it stays inside training and never reaches validation or test.
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
    """Truncate every class to the size of the smallest.

    The only balancing option that throws data away -- 28% of training here --
    and it is included so the ablation can measure whether that cost buys
    anything, not because it is recommended.
    """
    rng = np.random.default_rng(seed)
    lab = labels[indices]
    smallest = np.bincount(lab, minlength=int(lab.max()) + 1)
    smallest = smallest[smallest > 0].min()
    picked = [rng.choice(indices[lab == c], smallest, replace=False)
              for c in np.unique(lab)]
    return np.sort(np.concatenate(picked))
