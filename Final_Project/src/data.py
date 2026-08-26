"""Data pipeline for the real Kaggle MRI scans.

The reference notebooks read images through ImageFolder on every access. That
is fine for 1000 synthetic 128px PNGs but wasteful here: these are 7,200 JPEGs
at up to 1000px, num_workers must be 0 on Windows, and the ablation in
notebook 04 re-reads the training set dozens of times. So each split is decoded
once into an in-RAM uint8 array and served from there.
"""
import random

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .config import BATCH_SIZE, CACHE_SIZE, IMG_SIZE, SEED, VAL_FRAC


def crop_to_content(img, thresh=10):
    """Trim the black border around a scan.

    Roughly 45% of a raw MRI slice is empty background, and it is not a
    consistent 45% — the scans come from different fields of view. Resizing
    without cropping therefore scales the brain itself by an arbitrary amount,
    so the network has to learn to ignore a nuisance variable we can simply
    remove. Cropping first also makes RandomAffine's fill=0 correct, since
    what remains outside the brain really is black.
    """
    arr = np.asarray(img)
    mask = arr > thresh
    if not mask.any():
        return img
    rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    return img.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))


def drop_augmented(samples):
    """Remove the pre-augmented duplicates the dataset author added.

    Only the meningioma class was padded this way (filenames contain '-aug-').
    Leaving them in the test split risks scoring the model on a rotated copy of
    an image it trained on, which would inflate the headline number.
    """
    return [(p, y) for p, y in samples if '-aug-' not in str(p).lower()]


def build_cache(root, size=CACHE_SIZE, keep_augmented=True):
    """Decode a split once into a (N, size, size) uint8 array.

    Grayscale conversion happens here and cannot be skipped: this dataset
    genuinely mixes 'L' and 'RGB' files, and a handful of the RGB ones carry
    real colour annotation marks rather than being grey stored in three channels.
    """
    base = ImageFolder(str(root))
    samples = base.samples if keep_augmented else drop_augmented(base.samples)

    cache  = np.zeros((len(samples), size, size), dtype=np.uint8)
    labels = np.zeros(len(samples), dtype=np.int64)
    for i, (path, label) in enumerate(samples):
        img = Image.open(path).convert("L")
        img = crop_to_content(img).resize((size, size), Image.BILINEAR)
        cache[i], labels[i] = np.asarray(img), label
    return cache, labels, base.classes


class CachedDataset(Dataset):
    """Serves images from the in-RAM cache, applying transforms per access.

    Transforms still run per access rather than being baked into the cache —
    that is the whole point of augmentation, the model must see a different
    version of each image every epoch.
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
        img = Image.fromarray(self.cache[j], mode="L")
        return self.transform(img), int(self.labels[j])


class RandomResample:
    """Downscale by a random factor and restore, destroying sharpness cues.

    This dataset is a merge of three collections with different native
    resolutions — the tumour classes are almost all 512x512, notumor almost
    never is. Resizing to a common size hides the dimensions but not the
    resampling signature, so how blurred an image is stays predictive of its
    label. Re-blurring by a random amount makes that signature uninformative.
    """

    def __init__(self, min_frac=0.35, p=0.8):
        self.min_frac, self.p = min_frac, p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        f = random.uniform(self.min_frac, 1.0)
        small = img.resize((max(8, int(w * f)), max(8, int(h * f))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


def make_transforms(mean, std, augment=False, img_size=IMG_SIZE, jitter=True,
                    resample=False):
    """Build the train or eval transform pipeline.

    No RandomHorizontalFlip. Axial and coronal slices are grossly symmetric so
    a flip would be harmless there, but this dataset mixes acquisition planes
    and a mirrored sagittal slice is anatomically impossible.

    ColorJitter is kept, mildly. MRI intensity is not calibrated in absolute
    units the way CT Hounsfield numbers are — scanner, sequence and windowing
    all shift brightness and contrast — so jittering them simulates real
    acquisition variance rather than inventing a fake one.
    """
    steps = [transforms.Resize((img_size, img_size))]
    if augment:
        if resample:
            steps.append(RandomResample())
        steps.append(transforms.RandomAffine(
            degrees=12, translate=(0.06, 0.06), scale=(0.92, 1.08), fill=0))
        if jitter:
            steps.append(transforms.ColorJitter(brightness=0.15, contrast=0.15))
    steps += [transforms.ToTensor(), transforms.Normalize([mean], [std])]
    return transforms.Compose(steps)


def stratified_split(labels, val_frac=VAL_FRAC, seed=SEED):
    """Split Training/ into train and val, preserving class proportions."""
    idx = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_frac, stratify=labels, random_state=seed)
    return train_idx, val_idx


def compute_stats(cache, indices, img_size=IMG_SIZE):
    """Normalisation statistics from the TRAINING split only.

    Using val or test images here would leak information about the data the
    model is meant to be judged on. Computed after cropping and resizing so the
    numbers describe the tensors the model actually sees.
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

    drop_last is True only for training, where a trailing batch of one would
    make BatchNorm's variance undefined. On val and test it must be False or
    evaluation would silently discard images.
    """
    loaders = [DataLoader(train_ds, batch_size, shuffle=True,
                          num_workers=0, drop_last=True),
               DataLoader(val_ds, batch_size, shuffle=False, num_workers=0)]
    if test_ds is not None:
        loaders.append(DataLoader(test_ds, batch_size, shuffle=False, num_workers=0))
    return tuple(loaders)
