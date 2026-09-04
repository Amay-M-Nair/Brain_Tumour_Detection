"""Reading the Cheng collection.

3,064 T1-weighted contrast-enhanced slices, 233 patients, two hospitals, one
protocol, every image 512x512, each .mat carrying a patient ID and a tumour
mask. One source, so the cross-collection signature problem is gone by
construction; that is also why there is no healthy class (see config.py).

The residual worry was internal: crop_to_content trims each image to its own
brain extent, so images reach the 224px cache by different resampling ratios,
and crop extent plausibly correlates with class. equalise() puts everything
through a common frequency ceiling first. Measured, it removes 0.005 of a
0.3187 texture lift, so it defaults OFF -- routing 512px through a 180px
ceiling costs real detail. The probe sits at 0.65 because of anatomy, not
resampling: mask centroid, area and extent alone predict class at 0.7063.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from .config import CACHE_SIZE

# Neither alphabetical nor zero-based, so mapped by name: an off-by-one here
# would silently relabel the dataset and nothing downstream would notice.
CHENG_LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary"}

# Below the smallest post-crop extent, so every image is downsampled into it.
# Upsampling some would leave them smoother than the rest, reintroducing the
# cue this removes.
EQUALISE_TO = 180


def equalise(img, cache_size=CACHE_SIZE, intermediate=EQUALISE_TO):
    """Common frequency ceiling, then cache size.

    LANCZOS down because it anti-aliases; a naive subsample aliases texture back
    differently depending on how much the image was cropped.
    """
    small = img.resize((intermediate, intermediate), Image.LANCZOS)
    return small.resize((cache_size, cache_size), Image.BILINEAR)


def _to_uint8(arr):
    """Per-image min-max to 0-255, matching the authors' own conversion."""
    a = arr.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return (255.0 * (a - lo) / (hi - lo)).astype(np.uint8)


def load_cheng(root, cache_size=CACHE_SIZE, equalise_first=True):
    """Read every .mat slice: images, labels, patient IDs, masks, names.

    MATLAB v7.3 is HDF5, hence h5py; arrays come out column-major, hence the .T.

    Masks travel through identical geometry -- same crop box, same resize,
    NEAREST so they stay binary. A mask that skips a step is misaligned with
    what it annotates, and every attention metric against it would be quietly
    wrong.
    """
    import h5py

    files = sorted(Path(root).glob("*.mat")) if not isinstance(root, list) else root
    images = np.zeros((len(files), cache_size, cache_size), dtype=np.uint8)
    masks  = np.zeros((len(files), cache_size, cache_size), dtype=bool)
    labels, pids, names = [], [], []

    for i, f in enumerate(files):
        with h5py.File(f, "r") as h:
            d = h["cjdata"]
            label = int(np.array(d["label"]).ravel()[0])
            pid_raw = np.array(d["PID"]).ravel()
            pid = "".join(chr(int(c)) for c in pid_raw).strip()
            img = _to_uint8(np.array(d["image"]).T)
            msk = np.array(d["tumorMask"]).T.astype(bool)

        pil = Image.fromarray(img, mode="L")
        box = _content_box(img)
        pil = pil.crop(box) if box else pil
        mpil = Image.fromarray(msk.astype(np.uint8) * 255, mode="L")
        mpil = mpil.crop(box) if box else mpil

        images[i] = np.asarray(equalise(pil, cache_size) if equalise_first
                               else pil.resize((cache_size, cache_size), Image.BILINEAR))
        masks[i] = np.asarray(mpil.resize((cache_size, cache_size), Image.NEAREST)) > 127
        labels.append(CHENG_LABELS[label])
        pids.append(pid)
        names.append(f.name)

    return images, labels, pids, masks, names


def _content_box(arr, thresh=10):
    """Crop box, so an image and its mask share one geometry."""
    m = arr > thresh
    if not m.any():
        return None
    rows, cols = np.where(m.any(1))[0], np.where(m.any(0))[0]
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
