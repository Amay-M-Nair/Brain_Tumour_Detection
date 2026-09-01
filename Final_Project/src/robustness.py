"""How far does the accuracy travel outside this dataset's conditions?

The standard objection to a high accuracy on a curated public MRI benchmark is
not that it is fabricated, it is that the benchmark is clean in ways a hospital
is not. A model trained on uniformly sharp, consistently windowed images can
learn to depend on that, and no ordinary evaluation reveals the dependence,
because the test split is clean in exactly the same ways.

So the test set is degraded and scored again. This is not external validation
and does not pretend to be -- simulating another site properly needs another
site's data. It answers a narrower question honestly: does the accuracy survive
the variation that separates one scanner from another?

The corruptions are the families that actually differ between acquisitions
rather than a generic benchmark: noise for field strength and acquisition time,
blur for slice thickness and motion, gamma for windowing conventions that are
not standardised between sites, and a bias field for the smooth B1
inhomogeneity that is specific to MRI.
"""
import numpy as np
import torch
from PIL import Image, ImageFilter

from .data import CachedDataset, make_transforms
from .metrics import macro_f1, predict


def add_noise(images, sigma, rng):
    return np.clip(images.astype(np.float32) + rng.normal(0, sigma, images.shape),
                   0, 255).astype(np.uint8)


def add_blur(images, radius):
    return np.stack([np.asarray(Image.fromarray(a, mode="L")
                                .filter(ImageFilter.GaussianBlur(radius)))
                     for a in images])


def add_gamma(images, gamma):
    return np.clip(((images / 255.0) ** gamma) * 255, 0, 255).astype(np.uint8)


def add_bias_field(images, amplitude):
    """Smooth multiplicative intensity drift -- the classic MRI inhomogeneity."""
    h, w = images.shape[1:]
    yy, xx = np.mgrid[0:h, 0:w]
    field = 1 + amplitude * np.sin(2 * np.pi * xx / w) * np.cos(2 * np.pi * yy / h)
    return np.clip(images.astype(np.float32) * field, 0, 255).astype(np.uint8)


def corruptions(seed=0):
    rng = np.random.default_rng(seed)
    return [("noise sigma=5",     lambda a: add_noise(a, 5, rng)),
            ("noise sigma=15",    lambda a: add_noise(a, 15, rng)),
            ("blur radius 1px",   lambda a: add_blur(a, 1)),
            ("blur radius 2px",   lambda a: add_blur(a, 2)),
            ("gamma 0.7",         lambda a: add_gamma(a, 0.7)),
            ("gamma 1.4",         lambda a: add_gamma(a, 1.4)),
            ("bias field +/-20%", lambda a: add_bias_field(a, 0.20))]


def robustness_test(model, cache, labels, mean, std, img_size, n_classes=4,
                    batch_size=64, seed=0):
    """Accuracy and macro F1 on the clean test set and under each corruption."""
    tf = make_transforms(mean, std, augment=False, img_size=img_size)

    def score(arrays):
        loader = torch.utils.data.DataLoader(
            CachedDataset(arrays, labels, tf), batch_size=batch_size, shuffle=False)
        y, p, _ = predict(model, loader)
        return {"acc": float((y == p).mean()), "f1": macro_f1(y, p, n_classes)}

    out = {"clean": score(cache)}
    for label, fn in corruptions(seed):
        out[label] = fn(cache) if False else score(fn(cache))
    return out


def print_robustness(rows):
    clean = rows["clean"]
    print(f"  {'condition':<24}{'accuracy':>10}{'macro F1':>10}{'drop (F1)':>12}")
    print("  " + "-" * 56)
    for label, r in rows.items():
        drop = "" if label == "clean" else f"{clean['f1'] - r['f1']:>+12.4f}"
        print(f"  {label:<24}{r['acc']:>10.4f}{r['f1']:>10.4f}{drop}")
    worst = min((v["f1"] for k, v in rows.items() if k != "clean"), default=clean["f1"])
    print(f"\n  worst case across the suite: macro F1 {worst:.4f} "
          f"against a clean {clean['f1']:.4f}")
