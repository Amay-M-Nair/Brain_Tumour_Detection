"""Attention, and what can honestly be claimed about it without masks.

This dataset ships 7,200 JPEGs and nothing else. There are no tumour
annotations, which removes the strongest evidence available for a question like
"is the model looking at the lesion": with masks you can score the fraction of
attention inside the tumour, the hit rate of the hottest pixel, and the overlap
against a centre-blob control. None of that is possible here.

What remains possible is worth being precise about, because the gap matters.

  Anatomical containment. A brain mask can be derived from the image itself, so
  "how much of the evidence is on skull, orbit and background rather than brain"
  becomes a measurement rather than an impression.

  Occlusion sensitivity. Sliding a patch across the image and watching the
  predicted probability fall needs no annotation at all, and answers the
  stronger question -- not where the model looks, but which regions it needs.
  Being a perturbation of the input rather than an inspection of gradients, it
  is also causal: the prediction actually changes.

The distinction that keeps this honest: a heatmap can sit on a lesion the
classifier would happily do without. Attention is not reliance, and only the
occlusion map speaks to the second.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from .config import DEVICE


def _normalise(cam):
    return (cam - cam.min()) / (cam.max() - cam.min()) if cam.max() > cam.min() else cam


# ── anatomy, derived from the image ─────────────────────

def brain_mask(image, percentile=35, min_frac=0.03):
    """Approximate intracranial region: threshold, open, largest component, fill.

    Crude, and deliberately not presented as segmentation. Its only job is to
    separate "inside the head" from skull margin, orbits and background well
    enough to ask whether the heat is on tissue. It is derived from the image
    itself, so it introduces no external labels and cannot leak.

    Returns None when the result is implausibly small, so a failed extraction is
    skipped rather than silently scored as a model failure.
    """
    body = image > 10
    if not body.any():
        return None
    t = np.percentile(image[body], percentile)
    m = ndimage.binary_opening(image > t, structure=np.ones((7, 7)))
    lab, n = ndimage.label(m)
    if n == 0:
        return None
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    m = ndimage.binary_fill_holes(lab == (np.argmax(sizes) + 1))
    return m if m.mean() >= min_frac else None


def containment(cam, brain):
    """How much of the attention sits on brain rather than skull or background.

    Returns the mass fraction inside the mask, the mask's own area fraction as
    the null, and the ratio. A ratio of 1 means the heat is spread as if it
    ignored anatomy entirely; below 1 means it is concentrated *outside* the
    brain, which is the case worth catching.
    """
    total = cam.sum()
    if total <= 0 or brain is None or not brain.any():
        return None
    inside = float(cam[brain].sum() / total)
    area   = float(brain.mean())
    return {"inside": inside, "area": area, "ratio": inside / area if area else 0.0}


def concentration(cam, level=0.5):
    """Fraction of the image above `level` -- how diffuse the map is.

    A normalised CAM always reaches 1 somewhere, so this measures spread rather
    than intensity. Lower is tighter.
    """
    return float((cam >= level).mean())


# ── occlusion sensitivity, no annotations required ──────

def occlusion_map(model, image, transform, patch=24, stride=12, class_idx=None,
                  batch_size=64):
    """Slide a patch across the image and record how far the class score falls.

    This needs no ground truth at all, and answers the question a CAM cannot:
    which regions the prediction actually depends on. The patch is filled with
    the median brain intensity rather than black, because a black square is a
    high-contrast object that appears in no training image, and the drop it
    causes would partly measure surprise at the artefact.

    Higher values mean the model needed that region.
    """
    from PIL import Image as PILImage

    model.eval()
    fill = int(np.median(image[image > 10])) if (image > 10).any() else 0
    with torch.no_grad():
        base = model(transform(PILImage.fromarray(image, mode="L")
                               ).unsqueeze(0).to(DEVICE))
        idx = int(base.argmax(1)) if class_idx is None else int(class_idx)
        base_p = float(F.softmax(base, 1)[0, idx])

    positions, patches = [], []
    h, w = image.shape
    for y in range(0, h - patch + 1, stride):
        for x in range(0, w - patch + 1, stride):
            occ = image.copy()
            occ[y:y + patch, x:x + patch] = fill
            positions.append((y, x))
            patches.append(transform(PILImage.fromarray(occ, mode="L")))

    drops = []
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = torch.stack(patches[i:i + batch_size]).to(DEVICE)
            probs = F.softmax(model(batch), 1)[:, idx].cpu().numpy()
            drops.extend(base_p - probs)

    heat = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    for (y, x), d in zip(positions, drops):
        heat[y:y + patch, x:x + patch] += d
        count[y:y + patch, x:x + patch] += 1
    heat /= np.maximum(count, 1)
    return _normalise(np.maximum(heat, 0)), idx, base_p
