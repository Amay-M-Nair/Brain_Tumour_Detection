"""Attention, and what can honestly be claimed about it.

Four things live here, in increasing order of evidential weight:

  brain_mask / containment   anatomy derived from the image itself, so "is the
                             heat on tissue or on skull" is a measurement. No
                             external labels, so it cannot leak.
  occlusion_map              slide a patch, watch the class score fall. Needs no
                             annotation, and is causal: the prediction changes.
  grad_cam                   pytorch-grad-cam, with the layer chosen by
                             measurement rather than intuition.
  occlusion_effect           blank the radiologist's actual mask and measure
                             what the prediction loses, against a matched
                             control elsewhere in the brain.

The distinction that keeps this honest: a heatmap can sit on a lesion the
classifier would happily do without. Attention is not reliance, and only the
occlusion tests speak to reliance.
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

    Crude, and deliberately not segmentation -- it only separates "inside the
    head" from skull, orbits and background. Returns None when implausibly
    small, so a failed extraction is skipped rather than scored as a failure.
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
    """Mass fraction of the map inside `brain`, against the area null.

    Ratio 1 means the heat is spread as if it ignored anatomy; below 1 means it
    is concentrated outside the brain, which is the case worth catching.
    """
    total = cam.sum()
    if total <= 0 or brain is None or not brain.any():
        return None
    inside = float(cam[brain].sum() / total)
    area   = float(brain.mean())
    return {"inside": inside, "area": area, "ratio": inside / area if area else 0.0}


def concentration(cam, level=0.5):
    """Fraction of the image above `level` -- how diffuse the map is.

    A normalised CAM always reaches 1 somewhere, so this measures spread, not
    intensity. Lower is tighter.
    """
    return float((cam >= level).mean())


# ── occlusion sensitivity, no annotations required ──────

def occlusion_map(model, image, transform, patch=24, stride=12, class_idx=None,
                  batch_size=64):
    """Slide a patch and record how far the class score falls.

    Answers what a CAM cannot: which regions the prediction depends on. Higher
    means the model needed that region. Filled with median brain intensity, not
    black -- a black square appears in no training image, so the drop would
    partly measure surprise at the artefact.
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


# ── occlusion against ground truth ──────────────────────

def shifted_control(mask, brain, rng, tries=300):
    """A same-shape, same-area region placed elsewhere inside the brain.

    Occluding anything costs probability, so the lesion's cost only means
    something against an equal occlusion of non-lesion tissue. Translated rather
    than redrawn, so the two differ in location and nothing else. Returns None
    when no placement avoids the lesion, so that image is skipped rather than
    scored against a contaminated control.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    h, w = mask.shape
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    for _ in range(tries):
        dy = int(rng.integers(-y0, h - y1)) if h - y1 > -y0 else 0
        dx = int(rng.integers(-x0, w - x1)) if w - x1 > -x0 else 0
        shifted = np.zeros_like(mask)
        shifted[ys + dy, xs + dx] = True
        if (shifted & mask).any():
            continue
        if brain is not None and int(shifted[brain].sum()) != int(shifted.sum()):
            continue
        return shifted
    return None


def occlusion_effect(model, image, mask, transform, class_idx=None, control=None):
    """Blank a region and measure what the prediction loses.

    The strongest attention test available here: it uses the radiologist's
    annotation rather than a heuristic, and perturbs the input rather than
    inspecting gradients.

    Returns base probability, probability after occluding `mask`, and after
    occluding `control` if given.
    """
    from PIL import Image as PILImage

    model.eval()
    fill = int(np.median(image[image > 10])) if (image > 10).any() else 0

    def prob(img, idx=None):
        with torch.no_grad():
            out = model(transform(PILImage.fromarray(img, mode="L")
                                  ).unsqueeze(0).to(DEVICE))
            i = int(out.argmax(1)) if idx is None else int(idx)
            return float(F.softmax(out, 1)[0, i]), i

    base_p, idx = prob(image, class_idx)

    lesion_img = image.copy()
    lesion_img[mask] = fill
    lesion_p, _ = prob(lesion_img, idx)

    control_p = None
    if control is not None:
        control_img = image.copy()
        control_img[control] = fill
        control_p, _ = prob(control_img, idx)

    return {"base": base_p, "lesion": lesion_p, "control": control_p, "class": idx}


# ── Grad-CAM, and the controls that decide whether it means anything ──

CAM_LAYERS = ("block2", "block3", "block3b", "block4", "block4b")


def grad_cam(model, image, transform, layer="block4b", class_idx=None):
    """Grad-CAM at a chosen stage, upsampled to the image.

    Delegates to pytorch-grad-cam (jacobgil); a hand-rolled copy is one more
    thing that can drift from what the name implies. Verified against the
    version it replaced: 80 scans, IoU identical to four decimals, corr 1.0000.

    What is local is what the library does not provide -- centre_blob and
    area_matched_iou below, which turn a heatmap into a measurement.

    The default layer is a result, not a preference. Global average pooling
    makes the final stage's gradient spatially constant per channel, which
    argues for going shallower; measured against the masks that is wrong, with
    the finest stage worst by a factor of fifteen and the coarsest best.
    """
    from PIL import Image as PILImage
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    target = dict(model.features.named_children()).get(layer)
    if target is None:
        raise ValueError(f"no layer {layer!r}; available: "
                         f"{[n for n, _ in model.features.named_children()]}")

    model.eval()
    x = transform(PILImage.fromarray(image, mode="L")).unsqueeze(0).to(DEVICE)
    if class_idx is None:
        with torch.no_grad():
            class_idx = int(model(x).argmax(1))
    idx = int(class_idx)

    # context manager so hooks are released; leaving them attached across a few
    # thousand calls leaks memory and nothing else, so nobody looks for it
    with GradCAM(model=model, target_layers=[target]) as cam:
        heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(idx)])[0]

    h, w = image.shape
    heat = np.asarray(PILImage.fromarray(heat).resize((w, h), PILImage.BILINEAR))
    return _normalise(heat), idx


def centre_blob(shape, sigma_frac=0.22):
    """A Gaussian at the image centre -- the control every CAM has to beat.

    Tumours are not uniformly distributed across a slice, so a map that just
    points at the middle scores well against any lesion mask. A CAM's overlap
    reported without this beside it says nothing.
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    sigma = sigma_frac * max(h, w)
    return _normalise(np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))))


def area_matched_iou(cam, mask):
    """IoU after selecting exactly as many pixels as the mask contains.

    A fixed threshold caps IoU artificially: `cam >= 0.5` selects ~28% of the
    frame against masks averaging 2.4%, so the ceiling is ~0.08 and the metric
    measures the threshold. Top-N selection removes that ceiling and makes maps
    of different spread comparable -- perfect overlap 1.0, disjoint 0.0.
    """
    n = int(mask.sum())
    if n == 0 or n >= mask.size:
        return None
    flat = np.asarray(cam, dtype=np.float64).ravel()
    top = np.argpartition(-flat, n - 1)[:n]
    sel = np.zeros(flat.size, dtype=bool)
    sel[top] = True
    sel = sel.reshape(mask.shape)
    union = (sel | mask).sum()
    return float((sel & mask).sum() / union) if union else None
