"""Grad-CAM galleries, kept out of the notebooks deliberately.

Phase 5 already reports the numbers that decide whether Grad-CAM means anything
here -- the layer sweep, the per-class IoU, the centre-blob control. These are
the pictures, and pictures are the part most likely to be quoted without their
qualifiers, so they live in one place with the qualifier printed on the figure.

Three figures, and the first is the important one:

  gradcam_spectrum.png    best to worst per class, chosen by quantile rather
                          than by eye. Showing the best example per class is how
                          heatmap figures usually mislead; this shows the whole
                          range, so the typical case and the failure are visible
                          in the same picture.
  gradcam_meningioma.png  the class that genuinely localises, paired with the
                          annotation it is being scored against.
  gradcam_pituitary_failure.png
                          the class where a Gaussian blob that knows nothing
                          scores higher than the model.

Run after Phase 3 has written a checkpoint:  python make_gradcam_gallery.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import data, engine, explain, metrics, splits, viz
from src.config import CHENG_DIR, CKPT_PATH, CLASSES

LAYER = "block4b"      # measured best of five stages; see Phase 5 section 9


def main():
    D = splits.build_dataset(CHENG_DIR)
    S = splits.build_splits(D["images"], D["labels"], D["groups"])
    TEST = S["test_idx"]
    img, lab, msk = D["images"][TEST], D["labels"][TEST], D["masks"][TEST]

    model, ckpt = engine.load_checkpoint(CKPT_PATH)
    tf = data.make_transforms(*ckpt["norm"], augment=False,
                              img_size=ckpt["img_size"])
    ds = data.CachedDataset(img, lab, tf)
    y_true, y_pred, _ = metrics.predict(
        model, torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False))
    ok = y_true == y_pred

    blob = explain.centre_blob(msk[0].shape)
    blob_iou = np.array([explain.area_matched_iou(blob, m) or 0.0 for m in msk])

    cams, iou = [], []
    for i in range(len(TEST)):
        c, _ = explain.grad_cam(model, img[i], tf, layer=LAYER,
                                class_idx=int(lab[i]))
        cams.append(c)
        iou.append(explain.area_matched_iou(c, msk[i]) or 0.0)
    iou = np.array(iou)

    def heat(ax, i, use_blob=False):
        ax.imshow(img[i], cmap="gray")
        ax.imshow(blob if use_blob else cams[i], cmap="jet", alpha=0.5)
        ax.contour(msk[i], levels=[0.5], colors=["w"], linewidths=1.1)
        ax.axis("off")

    # 1. the whole range, per class
    NC = 6
    fig, axes = viz.styled_fig(len(CLASSES), NC, figsize=(2.05 * NC, 2.35 * len(CLASSES)))
    for r, name in enumerate(CLASSES):
        pool = np.where((lab == r) & ok)[0]
        pool = pool[np.argsort(-iou[pool])]
        for c, q in enumerate(np.linspace(0, 1, NC)):
            i = pool[int(round(q * (len(pool) - 1)))]
            heat(axes[r, c], i)
            axes[r, c].set_title(f"IoU {iou[i]:.3f}   (blob {blob_iou[i]:.3f})",
                                 fontsize=7.5)
            if c == 0:
                axes[r, c].text(-0.08, 0.5, name, transform=axes[r, c].transAxes,
                                rotation=90, va="center", ha="right",
                                fontsize=10, fontweight="bold")
    plt.suptitle("Grad-CAM across the whole IoU range, best (left) to worst (right)\n"
                 "white outline = radiologist mask   |   correctly classified scans only",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0.02, 0, 1, 0.94])
    viz.save(fig, "gradcam_spectrum.png")

    # 2. meningioma, paired with the annotation it is scored against
    men = np.where((lab == CLASSES.index("meningioma")) & ok)[0]
    men = men[np.argsort(-iou[men])][:12]
    fig, axes = viz.styled_fig(3, 8, figsize=(16, 6.6))
    for k, i in enumerate(men):
        row, col = divmod(k, 4)
        heat(axes[row, col * 2], i)
        axes[row, col * 2].set_title(f"scan   IoU {iou[i]:.3f}", fontsize=8)
        axes[row, col * 2 + 1].imshow(img[i], cmap="gray")
        axes[row, col * 2 + 1].contour(msk[i], levels=[0.5], colors=["#D85A30"],
                                       linewidths=1.3)
        axes[row, col * 2 + 1].set_title("annotation", fontsize=8)
        axes[row, col * 2 + 1].axis("off")
    m = lab == CLASSES.index("meningioma")
    plt.suptitle(f"Meningioma \u2014 the one class Grad-CAM localises "
                 f"(mean IoU {iou[m].mean():.3f} vs centre-blob {blob_iou[m].mean():.3f})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    viz.save(fig, "gradcam_meningioma.png")

    # 3. the failure, shown rather than described
    pit = np.where((lab == CLASSES.index("pituitary")) & ok)[0]
    pit = sorted(pit, key=lambda i: blob_iou[i] - iou[i])[-6:]
    fig, axes = viz.styled_fig(2, 6, figsize=(13.5, 5.8))
    for c, i in enumerate(pit):
        heat(axes[0, c], i)
        axes[0, c].set_title(f"Grad-CAM  IoU {iou[i]:.3f}", fontsize=8, pad=4)
        heat(axes[1, c], i, use_blob=True)
        axes[1, c].set_title(f"centre blob  IoU {blob_iou[i]:.3f}", fontsize=8, pad=4)
    plt.suptitle("Pituitary \u2014 where a Gaussian blob that knows nothing beats the model\n"
                 "sellar tumours sit at the slice centre, so pointing at the middle already works",
                 fontsize=11, fontweight="bold")
    plt.subplots_adjust(top=0.82, hspace=0.18, wspace=0.04,
                        left=0.01, right=0.99, bottom=0.02)
    viz.save(fig, "gradcam_pituitary_failure.png")

    print(f"\n{'class':<14}{'n':>5}{'mean IoU':>10}{'blob':>9}{'beats blob':>12}")
    print("-" * 50)
    for r, name in enumerate(CLASSES):
        s = (lab == r) & ok
        print(f"{name:<14}{s.sum():>5}{iou[s].mean():>10.4f}"
              f"{blob_iou[s].mean():>9.4f}{(iou[s] > blob_iou[s]).mean():>12.3f}")


if __name__ == "__main__":
    main()
