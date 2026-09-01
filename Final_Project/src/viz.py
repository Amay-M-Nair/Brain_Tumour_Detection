"""Figures. Every one is stamped with the run that produced it.

The stamp is small and in the corner, and it is there because two figures from
two different checkpoints once sat in an outputs folder looking equally
authoritative and disagreeing with each other. Reconciling them took longer than
the analysis they belonged to. A figure whose stamp does not match
outputs/run_manifest.json was made by a different run.
"""
import matplotlib.pyplot as plt
import numpy as np

from .config import FACE, OUTPUTS, PALETTE


def styled_fig(*args, **kwargs):
    fig, ax = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(FACE)
    return fig, ax


def save(fig, name, stamp=True):
    if stamp:
        from .manifest import current_hash
        fig.text(0.995, 0.005, current_hash(), ha="right", va="bottom",
                 fontsize=5, color="#999999", family="monospace")
    path = OUTPUTS / name
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  saved -> outputs/{name}")
    plt.close(fig)
    return path


def denorm(tensor, mean, std):
    return (tensor.squeeze().cpu().numpy() * std + mean).clip(0, 1)


def show_batch(dataset, classes, mean, std, n=8, name="sample_batch.png",
               title="Training samples"):
    """A batch after the full pipeline.

    Every figure in this project is produced by code that could be wrong in a
    way that still runs. Plotting a batch after cache, crop, resize, augment,
    normalise and denormalise is the cheapest check that labels still line up
    with images and nothing was silently corrupted.
    """
    fig, axes = styled_fig(2, n // 2, figsize=(1.7 * n // 2, 4))
    for ax, i in zip(axes.ravel(), np.linspace(0, len(dataset) - 1, n).astype(int)):
        x, y = dataset[i]
        ax.imshow(denorm(x, mean, std), cmap="gray")
        ax.set_title(classes[y], fontsize=8)
        ax.axis("off")
    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return save(fig, name)


def plot_class_balance(counts, classes, name="class_balance.png",
                       title="Class balance"):
    fig, ax = styled_fig(figsize=(6, 3.6))
    colours = [PALETTE["train"], PALETTE["val"], PALETTE["good"], PALETTE["accent"]]
    ax.bar(classes, counts, color=[colours[i % 4] for i in range(len(classes))])
    ax.set_ylabel("images"); ax.set_facecolor(FACE)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i, v in enumerate(counts):
        ax.text(i, v + max(counts) * 0.01, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    return save(fig, name)


def plot_pairs(rows, name="leakage_examples.png",
               title="Same scans on both sides of the shipped split"):
    """Query/match image pairs, laid out one band per block.

    `rows` is a list of (band_title, [(query_img, query_label, match_img,
    match_label, caption), ...]).
    """
    n_band = len(rows)
    n_col = max(len(r[1]) for r in rows)
    fig = plt.figure(figsize=(2.8 * n_col, 3.6 * n_band))
    fig.patch.set_facecolor(FACE)
    gs = fig.add_gridspec(2 * n_band, n_col, left=0.03, right=0.99,
                          top=0.945, bottom=0.02, hspace=0.55, wspace=0.08)
    for b, (band, pairs) in enumerate(rows):
        for j, (qi, ql, mi, ml, cap) in enumerate(pairs):
            a1 = fig.add_subplot(gs[b * 2, j])
            a1.imshow(qi, cmap="gray"); a1.axis("off")
            a1.set_title(ql, fontsize=8)
            a2 = fig.add_subplot(gs[b * 2 + 1, j])
            a2.imshow(mi, cmap="gray"); a2.axis("off")
            a2.set_title(f"{ml}\n{cap}", fontsize=8)
        fig.text(0.03, 0.958 - b * (0.94 / n_band), band,
                 fontsize=12, fontweight="bold", va="bottom")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.988)
    return save(fig, name)


def plot_curves(history, name="training_curves.png", title="Training dynamics"):
    """Loss, accuracy and learning rate over the run.

    The learning-rate panel is not decoration. A cosine schedule stepped per
    batch instead of per epoch completes its whole cycle inside the first epoch
    and leaves everything after it at eta_min, which looks identical to a model
    that simply stopped improving. Plotting the rate makes that visible instead
    of leaving it to be inferred from a flat loss curve.
    """
    ep = range(1, len(history["train_loss"]) + 1)
    fig, axes = styled_fig(1, 3, figsize=(15, 4))
    best = history.get("best_epoch", int(np.argmin(history["val_loss"])) + 1)

    for ax, (key, label) in zip(axes[:2], [("loss", "Loss"), ("acc", "Accuracy")]):
        ax.plot(ep, history[f"train_{key}"], color=PALETTE["train"], label="train", lw=1.8)
        ax.plot(ep, history[f"val_{key}"],   color=PALETTE["val"],   label="val",   lw=1.8)
        ax.axvline(best, color=PALETTE["good"], ls="--", lw=1.2, label=f"best (ep {best})")
        ax.set_xlabel("epoch"); ax.set_title(label, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8); ax.set_facecolor(FACE)

    axes[2].plot(ep, history["lr"], color=PALETTE["accent"], lw=1.8)
    axes[2].set_xlabel("epoch"); axes[2].set_yscale("log")
    axes[2].set_title("Learning rate", fontsize=10, fontweight="bold")
    axes[2].set_facecolor(FACE)

    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return save(fig, name)


def plot_confusion(cm, classes, name="confusion_matrix.png"):
    """Counts alongside row-normalised recall.

    The normalised panel is the one to read. With uneven classes the raw counts
    are dominated by the large ones, but a row that sends 8% of its scans
    elsewhere is obvious once every row sums to one.
    """
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = styled_fig(1, 2, figsize=(12, 5))
    for ax, mat, title, fmt in ((axes[0], cm, "Counts", "d"),
                                (axes[1], norm, "Row-normalised (recall)", ".2f")):
        im = ax.imshow(mat, cmap="Blues", vmin=0)
        ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(title, fontsize=10, fontweight="bold")
        thresh = mat.max() / 2
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                        fontsize=9, color="white" if mat[i, j] > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle("Confusion matrix — test set", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return save(fig, name)


def plot_roc(curves, aucs, macro, classes, name="roc_curves.png"):
    fig, ax = styled_fig(figsize=(6, 5.5))
    colours = [PALETTE["train"], PALETTE["val"], PALETTE["good"], PALETTE["accent"]]
    for i, cls in enumerate(classes):
        fpr, tpr = curves[i]
        ax.plot(fpr, tpr, color=colours[i % 4], lw=1.8,
                label=f"{cls} (AUC {aucs[i]:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="chance")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"One-vs-rest ROC — macro AUC {macro:.4f}", fontsize=11,
                 fontweight="bold")
    ax.legend(fontsize=8, loc="lower right"); ax.set_facecolor(FACE)
    plt.tight_layout()
    return save(fig, name)


def plot_bars(labels, values, name, title, ylabel="accuracy", hline=None,
              colours=None):
    fig, ax = styled_fig(figsize=(max(6, 1.1 * len(labels)), 4))
    ax.bar(range(len(labels)), values,
           color=colours or [PALETTE["train"]] * len(labels))
    if hline is not None:
        ax.axhline(hline, color="k", ls="--", lw=1, alpha=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(ylabel); ax.set_facecolor(FACE)
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    return save(fig, name)
