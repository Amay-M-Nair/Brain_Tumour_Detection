"""Figures for the report. House style carried over from Days 1-8."""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .config import DEVICE, FACE, OUTPUTS, PALETTE


def styled_fig(*args, **kwargs):
    fig, ax = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(FACE)
    return fig, ax


def save(fig, name):
    path = OUTPUTS / name
    fig.savefig(path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"  saved -> outputs/{name}")
    return path


def plot_curves(history, name="training_curves.png", title="Training dynamics"):
    ep = range(1, len(history["train_loss"]) + 1)
    fig, axes = styled_fig(1, 3, figsize=(15, 4))
    best = history.get("best_epoch", int(np.argmin(history["val_loss"])) + 1)

    for ax, (key, label) in zip(axes[:2], [("loss", "Loss"), ("acc", "Accuracy")]):
        ax.plot(ep, history[f"train_{key}"], color=PALETTE["train"], label="train", lw=1.8)
        ax.plot(ep, history[f"val_{key}"],   color=PALETTE["val"],   label="val",   lw=1.8)
        ax.axvline(best, color=PALETTE["good"], ls='--', lw=1.2, label=f"best (ep {best})")
        ax.set_xlabel("epoch"); ax.set_title(label, fontsize=10, fontweight='bold')
        ax.legend(fontsize=8); ax.set_facecolor(FACE)

    axes[2].plot(ep, history["lr"], color=PALETTE["accent"], lw=1.8)
    axes[2].set_xlabel("epoch"); axes[2].set_yscale("log")
    axes[2].set_title("Learning rate", fontsize=10, fontweight='bold')
    axes[2].set_facecolor(FACE)

    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    return save(fig, name)


def plot_confusion(cm, classes, name="confusion_matrix.png"):
    """Counts alongside row-normalised recall.

    The normalised panel is the one to read: with balanced classes the raw
    counts look similar, but a row that sends 8% of its scans elsewhere is
    obvious once each row sums to one.
    """
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = styled_fig(1, 2, figsize=(12, 5))

    for ax, mat, title, fmt in ((axes[0], cm, "Counts", "d"),
                                (axes[1], norm, "Row-normalised (recall)", ".2f")):
        im = ax.imshow(mat, cmap="Blues", vmin=0)
        ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(title, fontsize=10, fontweight='bold')
        thresh = mat.max() / 2
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, format(mat[i, j], fmt), ha='center', va='center',
                        fontsize=9, color='white' if mat[i, j] > thresh else 'black')
        fig.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle("Confusion matrix — test set", fontsize=12, fontweight='bold')
    plt.tight_layout()
    return save(fig, name)


def plot_roc(curves, aucs, macro, classes, name="roc_curves.png"):
    fig, ax = styled_fig(figsize=(6, 5.5))
    colours = [PALETTE["train"], PALETTE["val"], PALETTE["good"], PALETTE["accent"]]
    for i, cls in enumerate(classes):
        fpr, tpr = curves[i]
        ax.plot(fpr, tpr, color=colours[i % 4], lw=1.8, label=f"{cls} (AUC {aucs[i]:.4f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label="chance")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"One-vs-rest ROC — macro AUC {macro:.4f}",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc="lower right"); ax.set_facecolor(FACE)
    plt.tight_layout()
    return save(fig, name)


def denorm(tensor, mean, std):
    return (tensor.squeeze().cpu().numpy() * std + mean).clip(0, 1)


def show_batch(dataset, classes, mean, std, n=8, name="sample_batch.png",
               title="Training samples"):
    fig, axes = styled_fig(2, n // 2, figsize=(1.7 * n // 2, 4))
    for ax, i in zip(axes.ravel(), np.linspace(0, len(dataset) - 1, n).astype(int)):
        x, y = dataset[i]
        ax.imshow(denorm(x, mean, std), cmap='gray')
        ax.set_title(classes[y], fontsize=8)
        ax.axis('off')
    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    return save(fig, name)


def plot_feature_maps(model, x, mean, std, name="feature_maps.png", n=16):
    """First-block and last-block activations for one scan.

    Block 1 should look like edge and texture detectors — recognisably the
    input. Block 4 should not: by then the maps are sparse and abstract, which
    is what it looks like when a network has stopped describing pixels and
    started describing evidence.
    """
    acts = {}
    h1 = model.features.block1.register_forward_hook(
        lambda m, i, o: acts.__setitem__("block1", o.detach()))
    last = model.features.block4b if model.features.deep else model.features.block4
    h4 = last.register_forward_hook(
        lambda m, i, o: acts.__setitem__("block4", o.detach()))
    model.eval()
    with torch.no_grad():
        model(x.unsqueeze(0).to(DEVICE))
    h1.remove(); h4.remove()

    fig = plt.figure(figsize=(12, 6.5))
    fig.patch.set_facecolor(FACE)
    for row, key in enumerate(["block1", "block4"]):
        maps = acts[key][0].cpu().numpy()
        for c in range(n):
            ax = fig.add_subplot(2, n, row * n + c + 1)
            ax.imshow(maps[c], cmap='viridis'); ax.axis('off')
            if c == 0:
                ax.set_title(f"{key}\n({maps.shape[1]}x{maps.shape[2]})",
                             fontsize=8, loc='left')
    plt.suptitle("Feature maps — early filters vs deep filters",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return save(fig, name)


def grad_cam(model, x, class_idx=None):
    """Grad-CAM heatmap over the last convolutional block.

    Channels of the final feature map are weighted by how strongly the target
    class score responds to them, then summed. It answers the question a marker
    on a report cannot: not whether the prediction was right, but whether it was
    right for the anatomically correct reason.
    """
    model.eval()
    acts = {}
    last = model.features.block4b if model.features.deep else model.features.block4
    handle = last.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))

    logits = model(x.unsqueeze(0).to(DEVICE))
    if class_idx is None:
        class_idx = int(logits.argmax(1))
    grads = torch.autograd.grad(logits[0, class_idx], acts["a"])[0]
    handle.remove()

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * acts["a"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam, class_idx, F.softmax(logits, 1)[0, class_idx].item()


def plot_cam_grid(model, samples, classes, mean, std, name="grad_cam.png"):
    """samples: list of (tensor, true_label, tag) triples, laid out one row per class."""
    n_cols = 3
    n_rows = len(samples) // n_cols
    fig, axes = styled_fig(n_rows, n_cols, figsize=(3.1 * n_cols, 3.2 * n_rows))
    for ax, (x, y_true, tag) in zip(axes.ravel(), samples):
        cam, pred, conf = grad_cam(model, x)
        ax.imshow(denorm(x, mean, std), cmap='gray')
        ax.imshow(cam, cmap='jet', alpha=0.42)
        ok = "OK" if pred == y_true else "WRONG"
        ax.set_title(f"true {classes[y_true]} / pred {classes[pred]}\n"
                     f"{conf:.2f} confidence  [{ok}] {tag}", fontsize=8)
        ax.axis('off')
    plt.suptitle("Grad-CAM — what the network looked at",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return save(fig, name)


def plot_confidence(y_true, y_pred, probs, name="confidence.png"):
    """Confidence split by whether the prediction was right.

    A well-calibrated model is unsure when it is wrong. Wrong predictions
    clustered up at 0.99 mean the confidence score carries no information and
    cannot be used as a referral threshold.
    """
    conf = probs.max(axis=1)
    right, wrong = conf[y_true == y_pred], conf[y_true != y_pred]
    fig, ax = styled_fig(figsize=(7, 4.5))
    bins = np.linspace(0.25, 1.0, 31)
    ax.hist(right, bins=bins, color=PALETTE["good"], alpha=0.75,
            label=f"correct (n={len(right)})")
    ax.hist(wrong, bins=bins, color=PALETTE["val"], alpha=0.85,
            label=f"incorrect (n={len(wrong)})")
    ax.set_yscale("log")
    ax.set_xlabel("max softmax probability"); ax.set_ylabel("count (log)")
    ax.set_title("Prediction confidence, correct vs incorrect",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.set_facecolor(FACE)
    plt.tight_layout()
    return save(fig, name)


def plot_misclassified(dataset, y_true, y_pred, probs, classes, mean, std,
                       name="misclassified.png", n=12):
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        print("  no misclassified images to show")
        return None
    order = wrong[np.argsort(-probs[wrong].max(axis=1))][:n]   # most confident errors first
    cols = 4
    rows = int(np.ceil(len(order) / cols))
    fig, axes = styled_fig(rows, cols, figsize=(2.6 * cols, 2.9 * rows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis('off')
    for ax, i in zip(np.atleast_1d(axes).ravel(), order):
        x, _ = dataset[i]
        ax.imshow(denorm(x, mean, std), cmap='gray')
        ax.set_title(f"true {classes[y_true[i]]}\npred {classes[y_pred[i]]} "
                     f"({probs[i].max():.2f})", fontsize=8)
    plt.suptitle("Most confident mistakes", fontsize=12, fontweight='bold')
    plt.tight_layout()
    return save(fig, name)
