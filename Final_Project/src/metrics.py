"""Evaluation metrics.

Accuracy alone is a poor summary for a medical classifier: it hides which
class is being missed, and a miss is not symmetric — calling a tumour healthy
costs more than the reverse. Everything here reports per class.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from .config import DEVICE


@torch.no_grad()
def predict(model, loader):
    """Run the model over a loader, returning true labels, predictions and probabilities."""
    model.eval()
    y_true, y_pred, probs = [], [], []
    for images, targets in loader:
        logits = model(images.to(DEVICE))
        p = F.softmax(logits, dim=1)
        y_true.append(targets.numpy())
        y_pred.append(logits.argmax(1).cpu().numpy())
        probs.append(p.cpu().numpy())
    return (np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(probs))


def confusion(y_true, y_pred, n_classes=4):
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


def per_class_report(y_true, y_pred, classes):
    """Precision, recall, F1 and support per class, plus macro and weighted means.

    Recall is the one to read first here: it answers 'of the scans that really
    were this class, how many did we catch', which is the question a missed
    diagnosis turns on.
    """
    cm = confusion(y_true, y_pred, len(classes))
    rows = []
    for i, name in enumerate(classes):
        tp = cm[i, i]
        support   = cm[i].sum()
        precision = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        recall    = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({"class": name, "precision": precision, "recall": recall,
                     "f1": f1, "support": int(support)})

    support = np.array([r["support"] for r in rows], dtype=float)
    for name, w in (("macro avg", None), ("weighted avg", support / support.sum())):
        rows.append({"class": name,
                     "precision": np.average([r["precision"] for r in rows[:len(classes)]], weights=w),
                     "recall":    np.average([r["recall"]    for r in rows[:len(classes)]], weights=w),
                     "f1":        np.average([r["f1"]        for r in rows[:len(classes)]], weights=w),
                     "support":   int(support.sum())})
    return rows


def print_report(rows):
    print(f"{'class':<14}{'precision':>11}{'recall':>9}{'f1':>9}{'support':>9}")
    print("-" * 52)
    for r in rows:
        if r["class"] == "macro avg":
            print("-" * 52)
        print(f"{r['class']:<14}{r['precision']:>11.4f}{r['recall']:>9.4f}"
              f"{r['f1']:>9.4f}{r['support']:>9}")


def roc_ovr(y_true, probs, n_classes=4):
    """One-vs-rest ROC curves and AUCs, plus the macro-average AUC."""
    curves, aucs = {}, {}
    for i in range(n_classes):
        binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(binary, probs[:, i])
        curves[i] = (fpr, tpr)
        aucs[i]   = roc_auc_score(binary, probs[:, i])
    macro = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    return curves, aucs, macro


def bootstrap_ci(y_true, y_pred, class_idx=None, n_boot=1000, seed=0):
    """95% confidence interval for accuracy (or one class's recall) by resampling.

    A single accuracy figure on 1,497 test images is a point estimate with real
    uncertainty attached. Resampling the test set with replacement shows how
    much of the number is signal and how much is which images happened to be
    in the split.
    """
    rng = np.random.default_rng(seed)
    if class_idx is None:
        pool = np.arange(len(y_true))
        score = lambda s: (y_true[s] == y_pred[s]).mean()
    else:
        pool = np.where(y_true == class_idx)[0]
        score = lambda s: (y_pred[s] == class_idx).mean()

    stats = np.array([score(rng.choice(pool, len(pool), replace=True))
                      for _ in range(n_boot)])
    return score(pool), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))
