"""Scoring. Macro averages are the default, and that is a decision.

Cheng is uneven by nature: 286 glioma against 142 meningioma in the test split.
Plain accuracy on an imbalanced set is a weighted average that hides the small
classes -- a model that handled glioma and pituitary well and failed meningioma
entirely would still post a respectable figure. Macro averaging weights every
class equally, so the failure shows.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from .config import DEVICE


def predict(model, loader):
    """True labels, predictions and probabilities for a whole loader.

    model.eval() is not cosmetic: with BatchNorm in training mode a batch is
    normalised by its own statistics, and on small batches that alone can drop
    accuracy from 0.97 to near chance.
    """
    model.eval()
    y_true, y_pred, probs = [], [], []
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(DEVICE))
            y_true.append(targets.numpy())
            y_pred.append(logits.argmax(1).cpu().numpy())
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
    return (np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(probs))


def confusion(y_true, y_pred, n_classes=3):
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


def per_class_report(y_true, y_pred, classes):
    """Precision, recall and F1 per class, plus macro and weighted averages.

    Precision drives unnecessary follow-up; recall's complement is the
    missed-diagnosis rate. F1 punishes buying one by sacrificing the other.
    """
    rows = []
    for c, name in enumerate(classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec  = tp / (tp + fn) if tp + fn else 0.0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"class": name, "precision": prec, "recall": rec,
                     "f1": f1, "support": int((y_true == c).sum())})

    support = np.array([r["support"] for r in rows], dtype=float)
    for tag, weights in (("macro avg", np.ones_like(support)),
                         ("weighted avg", support)):
        w = weights / weights.sum()
        rows.append({"class": tag,
                     "precision": float(np.dot(w, [r["precision"] for r in rows[:len(classes)]])),
                     "recall":    float(np.dot(w, [r["recall"] for r in rows[:len(classes)]])),
                     "f1":        float(np.dot(w, [r["f1"] for r in rows[:len(classes)]])),
                     "support":   int(support.sum())})
    return rows


def macro_f1(y_true, y_pred, n_classes=3):
    """Macro F1 as a single number, for model selection."""
    scores = []
    for c in range(n_classes):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec  = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(scores))


def print_report(rows):
    print(f"{'class':<16}{'precision':>11}{'recall':>9}{'f1':>9}{'support':>9}")
    print("-" * 54)
    for r in rows:
        if r["class"] == "macro avg":
            print("-" * 54)
        print(f"{r['class']:<16}{r['precision']:>11.4f}{r['recall']:>9.4f}"
              f"{r['f1']:>9.4f}{r['support']:>9}")


def bootstrap_ci(y_true, y_pred, metric="accuracy", class_idx=None, n_classes=3,
                 n_boot=1000, seed=0):
    """95% interval by resampling the test set with replacement.

    Quoting a point estimate to four decimals implies a precision the sample
    size does not support. Note these resample images, not patients, so on a
    dataset with many slices per patient they are optimistic.
    """
    rng = np.random.default_rng(seed)
    if class_idx is None:
        pool = np.arange(len(y_true))
        if metric == "macro_f1":
            score = lambda s: macro_f1(y_true[s], y_pred[s], n_classes)
        else:
            score = lambda s: (y_true[s] == y_pred[s]).mean()
    else:
        pool = np.where(y_true == class_idx)[0]
        score = lambda s: (y_pred[s] == class_idx).mean()

    stats = np.array([score(rng.choice(pool, len(pool), replace=True))
                      for _ in range(n_boot)])
    return score(pool), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def roc_ovr(y_true, probs, n_classes=3):
    """One-vs-rest ROC curves and AUCs, plus the macro average."""
    from sklearn.metrics import auc, roc_curve
    curves, aucs = [], []
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve((y_true == c).astype(int), probs[:, c])
        curves.append((fpr, tpr)); aucs.append(auc(fpr, tpr))
    return curves, aucs, float(np.mean(aucs))
