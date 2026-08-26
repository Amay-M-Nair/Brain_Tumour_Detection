"""Training, checkpointing and the ablation harness.

train_one_epoch, evaluate, the checkpoint pair and EarlyStopper are carried
over from Days 7 and 8. run_experiment is the Day 8 harness with its globals
promoted to arguments — in the notebook it closed over the split and the
transforms, which meant changing either one silently changed every past result.
"""
import time

import numpy as np
import torch
import torch.nn as nn

from .config import (BATCH_SIZE, CKPT_PATH, DEVICE, EPOCHS, IMG_SIZE, LR,
                     PATIENCE, SEED, WD)
from .data import CachedDataset, make_loaders, make_transforms
from .model import BrainTumourNet


def train_one_epoch(model, loader, criterion, optimizer):
    """One full pass over the training set."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct      += (logits.argmax(1) == targets).sum().item()
        total        += targets.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Validation pass: eval mode, no gradients, no augmentation."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        logits = model(images)
        loss   = criterion(logits, targets)

        running_loss += loss.item() * images.size(0)
        correct      += (logits.argmax(1) == targets).sum().item()
        total        += targets.size(0)

    return running_loss / total, correct / total


class EarlyStopper:
    """Stops training once validation loss has stalled for `patience` epochs.

    Tracks loss rather than accuracy because loss moves before accuracy does —
    a model can keep its argmax correct while growing steadily more
    overconfident, and that is the point at which it has started to memorise.
    """

    def __init__(self, patience=PATIENCE, min_delta=1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.best_epoch, self.counter = float('inf'), 0, 0

    def step(self, val_loss, epoch):
        if val_loss < self.best - self.min_delta:
            self.best, self.best_epoch, self.counter = val_loss, epoch, 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def save_checkpoint(model, optimizer, epoch, val_loss, val_acc,
                    classes, mean, std, img_size=IMG_SIZE, deep=False,
                    path=CKPT_PATH):
    """Persist everything needed to reproduce an evaluation.

    The normalisation statistics and image size travel with the weights: a
    model scored under different preprocessing than it trained under is being
    measured on a distribution it never saw.
    """
    torch.save({"epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss, "val_acc": val_acc,
                "classes": classes, "img_size": img_size,
                "norm": (mean, std), "deep": deep}, path)


def load_checkpoint(path=CKPT_PATH, model=None, optimizer=None):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    if model is None:
        model = BrainTumourNet(num_classes=len(ckpt["classes"]),
                               deep=ckpt.get("deep", False)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return model, ckpt


def fit(model, train_loader, val_loader, *, epochs=EPOCHS, lr=LR, weight_decay=WD,
        label_smoothing=0.0, class_weights=None, patience=PATIENCE,
        checkpoint=None, verbose=True):
    """Full training run with cosine LR, early stopping and best-loss checkpointing.

    The scheduler steps once per epoch, not once per batch. Stepping per batch
    would complete the whole cosine cycle within the first epoch and leave the
    remaining epochs training at eta_min — a silent failure that looks like a
    model which simply stopped improving.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    # Validation is always scored unsmoothed and unweighted so that runs with
    # different training losses stay comparable to each other.
    eval_crit = nn.CrossEntropyLoss()

    stopper = EarlyStopper(patience=patience)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_loss, best_epoch, t0 = float('inf'), 0, time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = evaluate(model, val_loader, eval_crit)

        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss);   history["val_acc"].append(va_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        scheduler.step()

        if va_loss < best_loss:
            best_loss, best_epoch = va_loss, epoch
            if checkpoint is not None:
                save_checkpoint(model, optimizer, epoch, va_loss, va_acc, **checkpoint)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  epoch {epoch:>3}/{epochs}  "
                  f"train {tr_loss:.4f}/{tr_acc:.4f}   "
                  f"val {va_loss:.4f}/{va_acc:.4f}"
                  f"{'  <- best' if epoch == best_epoch else ''}")

        if stopper.step(va_loss, epoch):
            if verbose:
                print(f"  early stop at epoch {epoch} "
                      f"(no improvement since {stopper.best_epoch})")
            break

    history["best_epoch"] = best_epoch
    history["stopped_at"] = len(history["val_loss"])
    history["seconds"]    = time.time() - t0
    return history


def summarise(h):
    """Best-epoch and end-of-run figures for one training history.

    overfit_ratio is final val loss over its minimum: 1.0 means the run never
    turned around, and anything much above it means the model spent its later
    epochs memorising.
    """
    b = int(np.argmin(h["val_loss"]))
    return {"best_epoch":      b + 1,
            "best_val_loss":   h["val_loss"][b],
            "best_val_acc":    h["val_acc"][b],
            "final_val_loss":  h["val_loss"][-1],
            "final_val_acc":   h["val_acc"][-1],
            "final_train_acc": h["train_acc"][-1],
            "gap":             h["val_loss"][-1] - h["train_loss"][-1],
            "overfit_ratio":   h["val_loss"][-1] / h["val_loss"][b],
            "seconds":         h.get("seconds", float('nan'))}


def run_experiment(cache, labels, train_idx, val_idx, mean, std, *,
                   dropout=0.0, weight_decay=0.0, augment=False, jitter=True,
                   label_smoothing=0.0, deep=False, img_size=IMG_SIZE,
                   resample=False, epochs=25, batch_size=BATCH_SIZE, lr=LR,
                   seed=SEED, patience=10**6, verbose=False):
    """Train one configuration end to end and return its history.

    Everything the run depends on is an argument: two calls with the same
    arguments give the same numbers, and no call can be changed by editing an
    unrelated cell. Early stopping is off by default so that every ablation
    config is compared over the same number of epochs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_tf = make_transforms(mean, std, augment=augment, img_size=img_size,
                               jitter=jitter, resample=resample)
    eval_tf  = make_transforms(mean, std, augment=False,   img_size=img_size)
    train_ds = CachedDataset(cache, labels, train_tf, train_idx)
    val_ds   = CachedDataset(cache, labels, eval_tf,  val_idx)
    train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size=batch_size)

    model = BrainTumourNet(num_classes=len(set(labels.tolist())),
                           dropout=dropout, deep=deep).to(DEVICE)
    h = fit(model, train_loader, val_loader, epochs=epochs, lr=lr,
            weight_decay=weight_decay, label_smoothing=label_smoothing,
            patience=patience, verbose=verbose)
    h["model"] = model
    return h
