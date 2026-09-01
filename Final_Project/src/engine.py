"""Training loop, checkpointing, and one-shot experiments.

Two decisions in here are worth reading before the code.

Early stopping cannot improve model selection. The checkpoint already keeps the
best-validation-loss epoch, so stopping early can only forfeit improvement that
would have come later -- it never picks a better model, it only saves time. That
makes it a runaway guard, and its patience should be generous. A tight patience
previously halted a 60-epoch cosine schedule at epoch 20, before the low
learning-rate phase where the best epoch actually appears, and cost four points
of accuracy that were simply left on the table.

The scheduler steps once per epoch, not once per batch. Stepping per batch
completes the whole cosine cycle inside the first epoch and leaves everything
after it training at eta_min -- a silent failure that looks exactly like a model
which stopped improving.
"""
import time

import numpy as np
import torch
import torch.nn as nn

from .config import (BALANCE, BATCH_SIZE, CKPT_PATH, DEVICE, EPOCHS, IMG_SIZE,
                     LR, PATIENCE, SEED, WD)
from .data import (CachedDataset, balanced_sampler, class_weights, make_loaders,
                   make_transforms, undersample)
from .model import BrainTumourNet


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = correct = seen = 0
    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        # Clipping is cheap insurance rather than a tuned hyperparameter: a
        # single exploding batch early on can undo an epoch of progress.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(targets)
        correct    += (logits.argmax(1) == targets).sum().item()
        seen       += len(targets)
    return total_loss / seen, correct / seen


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = correct = seen = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            logits = model(images)
            total_loss += criterion(logits, targets).item() * len(targets)
            correct    += (logits.argmax(1) == targets).sum().item()
            seen       += len(targets)
    return total_loss / seen, correct / seen


class EarlyStopper:
    """Stops when validation loss has not improved for `patience` epochs."""

    def __init__(self, patience=PATIENCE, min_delta=0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.best_epoch, self.waited = float("inf"), 0, 0

    def step(self, value, epoch):
        if value < self.best - self.min_delta:
            self.best, self.best_epoch, self.waited = value, epoch, 0
            return False
        self.waited += 1
        return self.waited >= self.patience


def save_checkpoint(model, optimizer, epoch, val_loss, val_acc, classes, mean,
                    std, img_size=IMG_SIZE, deep=False, activation="relu",
                    path=CKPT_PATH):
    """Persist everything needed to reproduce an evaluation.

    The normalisation statistics and image size travel with the weights: a model
    scored under different preprocessing than it trained under is being measured
    on a distribution it never saw.

    The manifest hash travels too, so evaluation can refuse to score a checkpoint
    that belongs to a different run than the figures beside it.
    """
    from .manifest import current_hash
    torch.save({"epoch": epoch, "manifest": current_hash(),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss, "val_acc": val_acc,
                "classes": classes, "img_size": img_size,
                "norm": (mean, std), "deep": deep,
                "activation": activation}, path)


def load_checkpoint(path=CKPT_PATH, model=None, optimizer=None):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    if model is None:
        model = BrainTumourNet(num_classes=len(ckpt["classes"]), dropout=0.0,
                               deep=ckpt.get("deep", False),
                               activation=ckpt.get("activation", "relu")).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return model, ckpt


def fit(model, train_loader, val_loader, *, epochs=EPOCHS, lr=LR, weight_decay=WD,
        label_smoothing=0.0, class_weight=None, patience=PATIENCE,
        checkpoint=None, verbose=True):
    """Full run with cosine LR, early stopping and best-loss checkpointing."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)
    weight = None if class_weight is None else class_weight.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    # Validation is always scored unsmoothed and unweighted, so runs with
    # different training losses stay comparable to each other. Without this,
    # changing the class weights would change the validation number even if the
    # model were identical.
    eval_crit = nn.CrossEntropyLoss()

    stopper = EarlyStopper(patience=patience)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_loss, best_epoch, t0 = float("inf"), 0, time.time()

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
            print(f"  epoch {epoch:>3}/{epochs}  train {tr_loss:.4f}/{tr_acc:.4f}   "
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
    best = h["best_epoch"] - 1
    return {"best_epoch": h["best_epoch"], "stopped_at": h["stopped_at"],
            "best_val_loss": h["val_loss"][best], "best_val_acc": h["val_acc"][best],
            "final_train_acc": h["train_acc"][-1], "final_val_acc": h["val_acc"][-1],
            "gap": h["train_acc"][-1] - h["val_acc"][-1],
            "seconds": h["seconds"]}


def build_loaders(cache, labels, train_idx, val_idx, mean, std, *,
                  augment=True, jitter=True, corrupt=False, img_size=IMG_SIZE,
                  balance=BALANCE, batch_size=BATCH_SIZE, seed=SEED):
    """Loaders plus the class weight implied by the balancing choice.

    Returns (train_loader, val_loader, class_weight). Only one correction is
    ever applied: 'weights' returns a weight vector and an ordinary shuffled
    loader, 'sampler' returns a balanced sampler and no weight, 'undersample'
    truncates the index list. Applying two of them at once would correct the
    same imbalance twice.
    """
    train_tf = make_transforms(mean, std, augment=augment, img_size=img_size,
                               jitter=jitter, corrupt=corrupt)
    eval_tf  = make_transforms(mean, std, augment=False, img_size=img_size)

    weight, sampler = None, None
    if balance == "weights":
        weight = class_weights(labels[train_idx], int(labels.max()) + 1)
    elif balance == "sampler":
        sampler = balanced_sampler(labels, train_idx, seed=seed)
    elif balance == "undersample":
        train_idx = undersample(labels, train_idx, seed=seed)

    train_ds = CachedDataset(cache, labels, train_tf, train_idx)
    val_ds   = CachedDataset(cache, labels, eval_tf,  val_idx)

    if sampler is None:
        train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size=batch_size)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size, sampler=sampler, num_workers=0, drop_last=True)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, weight


def run_experiment(cache, labels, train_idx, val_idx, mean, std, *, dropout=0.0,
                   weight_decay=0.0, augment=False, jitter=True, corrupt=False,
                   label_smoothing=0.0, deep=False, activation="relu",
                   img_size=IMG_SIZE, balance=None, epochs=25,
                   batch_size=BATCH_SIZE, lr=LR, seed=SEED, patience=10**6,
                   verbose=False):
    """Train one configuration end to end and return its history.

    Everything the run depends on is an argument, so two calls with the same
    arguments give the same numbers and no call can be changed by editing an
    unrelated cell. Early stopping is off by default so every ablation
    configuration is compared over an identical number of epochs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, val_loader, weight = build_loaders(
        cache, labels, train_idx, val_idx, mean, std, augment=augment,
        jitter=jitter, corrupt=corrupt, img_size=img_size, balance=balance,
        batch_size=batch_size, seed=seed)

    model = BrainTumourNet(num_classes=int(labels.max()) + 1, dropout=dropout,
                           deep=deep, activation=activation).to(DEVICE)
    h = fit(model, train_loader, val_loader, epochs=epochs, lr=lr,
            weight_decay=weight_decay, label_smoothing=label_smoothing,
            class_weight=weight, patience=patience, verbose=verbose)
    h["model"] = model
    h["val_loader"] = val_loader
    return h
