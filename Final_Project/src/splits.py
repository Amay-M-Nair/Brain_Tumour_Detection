"""The one place a split is constructed, and where leakage is removed.

Logic that decides what the model is measured against does not belong in a
notebook cell. In an earlier version it did, and a stale editor tab wrote an
older copy of that notebook back to disk, silently reverting the split to a
leaky one. The training run that followed looked entirely normal and reported a
better number, which is what a silent regression looks like.

The requirement this module carries is that `Training/` and `Testing/` share no
images. On this dataset that is not the default: 293 of the shipped test images
are pixel-identical to training images under different filenames. Those are
found, removed, and then their absence is asserted -- leakage is not written up
as a caveat, it is eliminated and the elimination is checked.

What cannot be eliminated is patient identity. This dataset ships no patient
IDs, so two different slices of one patient are invisible to a pixel comparison
and may still sit on both sides. `leak_report` bounds that residual rather than
ignoring it.
"""
import hashlib
from pathlib import Path

import numpy as np

from .config import (CACHE_DIR, DEDUPE_TRAIN, DUP_COSINE, DUP_L1, SEED,
                     STRICT_LEAK, STRICT_LEAK_COSINE, VAL_FRAC)
from .data import build_cache, duplicate_groups, find_duplicates, grouped_split


def _hash(*arrays):
    """Stable digest of the arrays defining a split, for the run manifest."""
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def build_caches(train_dir, test_dir, rebuild=False):
    """Decode both splits once and memoise them under data/cache.

    Returns pixels, labels, paths and the class list for each split. The
    augmented copies are dropped from *both* splits here, not just Testing.
    """
    files = {n: CACHE_DIR / f"{n}.npy" for n in
             ("train_img", "train_lab", "test_img", "test_lab")}
    paths_file = CACHE_DIR / "paths.npz"

    if not rebuild and all(f.exists() for f in files.values()) and paths_file.exists():
        a = {n: np.load(f) for n, f in files.items()}
        p = np.load(paths_file, allow_pickle=True)
        return (a["train_img"], a["train_lab"], list(p["train"]),
                a["test_img"], a["test_lab"], list(p["test"]), list(p["classes"]))

    tr_img, tr_lab, classes, tr_paths = build_cache(train_dir)
    te_img, te_lab, te_classes, te_paths = build_cache(test_dir)
    assert classes == te_classes, "Training/ and Testing/ disagree on class order"

    np.save(files["train_img"], tr_img); np.save(files["train_lab"], tr_lab)
    np.save(files["test_img"], te_img);  np.save(files["test_lab"], te_lab)
    np.savez(paths_file, train=np.array(tr_paths, dtype=object),
             test=np.array(te_paths, dtype=object),
             classes=np.array(classes, dtype=object))
    return tr_img, tr_lab, tr_paths, te_img, te_lab, te_paths, classes


def build_splits(train_img, train_lab, test_img, val_frac=VAL_FRAC, seed=SEED,
                 rebuild=False):
    """Grouped train/val indices plus the cross-split leak mask.

    train_idx / val_idx -- whole duplicate clusters land on one side, so the
    validation number cannot be inflated by scoring the model on a scan it
    memorised.

    test_leak -- test images that repeat a training scan. Everything downstream
    excludes them, and Phase 1 asserts that nothing survives the exclusion.
    """
    cached = CACHE_DIR / "splits.npz"
    if not rebuild and cached.exists():
        d = np.load(cached)
        out = {k: d[k] for k in d.files}
        out["split_hash"] = str(out["split_hash"])
        return out

    # Cluster at the strict threshold, not the duplicate threshold. The
    # train/val boundary needs the same rule the test set gets: grouping only
    # exact duplicates leaves same-patient adjacent slices free to straddle it,
    # which is leakage into the number every training decision is made against.
    # Connected components stay tight here because contrast_l1 <= 0.15 still
    # has to hold -- measured, the largest cluster is 9 images even at 0.92.
    split_cos = STRICT_LEAK_COSINE if STRICT_LEAK else DUP_COSINE
    groups = duplicate_groups(train_img, cos_thresh=split_cos)

    # 1. drop redundant copies inside Training/, keeping one image per cluster
    train_keep = np.zeros(len(train_lab), dtype=bool)
    _, first = np.unique(groups, return_index=True)
    train_keep[first] = True
    if not DEDUPE_TRAIN:
        train_keep[:] = True

    # 2. drop test images that match anything still in Training/. Matching
    #    against the kept subset, not the original, so the exclusion describes
    #    the data the model will actually see.
    kept_idx = np.where(train_keep)[0]
    test_leak, _, cos = find_duplicates(test_img, train_img[kept_idx])
    if STRICT_LEAK:
        # near-neighbours are not duplicates and no pixel test calls them one,
        # but inspection showed same-patient adjacent slices, and 100% of pairs
        # above this threshold share a class against a 77% baseline
        test_leak = test_leak | (cos >= STRICT_LEAK_COSINE)

    # 3. drop repeats *inside* Testing/ as well. These do not leak between
    #    splits, but a scan present twice is scored twice, so it silently gets
    #    double weight in every metric.
    test_groups = duplicate_groups(test_img, cos_thresh=split_cos)
    seen = set()
    for i, g in enumerate(test_groups):
        if test_leak[i]:
            continue
        if g in seen:
            test_leak[i] = True
        else:
            seen.add(g)

    # 4. split what survives
    train_idx, val_idx = grouped_split(train_lab[train_keep],
                                       groups[train_keep], val_frac, seed)
    train_idx, val_idx = kept_idx[train_idx], kept_idx[val_idx]

    assert not (set(groups[train_idx]) & set(groups[val_idx])), \
        "a duplicate cluster straddles the train/val boundary"

    out = {"train_idx": train_idx, "val_idx": val_idx, "groups": groups,
           "train_keep": train_keep, "test_leak": test_leak,
           "test_cosine": cos,
           "split_hash": _hash(train_idx, val_idx, test_leak)}
    np.savez(cached, **out)
    return out


def write_exclusions(train_paths, test_paths, S, path=None):
    """Persist exactly which files were dropped and why, as a checkable record.

    A count in a notebook cell is not auditable. This writes the filenames, so
    the exclusion can be re-derived, diffed between runs, or argued with.
    """
    import json
    from .config import OUTPUTS

    path = path or OUTPUTS / "excluded.json"
    keep = S["train_keep"]
    body = {
        "train_dropped_as_redundant": sorted(
            Path(p).name for p, k in zip(train_paths, keep) if not k),
        "test_dropped_as_leaked": sorted(
            Path(p).name for p, k in zip(test_paths, S["test_leak"]) if k),
        "rule": {"duplicate_cosine": float(DUP_COSINE),
                 "duplicate_pixel_l1": float(DUP_L1),
                 "strict_leak": bool(STRICT_LEAK),
                 "strict_leak_cosine": float(STRICT_LEAK_COSINE),
                 "dedupe_train": bool(DEDUPE_TRAIN)},
    }
    body["counts"] = {k: len(v) for k, v in body.items() if isinstance(v, list)}
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body


def assert_no_leakage(train_img, test_img, keep):
    """The hard gate. Re-checks the *surviving* test set against training.

    Deliberately independent of build_splits rather than reusing its result: the
    point is to verify the exclusion actually worked, and a check that trusts
    the thing it is checking verifies nothing.
    """
    remaining, _, cos = find_duplicates(test_img[keep], train_img)
    n = int(remaining.sum())
    if n:
        raise AssertionError(
            f"{n} test images still duplicate a training scan after exclusion "
            f"(max cosine {cos[remaining].max():.4f}) -- the split is not clean")
    return {"checked": int(keep.sum()), "max_cosine": float(cos.max())}


def leak_report(train_img, train_lab, test_img, test_lab, keep, classes):
    """Bound the leakage that cannot be removed, rather than ignoring it.

    Exact repeats are gone by the time this runs. What remains possible is two
    different slices of the same patient, which no pixel comparison can identify
    without patient IDs. This measures how close the surviving test images get
    to the training set, against a null: the nearest training image of a
    *different* class, which is almost certainly a different patient.

    A same-class distribution that sits far above that null, with mass at high
    similarity, is the signature of same-patient bleed.
    """
    from .data import fingerprints
    f_tr, f_te = fingerprints(train_img), fingerprints(test_img[keep])
    lab = test_lab[keep]
    same, diff = [], []
    for i in range(len(lab)):
        row = f_te[i] @ f_tr.T
        m = train_lab == lab[i]
        same.append(row[m].max())
        diff.append(row[~m].max())
    same, diff = np.array(same), np.array(diff)
    return {"same_class": same, "other_class": diff,
            "bands": {t: float((same >= t).mean()) for t in (0.99, 0.98, 0.95, 0.90)},
            "null_bands": {t: float((diff >= t).mean()) for t in (0.99, 0.98, 0.95, 0.90)}}
