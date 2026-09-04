"""The one place a split is constructed, and where leakage is removed.

Split logic does not belong in a notebook cell: it lived there once, a stale
editor tab wrote an older copy back to disk, and the run that followed looked
entirely normal while reporting a better number.

Cheng ships a patient ID per slice, so same-patient bleed -- undetectable by any
pixel comparison, and only boundable on the previous dataset -- is preventable.
Patients are the grouping unit and no patient crosses a boundary; asserted, not
hoped for. The old STRICT_LEAK cosine threshold guessed at patient identity from
near-neighbours and is gone: the metadata exists, so the guess is replaced.
"""
import hashlib
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

from .config import (CLASSES, DUP_COSINE, DUP_L1, SEED, TEST_FRAC,
                     VAL_FRAC)
from .data import duplicate_groups, find_duplicates


def _hash(*arrays):
    """Stable digest of the arrays defining a split, for the run manifest."""
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def build_dataset(cheng_dir, classes=None, equalise=False):
    """Load Cheng into one cache, with patient groups and tumour masks.

    Returns a dict of images, labels, group ids, masks and file names. Groups
    are the real patient ID, which is the whole reason this dataset was worth
    moving to: a split can finally separate patients rather than slices.

    `equalise` defaults off: measured, it removes 0.005 of a 0.3187 texture lift
    while costing real resolution. The switch stays so it can be re-measured.

    Nothing is cached. Reading 3,064 HDF5 files takes ~20s and the result is a
    pure function of the .mat files, so a cache buys little and costs
    correctness -- a stale .npz is indistinguishable from a fresh one at the
    call site. The manifest catches that after the fact; not caching prevents it.
    """
    from .sources import load_cheng

    classes = classes or list(CLASSES)
    images, lab, pids, masks, names = load_cheng(cheng_dir, equalise_first=equalise)
    labels = np.array([classes.index(l) for l in lab], dtype=np.int64)

    pid_index = {p: i for i, p in enumerate(sorted(set(pids)))}
    groups = np.array([pid_index[p] for p in pids], dtype=np.int64)

    return {"images": images, "labels": labels, "groups": groups, "masks": masks,
            "names": np.array(names, dtype=object),
            "pids": np.array(pids, dtype=object),
            "n_patients": np.array(len(pid_index))}


def merge_groups_by_duplicate(images, groups, cos_thresh=DUP_COSINE,
                              l1_thresh=DUP_L1):
    """Fuse patient groups that share a duplicated scan.

    Patient grouping closes the leak only if IDs are consistent. One scan filed
    under two identifiers looks like two independent patients, and the split is
    then free to put one on each side.

    Returns merged groups, duplicate cluster ids, and the merge count --
    reported rather than hidden, since a non-zero value says something real.
    """
    dup = duplicate_groups(images, cos_thresh=cos_thresh, l1_thresh=l1_thresh)
    n = int(groups.max()) + 1
    rows, cols = [], []
    for cluster in np.unique(dup):
        members = np.unique(groups[dup == cluster])
        for other in members[1:]:
            rows.append(members[0])
            cols.append(other)
    if rows:
        graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        n_merged, relabel = connected_components(graph, directed=False)
    else:
        n_merged, relabel = n, np.arange(n)
    return relabel[groups], dup, int(n - n_merged)


def build_splits(images, labels, groups, test_frac=TEST_FRAC, val_frac=VAL_FRAC,
                 seed=SEED):
    """Patient-wise train/val/test indices over a single pool.

    Cheng ships no train/test division, so one is drawn here -- twice, both
    grouped by patient. Test is carved off first and never touched again, then
    validation from the remainder; so TEST_FRAC is of everything and VAL_FRAC of
    what is left.

    Exact repeats are thinned first. A scan present twice is not extra evidence:
    in training it gets double weight in the loss, in test it is scored twice.

    Not cached, like build_dataset: deterministic in its arguments and the seed,
    so recomputing is always right and reloading only sometimes is.
    """
    merged, dup, n_merges = merge_groups_by_duplicate(images, groups)

    # thin exact repeats: one image per duplicate cluster
    keep = np.zeros(len(labels), dtype=bool)
    _, first = np.unique(dup, return_index=True)
    keep[first] = True
    idx = np.where(keep)[0]

    lab_k, grp_k = labels[idx], merged[idx]

    # test first, from the whole pool
    n_test = max(2, round(1 / test_frac))
    tv_rel, te_rel = next(StratifiedGroupKFold(
        n_test, shuffle=True, random_state=seed).split(
            np.zeros(len(lab_k)), lab_k, grp_k))

    # then validation, from what is left
    n_val = max(2, round(1 / val_frac))
    tr_sub, va_sub = next(StratifiedGroupKFold(
        n_val, shuffle=True, random_state=seed).split(
            np.zeros(len(tv_rel)), lab_k[tv_rel], grp_k[tv_rel]))

    train_idx = idx[tv_rel[tr_sub]]
    val_idx = idx[tv_rel[va_sub]]
    test_idx = idx[te_rel]

    out = {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
           "keep": keep, "dup": dup, "merged_groups": merged,
           "n_group_merges": np.array(n_merges),
           "split_hash": _hash(train_idx, val_idx, test_idx)}
    return out


def assert_patient_disjoint(groups, train_idx, val_idx, test_idx):
    """The hard gate: no patient may appear in more than one split.

    The claim the dataset change was made to be able to assert. Fails loudly
    rather than returning a flag: a violation produces numbers that look better
    and mean less.
    """
    sets = {"train": set(groups[train_idx].tolist()),
            "val": set(groups[val_idx].tolist()),
            "test": set(groups[test_idx].tolist())}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = sets[a] & sets[b]
        if shared:
            raise AssertionError(
                f"{len(shared)} patient group(s) appear in both {a} and {b} "
                f"-- the split is not patient-wise: {sorted(shared)[:8]}")
    return {k: len(v) for k, v in sets.items()}


def assert_no_leakage(images, train_idx, test_idx):
    """Re-check test images against training, independently.

    Deliberately does not reuse build_splits' result: a check that trusts the
    thing it is checking verifies nothing.
    """
    remaining, _, cos = find_duplicates(images[test_idx], images[train_idx])
    n = int(remaining.sum())
    if n:
        raise AssertionError(
            f"{n} test images still duplicate a training scan after exclusion "
            f"(max cosine {cos[remaining].max():.4f}) -- the split is not clean")
    return {"checked": len(test_idx), "max_cosine": float(cos.max())}


def neighbour_report(images, labels, train_idx, test_idx):
    """How close held-out images get to training ones, against a null.

    The different-class nearest neighbour is near-certainly a different patient,
    so it is the null. With patients disjoint the two distributions should sit
    close together; same-class mass far above the null is the signature this
    split exists to prevent.
    """
    from .data import fingerprints
    f_tr = fingerprints(images[train_idx])
    f_te = fingerprints(images[test_idx])
    lab_tr, lab_te = labels[train_idx], labels[test_idx]
    same, diff = [], []
    for i in range(len(lab_te)):
        row = f_te[i] @ f_tr.T
        m = lab_tr == lab_te[i]
        same.append(row[m].max())
        diff.append(row[~m].max())
    same, diff = np.array(same), np.array(diff)
    bands = (0.99, 0.98, 0.95, 0.90)
    return {"same_class": same, "other_class": diff,
            "bands": {t: float((same >= t).mean()) for t in bands},
            "null_bands": {t: float((diff >= t).mean()) for t in bands}}


def write_exclusions(names, S, path=None):
    """Persist which files were dropped and why.

    A count in a notebook cell is not auditable; filenames can be re-derived,
    diffed between runs, or argued with.
    """
    import json

    from .config import OUTPUTS
    path = path or OUTPUTS / "excluded.json"
    body = {
        "dropped_as_duplicate": sorted(
            str(n) for n, k in zip(names, S["keep"]) if not k),
        "rule": {"duplicate_cosine": float(DUP_COSINE),
                 "duplicate_pixel_l1": float(DUP_L1),
                 "grouping": "patient id, merged across duplicate clusters",
                 "patient_group_merges": int(S["n_group_merges"])},
    }
    body["counts"] = {"dropped_as_duplicate": len(body["dropped_as_duplicate"]),
                      "kept": int(S["keep"].sum())}
    Path(path).write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body
