"""Dataset audit, run before anything is trained.

One question: can something other than the anatomy predict the label?

It runs first because the alternative was tried. In an earlier version of this
project a model was trained, evaluated, written up, and only then found to be
reading a source signature -- image resolution correlated almost perfectly with
class, because the tumour classes and the healthy class came from different
source collections digitised at different sizes. Accuracy was 0.9764 on the
scans where that cue pointed the right way and 0.8019 on the rest, and no split
and no augmentation repaired it, because the training data contained no examples
of one class in the other's style.

The decisive check is the last one. If a classifier that never sees a pixel --
only width, height, file size and compression -- beats chance, then the labels
are partly recoverable from the files themselves, and whatever the CNN reports
is partly a measurement of that.

On this dataset the probe is *expected to fail*. That is the point: the failure
is recorded before training rather than discovered afterwards, and evaluation
stratifies by the offending variable so the shortcut stays visible.
"""
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def scan(root, drop_augmented=True):
    """Per-file metadata, without decoding pixels.

    PIL reads the JPEG header for .size and .quantization, so this stays cheap.
    Nothing here touches image content -- that is the point, since everything
    collected is information a model should not be able to use.
    """
    records = []
    for cls_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for f in sorted(cls_dir.iterdir()):
            if drop_augmented and "-aug-" in f.name.lower():
                continue
            with Image.open(f) as img:
                w, h = img.size
                qt = getattr(img, "quantization", None)
            records.append({
                "path": f, "label": cls_dir.name, "width": w, "height": h,
                "filesize": f.stat().st_size, "bpp": f.stat().st_size / (w * h),
                "qhash": hashlib.md5(
                    repr(sorted((k, tuple(v)) for k, v in qt.items())).encode()
                ).hexdigest()[:8] if qt else "none",
            })
    return records


def report_resolution(records, classes):
    """A1 -- is any class separable by native image size?"""
    print("A1  native resolution by class")
    sizes = sorted({(r["width"], r["height"]) for r in records})
    print(f"    distinct (w,h) across both splits: {len(sizes)}")
    is512 = {}
    for c in classes:
        rows = [r for r in records if r["label"] == c]
        n512 = sum(r["width"] == 512 and r["height"] == 512 for r in rows)
        is512[c] = n512 / len(rows)
        per = Counter((r["width"], r["height"]) for r in rows)
        print(f"    {c:<12}{n512:>6}/{len(rows):<6} at 512x512 ({n512/len(rows):>6.1%})"
              f"   {len(per)} distinct")
    spread = max(is512.values()) - min(is512.values())
    print(f"    -> spread in 512x512 share across classes: {spread:.1%}")
    return {"uniform": len(sizes) == 1, "share_512": is512, "spread_512": spread}


def report_signature(records, classes):
    """A2 -- do the JPEG encoder settings cluster by class?

    Resizing hides the dimensions but not the compression history. If one class
    was written by a different pipeline its quantization tables differ, and that
    is a per-class watermark readable straight out of the texture.
    """
    print("\nA2  JPEG quantization tables and bytes-per-pixel by class")
    tables = defaultdict(Counter)
    for r in records:
        tables[r["label"]][r["qhash"]] += 1
    print(f"    {'class':<12}{'distinct tables':>17}{'top share':>12}{'mean bpp':>11}")
    for c in classes:
        t = tables[c]
        rows = [r for r in records if r["label"] == c]
        print(f"    {c:<12}{len(t):>17}{t.most_common(1)[0][1]/sum(t.values()):>11.1%}"
              f"{np.mean([r['bpp'] for r in rows]):>11.4f}")
    shared = set.intersection(*(set(tables[c]) for c in classes)) if classes else set()
    print(f"    -> tables shared by all classes: {len(shared)}")
    return {"shared_qtables": len(shared)}


def report_duplicates(train_cache, test_cache, train_lab, test_lab, classes):
    """A3 -- repeats within and across the shipped splits."""
    from .data import duplicate_groups, find_duplicates

    print("\nA3  repeated scans")
    g = duplicate_groups(train_cache)
    extra = len(g) - len(np.unique(g))
    gt = duplicate_groups(test_cache)
    leak, _, _ = find_duplicates(test_cache, train_cache)
    print(f"    redundant copies inside Training/   {extra} of {len(g)}")
    print(f"    redundant copies inside Testing/    {len(gt) - len(np.unique(gt))} of {len(gt)}")
    print(f"    test images repeating a train scan  {leak.sum()} of {len(leak)}"
          f"  ({leak.mean():.1%})")
    if leak.any():
        for c, n in enumerate(classes):
            m = test_lab == c
            print(f"      {n:<12}{leak[m].sum():>5} / {m.sum()}")
    return {"train_redundant": int(extra), "cross_split": int(leak.sum()),
            "groups": g, "test_leak": leak}


def metadata_probe(records, classes, seed=42):
    """A4 -- can the label be predicted from metadata alone?

    A random forest on width, height, file size and bytes-per-pixel. It never
    sees a pixel, so anything above chance means the label leaks through the
    file itself, and the CNN will find that signal faster than it finds anatomy.

    A forest rather than a linear model deliberately: the shortcut here is a
    threshold ("is it 512x512"), which a tree splits out immediately and a
    linear model on raw sizes can miss.
    """
    print("\nA4  metadata-only probe (no pixels)")
    X = np.array([[r["width"], r["height"], r["filesize"], r["bpp"]]
                  for r in records], dtype=np.float64)
    y = np.array([classes.index(r["label"]) for r in records])
    clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    chance = 1.0 / len(classes)
    acc = float(scores.mean())
    print(f"    5-fold accuracy {acc:.4f} +/- {scores.std():.4f}   chance {chance:.4f}")
    clf.fit(X, y)
    names = ["width", "height", "filesize", "bpp"]
    order = np.argsort(-clf.feature_importances_)
    print("    importance: " + "  ".join(
        f"{names[i]} {clf.feature_importances_[i]:.3f}" for i in order))
    print(f"    -> lift over chance: {acc - chance:+.4f}")
    return {"accuracy": acc, "chance": chance, "lift": acc - chance}


def run_audit(train_dir, test_dir, classes, train_cache=None, test_cache=None,
              train_lab=None, test_lab=None):
    """Run every check and return the findings with a verdict.

    The verdict is strict on purpose. It is easier to argue down a failed check
    now than to explain a headline number later.
    """
    records = scan(train_dir) + scan(test_dir)
    print("=" * 68)
    print(f"DATASET AUDIT  --  {len(records)} images after dropping '-aug-' copies")
    print("=" * 68)
    res = {"resolution": report_resolution(records, classes),
           "signature":  report_signature(records, classes)}
    if train_cache is not None:
        res["duplicates"] = report_duplicates(train_cache, test_cache,
                                              train_lab, test_lab, classes)
    res["probe"] = metadata_probe(records, classes)

    checks = [("metadata probe near chance", res["probe"]["lift"] < 0.10),
              ("no class separable by size", res["resolution"]["spread_512"] < 0.50),
              ("quantization tables shared", res["signature"]["shared_qtables"] > 0)]
    if "duplicates" in res:
        checks.append(("no cross-split repeats", res["duplicates"]["cross_split"] == 0))

    print("\n" + "=" * 68)
    print("VERDICT")
    print("=" * 68)
    for label, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    res["failed"] = [label for label, ok in checks if not ok]
    if res["failed"]:
        print("\n  -> Something other than anatomy can predict the label here.")
        print("     Duplicates are removed downstream and re-checked. The size")
        print("     shortcut cannot be removed, so evaluation reports every")
        print("     headline number beside its size-stratified breakdown.")
    return res
