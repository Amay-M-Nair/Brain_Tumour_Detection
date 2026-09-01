"""A record of what produced each result.

Dataset digest, split digest, config, git SHA, timestamp. The short hash is
stamped on every figure and embedded in every checkpoint, so a figure and a
model can be checked against each other instead of being assumed to match.

This exists because they once did not match, silently: a ROC curve and an
attention figure in the same folder turned out to come from two different
checkpoints several hours and one reverted notebook apart, and nothing on disk
said so.
"""
import hashlib
import json
import subprocess
from datetime import datetime

from . import config
from .config import MANIFEST_PATH


def _git_sha():
    """Current commit, or 'unknown'. Never raises -- a missing SHA must not be
    able to fail a run."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _config_snapshot():
    """Every upper-case scalar in config.py -- the knobs, without the paths."""
    skip = {"CLASSES", "KAGGLE_DATASET"}
    return {k: v for k, v in vars(config).items()
            if k.isupper() and k not in skip and isinstance(v, (int, float, str, bool))}


def write(dataset_hash, split_hash, extra=None):
    """Write run_manifest.json and return its short hash.

    The hash is computed from the manifest's content rather than assigned, so
    two different runs cannot share a stamp.
    """
    body = {
        "written":      datetime.now().isoformat(timespec="seconds"),
        "git_sha":      _git_sha(),
        "dataset":      config.KAGGLE_DATASET,
        "dataset_hash": dataset_hash,
        "split_hash":   split_hash,
        "classes":      list(config.CLASSES),
        "config":       _config_snapshot(),
        **(extra or {}),
    }
    body["hash"] = hashlib.sha256(
        json.dumps({k: v for k, v in body.items() if k != "written"},
                   sort_keys=True).encode()).hexdigest()[:12]
    MANIFEST_PATH.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body["hash"]


def read():
    if not MANIFEST_PATH.exists():
        return None
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def current_hash():
    m = read()
    return m["hash"] if m else "no-manifest"
