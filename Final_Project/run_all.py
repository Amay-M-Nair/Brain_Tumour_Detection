"""Execute the phase notebooks in order, checking each one actually succeeded.

This exists because of a specific mistake worth not repeating. The notebooks
were once run from a shell one-liner that piped nbconvert through `tail`, and a
pipeline's exit code is the last command's -- so `tail` returning 0 masked a
dead kernel completely. The chain carried on to the next notebook, which kept
its stale outputs from an earlier dataset while everything downstream treated
them as current.

So: one notebook at a time, exit code checked, stop on the first failure, and
report which cell died rather than only that something did.

    python run_all.py            # every phase, in order
    python run_all.py 01 02      # just these, still in order
    python run_all.py --check    # inspect stored outputs, run nothing
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT      = Path(__file__).resolve().parent
NOTEBOOKS = ROOT / "notebooks"
ORDER     = ["01_data", "02_model", "03_train", "04_ablation", "05_evaluation"]
TIMEOUT   = {"01_data": 1800, "02_model": 900, "03_train": 3600,
             "04_ablation": 7200, "05_evaluation": 1800}


def existing():
    return [n for n in ORDER if (NOTEBOOKS / f"{n}.ipynb").exists()]


def stored_errors(path):
    """Error outputs saved in a notebook, as (cell index, exception) triples.

    An exit code says the run failed; this says where. It also catches the other
    case worth catching -- a notebook whose committed outputs contain a
    traceback nobody noticed.
    """
    nb = json.loads(path.read_text(encoding="utf-8"))
    return [(i, o.get("ename"), o.get("evalue"))
            for i, c in enumerate(nb["cells"])
            for o in c.get("outputs", [])
            if o.get("output_type") == "error"]


def unexecuted(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return sum(1 for c in nb["cells"]
               if c["cell_type"] == "code" and c.get("execution_count") is None)


def run(name):
    path = NOTEBOOKS / f"{name}.ipynb"
    print(f"\n{'=' * 64}\n  {name}\n{'=' * 64}", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
         "--inplace", f"--ExecutePreprocessor.timeout={TIMEOUT.get(name, 1800)}",
         path.name],
        cwd=NOTEBOOKS, capture_output=True, text=True)
    mins = (time.time() - t0) / 60

    if proc.returncode != 0:
        tail = [l for l in proc.stderr.splitlines() if l.strip()][-12:]
        print(f"  FAILED after {mins:.1f} min (exit {proc.returncode})")
        print("\n".join(f"    {l}" for l in tail))
        return False

    errs = stored_errors(path)
    if errs:
        print(f"  FAILED after {mins:.1f} min — error output in cells "
              f"{[e[0] for e in errs]}")
        for i, ename, val in errs:
            print(f"    cell {i}: {ename}: {val}")
        return False

    print(f"  ok  ({mins:.1f} min)")
    return True


def check():
    ok = True
    for name in existing():
        path = NOTEBOOKS / f"{name}.ipynb"
        errs, pending = stored_errors(path), unexecuted(path)
        if errs:
            status, ok = f"{len(errs)} error cell(s): {[e[1] for e in errs]}", False
        elif pending:
            status, ok = f"{pending} cell(s) never executed", False
        else:
            status = "ok"
        print(f"  {name:<16} {status}")
    missing = [n for n in ORDER if n not in existing()]
    if missing:
        print(f"  not built yet: {', '.join(missing)}")
    return ok


def main():
    if "--check" in sys.argv:
        return 0 if check() else 1

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    names = [n for n in existing() if not args or any(a in n for a in args)]
    if not names:
        print(f"nothing matched {args}; available: {existing()}")
        return 2

    for name in names:
        if not run(name):
            print(f"\nstopped at {name}. Nothing after it was run.")
            return 1
    print(f"\n{'=' * 64}\n  {len(names)} notebook(s) completed\n{'=' * 64}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
