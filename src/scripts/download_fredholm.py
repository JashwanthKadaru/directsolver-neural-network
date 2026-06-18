"""Download the Fredholm IE datasets from Kaggle Hub into
``datasets/fredholm/``. The Kaggle archive ships five sizes; this script
copies the three used by ``runner.run_fredholm_sweeps`` (``N = 1600,
6400, 14400``).

Authentication: reads ``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` from the
environment (or ``~/.kaggle/kaggle.json``).

Usage:
    python -m scripts.download_fredholm
    python -m scripts.download_fredholm --force
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import kagglehub
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FREDHOLM_DIR = ROOT / "datasets" / "fredholm"

KAGGLE_SLUG = "jashwanthreddykadaru/fredholm-ie-datasets"

FREDHOLM_FILES = [
    "Fredholm_IE_dataset_N=1600.txt.npz",
    "Fredholm_IE_dataset_N=6400.txt.npz",
    "Fredholm_IE_dataset_N=14400.txt.npz",
]

EXPECTED_KEYS = ("b", "x")


def setup_auth() -> None:
    has_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    has_file = (Path.home() / ".kaggle" / "kaggle.json").exists()
    if not (has_env or has_file):
        sys.exit(
            "ERROR: No Kaggle credentials found.\n"
            "Export KAGGLE_USERNAME and KAGGLE_KEY in your shell, OR place a\n"
            "kaggle.json at ~/.kaggle/kaggle.json."
        )


def verify_npz(path: Path) -> tuple[int, tuple]:
    with np.load(path) as f:
        missing = [k for k in EXPECTED_KEYS if k not in f.files]
        if missing:
            raise KeyError(
                f"{path.name}: missing expected NPZ keys {missing}; "
                f"file contains {f.files}"
            )
        b = f["b"]
        x = f["x"]
        if b.shape != x.shape:
            raise ValueError(f"{path.name}: shape mismatch b={b.shape} vs x={x.shape}")
        return int(b.shape[0]), tuple(b.shape[1:])


def cleanup_cache(cache_dir: Path) -> None:
    target = cache_dir
    parts = target.parts
    if "versions" in parts:
        idx = parts.index("versions")
        target = Path(*parts[:idx])
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if canonical files already exist.")
    ap.add_argument("--keep-cache", action="store_true",
                    help="Skip removing the kagglehub cache after copying.")
    args = ap.parse_args()

    setup_auth()
    FREDHOLM_DIR.mkdir(parents=True, exist_ok=True)

    targets = {name: FREDHOLM_DIR / name for name in FREDHOLM_FILES}
    if not args.force and all(p.exists() for p in targets.values()):
        print("[fredholm] all 3 files already present; verifying...")
        for name, p in targets.items():
            n, shape = verify_npz(p)
            print(f"  {name:<40}  n={n}  sample_shape={shape}")
        print(f"\nAll Fredholm datasets ready under: {FREDHOLM_DIR}")
        return 0

    print(f"[fredholm] downloading {KAGGLE_SLUG} ...")
    cache_path = Path(kagglehub.dataset_download(KAGGLE_SLUG))
    print(f"[fredholm] cache: {cache_path}")

    results = []
    any_failed = False
    for name, target in targets.items():
        if target.exists() and not args.force:
            try:
                n, shape = verify_npz(target)
                results.append((name, "skipped (exists)", n, shape))
                continue
            except Exception as e:
                print(f"[fredholm] {name}: existing file failed verify ({e!r}); re-copying.")

        matches = list(cache_path.rglob(name))
        if not matches:
            results.append((name, f"FAILED: not found in cache {cache_path}", -1, ()))
            any_failed = True
            continue
        if len(matches) > 1:
            print(f"[fredholm] {name}: multiple matches; using {matches[0]}")
        src = matches[0]
        shutil.copy2(src, target)
        try:
            n, shape = verify_npz(target)
            results.append((name, "ok", n, shape))
        except Exception as e:
            results.append((name, f"FAILED verify: {e!r}", -1, ()))
            any_failed = True

    if not args.keep_cache and not any_failed:
        cleanup_cache(cache_path)

    print("\n" + "=" * 78)
    print(f"{'file':<42}{'status':<22}{'details'}")
    print("-" * 78)
    for name, status, n, shape in results:
        if status.startswith("FAILED"):
            print(f"{name:<42}{status}")
        else:
            print(f"{name:<42}{status:<22}n={n}, sample_shape={shape}")
    print("=" * 78)

    if any_failed:
        print("\nOne or more files failed. See messages above.")
        return 1
    print(f"\nAll Fredholm datasets ready under: {FREDHOLM_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
