"""Download the four main PDE datasets (and the NLSE-1D auxiliary
``betas`` dataset) from Kaggle Hub into ``datasets/`` using the canonical
filenames expected by ``runner.data``.

Authentication: reads ``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` from the
environment (or ``~/.kaggle/kaggle.json``).

Usage:
    python -m scripts.download_datasets
    python -m scripts.download_datasets --force
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
DATASETS_DIR = ROOT / "datasets"
BETAS_SUBDIR = DATASETS_DIR / "nlse_1d_betas"


DATASETS = [
    {
        "label": "nlse_1d",
        "slug": "jashwanthreddykadaru/nlse-1d-dataset/versions/1",
        "target_name": "nlse_1d_dataset.npz",
        "source_match": None,
        "keys": ("V", "uG"),
    },
    {
        "label": "burgers_1d",
        "slug": "jashwanthreddykadaru/burgers-1d-dataset-and-python-code",
        "target_name": "burgers_1d_dataset.npz",
        "source_match": "burgers_1d_40k.npz",
        "keys": ("a", "u"),
    },
    {
        "label": "nlse_2d",
        "slug": "jashwanthreddykadaru/nlse-2d-dataset",
        "target_name": "nlse_2d_dataset.npz",
        "source_match": None,
        "keys": ("V", "uG"),
    },
    {
        "label": "darcy_2d",
        "slug": "jashwanthreddykadaru/darcys-flow-2d-40k-dataset",
        "target_name": "darcys_flow_2d_dataset.npz",
        "source_match": None,
        "keys": ("V", "uG"),
    },
]

BETAS_SPEC = {
    "label": "nlse_1d_betas",
    "slug": "jashwanthreddykadaru/nlse-1d-dataset/versions/4",
    "target_dir": BETAS_SUBDIR,
}


def setup_auth() -> None:
    has_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    has_file = (Path.home() / ".kaggle" / "kaggle.json").exists()
    if not (has_env or has_file):
        sys.exit(
            "ERROR: No Kaggle credentials found.\n"
            "Export KAGGLE_USERNAME and KAGGLE_KEY in your shell, OR place a\n"
            "kaggle.json at ~/.kaggle/kaggle.json."
        )


def find_npz(cache_dir: Path, source_match: str | None) -> Path:
    if source_match is not None:
        candidates = list(cache_dir.rglob(source_match))
        if not candidates:
            raise FileNotFoundError(f"{source_match!r} not found under {cache_dir}")
        return candidates[0]
    npzs = list(cache_dir.rglob("*.npz"))
    if len(npzs) == 0:
        raise FileNotFoundError(f"no .npz files under {cache_dir}")
    if len(npzs) > 1:
        raise RuntimeError(
            f"expected exactly one .npz under {cache_dir}, found {len(npzs)}:\n  "
            + "\n  ".join(str(p) for p in npzs)
        )
    return npzs[0]


def verify_npz(path: Path, keys: tuple[str, str]) -> tuple[int, tuple]:
    with np.load(path) as f:
        missing = [k for k in keys if k not in f.files]
        if missing:
            raise KeyError(
                f"{path.name}: missing expected NPZ keys {missing}; "
                f"file contains {f.files}"
            )
        x = f[keys[0]]
        return int(x.shape[0]), tuple(x.shape[1:])


def cleanup_cache(cache_dir: Path) -> None:
    target = cache_dir
    parts = target.parts
    if "versions" in parts:
        idx = parts.index("versions")
        target = Path(*parts[:idx])
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)


def download_main(spec: dict, force: bool) -> dict:
    target = DATASETS_DIR / spec["target_name"]
    if target.exists() and not force:
        n, shape = verify_npz(target, spec["keys"])
        return {"label": spec["label"], "status": "skipped (exists)",
                "path": target, "n": n, "shape": shape, "cache": None}

    print(f"[{spec['label']}] downloading {spec['slug']} ...")
    cache_path = Path(kagglehub.dataset_download(spec["slug"]))
    src = find_npz(cache_path, spec["source_match"])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)
    n, shape = verify_npz(target, spec["keys"])
    return {"label": spec["label"], "status": "ok", "path": target,
            "n": n, "shape": shape, "cache": cache_path}


def download_betas(force: bool) -> dict:
    BETAS_SUBDIR.mkdir(parents=True, exist_ok=True)
    existing = list(BETAS_SUBDIR.glob("*.npz"))
    if existing and not force:
        return {"label": BETAS_SPEC["label"], "status": "skipped (exists)",
                "path": BETAS_SUBDIR, "n_files": len(existing), "cache": None}

    print(f"[{BETAS_SPEC['label']}] downloading {BETAS_SPEC['slug']} ...")
    cache_path = Path(kagglehub.dataset_download(BETAS_SPEC["slug"]))
    npzs = list(cache_path.rglob("*.npz"))
    if not npzs:
        raise FileNotFoundError(f"no .npz files under {cache_path}")
    for src in npzs:
        shutil.copy2(src, BETAS_SUBDIR / src.name)
    return {"label": BETAS_SPEC["label"], "status": "ok", "path": BETAS_SUBDIR,
            "n_files": len(npzs), "cache": cache_path}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if canonical files already exist.")
    ap.add_argument("--keep-cache", action="store_true",
                    help="Skip removing the kagglehub cache after copying.")
    args = ap.parse_args()

    setup_auth()
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for spec in DATASETS:
        try:
            results.append(download_main(spec, args.force))
        except Exception as e:
            results.append({"label": spec["label"], "status": f"FAILED: {e}",
                            "path": None, "cache": None})

    try:
        results.append(download_betas(args.force))
    except Exception as e:
        results.append({"label": BETAS_SPEC["label"], "status": f"FAILED: {e}",
                        "path": None, "cache": None})

    if not args.keep_cache:
        for r in results:
            cache = r.get("cache")
            if cache is not None and r["status"] == "ok":
                cleanup_cache(cache)

    print("\n" + "=" * 78)
    print(f"{'dataset':<18}{'status':<22}{'details'}")
    print("-" * 78)
    any_failed = False
    for r in results:
        if r["status"].startswith("FAILED"):
            any_failed = True
            print(f"{r['label']:<18}{r['status']}")
            continue
        if "n_files" in r:
            details = f"{r['n_files']} files at {r['path']}"
        else:
            details = f"n={r['n']}, sample_shape={r['shape']}, at {r['path'].name}"
        print(f"{r['label']:<18}{r['status']:<22}{details}")
    print("=" * 78)
    if any_failed:
        print("\nOne or more datasets failed. See messages above.")
        return 1
    print(f"\nAll datasets ready under: {DATASETS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
