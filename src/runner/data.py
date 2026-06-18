"""Dataset loaders. Each ``.npz`` archive contains two arrays (input and
target) of shape ``(N_samples, ...)``. The trailing singleton channel, if
present, is squeezed so the runner sees ``(B, N)`` for 1D and ``(B, H, W)``
for 2D inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


DATASET_FILES = {
    "nlse_1d":    "nlse_1d_dataset.npz",
    "burgers_1d": "burgers_1d_dataset.npz",
    "nlse_2d":    "nlse_2d_dataset.npz",
    "darcy_2d":   "darcys_flow_2d_dataset.npz",
    "fredholm_N1600":  "fredholm/Fredholm_IE_dataset_N=1600.txt.npz",
    "fredholm_N6400":  "fredholm/Fredholm_IE_dataset_N=6400.txt.npz",
    "fredholm_N14400": "fredholm/Fredholm_IE_dataset_N=14400.txt.npz",
}

DATASET_KEYS = {
    "nlse_1d":    ("V", "uG"),
    "burgers_1d": ("a", "u"),
    "nlse_2d":    ("V", "uG"),
    "darcy_2d":   ("V", "uG"),
    "fredholm_N1600":  ("b", "x"),
    "fredholm_N6400":  ("b", "x"),
    "fredholm_N14400": ("b", "x"),
}


def load_split(
    datasets_dir: Path,
    dataset_name: str,
    n_train: int,
    n_test: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(X_train, Y_train, X_test, Y_test)`` as float32 CPU tensors.

    Train indices are the first ``n_train`` of a seeded permutation; test
    indices are the next ``n_test``. Same seed across models guarantees
    identical splits.
    """
    path = datasets_dir / DATASET_FILES[dataset_name]
    x_key, y_key = DATASET_KEYS[dataset_name]
    with np.load(path) as f:
        X = f[x_key]
        Y = f[y_key]

    assert X.shape == Y.shape, f"shape mismatch: {X.shape} vs {Y.shape}"
    if X.ndim >= 2 and X.shape[-1] == 1:
        X = np.squeeze(X, axis=-1)
        Y = np.squeeze(Y, axis=-1)
    N_total = X.shape[0]
    if n_train + n_test > N_total:
        raise ValueError(f"n_train+n_test={n_train+n_test} exceeds dataset {N_total}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N_total)
    train_idx = perm[:n_train]
    test_idx = perm[n_train : n_train + n_test]

    Xtr = torch.from_numpy(np.ascontiguousarray(X[train_idx])).float()
    Ytr = torch.from_numpy(np.ascontiguousarray(Y[train_idx])).float()
    Xte = torch.from_numpy(np.ascontiguousarray(X[test_idx])).float()
    Yte = torch.from_numpy(np.ascontiguousarray(Y[test_idx])).float()
    return Xtr, Ytr, Xte, Yte


class TensorDataset2(Dataset):
    """Minimal ``(X, Y)`` Dataset returning CPU float32 tensors."""

    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        assert X.size(0) == Y.size(0)
        self.X = X
        self.Y = Y

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, i: int):
        return self.X[i], self.Y[i]
