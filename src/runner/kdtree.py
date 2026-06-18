"""HODLR k-d-tree permutation for linearizing 2D grids while preserving
spatial locality. The permutation depends only on the grid size and is
cached per N.
"""

from __future__ import annotations

import numpy as np


def _kdtree_sort_indices(coords: np.ndarray, indices: np.ndarray, axis: int, ndim: int) -> np.ndarray:
    n = len(indices)
    if n <= 1:
        return indices
    sort_dim = axis % ndim
    order = np.argsort(coords[indices, sort_dim], kind="stable")
    indices = indices[order]
    n_left = n // 2
    left = _kdtree_sort_indices(coords, indices[:n_left], axis + 1, ndim)
    right = _kdtree_sort_indices(coords, indices[n_left:], axis + 1, ndim)
    return np.concatenate([left, right])


_perm_cache: dict[int, np.ndarray] = {}


def get_hodlr_permutation(N: int) -> np.ndarray:
    """Return the HODLR k-d-tree permutation for an N x N grid (flat length N^2)."""
    if N in _perm_cache:
        return _perm_cache[N]
    total = N * N
    rows, cols = np.divmod(np.arange(total), N)
    coords = np.column_stack([cols, rows]).astype(float)
    indices = np.arange(total)
    perm = _kdtree_sort_indices(coords, indices, axis=0, ndim=2)
    _perm_cache[N] = perm
    return perm


def get_inverse_permutation(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv
