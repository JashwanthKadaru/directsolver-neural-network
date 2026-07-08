"""
2D Darcy flow dataset generator.

Output: an .npz with a, u of shape (n_samples, N, N), float32.

References:
  [1] Li et al., "Fourier Neural Operator for Parametric PDEs", ICLR 2021
      (arXiv:2010.08895).
"""

from __future__ import annotations

import math
import os
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np
import numpy.fft as npfft
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# Constants
BATCH_SIZE: int   = 500    # samples per parallel task
_N_DEFAULT: int   = 96     # interior grid side-length
_ALPHA:     float = 2.0    # GRF Sobolev exponent      (FNO paper Sec A.3.2)
_TAU:       float = 3.0    # GRF length-scale          (FNO paper Sec A.3.2)
_A_LOW:     float = 4.0    # permeability below GRF threshold
_A_HIGH:    float = 12.0   # permeability above GRF threshold
_FORCE:     float = 1.0   
_SEED:      int   = 42     # base random seed


class GaussianRF:
    def __init__(
        self,
        size:  int,
        alpha: float = _ALPHA,
        tau:   float = _TAU,
        sigma: Optional[float] = None,
    ) -> None:
        dim = 2
        if sigma is None:
            sigma = tau ** (0.5 * (2.0 * alpha - dim))

        k_max = size // 2
        k = np.concatenate([np.arange(0, k_max), np.arange(-k_max, 0)])

        # 2D wavenumber grids -- matches FNO repo exactly
        k_x = np.tile(k[:, None], (1, size))   # (size, size): varies in axis-0
        k_y = np.tile(k[None, :], (size, 1))   # (size, size): varies in axis-1

        self.sqrt_eig: np.ndarray = (
            (size ** 2)
            * math.sqrt(2.0)
            * sigma
            * (4.0 * math.pi ** 2 * (k_x ** 2 + k_y ** 2) + tau ** 2)
            ** (-alpha / 2.0)
        )
        self.sqrt_eig[0, 0] = 0.0          # enforce zero mean
        self._shape = (size, size)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        xi = rng.standard_normal((n, *self._shape))
        return npfft.ifft2(self.sqrt_eig * xi).real   # broadcast over batch


def _build_fd_matrix(a: np.ndarray, h: float) -> sp.csr_matrix:
    N   = a.shape[0]
    ih2 = 1.0 / (h * h)

    # Pad permeability with edge values -> (N+2, N+2)
    # a_pad[i+1, j+1] = a[i, j] for i, j in {0, ..., N-1}
    ap = np.pad(a, 1, mode="edge")

    # 0-indexed interior coordinate meshgrid
    ii, jj = np.mgrid[0:N, 0:N]          # both (N, N)

    # Half-point conductances (arithmetic mean * 1/h^2)
    # Notation: W=west(j-1), E=east(j+1), S=south(i-1), N=north(i+1)
    cW = 0.5 * ih2 * (ap[ii+1, jj+1] + ap[ii+1, jj  ])
    cE = 0.5 * ih2 * (ap[ii+1, jj+1] + ap[ii+1, jj+2])
    cS = 0.5 * ih2 * (ap[ii+1, jj+1] + ap[ii,   jj+1])
    cN = 0.5 * ih2 * (ap[ii+1, jj+1] + ap[ii+2, jj+1])

    k    = (ii * N + jj).ravel()          # global DOF index (row-major)
    cWr  = cW.ravel(); cEr = cE.ravel()
    cSr  = cS.ravel(); cNr = cN.ravel()

    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    data_list: list[np.ndarray] = []

    def _add(r: np.ndarray, c: np.ndarray, d: np.ndarray) -> None:
        rows_list.append(r); cols_list.append(c); data_list.append(d)

    # Diagonal  (sum of all four conductances)
    _add(k, k, cWr + cEr + cSr + cNr)

    # West  off-diagonal  (j > 0)
    m = jj.ravel() > 0
    _add(k[m], k[m] - 1,  -cWr[m])

    # East  off-diagonal  (j < N-1)
    m = jj.ravel() < N - 1
    _add(k[m], k[m] + 1,  -cEr[m])

    # South off-diagonal  (i > 0)
    m = ii.ravel() > 0
    _add(k[m], k[m] - N,  -cSr[m])

    # North off-diagonal  (i < N-1)
    m = ii.ravel() < N - 1
    _add(k[m], k[m] + N,  -cNr[m])

    n_dof = N * N
    A = sp.coo_matrix(
        (np.concatenate(data_list),
         (np.concatenate(rows_list), np.concatenate(cols_list))),
        shape=(n_dof, n_dof),
    ).tocsr()
    return A


def _worker(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    start_idx, count, N, alpha, tau, a_low, a_high, force, seed = args

    rng   = np.random.default_rng(seed)
    grf   = GaussianRF(size=N, alpha=alpha, tau=tau)
    h     = 1.0 / (N + 1)
    f_rhs = np.full(N * N, force, dtype=np.float64)   # flat RHS vector

    a_out = np.empty((count, N, N), dtype=np.float32)
    u_out = np.empty((count, N, N), dtype=np.float32)

    for s in range(count):
        raw     = grf.sample(1, rng)[0]                      # (N, N)
        a_field = np.where(raw >= 0.0, a_high, a_low).astype(np.float64)
        A = _build_fd_matrix(a_field, h)
        u_flat = spla.spsolve(A, f_rhs)                      # (N^2,)

        a_out[s] = a_field.astype(np.float32)
        u_out[s] = u_flat.reshape(N, N).astype(np.float32)

    return a_out, u_out


def generate_darcy_dataset(
    n_samples:   int   = 40_000,
    N:           int   = _N_DEFAULT,
    alpha:       float = _ALPHA,
    tau:         float = _TAU,
    a_low:       float = _A_LOW,
    a_high:      float = _A_HIGH,
    force:       float = _FORCE,
    n_cores:     int   = 10,
    base_seed:   int   = _SEED,
    output_path: str   = "darcy_2d_96x96_40K.npz",
    verbose:     bool  = True,
) -> None:
    a_all = np.empty((n_samples, N, N), dtype=np.float32)
    u_all = np.empty((n_samples, N, N), dtype=np.float32)

    batches: list[tuple] = []
    start = 0
    while start < n_samples:
        count = min(BATCH_SIZE, n_samples - start)
        batches.append((
            start, count, N, alpha, tau, a_low, a_high, force,
            base_seed + start,
        ))
        start += count

    n_workers = min(n_cores, len(batches))
    done = 0
    t0   = time.time()

    if verbose:
        print(
            f"N={N}x{N} alpha={alpha} tau={tau} a in {{{a_low},{a_high}}} "
            f"f={force} samples={n_samples} "
            f"batches={len(batches)}x{BATCH_SIZE} workers={n_workers}"
        )

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        future_to_batch = {ex.submit(_worker, b): b for b in batches}

        for fut in as_completed(future_to_batch):
            b = future_to_batch[fut]
            b_start, b_count = b[0], b[1]

            try:
                a_batch, u_batch = fut.result()
            except Exception as exc:
                raise RuntimeError(
                    f"Worker for batch starting at {b_start} failed: {exc}"
                ) from exc

            a_all[b_start : b_start + b_count] = a_batch
            u_all[b_start : b_start + b_count] = u_batch
            done += b_count

            if verbose:
                elapsed = time.time() - t0
                rate    = done / elapsed if elapsed > 0 else 0.0
                eta     = (n_samples - done) / rate if rate > 0 else float("inf")
                print(
                    f"  [{elapsed:7.1f}s]  {done:>6}/{n_samples}  "
                    f"({rate:5.1f} samples/s   ETA {eta:5.0f}s)"
                )

    # Sanity Check
    nan_a = int(np.isnan(a_all).sum())
    nan_u = int(np.isnan(u_all).sum())
    if nan_a or nan_u:
        print(f"WARNING: NaN values detected -- a: {nan_a}, u: {nan_u}")
    else:
        # u must be non-negative everywhere (f>0, Dirichlet BC u=0, a>0)
        neg_u = int((u_all < 0).sum())
        if neg_u > 0:
            print(f"WARNING: {neg_u} negative u values (solver instability?)")
        if verbose:
            print(
                f"\n  Sanity OK\n"
                f"  a in [{a_all.min():.1f}, {a_all.max():.1f}]  "
                f"(expected {{{a_low}, {a_high}}})\n"
                f"  u in [{u_all.min():.6f}, {u_all.max():.6f}]  "
                f"(must be >= 0)"
            )

    # Save to disk
    np.savez_compressed(output_path, a=a_all, u=u_all)
    size_mb  = os.path.getsize(output_path) / 1e6
    total_s  = time.time() - t0
    if verbose:
        print(
            f"\n  Saved -> {output_path}\n"
            f"  a shape: {a_all.shape}   u shape: {u_all.shape}\n"
            f"  File size: {size_mb:.1f} MB   Total time: {total_s:.1f}s\n"
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Generate 2D Darcy flow (a -> u) dataset at N x N grid.\n"
            "Reference: Li et al., ICLR 2021  (arXiv:2010.08895)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n-samples", type=int,   default=40_000,
                   help="Total number of (a, u) pairs (default: 40000)")
    p.add_argument("--N",         type=int,   default=_N_DEFAULT,
                   help="Interior grid side-length (default: 96; must = 2^L*m, m<7)")
    p.add_argument("--alpha",     type=float, default=_ALPHA,
                   help="GRF Sobolev exponent (default: 2.0)")
    p.add_argument("--tau",       type=float, default=_TAU,
                   help="GRF length-scale (default: 3.0)")
    p.add_argument("--a-low",     type=float, default=_A_LOW,
                   help="Permeability below GRF=0 threshold (default: 4.0)")
    p.add_argument("--a-high",    type=float, default=_A_HIGH,
                   help="Permeability above GRF=0 threshold (default: 12.0)")
    p.add_argument("--force",     type=float, default=_FORCE,
                   help="Constant forcing f(x)=force (default: 1.0)")
    p.add_argument("-j", "--num-cores", type=int, default=10,
                   help="Number of parallel worker processes (default: 10)")
    p.add_argument("--seed",      type=int,   default=_SEED,
                   help="Base random seed (default: 42)")
    p.add_argument("--output",    type=str,   default="darcy_2d_96x96_40K.npz",
                   help="Output .npz file path")
    p.add_argument("--quiet",     action="store_true",
                   help="Suppress progress output")
    args = p.parse_args()

    generate_darcy_dataset(
        n_samples   = args.n_samples,
        N           = args.N,
        alpha       = args.alpha,
        tau         = args.tau,
        a_low       = args.a_low,
        a_high      = args.a_high,
        force       = args.force,
        n_cores     = args.num_cores,
        base_seed   = args.seed,
        output_path = args.output,
        verbose     = not args.quiet,
    )


if __name__ == "__main__":
    main()
