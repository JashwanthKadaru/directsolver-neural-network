"""
Burgers 1D Data Generator

Generates training data with the same broad setup used in:
    Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations",
    ICLR 2021, arXiv:2010.08895
    https://github.com/neuraloperator/neuraloperator

Usage:
    Generate 10,000 samples from the command line:
    python burgers_1d_data_generator.py -n 10000 -s 1024 -o burgers_1d_10k.npz

    Generate samples from Python:
    from burgers_1d_data_generator import generate_burgers_dataset, save_dataset
    a, u = generate_burgers_dataset(n_samples=10000, s=1024, seed=42)
    save_dataset(a, u, "burgers_1d_10k.npz")
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

BATCH_SIZE = 500


def grf_periodic_1d(
    n_modes: int,
    gamma: float = 2.5,
    tau: float = 7.0,
    sigma: float = 49.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample a periodic 1D Gaussian random field on [0, 1].

    Parameters
    ----------
    n_modes : int
        Number of Fourier modes (typically s/2 for grid size s).
    gamma, tau, sigma : float
        GRF parameters controlling smoothness and variance.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    u : np.ndarray of shape (n_modes * 2,)
        Real-valued field evaluated on a grid over [0, 1).
    """
    if rng is None:
        rng = np.random.default_rng()

    k = np.arange(1, n_modes + 1, dtype=float)
    my_eigs = np.sqrt(2) * sigma * ((2 * np.pi * k) ** 2 + tau**2) ** (-gamma / 2)

    xi_alpha = rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)
    alpha = my_eigs * xi_alpha

    n_pts = 2 * n_modes
    u_hat = np.zeros(n_pts, dtype=complex)

    u_hat[0] = 0.0
    u_hat[1:n_modes] = alpha[: n_modes - 1]
    u_hat[n_modes] = np.real(alpha[n_modes - 1]) 
    u_hat[n_modes + 1 :] = np.conj(alpha[n_modes - 2 :: -1])

    u = np.fft.ifft(u_hat * n_pts).real
    return u


def _apply_dealiasing(u_hat: np.ndarray, dealias_cutoff: int) -> np.ndarray:
    """Apply 2/3 dealiasing rule: zero modes with |k| > N/3."""
    u_hat = u_hat.copy()
    n = len(u_hat)
    u_hat[dealias_cutoff : n - dealias_cutoff + 1] = 0
    return u_hat


def burgers_rhs_spectral(
    u_hat: np.ndarray,
    visc: float,
    k: np.ndarray,
    dealias_cutoff: int,
) -> np.ndarray:
    """
    Compute d(u_hat)/dt for the viscous Burgers equation.

    The equation is u_t + u * u_x = visc * u_xx.
    In Fourier space, the right hand side is -visc * k**2 * u_hat
    minus the transform of u * u_x.
    """
    n = len(u_hat)
    rhs = -visc * (k**2) * u_hat

    ux_hat = 1j * k * u_hat
    ux_hat = _apply_dealiasing(ux_hat, dealias_cutoff)

    u = np.fft.ifft(u_hat).real
    ux = np.fft.ifft(ux_hat).real
    conv = u * ux

    conv_hat = np.fft.fft(conv) / n
    conv_hat = _apply_dealiasing(conv_hat, dealias_cutoff)
    rhs = rhs - conv_hat

    return rhs


def _compute_stable_dt(u: np.ndarray, dx: float, visc: float, cfl: float = 0.5) -> float:
    """
    Compute CFL-stable time step for Burgers: dt < CFL * dx / max|u|.
    Also respect the spectral diffusion limit.
    """
    u_max = np.abs(u).max()
    if u_max < 1e-10:
        u_max = 1e-10
    dt_adv = cfl * dx / u_max
    n = len(u)
    k_max = n / 3
    dt_diff = 0.5 / (visc * (np.pi * k_max) ** 2)
    return min(dt_adv, dt_diff)


def solve_burgers_1d(
    u0: np.ndarray,
    t_span: tuple[float, float],
    n_steps: int,
    visc: float,
    rng: Optional[np.random.Generator] = None,
    adaptive_dt: bool = True,
) -> np.ndarray:
    """
    Solve 1D viscous Burgers equation using pseudo-spectral method with RK4.

    Parameters
    ----------
    u0 : np.ndarray
        Initial condition (periodic, evaluated on grid).
    t_span : (t0, t1)
        Time interval.
    n_steps : int
        Number of time steps (minimum; may increase if adaptive_dt).
    visc : float
        Viscosity.
    adaptive_dt : bool
        If True, use CFL-based dt (may use more steps than n_steps).

    Returns
    -------
    u_trajectory : np.ndarray of shape (n_steps + 1, n)
        Solution at t_0, t_1, ..., t_{n_steps}.
    """
    n = len(u0)
    dx = 1.0 / n
    t0, t1 = t_span
    total_time = t1 - t0

    if adaptive_dt:
        dt_max = _compute_stable_dt(u0, dx, visc)
        n_steps = max(n_steps, int(np.ceil(total_time / dt_max)))

    dt = total_time / n_steps

    k = np.fft.fftfreq(n) * 2 * np.pi * n
    k = k.astype(complex)

    dealias_cutoff = int(np.ceil(n / 3))

    u_hat = np.fft.fft(u0) / n
    u_hat = _apply_dealiasing(u_hat, dealias_cutoff)

    trajectory = [u0.copy()]
    u_hat_ = u_hat.copy()

    for _ in range(n_steps):
        k1 = burgers_rhs_spectral(u_hat_, visc, k, dealias_cutoff)
        k2 = burgers_rhs_spectral(u_hat_ + 0.5 * dt * k1, visc, k, dealias_cutoff)
        k3 = burgers_rhs_spectral(u_hat_ + 0.5 * dt * k2, visc, k, dealias_cutoff)
        k4 = burgers_rhs_spectral(u_hat_ + dt * k3, visc, k, dealias_cutoff)

        u_hat_ = u_hat_ + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        u_hat_ = _apply_dealiasing(u_hat_, dealias_cutoff)

        u_phys = np.fft.ifft(u_hat_ * n).real
        trajectory.append(u_phys.copy())

    return np.array(trajectory)


def _generate_batch(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Generate one batch for a worker process."""
    (start_idx, count, s, n_steps, visc, gamma, tau, sigma, seed, adaptive_dt) = args
    rng = np.random.default_rng(seed)
    n_modes = s // 2
    a_batch = np.zeros((count, s), dtype=np.float64)
    u_batch = np.zeros((count, s), dtype=np.float64)
    for i in range(count):
        u0 = grf_periodic_1d(n_modes, gamma, tau, sigma, rng)
        traj = solve_burgers_1d(u0, (0.0, 1.0), n_steps, visc, adaptive_dt=adaptive_dt)
        a_batch[i] = traj[0]
        u_batch[i] = traj[-1]
    return a_batch, u_batch


def generate_burgers_dataset(
    n_samples: int = 10_000,
    s: int = 1024,
    n_steps: int = 200,
    visc: float = 1.0 / 1000,
    gamma: float = 2.5,
    tau: float = 7.0,
    sigma: float = 49.0,
    seed: Optional[int] = None,
    verbose: bool = True,
    adaptive_dt: bool = True,
    n_cores: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate (input, output) pairs for Burgers 1D.

    Input  = initial condition u(x, 0) on grid of size s.
    Output = solution u(x, 1) at final time.

    Parameters
    ----------
    n_samples : int
        Number of (input, output) pairs.
    s : int
        Spatial grid size (number of points).
    n_steps : int
        Time steps from t=0 to t=1.
    visc : float
        Viscosity.
    gamma, tau, sigma : float
        GRF parameters for initial conditions.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool
        Print progress.
    n_cores : int
        Number of CPU cores for parallel batches (batch size 500).

    Returns
    -------
    a : np.ndarray of shape (n_samples, s)
        Initial conditions (inputs).
    u : np.ndarray of shape (n_samples, s)
        Solutions at t=1 (outputs).
    """
    base_seed = seed if seed is not None else np.random.randint(0, 2**31)
    a = np.zeros((n_samples, s), dtype=np.float64)
    u = np.zeros((n_samples, s), dtype=np.float64)

    if n_cores <= 1:
        rng = np.random.default_rng(base_seed)
        n_modes = s // 2
        for j in range(n_samples):
            u0 = grf_periodic_1d(n_modes, gamma, tau, sigma, rng)
            traj = solve_burgers_1d(u0, (0.0, 1.0), n_steps, visc, adaptive_dt=adaptive_dt)
            a[j], u[j] = traj[0], traj[-1]
            if verbose and (j + 1) % BATCH_SIZE == 0:
                print(f"  Generated {j + 1}/{n_samples} samples")
        return a, u

    batches = []
    start = 0
    while start < n_samples:
        count = min(BATCH_SIZE, n_samples - start)
        batches.append((start, count, s, n_steps, visc, gamma, tau, sigma, base_seed + start, adaptive_dt))
        start += count

    n_workers = min(n_cores, len(batches))
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_generate_batch, b): b for b in batches}
        for fut in as_completed(futures):
            b = futures[fut]
            start_idx, count = b[0], b[1]
            a_batch, u_batch = fut.result()
            a[start_idx : start_idx + count] = a_batch
            u[start_idx : start_idx + count] = u_batch
            done += count
            if verbose:
                print(f"  Generated {done}/{n_samples} samples")

    return a, u


def save_dataset(
    a: np.ndarray,
    u: np.ndarray,
    path: str | Path,
    format: str = "npz",
) -> None:
    """
    Save dataset to disk.

    Parameters
    ----------
    a, u : np.ndarray
        Inputs and outputs.
    path : str or Path
        Output path (without extension for npz).
    format : "npz" or "mat"
        Output format.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "npz":
        out_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
        np.savez_compressed(out_path, a=a, u=u)
        print(f"Saved to {out_path}")
    elif format == "mat":
        try:
            import scipy.io
        except ImportError:
            raise ImportError("scipy required for .mat export: pip install scipy")
        out_path = path if path.suffix == ".mat" else path.with_suffix(".mat")
        scipy.io.savemat(out_path, {"a": a, "u": u})
        print(f"Saved to {out_path}")
    else:
        raise ValueError(f"Unknown format: {format}")


def validate_solver() -> bool:
    """
    Run a quick sanity check: known solution or conservation check.
    Returns True if validation passes.
    """
    n = 256
    u0 = 0.1 * np.sin(2 * np.pi * np.linspace(0, 1, n, endpoint=False))

    traj = solve_burgers_1d(
        u0, (0.0, 0.1), 50, visc=0.05, adaptive_dt=True
    )
    u_final = traj[-1]

    if not np.isfinite(u_final).all():
        print("VALIDATION FAILED: NaN/Inf in solution")
        return False

    if np.abs(u_final).max() > 10 * np.abs(u0).max():
        print("VALIDATION WARNING: Solution grew unexpectedly")
        return False

    print("Validation passed.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate Burgers 1D dataset for neural operator training"
    )
    parser.add_argument(
        "-n", "--n-samples",
        type=int,
        default=10_000,
        help="Number of (input, output) pairs (default: 10000)",
    )
    parser.add_argument(
        "-s", "--grid-size",
        type=int,
        default=1024,
        help="Spatial grid size (default: 1024)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Time steps from t=0 to t=1 (default: 200)",
    )
    parser.add_argument(
        "--visc",
        type=float,
        default=1e-3,
        help="Viscosity (default: 0.001)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="burgers_1d_10k.npz",
        help="Output path (default: burgers_1d_10k.npz)",
    )
    parser.add_argument(
        "--format",
        choices=["npz", "mat"],
        default="npz",
        help="Output format (default: npz)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run solver validation before generating",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--no-adaptive",
        action="store_true",
        help="Use fixed dt (steps only); may be unstable for large |u|",
    )
    parser.add_argument(
        "-j", "--num-cores",
        type=int,
        default=1,
        help="Number of CPU cores for parallel batches of 500 (default: 1)",
    )
    args = parser.parse_args()

    if args.validate:
        validate_solver()

    print(f"Generating {args.n_samples} Burgers 1D samples (s={args.grid_size}, steps={args.steps}, cores={args.num_cores})...")
    a, u = generate_burgers_dataset(
        n_samples=args.n_samples,
        s=args.grid_size,
        n_steps=args.steps,
        visc=args.visc,
        seed=args.seed,
        verbose=not args.quiet,
        adaptive_dt=not args.no_adaptive,
        n_cores=args.num_cores,
    )
    print(f"Shape: a={a.shape}, u={u.shape}")

    save_dataset(a, u, args.output, format=args.format)


if __name__ == "__main__":
    main()
