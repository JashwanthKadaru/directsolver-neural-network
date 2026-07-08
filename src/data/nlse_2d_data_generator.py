"""
Generate 2D NLSE ground-state datasets.

The script samples low-resolution Gaussian fields, uses 2D Fourier
interpolation to build smooth high-resolution potentials, sets
V = -20 * exp(IF2D(v)), and solves the 2D nonlinear Schrodinger ground state
with normalized gradient flow and an FFT2-based spectral kinetic step. Paper
defaults are Nx=Ny=80, low-resolution 10x10 fields, and beta=10.

Reference:
    1. Fan et al. (2019) "A multiscale neural network based on hierarchical matrices"
"""
import numpy as np, math, time, json
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

def fourier_interpolate_2d(v_low, Nx, Ny):
    v_low = np.asarray(v_low, dtype=np.float64)
    m1, m2 = v_low.shape
    if (Ny == m1) and (Nx == m2):
        return v_low.astype(np.float64)
    V = np.fft.fft2(v_low)
    V_shift = np.fft.fftshift(V)
    pad_y = Ny - m1
    pad_x = Nx - m2
    top = pad_y // 2
    bottom = pad_y - top
    left = pad_x // 2
    right = pad_x - left
    V_shift_padded = np.pad(V_shift, ((top,bottom),(left,right)), mode='constant', constant_values=0.0)
    V_padded = np.fft.ifftshift(V_shift_padded)
    v_high = np.fft.ifft2(V_padded).real
    v_high = v_high * ((Ny * Nx) / (m1 * m2))
    return v_high.real

# Sanity test
m1, m2 = 10, 10
Ny, Nx = 80, 80
v_low = np.zeros((m1,m2))
v_low[0,0] = 1.0
v_high = fourier_interpolate_2d(v_low, Nx, Ny)
print('v_high shape:', v_high.shape)

def ground_state_nlse_2d(V, Nx=80, Ny=80, beta=10.0, tau=1e-3, tol=1e-9, maxiter=20000, verbose=False):
    """
    Normalized gradient flow / imaginary-time solver (2D) with spectral kinetic step.
    Returns real-valued u (ground-state wavefunction) and a stats dict including rho integral.
    Domain assumed [0,1]^2 -> dx = 1/Nx, dy = 1/Ny
    """
    V = np.asarray(V, dtype=np.float64).reshape((Ny, Nx))
    dx = 1.0 / Nx
    dy = 1.0 / Ny
    # FFT frequencies consistent with domain spacing
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    k2 = KX**2 + KY**2
    denom = 1.0 + tau * 0.5 * k2

    # constant normalized with dx*dy
    psi = np.ones((Ny, Nx), dtype=np.complex128)
    norm0 = np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)
    psi /= norm0

    for it in range(int(maxiter)):
        psi_old = psi.copy()
        nonlinear = beta * (np.abs(psi_old)**2) * psi_old
        RHS = psi_old - tau * (V * psi_old + nonlinear)
        RHS_hat = np.fft.fft2(RHS)
        psi_star = np.fft.ifft2(RHS_hat / denom)
        norm_star = np.sqrt(np.sum(np.abs(psi_star)**2) * dx * dy)
        if norm_star == 0 or not np.isfinite(norm_star):
            raise RuntimeError(f'psi_star norm invalid at iter {it}: {norm_star}')
        psi = psi_star / norm_star
        diff = np.linalg.norm(psi - psi_old)
        if verbose and (it % 200 == 0):
            rho_tmp = np.abs(psi)**2
            print(f'iter {it}, diff={diff:.3e}')
        if diff < tol:
            break

    u = psi.real
    rho = np.abs(psi)**2
    integral_rho = float(np.sum(rho) * dx * dy)
    return u, {'iterations': it+1, 'convergence_diff': float(diff), 'rho_integral': integral_rho}

# Demo sample (2D)
Nx = Ny = 80
m1 = m2 = 10
sigma = 0.2
beta = 10.0

v_low = np.random.randn(m1, m2)
# use IF to upsample; uses V = -20 * exp(IF(v))
V_field = -20.0 * np.exp(fourier_interpolate_2d(v_low, Nx, Ny))

uG, stats = ground_state_nlse_2d(V_field, Nx=Nx, Ny=Ny, beta=beta, tau=1e-3, tol=1e-11, maxiter=8000, verbose=True)
print('stats', stats)

# compute density and integral
dx = 1.0 / Nx
dy = 1.0 / Ny
rho = uG**2 
print('rho integral (should be 1.0):', np.sum(rho) * dx * dy)

# 3D surface plots (or use imshow if preferred)
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
x = np.linspace(0, 1, Nx, endpoint=False)
y = np.linspace(0, 1, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='xy')

fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(1,2,1, projection='3d')
ax1.plot_surface(X, Y, V_field, cmap=cm.viridis, linewidth=0, antialiased=True)
ax1.set_title('V sample')

ax2 = fig.add_subplot(1,2,2, projection='3d')
ax2.plot_surface(X, Y, uG, cmap=cm.plasma, linewidth=0, antialiased=True)
ax2.set_title('uG')

ax1.view_init(elev=10, azim=250)
ax2.view_init(elev=10, azim=250)

plt.tight_layout()
plt.show()


def generate_nlse_dataset_2d(out_path, n_samples=200, Nx=80, Ny=80, m1=10, m2=10, beta=10.0, tau=1e-3, tol=1e-8, maxiter=10000, seed=123):
    rng = np.random.RandomState(seed)
    Vs = np.zeros((n_samples, Ny, Nx), dtype=np.float32)
    uGs = np.zeros((n_samples, Ny, Nx), dtype=np.float32)
    for i in tqdm(range(n_samples), desc='generating 2d samples'):
        v_low = rng.randn(m1, m2)
        V = -20.0 * np.exp(fourier_interpolate_2d(v_low, Nx, Ny))
        uG, stats = ground_state_nlse_2d(V, Nx=Nx, Ny=Ny, beta=beta, tau=tau, tol=tol, maxiter=maxiter)
        Vs[i,:,:] = V.astype(np.float32)
        uGs[i,:,:] = uG.astype(np.float32)
    np.savez_compressed(out_path, V=Vs, uG=uGs, meta=json.dumps({'Nx':Nx,'Ny':Ny,'m1':m1,'m2':m2,'beta':beta,'tau':tau,'seed':seed,'samples':n_samples}))
    print(f'Saved {n_samples} samples to {out_path}')
    return out_path

# Example (commented):
# generate_nlse_dataset_2d('nlse_2d_demo_200.npz', n_samples=200)
# print('2D generator ready.')

generate_nlse_dataset_2d('nlse_2d_demo_40K.npz', n_samples=40000)
