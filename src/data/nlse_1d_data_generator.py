"""
Generate 1D NLSE ground-state datasets.

The script samples low-resolution Gaussian coefficients, uses Fourier
interpolation to build smooth high-resolution potentials, sets
V = -20 * exp(IF(v)), and solves the 1D nonlinear Schrodinger ground state
with normalized gradient flow. Paper defaults are N=320, M=40, and beta=10.

Reference:
    1. Fan et al. (2019) "A multiscale neural network based on hierarchical matrices"
"""

import numpy as np, math, time, json
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

N=320
L = 1.0 # length of domain [0, 1]
dx = L / N

def fourier_interpolate_1d(v_low, N_high):
    v_low = np.asarray(v_low, dtype=np.float64)
    M = v_low.shape[0]
    if N_high == M:
        return v_low.astype(np.float64)
    V = np.fft.fft(v_low)
    V_shift = np.fft.fftshift(V)
    pad_total = N_high - M
    left = pad_total // 2
    right = pad_total - left
    V_shift_padded = np.pad(V_shift, (left, right), mode='constant', constant_values=0.0)
    V_padded = np.fft.ifftshift(V_shift_padded)
    v_high = np.fft.ifft(V_padded)
    v_high = v_high * (N_high / M)
    return v_high.real

# Sanity test
M = 40; N = 320
x_low = np.linspace(0,1,M,endpoint=False)
x_high = np.linspace(0,1,N,endpoint=False)
v_low = np.sin(2*np.pi*3*x_low)
v_high_true = np.sin(2*np.pi*3*x_high)
v_high = fourier_interpolate_1d(v_low, N)
print('max abs err:', np.max(np.abs(v_high - v_high_true)))

def ground_state_nlse_1d(V, N=320, beta=10.0, tau=1e-3, tol=1e-11, maxiter=20000, verbose=False):
    V = np.asarray(V, dtype=np.float64).reshape(N)
    L = 1.0
    dx = L / N
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    k2 = k**2
    denom = 1.0 + tau * 0.5 * k2
    psi = np.ones(N, dtype=np.complex128)
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)

    for it in range(int(maxiter)):
        psi_old = psi.copy()
        nonlinear = beta * (np.abs(psi_old)**2) * psi_old
        RHS = psi_old - tau * (V * psi_old + nonlinear)
        RHS_hat = np.fft.fft(RHS)
        psi_star = np.fft.ifft(RHS_hat / denom)
        psi = psi_star / np.sqrt(np.sum(np.abs(psi_star)**2) * dx)
        diff = np.linalg.norm(psi - psi_old)
        if verbose and (it % 500 == 0):
            print(f'iter {it}, diff={diff:.3e}')
        if diff < tol:
            break
    rho = np.abs(psi)**2
    auc = np.sum(rho) * dx 
    print('AUC rho =', auc)
    u = psi.real
    if np.sum(u) < 0:
        u = -u
    return u, {'iterations': it+1, 'convergence_diff': float(diff)}

# Demo sample
N = 320; M = 40; beta = 10.0
v_low = np.random.randn(M)
V_field = -20.0 * np.exp(fourier_interpolate_1d(v_low, N))
uG, stats = ground_state_nlse_1d(V_field, N=N, beta=beta, tau=1e-3, tol=1e-11, maxiter=15000, verbose=True)
print('stats', stats)
x = np.linspace(0,1,N,endpoint=False)
plt.figure(figsize=(10,3))
plt.subplot(1,2,1); plt.plot(x, V_field); plt.title('V sample')
plt.subplot(1,2,2); plt.plot(x, uG); plt.title('u_G sample')
plt.tight_layout()

def generate_nlse_dataset_1d(out_path, n_samples=100, N=320, M=40, beta=10.0, tau=1e-3, tol=1e-11, maxiter=20000, seed=12345):
    rng = np.random.RandomState(seed)
    Vs = np.zeros((n_samples, N), dtype=np.float32)
    uGs = np.zeros((n_samples, N), dtype=np.float32)
    for i in tqdm(range(n_samples), desc='generating 1d samples'):
        v_low = rng.randn(M)
        V = -20.0 * np.exp(fourier_interpolate_1d(v_low, N))
        uG, stats = ground_state_nlse_1d(V, N=N, beta=beta, tau=tau, tol=tol, maxiter=maxiter)
        Vs[i,:] = V.astype(np.float32)
        uGs[i,:] = uG.astype(np.float32)
    np.savez_compressed(out_path, V=Vs, uG=uGs, meta=json.dumps({'N':N,'M':M,'beta':beta,'tau':tau,'seed':seed,'samples':n_samples}))
    print(f'Saved {n_samples} samples to {out_path}')
    return out_path

# Test Example (commented):
# generate_nlse_dataset_1d('nlse_1d_demo_200.npz', n_samples=200)

generate_nlse_dataset_1d('nlse_1d_demo_40K.npz', n_samples=40000)
