"""Multiscale Neural Network based on H-matrices (MNN-H) baseline.

Implements the architecture of Fan, Lin, Ying, Zepeda-Nunez. Provides 1D
and 2D variants; both use periodic padding and follow the band-size
schedule from the reference (kernel widths grow with hierarchy level).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _periodic_pad_1d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    s = kernel_size // 2
    left = x[..., x.size(-1) - s:]
    right = x[..., : kernel_size - s - 1]
    return torch.cat([left, x, right], dim=-1)


def _periodic_pad_2d(x: torch.Tensor, kx: int, ky: int) -> torch.Tensor:
    sx = kx // 2
    x = torch.cat([x[..., x.size(-2) - sx:, :], x, x[..., : kx - sx - 1, :]], dim=-2)
    sy = ky // 2
    x = torch.cat([x[..., :, x.size(-1) - sy:], x, x[..., :, : ky - sy - 1]], dim=-1)
    return x


def _matrix2tensor(x: torch.Tensor, wx: int, wy: int) -> torch.Tensor:
    B, C, Nx, Ny = x.shape
    assert C == 1 and Nx % wx == 0 and Ny % wy == 0
    y = x.permute(0, 2, 3, 1).contiguous()
    y = y.reshape(B, Nx // wx, wx, Ny // wy, wy)
    y = y.permute(0, 1, 3, 2, 4).contiguous()
    y = y.reshape(B, Nx // wx, Ny // wy, wx * wy)
    return y.permute(0, 3, 1, 2).contiguous()


def _tensor2matrix(x: torch.Tensor, wx: int, wy: int) -> torch.Tensor:
    B, C, Nx, Ny = x.shape
    assert C == wx * wy
    y = x.permute(0, 2, 3, 1).contiguous()
    y = y.reshape(B, Nx, Ny, wx, wy)
    y = y.permute(0, 1, 3, 2, 4).contiguous()
    return y.reshape(B, Nx * wx, Ny * wy)


class MNNH1d(nn.Module):
    """MNN-H for 1D periodic problems."""

    def __init__(
        self,
        N: int,
        L: int,
        alpha: int,
        n_cnn: int,
        n_b_ad: int = 1,
        n_b_2: int = 2,
        n_b_l: int = 3,
    ):
        super().__init__()
        if N % (2 ** L) != 0:
            raise ValueError(f"N ({N}) must be divisible by 2**L ({2 ** L}).")
        if n_cnn < 1:
            raise ValueError("n_cnn must be >= 1.")

        self.N = N
        self.L = L
        self.alpha = alpha
        self.n_cnn = n_cnn
        self.m = N // (2 ** L)

        self.level_keys = list(range(2, L + 1))
        self.V = nn.ModuleDict()
        self.kernels = nn.ModuleDict()
        self.U = nn.ModuleDict()
        self.kernel_sizes_lvl: dict[str, int] = {}

        for ll in self.level_keys:
            w = self.m * (2 ** (L - ll))
            n_b = n_b_2 if ll == 2 else n_b_l
            k_size = 2 * n_b + 1
            key = str(ll)

            self.V[key] = nn.Conv1d(in_channels=1, out_channels=alpha,
                                    kernel_size=w, stride=w, padding=0, bias=True)
            stack = nn.ModuleList([
                nn.Conv1d(in_channels=alpha, out_channels=alpha,
                          kernel_size=k_size, stride=1, padding=0, bias=True)
                for _ in range(n_cnn)
            ])
            self.kernels[key] = stack
            self.U[key] = nn.Conv1d(in_channels=alpha, out_channels=w,
                                    kernel_size=1, stride=1, padding=0, bias=True)
            self.kernel_sizes_lvl[key] = k_size

        self.k_ad = 2 * n_b_ad + 1
        self.adjacent = nn.ModuleList([
            nn.Conv1d(in_channels=self.m, out_channels=self.m,
                      kernel_size=self.k_ad, stride=1, padding=0, bias=True)
            for _ in range(n_cnn)
        ])

    def _level_branch(self, x_in: torch.Tensor, ll: int) -> torch.Tensor:
        key = str(ll)
        k_size = self.kernel_sizes_lvl[key]

        v = self.V[key](x_in)
        h = v
        for conv in self.kernels[key]:
            h = _periodic_pad_1d(h, k_size)
            h = F.relu(conv(h))
        u = self.U[key](h)
        B = u.size(0)
        return u.permute(0, 2, 1).contiguous().reshape(B, self.N)

    def _adjacent_branch(self, x_in: torch.Tensor) -> torch.Tensor:
        B = x_in.size(0)
        h = x_in.reshape(B, self.N // self.m, self.m).permute(0, 2, 1).contiguous()
        last = self.n_cnn - 1
        for k, conv in enumerate(self.adjacent):
            h = _periodic_pad_1d(h, self.k_ad)
            h = conv(h)
            if k != last:
                h = F.relu(h)
        return h.permute(0, 2, 1).contiguous().reshape(B, self.N)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        if v.dim() == 3:
            assert v.size(-1) == 1
            v = v.squeeze(-1)
        x_in = v.unsqueeze(1)

        u_total = self._adjacent_branch(x_in)
        for ll in self.level_keys:
            u_total = u_total + self._level_branch(x_in, ll)
        return u_total


class MNNH2d(nn.Module):
    """MNN-H for 2D periodic problems."""

    def __init__(
        self,
        Nx: int,
        Ny: int,
        L: int,
        alpha: int,
        n_cnn: int,
        w_b_ad: tuple[int, int] = (3, 3),
        w_b_2: tuple[int, int] = (5, 5),
        w_b_l: tuple[int, int] = (7, 7),
    ):
        super().__init__()
        if Nx % (2 ** L) != 0 or Ny % (2 ** L) != 0:
            raise ValueError(
                f"Nx ({Nx}) and Ny ({Ny}) must each be divisible by 2**L ({2 ** L})."
            )
        if n_cnn < 1:
            raise ValueError("n_cnn must be >= 1.")

        self.Nx = Nx
        self.Ny = Ny
        self.L = L
        self.alpha = alpha
        self.n_cnn = n_cnn
        self.m = (Nx // (2 ** L), Ny // (2 ** L))

        self.level_keys = list(range(2, L + 1))
        self.V = nn.ModuleDict()
        self.kernels = nn.ModuleDict()
        self.U = nn.ModuleDict()
        self.k_lvl: dict[str, tuple[int, int]] = {}
        self.w_lvl: dict[str, tuple[int, int]] = {}

        for ll in self.level_keys:
            w = (self.m[0] * (2 ** (L - ll)), self.m[1] * (2 ** (L - ll)))
            w_b = w_b_2 if ll == 2 else w_b_l
            key = str(ll)

            self.V[key] = nn.Conv2d(in_channels=1, out_channels=alpha,
                                    kernel_size=w, stride=w, padding=0, bias=True)
            stack = nn.ModuleList([
                nn.Conv2d(in_channels=alpha, out_channels=alpha,
                          kernel_size=w_b, stride=1, padding=0, bias=True)
                for _ in range(n_cnn)
            ])
            self.kernels[key] = stack
            self.U[key] = nn.Conv2d(in_channels=alpha, out_channels=w[0] * w[1],
                                    kernel_size=1, stride=1, padding=0, bias=True)
            self.k_lvl[key] = w_b
            self.w_lvl[key] = w

        self.w_b_ad = w_b_ad
        self.adjacent = nn.ModuleList([
            nn.Conv2d(in_channels=self.m[0] * self.m[1],
                      out_channels=self.m[0] * self.m[1],
                      kernel_size=w_b_ad, stride=1, padding=0, bias=True)
            for _ in range(n_cnn)
        ])

    def _level_branch(self, x_in: torch.Tensor, ll: int) -> torch.Tensor:
        key = str(ll)
        kx, ky = self.k_lvl[key]
        wx, wy = self.w_lvl[key]

        v = self.V[key](x_in)
        h = v
        for conv in self.kernels[key]:
            h = _periodic_pad_2d(h, kx, ky)
            h = F.relu(conv(h))
        u = self.U[key](h)
        return _tensor2matrix(u, wx, wy)

    def _adjacent_branch(self, x_in: torch.Tensor) -> torch.Tensor:
        kx, ky = self.w_b_ad
        h = _matrix2tensor(x_in, self.m[0], self.m[1])
        last = self.n_cnn - 1
        for k, conv in enumerate(self.adjacent):
            h = _periodic_pad_2d(h, kx, ky)
            h = conv(h)
            if k != last:
                h = F.relu(h)
        return _tensor2matrix(h, self.m[0], self.m[1])

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        if v.dim() == 4:
            assert v.size(-1) == 1
            v = v.squeeze(-1)
        x_in = v.unsqueeze(1)

        u_total = self._adjacent_branch(x_in)
        for ll in self.level_keys:
            u_total = u_total + self._level_branch(x_in, ll)
        return u_total


@dataclass(frozen=True)
class MNNHConfig:
    name: str
    dim: int
    N: int | tuple[int, int]
    L: int
    alpha: int
    n_cnn: int

    def build(self) -> nn.Module:
        if self.dim == 1:
            return MNNH1d(N=self.N, L=self.L, alpha=self.alpha, n_cnn=self.n_cnn)
        Nx, Ny = self.N if isinstance(self.N, tuple) else (self.N, self.N)
        return MNNH2d(Nx=Nx, Ny=Ny, L=self.L, alpha=self.alpha, n_cnn=self.n_cnn)


ABLATION_CONFIGS: dict[str, MNNHConfig] = {
    "nlse_1d":    MNNHConfig("nlse_1d",    dim=1, N=320,        L=6, alpha=10, n_cnn=5),
    "burgers_1d": MNNHConfig("burgers_1d", dim=1, N=1024,       L=7, alpha=10, n_cnn=5),
    "nlse_2d":    MNNHConfig("nlse_2d",    dim=2, N=(80, 80),   L=4, alpha=12, n_cnn=7),
    "darcy_2d":   MNNHConfig("darcy_2d",   dim=2, N=(96, 96),   L=5, alpha=15, n_cnn=5),
}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
