"""Fourier Neural Operator (FNO) baseline.

Provides distinct 1D and 2D variants (``FNO1d``, ``FNO2d``) following the
reference architecture of Li et al. (ICLR 2021): lifting layer, a stack of
Fourier layers (spectral conv + pointwise 1x1 skip + GELU), and a projection
head. Per-dataset configurations use the FNO paper's default widths and
mode counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """1D spectral convolution with low-frequency mode truncation."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    @staticmethod
    def _compl_mul1d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bix,iox->box", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        N = x.size(-1)

        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(
            B, self.out_channels, N // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        out_ft[:, :, : self.modes] = self._compl_mul1d(
            x_ft[:, :, : self.modes], self.weights
        )
        return torch.fft.irfft(out_ft, n=N, dim=-1)


class SpectralConv2d(nn.Module):
    """2D spectral convolution with separate top-left and bottom-left
    corner parameterizations of the rFFT2 spectrum."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def _compl_mul2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        H, W = x.shape[-2], x.shape[-1]

        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self._compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self._compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )
        return torch.fft.irfft2(out_ft, s=(H, W))


class FNO1d(nn.Module):
    """1D Fourier Neural Operator."""

    def __init__(
        self,
        modes: int = 16,
        width: int = 64,
        n_layers: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        proj_hidden: int = 128,
    ):
        super().__init__()
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fc0 = nn.Linear(in_channels + 1, width)

        self.spectral_convs = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(n_layers)]
        )
        self.pointwise_convs = nn.ModuleList(
            [nn.Conv1d(width, width, kernel_size=1) for _ in range(n_layers)]
        )

        self.fc1 = nn.Linear(width, proj_hidden)
        self.fc2 = nn.Linear(proj_hidden, out_channels)

    @staticmethod
    def _get_grid(shape: tuple[int, int, int], device: torch.device) -> torch.Tensor:
        B, N, _ = shape
        grid = torch.linspace(0.0, 1.0, N, device=device).reshape(1, N, 1)
        return grid.expand(B, N, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.size(-1) != self.in_channels:
            raise ValueError(
                f"expected last dim {self.in_channels}, got {x.size(-1)}"
            )

        grid = self._get_grid(x.shape, x.device)
        x = torch.cat([x, grid], dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (sconv, pconv) in enumerate(zip(self.spectral_convs, self.pointwise_convs)):
            x = sconv(x) + pconv(x)
            if i != self.n_layers - 1:
                x = F.gelu(x)

        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1) if self.out_channels == 1 else x


class FNO2d(nn.Module):
    """2D Fourier Neural Operator.

    For non-periodic problems (e.g. Darcy flow with Dirichlet boundary),
    pass ``padding > 0`` to pad before the spectral layers and crop after.
    """

    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        n_layers: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        proj_hidden: int = 128,
        padding: int = 0,
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        self.fc0 = nn.Linear(in_channels + 2, width)

        self.spectral_convs = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)]
        )
        self.pointwise_convs = nn.ModuleList(
            [nn.Conv2d(width, width, kernel_size=1) for _ in range(n_layers)]
        )

        self.fc1 = nn.Linear(width, proj_hidden)
        self.fc2 = nn.Linear(proj_hidden, out_channels)

    @staticmethod
    def _get_grid(shape: tuple[int, int, int, int], device: torch.device) -> torch.Tensor:
        B, H, W, _ = shape
        gx = torch.linspace(0.0, 1.0, H, device=device).reshape(1, H, 1, 1).expand(B, H, W, 1)
        gy = torch.linspace(0.0, 1.0, W, device=device).reshape(1, 1, W, 1).expand(B, H, W, 1)
        return torch.cat([gx, gy], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        if x.size(-1) != self.in_channels:
            raise ValueError(
                f"expected last dim {self.in_channels}, got {x.size(-1)}"
            )

        grid = self._get_grid(x.shape, x.device)
        x = torch.cat([x, grid], dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        for i, (sconv, pconv) in enumerate(zip(self.spectral_convs, self.pointwise_convs)):
            x = sconv(x) + pconv(x)
            if i != self.n_layers - 1:
                x = F.gelu(x)

        if self.padding > 0:
            x = x[..., : -self.padding, : -self.padding]

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1) if self.out_channels == 1 else x


@dataclass(frozen=True)
class FNOConfig:
    """Carries the hyperparameters needed to instantiate one FNO model."""
    name: str
    dim: int
    modes: int | tuple[int, int]
    width: int
    n_layers: int = 4
    padding: int = 0

    def build(self) -> nn.Module:
        if self.dim == 1:
            return FNO1d(modes=self.modes, width=self.width, n_layers=self.n_layers)
        m1, m2 = self.modes if isinstance(self.modes, tuple) else (self.modes, self.modes)
        return FNO2d(
            modes1=m1, modes2=m2, width=self.width,
            n_layers=self.n_layers, padding=self.padding,
        )


ABLATION_CONFIGS: dict[str, FNOConfig] = {
    "nlse_1d":    FNOConfig("nlse_1d",    dim=1, modes=16, width=64, n_layers=4),
    "burgers_1d": FNOConfig("burgers_1d", dim=1, modes=16, width=64, n_layers=4),
    "nlse_2d":    FNOConfig("nlse_2d",    dim=2, modes=(12, 12), width=32, n_layers=4, padding=0),
    "darcy_2d":   FNOConfig("darcy_2d",   dim=2, modes=(12, 12), width=32, n_layers=4, padding=8),
}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
