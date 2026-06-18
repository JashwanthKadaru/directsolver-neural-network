"""Per-model forward adapters.

Each adapter accepts inputs in the canonical shape coming from the dataset
(1D: ``(B, N)``; 2D: ``(B, H, W)``) and returns the prediction in the same
shape, so loss and metric computation are uniform across models.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FnoAdapter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FDSNetAdapter(nn.Module):
    """For 2D problems applies a k-d-tree permutation before the FDSNet
    forward so the linearized vector preserves spatial locality (required
    for the off-diagonal low-rank assumption to hold)."""

    def __init__(
        self,
        model: nn.Module,
        dim: int,
        shape_2d: tuple[int, int] | None = None,
        perm: torch.Tensor | None = None,
        inv_perm: torch.Tensor | None = None,
    ):
        super().__init__()
        self.model = model
        self.dim = dim
        self.shape_2d = shape_2d
        if dim == 2:
            assert perm is not None and inv_perm is not None
            self.register_buffer("perm", perm, persistent=False)
            self.register_buffer("inv_perm", inv_perm, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 1:
            return self.model(x)
        B, H, W = x.shape
        flat = x.reshape(B, H * W)
        permuted = flat.index_select(dim=1, index=self.perm)
        y_perm = self.model(permuted)
        y = y_perm.index_select(dim=1, index=self.inv_perm)
        return y.reshape(B, H, W)


class MnnhAdapter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MlpAdapter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DeepONetAdapter(nn.Module):
    """Wraps DeepONet so its branch/trunk Cartesian-product output is
    reshaped to the canonical ``(B, N)`` or ``(B, H, W)`` shape."""

    def __init__(
        self,
        model: nn.Module,
        coords: torch.Tensor,
        out_shape: tuple[int, ...],
        branch_mode: str,
    ):
        super().__init__()
        self.model = model
        self.register_buffer("coords", coords, persistent=False)
        self.out_shape = out_shape
        self.branch_mode = branch_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.branch_mode == "1d":
            branch_in = x
        elif self.branch_mode == "2d_mlp":
            B, H, W = x.shape
            branch_in = x.reshape(B, H * W)
        elif self.branch_mode == "2d_cnn":
            branch_in = x
        else:
            raise ValueError(self.branch_mode)
        y = self.model(branch_in, self.coords)
        B = x.size(0)
        return y.reshape(B, *self.out_shape)
