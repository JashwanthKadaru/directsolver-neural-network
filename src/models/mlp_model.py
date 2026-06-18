"""Plain feedforward MLP baseline.

Zero-inductive-bias control: input is flattened, passed through ``depth``
hidden layers of width ``width`` with ReLU activations, then projected back
to the output shape. Used to isolate the contribution of architectural
inductive bias by matching FDSNet's parameter budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _kaiming_relu_(linear: nn.Linear) -> None:
    nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu")
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class MLP1d(nn.Module):
    """Plain MLP for 1D problems."""

    def __init__(self, N: int, width: int, depth: int):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.N = N
        layers = [nn.Linear(N, width)]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
        self.hidden = nn.ModuleList(layers)
        self.out = nn.Linear(width, N)
        for lin in self.hidden:
            _kaiming_relu_(lin)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        if v.dim() == 3:
            v = v.squeeze(-1)
        x = v
        for lin in self.hidden:
            x = F.relu(lin(x))
        return self.out(x)


class MLP2d(nn.Module):
    """Plain MLP for 2D problems."""

    def __init__(self, H: int, W: int, width: int, depth: int):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.H = H
        self.W = W
        N = H * W
        layers = [nn.Linear(N, width)]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
        self.hidden = nn.ModuleList(layers)
        self.out = nn.Linear(width, N)
        for lin in self.hidden:
            _kaiming_relu_(lin)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        if v.dim() == 4:
            v = v.squeeze(-1)
        B = v.size(0)
        x = v.reshape(B, self.H * self.W)
        for lin in self.hidden:
            x = F.relu(lin(x))
        return self.out(x).reshape(B, self.H, self.W)


@dataclass(frozen=True)
class MLPConfig:
    name: str
    dim: int
    N: int | tuple[int, int]
    width: int
    depth: int = 3

    def build(self) -> nn.Module:
        if self.dim == 1:
            return MLP1d(N=self.N, width=self.width, depth=self.depth)
        H, W = self.N if isinstance(self.N, tuple) else (self.N, self.N)
        return MLP2d(H=H, W=W, width=self.width, depth=self.depth)


ABLATION_CONFIGS: dict[str, MLPConfig] = {
    "nlse_1d":    MLPConfig("nlse_1d",    dim=1, N=320,      width=35, depth=3),
    "burgers_1d": MLPConfig("burgers_1d", dim=1, N=1024,     width=18, depth=3),
    "nlse_2d":    MLPConfig("nlse_2d",    dim=2, N=(80, 80), width=13, depth=3),
    "darcy_2d":   MLPConfig("darcy_2d",   dim=2, N=(96, 96), width=11, depth=3),
}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
