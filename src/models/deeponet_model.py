"""DeepONet baseline (vanilla unstacked, Cartesian-product mode).

Operator-learning architecture composed of a branch network (ingests
discretized input function values) and a trunk network (ingests query
coordinates). Supports either an MLP branch (1D problems and 2D periodic
problems) or a CNN branch (Darcy flow on a 2D grid). A periodic feature
transform on the trunk input is available for periodic problems.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _glorot_normal_(linear: nn.Linear) -> None:
    nn.init.xavier_normal_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class MLP(nn.Module):
    """Multilayer perceptron with the final layer linear (no activation)."""

    def __init__(self, layers: list[int], activation: str):
        super().__init__()
        if len(layers) < 2:
            raise ValueError("layers must have at least 2 entries")
        self.act = {"tanh": torch.tanh, "relu": F.relu, "gelu": F.gelu}[activation]
        self.linears = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )
        for lin in self.linears:
            _glorot_normal_(lin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i != len(self.linears) - 1:
                x = self.act(x)
        return x


class MLPBranch(nn.Module):
    """Vanilla MLP branch."""

    def __init__(self, layers: list[int], activation: str):
        super().__init__()
        self.net = MLP(layers, activation)
        self.p_out = layers[-1]

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u)


class CNNBranch2d(nn.Module):
    """2D CNN branch for grid-structured inputs."""

    def __init__(self, H: int, W: int, p: int = 128, activation: str = "relu"):
        super().__init__()
        self.act_name = activation
        self.act = {"tanh": torch.tanh, "relu": F.relu, "gelu": F.gelu}[activation]
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=0, bias=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=0, bias=True)
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, H, W)
            flat_dim = self.conv2(self.act(self.conv1(dummy))).reshape(1, -1).shape[1]

        self.fc1 = nn.Linear(flat_dim, 128, bias=True)
        self.fc2 = nn.Linear(128, p, bias=True)
        _glorot_normal_(self.fc1)
        _glorot_normal_(self.fc2)
        self.p_out = p

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        if u.dim() == 3:
            u = u.unsqueeze(1)
        x = self.act(self.conv1(u))
        x = self.act(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = self.act(self.fc1(x))
        return self.fc2(x)


class DeepONet(nn.Module):
    """Vanilla unstacked DeepONet in Cartesian-product mode.

    Args:
        branch: an ``nn.Module`` exposing a ``p_out`` attribute (inner dim).
        trunk_layers: layer sizes for the trunk MLP; last dim must equal
            ``branch.p_out``.
        activation: activation function name for both branch (if MLP) and
            trunk MLPs.
        periodic: if True, apply ``y -> (cos(2 pi y), sin(2 pi y))`` per
            axis before the trunk MLP. ``trunk_layers[0]`` must equal
            ``2 * d_raw`` in that case.
        d_raw: dimensionality of raw query coordinates (1 for 1D, 2 for 2D).
    """

    def __init__(
        self,
        branch: nn.Module,
        trunk_layers: list[int],
        activation: str,
        periodic: bool,
        d_raw: int,
    ):
        super().__init__()
        if not hasattr(branch, "p_out"):
            raise ValueError("branch module must expose `p_out` attribute")
        if trunk_layers[-1] != branch.p_out:
            raise ValueError(
                f"trunk inner dim {trunk_layers[-1]} != branch p_out {branch.p_out}"
            )
        if periodic and trunk_layers[0] != 2 * d_raw:
            raise ValueError(
                f"periodic=True requires trunk_layers[0]={2*d_raw}, got {trunk_layers[0]}"
            )
        if (not periodic) and trunk_layers[0] != d_raw:
            raise ValueError(
                f"periodic=False requires trunk_layers[0]={d_raw}, got {trunk_layers[0]}"
            )

        self.branch = branch
        self.trunk = MLP(trunk_layers, activation)
        self.periodic = periodic
        self.d_raw = d_raw
        self.bias = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _periodic_transform(y: torch.Tensor) -> torch.Tensor:
        two_pi = 2.0 * math.pi
        return torch.cat([torch.cos(two_pi * y), torch.sin(two_pi * y)], dim=-1)

    def forward(self, u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        if coords.dim() != 2 or coords.size(-1) != self.d_raw:
            raise ValueError(
                f"coords must be (Nq, {self.d_raw}); got {tuple(coords.shape)}"
            )
        b = self.branch(u)
        y = self._periodic_transform(coords) if self.periodic else coords
        t = self.trunk(y)
        return torch.einsum("bp,np->bn", b, t) + self.bias


@dataclass(frozen=True)
class DeepONetConfig:
    name: str
    dim: int
    m_or_grid: int | tuple[int, int]
    branch_layers: list[int] | None
    trunk_layers: list[int]
    activation: str
    periodic: bool
    use_cnn_branch: bool = False
    p: int = 128

    def build(self) -> DeepONet:
        if self.use_cnn_branch:
            H, W = (self.m_or_grid if isinstance(self.m_or_grid, tuple)
                    else (self.m_or_grid, self.m_or_grid))
            branch = CNNBranch2d(H=H, W=W, p=self.p, activation=self.activation)
        else:
            assert self.branch_layers is not None
            branch = MLPBranch(self.branch_layers, self.activation)
        d_raw = self.dim
        return DeepONet(
            branch=branch,
            trunk_layers=self.trunk_layers,
            activation=self.activation,
            periodic=self.periodic,
            d_raw=d_raw,
        )


ABLATION_CONFIGS: dict[str, DeepONetConfig] = {
    "nlse_1d": DeepONetConfig(
        name="nlse_1d", dim=1, m_or_grid=320,
        branch_layers=[320, 128, 128, 128, 128],
        trunk_layers=[2, 128, 128, 128],
        activation="tanh", periodic=True,
    ),
    "burgers_1d": DeepONetConfig(
        name="burgers_1d", dim=1, m_or_grid=1024,
        branch_layers=[1024, 128, 128, 128, 128],
        trunk_layers=[2, 128, 128, 128],
        activation="tanh", periodic=True,
    ),
    "nlse_2d": DeepONetConfig(
        name="nlse_2d", dim=2, m_or_grid=(80, 80),
        branch_layers=None,
        trunk_layers=[4, 128, 128, 128, 128],
        activation="tanh", periodic=True,
        use_cnn_branch=True,
    ),
    "darcy_2d": DeepONetConfig(
        name="darcy_2d", dim=2, m_or_grid=(96, 96),
        branch_layers=None,
        trunk_layers=[2, 128, 128, 128, 128],
        activation="relu", periodic=False,
        use_cnn_branch=True,
    ),
}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
