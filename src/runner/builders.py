"""Builds ``(model, adapter, n_params, config)`` for a ``(model_name,
dataset_name)`` pair using the per-model ``ABLATION_CONFIGS`` and the
dataset metadata.

FDSNet is 1D-only; for 2D problems the input vector is reordered via a HODLR
k-d-tree permutation to preserve spatial locality.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "models"))

from fdsnet_model import FDSNet_Linear_1D                   # noqa: E402
from fno_model import ABLATION_CONFIGS as FNO_CFGS                # noqa: E402
from mnnh_model import ABLATION_CONFIGS as MNNH_CFGS              # noqa: E402
from deeponet_model import ABLATION_CONFIGS as DON_CFGS           # noqa: E402
from mlp_model import ABLATION_CONFIGS as MLP_CFGS                # noqa: E402

from runner.adapters import (                                     # noqa: E402
    FnoAdapter, FDSNetAdapter, MnnhAdapter, DeepONetAdapter, MlpAdapter,
)
from runner.kdtree import (                                       # noqa: E402
    get_hodlr_permutation, get_inverse_permutation,
)


FDSNET_CONFIGS: dict[str, dict] = {
    "nlse_1d":    {"N": 320,        "L": 6,  "r": 10, "K": 5},
    "burgers_1d": {"N": 1024,       "L": 7,  "r": 10, "K": 5},
    "nlse_2d":    {"N": 80 * 80,    "L": 8,  "r": 10, "K": 7},
    "darcy_2d":   {"N": 96 * 96,    "L": 10, "r": 9,  "K": 7},
    "fredholm_N1600":  {"N": 1600,  "L": 6, "r": 10, "K": 1},
    "fredholm_N6400":  {"N": 6400,  "L": 8, "r": 10, "K": 1},
    "fredholm_N14400": {"N": 14400, "L": 6, "r": 10, "K": 1},
}


DATASETS: dict[str, dict] = {
    "nlse_1d":    {"dim": 1, "shape": (320,)},
    "burgers_1d": {"dim": 1, "shape": (1024,)},
    "nlse_2d":    {"dim": 2, "shape": (80, 80)},
    "darcy_2d":   {"dim": 2, "shape": (96, 96)},
    "fredholm_N1600":  {"dim": 1, "shape": (1600,)},
    "fredholm_N6400":  {"dim": 1, "shape": (6400,)},
    "fredholm_N14400": {"dim": 1, "shape": (14400,)},
}


def _make_coords(dataset: str, device: torch.device) -> torch.Tensor:
    info = DATASETS[dataset]
    if info["dim"] == 1:
        N, = info["shape"]
        return torch.linspace(0.0, 1.0, N, device=device).unsqueeze(-1)
    H, W = info["shape"]
    gx = torch.linspace(0.0, 1.0, H, device=device)
    gy = torch.linspace(0.0, 1.0, W, device=device)
    grid = torch.stack(torch.meshgrid(gx, gy, indexing="ij"), dim=-1)
    return grid.reshape(H * W, 2)


def build(model_name: str, dataset: str, device: torch.device) -> tuple[torch.nn.Module, int, dict]:
    """Returns ``(adapter, n_params, config_dict)``."""
    info = DATASETS[dataset]

    if model_name == "fdsnet":
        cfg = FDSNET_CONFIGS[dataset]
        N_flat = cfg["N"]
        model = FDSNet_Linear_1D(N=N_flat, L=cfg["L"], r=cfg["r"], K=cfg["K"])
        if info["dim"] == 1:
            adapter = FDSNetAdapter(model, dim=1)
        else:
            H, W = info["shape"]
            assert H == W, f"k-d-tree perm assumes square grid, got {info['shape']}"
            perm_np = get_hodlr_permutation(H)
            inv_np = get_inverse_permutation(perm_np)
            perm = torch.as_tensor(perm_np, dtype=torch.long)
            inv_perm = torch.as_tensor(inv_np, dtype=torch.long)
            adapter = FDSNetAdapter(model, dim=2, shape_2d=info["shape"],
                                  perm=perm, inv_perm=inv_perm)
        return adapter.to(device), sum(p.numel() for p in model.parameters()), cfg

    if model_name == "fno":
        cfg = FNO_CFGS[dataset]
        model = cfg.build().to(device)
        return FnoAdapter(model), sum(p.numel() for p in model.parameters()), cfg.__dict__

    if model_name == "mnnh":
        cfg = MNNH_CFGS[dataset]
        model = cfg.build().to(device)
        return MnnhAdapter(model), sum(p.numel() for p in model.parameters()), cfg.__dict__

    if model_name == "mlp":
        cfg = MLP_CFGS[dataset]
        model = cfg.build().to(device)
        return MlpAdapter(model), sum(p.numel() for p in model.parameters()), cfg.__dict__

    if model_name == "deeponet":
        cfg = DON_CFGS[dataset]
        model = cfg.build().to(device)
        coords = _make_coords(dataset, device)
        if info["dim"] == 1:
            mode = "1d"
        elif cfg.use_cnn_branch:
            mode = "2d_cnn"
        else:
            mode = "2d_mlp"
        adapter = DeepONetAdapter(model, coords=coords, out_shape=info["shape"], branch_mode=mode)
        return adapter, sum(p.numel() for p in model.parameters()), cfg.__dict__

    raise ValueError(f"unknown model: {model_name}")
