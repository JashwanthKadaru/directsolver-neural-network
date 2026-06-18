"""Fast Direct Solver Neural Network (1D, shared-kernel Conv1D variant).

Implements the SMW-based recursive factorization of HODLR matrices as a
parameterized neural network. ``K=1`` corresponds to the fully-linear
variant used for integral-equation problems; ``K>=2`` introduces depth-K
nonlinear sub-networks per level.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwisePermute(nn.Module):
    """Swap adjacent half-blocks along the length axis (NCW layout)."""

    def __init__(self, N: int, block_size: int):
        super().__init__()
        self.N = N
        self.block_size = block_size

        half_bs = block_size // 2
        n_blocks = N // half_bs
        block_indices = torch.arange(N).reshape(n_blocks, half_bs)
        swapped = torch.stack(
            [block_indices[1::2], block_indices[0::2]], dim=1
        ).reshape(-1)
        self.register_buffer("swap_idx", swapped, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(dim=2, index=self.swap_idx)


def _glorot_init_(conv: nn.Conv1d) -> None:
    nn.init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


class FDSNet_Linear_1D(nn.Module):
    """Shared-kernel Fast Direct Solver Neural Network for 1D problems.

    Args:
        N: input length (must be divisible by 2**L).
        L: number of SMW levels.
        r: off-diagonal rank.
        K: nonlinearity depth. ``K=1`` yields the fully-linear variant; for
            ``K>=2`` the K_kappa and S sub-networks become depth-K stacks.
        seed: optional RNG seed applied to weight initialization.
    """

    def __init__(self, N: int, L: int, r: int, K: int, seed: int | None = None):
        super().__init__()
        if N % (2 ** L) != 0:
            raise ValueError(f"N ({N}) must be divisible by 2**L ({2 ** L}).")
        if K == 0 or K is None:
            raise ValueError("K must be non-zero.")

        self.N = N
        self.L = L
        self.r = r
        self.K = K
        self.p = r
        self.kappa = L
        self.m = N // (2 ** L)

        p, m, kappa = self.p, self.m, self.kappa

        self.k_kappa_projection_shared = nn.Conv1d(
            in_channels=1, out_channels=m, kernel_size=m, stride=m, padding=0, bias=True
        )
        n_kkappa_extra = 0 if K == 1 else K
        self.k_kappa_projection_shared_nonlinear = nn.ModuleList([
            nn.Conv1d(
                in_channels=m, out_channels=m, kernel_size=1, stride=1, padding=0, bias=True
            )
            for _ in range(n_kkappa_extra)
        ])

        self.permute = nn.ModuleDict()
        self.smw_v = nn.ModuleDict()
        self.smw_s = nn.ModuleDict()
        self.smw_u = nn.ModuleDict()

        for i in range(kappa - 1, -1, -1):
            block_size = N // (2 ** i)
            key = str(i)

            self.permute[key] = PairwisePermute(N=N, block_size=block_size)

            self.smw_v[key] = nn.Conv1d(
                in_channels=1, out_channels=p,
                kernel_size=block_size // 2, stride=block_size // 2,
                padding=0, bias=True,
            )

            s_stack = nn.ModuleList()
            for _ in range(K):
                s_stack.append(nn.Conv1d(
                    in_channels=p, out_channels=2 * p,
                    kernel_size=2, stride=2, padding=0, bias=True,
                ))
            self.smw_s[key] = s_stack

            self.smw_u[key] = nn.Conv1d(
                in_channels=p, out_channels=block_size // 2,
                kernel_size=1, stride=1, padding=0, bias=True,
            )

        self._reset_parameters(seed)

    def _reset_parameters(self, seed: int | None) -> None:
        gen_state = None
        if seed is not None:
            gen_state = torch.random.get_rng_state()
            torch.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                _glorot_init_(m)
        if gen_state is not None:
            torch.random.set_rng_state(gen_state)

    @staticmethod
    def _row_major_reshape(x: torch.Tensor, new_len: int, new_ch: int) -> torch.Tensor:
        B = x.shape[0]
        x = x.permute(0, 2, 1).contiguous()
        x = x.reshape(B, new_len, new_ch)
        return x.permute(0, 2, 1).contiguous()

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        B, N = x_input.shape
        if N != self.N:
            raise ValueError(f"expected length {self.N}, got {N}")

        x = x_input.unsqueeze(1)

        x = self.k_kappa_projection_shared(x)
        for j, conv in enumerate(self.k_kappa_projection_shared_nonlinear):
            x = conv(x)
            if j != self.K - 1:
                x = F.relu(x)

        x = x.permute(0, 2, 1).contiguous().reshape(B, N)

        for i in range(self.kappa - 1, -1, -1):
            key = str(i)
            x_in = x

            x_r = x_in.unsqueeze(1)
            x_r = self.permute[key](x_r)

            v = self.smw_v[key](x_r)

            s = v
            for j, conv in enumerate(self.smw_s[key]):
                s = conv(s)
                if j != self.K - 1:
                    s = F.relu(s)
                s = self._row_major_reshape(s, new_len=2 ** (i + 1), new_ch=self.p)

            u = self.smw_u[key](s)

            u = u.permute(0, 2, 1).contiguous().reshape(B, self.N)

            x = x_in - u

        return x
