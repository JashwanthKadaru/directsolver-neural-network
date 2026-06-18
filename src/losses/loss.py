"""Loss and metric functions for relative L2 error.

Both reduce over the last axis of the input and average across the batch.
``relative_mean_squared_error`` is used as the training criterion; the
non-mean-squared variant is used as a logging metric and for selecting the
best epoch.
"""

import torch

_EPS = 1e-7


def relative_l2_non_meansquared_error(
    pred: torch.Tensor, target: torch.Tensor, eps: float = _EPS
) -> torch.Tensor:
    """Mean relative L2 error (no squaring)."""
    assert pred.shape == target.shape, f"shape mismatch: {pred.shape} vs {target.shape}"
    true_norm = torch.sqrt(torch.sum(target ** 2, dim=-1))
    error_norm = torch.sqrt(torch.sum((target - pred) ** 2, dim=-1))
    return torch.mean(error_norm / (true_norm + eps))


def relative_mean_squared_error(
    pred: torch.Tensor, target: torch.Tensor, eps: float = _EPS
) -> torch.Tensor:
    """Mean squared relative L2 error."""
    assert pred.shape == target.shape, f"shape mismatch: {pred.shape} vs {target.shape}"
    true_norm = torch.sqrt(torch.sum(target ** 2, dim=-1))
    error_norm = torch.sqrt(torch.sum((target - pred) ** 2, dim=-1))
    relative_error = error_norm / (true_norm + eps)
    return torch.mean(relative_error ** 2)
