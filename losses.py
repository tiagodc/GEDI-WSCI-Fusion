"""Masked losses for sparse targets: only pixels with `target > 0` contribute.

`pred` is (B, P, H, W) and `target` is (B, H, W). GEDI footprints cover ~1% of the pixels in a
chip, so every loss averages over valid target pixels only (NaN targets must be set to 0 first).
"""

import torch

EPS = 1e-8


def _masked_mean(loss, mask):
    return (loss * mask).sum() / mask.sum().clamp(min=EPS)


def gaussian_nll(pred, target):
    """Gaussian negative log-likelihood (constant dropped); pred = (mean, variance). Paper default."""
    mean, var = pred[:, 0], pred[:, 1] + EPS
    return _masked_mean(0.5 * torch.log(var) + 0.5 * (mean - target) ** 2 / var, target > 0)


def laplace_nll(pred, target):
    """Laplace negative log-likelihood; pred = (location, scale b). Heavier tails than Gaussian."""
    loc, scale = pred[:, 0], pred[:, 1] + EPS
    return _masked_mean(torch.log(2 * scale) + torch.abs(loc - target) / scale, target > 0)


def mse(pred, target):
    """Mean squared error on the first output channel (use with `n_params=1`)."""
    return _masked_mean((pred[:, 0] - target) ** 2, target > 0)


def r2(pred, target):
    """Squared Pearson correlation between predicted mean and target (monitoring metric)."""
    mask = target > 0
    return torch.corrcoef(torch.stack([pred[:, 0][mask], target[mask]]))[0, 1] ** 2


LOSSES = {"gaussian_nll": gaussian_nll, "laplace_nll": laplace_nll, "mse": mse}
