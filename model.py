# Copyright (c) 2026 University of Maryland. Released under the MIT License (see LICENSE).
"""WSCI Fusion CNN: a fully-convolutional EfficientNetV2-style regressor with Monte Carlo dropout.

The network keeps the input resolution throughout (no strides / pooling), predicts a per-pixel
(mean, variance) pair and crops a `buffer` of pixels from every border of its output, so a
40x40 input chip yields a 32x32 prediction (buffer=4).
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MCDropout(nn.Module):
    """Channel-wise (2D) dropout that stays active in eval mode while `monte_carlo=True`.

    Keeping dropout on at inference turns each forward pass into a sample from the model's
    approximate posterior (Gal & Ghahramani 2016); set `monte_carlo=False` for deterministic eval.
    """

    def __init__(self, p=0.2, monte_carlo=True):
        super().__init__()
        self.p = p
        self.monte_carlo = monte_carlo

    def forward(self, x):
        return F.dropout2d(x, p=self.p, training=self.training or self.monte_carlo)


class FixedNormalization(nn.Module):
    """Standardize input channels with fixed (non-trainable) per-channel mean and std."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: reweight channels by a gate computed from their global average."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        return x * self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)


class MBConvBlock(nn.Module):
    """Mobile inverted bottleneck: 1x1 expand -> depthwise kxk -> SE -> 1x1 project.

    Dropout + residual are only applied when input/output channels match and dropout > 0.
    """

    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, dropout=0.2, monte_carlo=True):
        super().__init__()
        exp = int(in_channels * expand_ratio)
        self.dropout_rate = dropout
        self.dropout = MCDropout(dropout, monte_carlo)
        self.expand_conv = nn.Conv2d(in_channels, exp, 1, bias=False)
        self.bn0 = nn.BatchNorm2d(exp)
        self.depthwise_conv = nn.Conv2d(exp, exp, kernel_size, padding=kernel_size // 2, groups=exp, bias=False)
        self.bn1 = nn.BatchNorm2d(exp)
        self.se = SEBlock(exp)
        self.project_conv = nn.Conv2d(exp, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip_connection = in_channels == out_channels

    def forward(self, x):
        identity = x
        x = F.relu6(self.bn0(self.expand_conv(x)))
        x = F.relu6(self.bn1(self.depthwise_conv(x)))
        x = self.se(x)
        x = self.bn2(self.project_conv(x))
        if self.skip_connection and self.dropout_rate > 0:
            x = self.dropout(x) + identity
        return x


class FusedMBConvBlock(nn.Module):
    """Fused MBConv: a single kxk conv replaces expand + depthwise.

    Dropout is applied whenever dropout > 0; the residual only when channels match.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.2, monte_carlo=True):
        super().__init__()
        self.dropout_rate = dropout
        self.dropout = MCDropout(dropout, monte_carlo)
        self.fused_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.skip_connection = in_channels == out_channels

    def forward(self, x):
        identity = x
        x = F.relu6(self.bn(self.fused_conv(x)))
        if self.dropout_rate > 0:
            x = self.dropout(x)
            if self.skip_connection:
                x = x + identity
        return x


class WSCIFusionNet(nn.Module):
    """EfficientNetV2-style fully-convolutional network used for the GEDI L4C WSCI Fusion product.

    Parameters
    ----------
    in_channels : int
        Number of input layers (10 for the released model: 7 SAR/DEM + 3 cyclic coordinates).
    start_filters : int, optional
        Width of the first convolution; block widths are 1x, 2x, 4x and 8x this value.
        Defaults to `in_channels` (the released model uses 10).
    n_params : int
        Output channels per pixel. 2 = (mean, variance), both passed through softplus;
        any other value returns raw outputs (e.g. 1 for a mean-only MSE model).
    dropout : float
        MC dropout rate.
    buffer : int
        Pixels cropped from each output border (negative buffer). 4 -> 40x40 in, 32x32 out.
    mean, std : sequence of float
        Per-channel normalization statistics (length `in_channels`).
    monte_carlo : bool
        Keep dropout active at inference (required for epistemic uncertainty).
    """

    def __init__(self, in_channels=10, start_filters=None, n_params=2, dropout=0.2, buffer=4,
                 mean=None, std=None, monte_carlo=True):
        super().__init__()
        f = start_filters or in_channels
        self.n_params = n_params
        self.buffer = buffer
        self.normalization = FixedNormalization(
            np.zeros(in_channels) if mean is None else mean, np.ones(in_channels) if std is None else std
        )
        self.initial_conv = nn.Conv2d(in_channels, f, 3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(f)
        kw = dict(dropout=dropout, monte_carlo=monte_carlo)
        self.blocks = nn.Sequential(
            FusedMBConvBlock(f, f, 3, **kw),
            FusedMBConvBlock(f, f * 2, 3, **kw),
            MBConvBlock(f * 2, f * 2, 4, 3, **kw),
            MBConvBlock(f * 2, f * 4, 6, 5, **kw),
            MBConvBlock(f * 4, f * 4, 6, 5, **kw),
            MBConvBlock(f * 4, f * 8, 6, 5, **kw),
            MBConvBlock(f * 8, f * 8, 4, 3, **kw),
            MBConvBlock(f * 8, f * 8, 4, 3, **kw),
        )
        self.final_conv = nn.Conv2d(f * 8, f * 4, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(f * 4)
        self.output_conv = nn.Conv2d(f * 4, n_params, 1)

    def forward(self, x):
        x = self.normalization(x)
        x = F.relu6(self.bn0(self.initial_conv(x)))
        x = self.blocks(x)
        x = F.relu6(self.bn1(self.final_conv(x)))
        x = self.output_conv(x)
        if self.n_params == 2:
            x = F.softplus(x)
        if self.buffer > 0:
            b = self.buffer
            x = x[:, :, b:-b, b:-b]
        return x

    def set_monte_carlo(self, enabled):
        for m in self.modules():
            if isinstance(m, MCDropout):
                m.monte_carlo = enabled
        return self


def save_weights(model, path, config, **metadata):
    """Write `config`, `metadata` and the model's state_dict to a plain-text JSON file.

    Floats are written with the shortest repr that round-trips to the same float32 value.
    """
    lines = []
    for k, v in model.state_dict().items():
        v = v.detach().cpu()
        if v.is_floating_point():
            data = ",".join(str(x) for x in v.float().numpy().ravel())
        else:
            data = ",".join(str(int(x)) for x in v.numpy().ravel())
        dtype = str(v.dtype).replace("torch.", "")
        lines.append(f'    "{k}": {{"dtype": "{dtype}", "shape": {list(v.shape)}, "data": [{data}]}}')
    head = json.dumps({"config": config, **metadata}, indent=2)[:-2]
    with open(path + ".tmp", "w") as f:
        f.write(head + ',\n  "state_dict": {\n' + ",\n".join(lines) + "\n  }\n}\n")
    os.replace(path + ".tmp", path)  # never leave a half-written file behind


def load_weights(path):
    """Read a JSON weights file -> (config dict, state_dict, full JSON dict)."""
    with open(path) as f:
        d = json.load(f)
    sd = {k: torch.tensor(v["data"], dtype=getattr(torch, v["dtype"])).reshape(v["shape"])
          for k, v in d["state_dict"].items()}
    return d["config"], sd, d


def load_model(path, monte_carlo=True, device="cpu"):
    """Build a `WSCIFusionNet` from a JSON weights file and return it in eval mode."""
    config, sd, _ = load_weights(path)
    model = WSCIFusionNet(**config, monte_carlo=monte_carlo)
    model.load_state_dict(sd)
    return model.to(device).eval()
