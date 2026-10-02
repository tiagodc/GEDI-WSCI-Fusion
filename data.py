# Copyright (c) 2026 University of Maryland. Released under the MIT License (see LICENSE).
"""Input helpers: cyclic coordinate encoding, training chips and normalization statistics."""

import numpy as np
import torch
from torch.utils.data import Dataset


def geo_layers(lon, lat):
    """Cyclic-encoded coordinates -> (3, H, W) float32: [lat (radians), sin(lon), cos(lon)].

    `lon` / `lat` are pixel-center coordinates in degrees, either 1-D axes (W,) / (H,) of a
    north-up grid or full 2-D (H, W) grids. sin/cos keep longitude continuous across +/-180.
    """
    lon, lat = np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    if lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    return np.stack([lat, np.sin(lon), np.cos(lon)]).astype(np.float32)


def load_chips(paths):
    """Concatenate `.npz` chip files with arrays x (N,C,H,W), y (N,H,W) and optional lon (N,W), lat (N,H)."""
    parts = [dict(np.load(p)) for p in paths]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


class ChipDataset(Dataset):
    """Training chips held in memory; cyclic coordinates are appended on the fly.

    Parameters
    ----------
    chips : dict
        `x` raw predictor layers (N, C, H, W) in the model's input order, WITHOUT coordinate layers;
        `y` target (N, H, W), NaN or <= 0 where there is no observation;
        `lon` (N, W) / `lat` (N, H) pixel-center coordinates, required when `geo=True`.
    buffer : int
        Border cropped from the target to match the model's negative buffer.
    geo : bool
        Append the 3 cyclic coordinate layers after the predictor layers.
    """

    def __init__(self, chips, buffer=4, geo=True):
        self.x, self.y = chips["x"], chips["y"].reshape(len(chips["x"]), *chips["x"].shape[-2:])
        self.lon, self.lat = chips.get("lon"), chips.get("lat")
        self.buffer, self.geo = buffer, geo
        if geo and (self.lon is None or self.lat is None):
            raise ValueError("geo=True needs 'lon' and 'lat' arrays in the chip files")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = np.nan_to_num(self.x[i].astype(np.float32), nan=0)
        if self.geo:
            x = np.concatenate([x, geo_layers(self.lon[i], self.lat[i])])
        y = np.nan_to_num(self.y[i].astype(np.float32), nan=0)
        if self.buffer > 0:
            b = self.buffer
            y = y[b:-b, b:-b]
        return torch.from_numpy(x), torch.from_numpy(y)


def channel_stats(chips, geo=True):
    """Per-channel mean/std of the raw predictors; coordinate layers get (0, 1).

    As in the original workflow, no-data (NaN) pixels count as 0, i.e. the statistics describe
    exactly what the network sees.
    """
    x = np.nan_to_num(chips["x"].astype(np.float32), nan=0)
    mean = x.mean(axis=(0, 2, 3), dtype=np.float64).tolist()
    std = np.maximum(x.std(axis=(0, 2, 3), dtype=np.float64), 1e-6).tolist()
    if geo:
        mean, std = mean + [0.0] * 3, std + [1.0] * 3
    return mean, std
