"""`predict` regressions, on small synthetic rasters (CPU, a few seconds): python -m pytest tests"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import wsci_fusion  # noqa: E402

RES = 2.2457882102988038e-04


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    """7-band stack on a 400 x 400 px piece of the Sentinel-1 export grid, plausible raw values."""
    rng = np.random.default_rng(0)
    path = tmp_path_factory.mktemp("in") / "stack.tif"
    loc = [700, 4700, 2300, 36, -10.8, -17.8, 38.5]
    scale = [300, 1500, 900, 5, 2, 2, 1]
    x = np.stack([rng.normal(m, s, (400, 400)) for m, s in zip(loc, scale)]).astype(np.float32)
    x[:, :40, :60] = 0  # a no-data corner
    with rasterio.open(path, "w", driver="GTiff", width=400, height=400, count=7, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(-51.2, 1.3, RES, RES)) as dst:
        dst.write(x)
    return str(path)


def run(stack, out, bbox):
    args = SimpleNamespace(input=[stack], output=str(out), weights=os.path.join(REPO, wsci_fusion.DEFAULT_WEIGHTS),
                           bbox=bbox, offsets=[8, 12, 16, 20, 24], no_mc=True, no_geo=False, chip=40, block=1600,
                           chip_phase=[23, 15], batch_size=256, device="cpu")
    wsci_fusion.predict(args)
    with rasterio.open(out) as src:
        return src.read(), src.transform


def test_chips_do_not_depend_on_bbox(stack, tmp_path):
    """Two boxes 5 x 11 px apart predict the same values where they overlap (the chip grid is fixed by --chip-phase).

    Before the fix, chips started at each box's own corner, so the same pixel got different chips (and values).
    """
    a, ta = run(stack, tmp_path / "a.tif", [-51.2 + 100 * RES, 1.3 - 300 * RES, -51.2 + 300 * RES, 1.3 - 100 * RES])
    b, tb = run(stack, tmp_path / "b.tif", [-51.2 + 105 * RES, 1.3 - 311 * RES, -51.2 + 305 * RES, 1.3 - 111 * RES])
    dc, dr = round((tb.c - ta.c) / RES), round((ta.f - tb.f) / RES)
    assert (dc, dr) == (5, 11)
    ov_a, ov_b = a[:, dr:, dc:], b[:, : a.shape[1] - dr, : a.shape[2] - dc]
    valid = ov_a[0] != wsci_fusion.NODATA
    assert valid.mean() > 0.5
    np.testing.assert_array_equal(ov_a, ov_b)


def test_chips_keep_the_phase_at_the_raster_edge(stack, tmp_path):
    """The full raster and a box well inside it predict the same values where they overlap.

    The full raster's first read starts before the raster's edge (snapped to --chip-phase); it is padded with no-data
    instead of clamped to the edge, which would shift every chip of the first block row and column off the phase.
    """
    a, _ = run(stack, tmp_path / "full.tif", None)
    b, tb = run(stack, tmp_path / "box.tif", [-51.2 + 100 * RES, 1.3 - 300 * RES, -51.2 + 300 * RES, 1.3 - 100 * RES])
    dc, dr = round((tb.c + 51.2) / RES), round((1.3 - tb.f) / RES)
    assert (dc, dr) == (100, 100)
    ov_a = a[:, dr : dr + b.shape[1], dc : dc + b.shape[2]]
    valid = b[0] != wsci_fusion.NODATA
    assert valid.mean() > 0.5
    np.testing.assert_array_equal(ov_a, b)


def test_predict_is_strict_fp32(stack, tmp_path):
    """TF32 is off after predict: PyTorch enables it for cuDNN convolutions by default on Ampere and newer GPUs."""
    torch.backends.cudnn.allow_tf32 = True
    run(stack, tmp_path / "c.tif", None)
    assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
