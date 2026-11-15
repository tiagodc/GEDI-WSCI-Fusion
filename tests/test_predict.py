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
LOC = [700, 4700, 2300, 36, -10.8, -17.8, 38.5]
SCALE = [300, 1500, 900, 5, 2, 2, 1]
BOX_A = [-51.2 + 100 * RES, 1.3 - 300 * RES, -51.2 + 300 * RES, 1.3 - 100 * RES]
BOX_B = [-51.2 + 105 * RES, 1.3 - 311 * RES, -51.2 + 305 * RES, 1.3 - 111 * RES]  # BOX_A moved by 5 x 11 px


def write_stack(path, x):
    with rasterio.open(path, "w", driver="GTiff", width=400, height=400, count=7, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(-51.2, 1.3, RES, RES)) as dst:
        dst.write(x)
    return str(path)


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    """7-band stack on a 400 x 400 px piece of the Sentinel-1 export grid, plausible raw values."""
    rng = np.random.default_rng(0)
    x = np.stack([rng.normal(m, s, (400, 400)) for m, s in zip(LOC, SCALE)]).astype(np.float32)
    x[:, :40, :60] = 0  # a no-data corner
    return write_stack(tmp_path_factory.mktemp("in") / "stack.tif", x)


@pytest.fixture(scope="module")
def sparse(tmp_path_factory):
    """The same grid with valid data only in a 20 x 20 px patch (rows 170-189, columns 180-199)."""
    rng = np.random.default_rng(0)
    x = np.zeros((7, 400, 400), np.float32)
    for k, (m, s) in enumerate(zip(LOC, SCALE)):
        x[k, 170:190, 180:200] = rng.normal(m, s, (20, 20))
    return write_stack(tmp_path_factory.mktemp("in") / "sparse.tif", x)


def run(stack, out, bbox, block=1600):
    args = SimpleNamespace(input=[stack], output=str(out), weights=os.path.join(REPO, wsci_fusion.DEFAULT_WEIGHTS),
                           bbox=bbox, offsets=[8, 12, 16, 20, 24], no_mc=True, no_geo=False, chip=40, block=block,
                           chip_phase=[23, 15], batch_size=256, device="cpu")
    wsci_fusion.predict(args)
    with rasterio.open(out) as src:
        return src.read(), src.transform


def assert_same(a, b):
    """Same no-data mask and the same values up to float rounding: CPU convolutions can differ in the last bit
    with the batch's makeup and the thread count (1.9e-6 seen); a chip grid off by a pixel differs by >= 1e-2."""
    np.testing.assert_array_equal(a == wsci_fusion.NODATA, b == wsci_fusion.NODATA)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-5)


def overlap(a, ta, b, tb):
    dc, dr = round((tb.c - ta.c) / RES), round((ta.f - tb.f) / RES)
    assert (dc, dr) == (5, 11)
    return a[:, dr:, dc:], b[:, : a.shape[1] - dr, : a.shape[2] - dc]


def test_chips_do_not_depend_on_bbox(stack, tmp_path):
    """Two boxes 5 x 11 px apart predict the same values where they overlap (the chip grid is fixed by --chip-phase).

    Before the fix, chips started at each box's own corner, so the same pixel got different chips (and values).
    """
    ov_a, ov_b = overlap(*run(stack, tmp_path / "a.tif", BOX_A), *run(stack, tmp_path / "b.tif", BOX_B))
    assert (ov_a[0] != wsci_fusion.NODATA).mean() > 0.5
    assert_same(ov_a, ov_b)


def test_blocks_do_not_depend_on_bbox(sparse, tmp_path):
    """Sparse data: the same two boxes, with 64 px blocks, predict the same pixels.

    A block is skipped when one MC offset has no valid chip in it. Blocks started at the box's corner, so whether the
    patch's block was skipped depended on --bbox: 664 px of the overlap were valid in one box and not in the other.
    """
    ov_a, ov_b = overlap(*run(sparse, tmp_path / "a.tif", BOX_A, 64), *run(sparse, tmp_path / "b.tif", BOX_B, 64))
    assert (ov_a[0] != wsci_fusion.NODATA).sum() > 1000
    assert_same(ov_a, ov_b)


def test_chips_keep_the_phase_at_the_raster_edge(stack, tmp_path):
    """The full raster and a box well inside it predict the same values where they overlap.

    The full raster's first read starts before the raster's edge (snapped to --chip-phase); it is padded with no-data
    instead of clamped to the edge, which would shift every chip of the first block row and column off the phase.
    """
    a, _ = run(stack, tmp_path / "full.tif", None)
    b, tb = run(stack, tmp_path / "box.tif", BOX_A)
    dc, dr = round((tb.c + 51.2) / RES), round((1.3 - tb.f) / RES)
    assert (dc, dr) == (100, 100)
    assert (b[0] != wsci_fusion.NODATA).mean() > 0.5
    assert_same(a[:, dr : dr + b.shape[1], dc : dc + b.shape[2]], b)


def test_predict_is_strict_fp32_and_restores_tf32(stack, tmp_path, monkeypatch):
    """TF32 is off while predict runs (PyTorch enables it for cuDNN convolutions by default on Ampere and newer GPUs)
    and back to the caller's setting afterwards: the flags are process-wide."""
    seen = []
    inner = wsci_fusion.predict_array

    def spy(*a, **k):
        seen.append((torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32))
        return inner(*a, **k)

    monkeypatch.setattr(wsci_fusion, "predict_array", spy)
    saved = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32  # not plain attributes
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
    try:
        run(stack, tmp_path / "c.tif", BOX_A)
        after = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = saved
    assert seen and all(f == (False, False) for f in seen)
    assert after == (True, True)
