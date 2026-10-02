# Copyright (c) 2026 University of Maryland. Released under the MIT License (see LICENSE).
"""Predict or train the GEDI L4C WSCI Fusion model.

    python wsci_fusion.py predict -i stack.tif -o wsci.tif
    python wsci_fusion.py train --train chips/*.npz --val val/*.npz -o weights/my_model.json

Run `python wsci_fusion.py {predict,train} -h` for all options.
"""

import argparse
import os
import warnings

import numpy as np
import rasterio
import torch
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window, from_bounds
from torch.utils.data import DataLoader, Subset

from data import ChipDataset, channel_stats, geo_layers, load_chips
from losses import LOSSES, mse, r2
from model import WSCIFusionNet, load_model, load_weights, save_weights

DEFAULT_WEIGHTS = "weights/wsci_fusion_effnet2.json"
BAND_NAMES = ("Mean WSCI estimate", "Aleatoric Standard Deviation", "Epistemic Standard Deviation",
              "Fraction of valid predictors at inference")
NODATA = -9999.0


def get_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


# --------------------------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def predict_array(model, x, offsets=(8, 12, 16, 20, 24), chip=40, batch_size=256, check_bands=None, device="cpu"):
    """Sliding-window MC-dropout ensemble over one image array.

    The image is cut into `chip` x `chip` windows on a grid with stride `chip - 2 * buffer`, so
    the cropped model outputs tile the image seamlessly. The grid is shifted by each offset in
    `offsets` (diagonally, in pixels); with MC dropout active every pass is also a different
    posterior sample. Per pixel, over passes:

    - mean           = average of predicted means
    - aleatoric std  = sqrt(average of predicted variances)
    - epistemic std  = std of predicted means (ddof=0)

    Parameters
    ----------
    x : ndarray (C, H, W)
        Full input stack (including coordinate layers), raw (un-normalized) values.
    check_bands : sequence of int, optional
        Windows where any of these bands is entirely zero/NaN are skipped (no-data screening).

    Returns
    -------
    ndarray (4, H, W) [mean, aleatoric std, epistemic std, valid-input fraction] for 2-parameter
    models, (3, H, W) without the aleatoric band otherwise; NaN where nothing was predicted.
    """
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0)
    _, h, w = x.shape
    b = model.buffer
    step = chip - 2 * b
    n_out = min(model.n_params, 2)
    passes = np.full((len(offsets), n_out, h, w), np.nan, dtype=np.float32)
    xt = torch.from_numpy(x)
    for k, o in enumerate(offsets):
        ij = [(i, j) for i in range(o, h - chip + 1, step) for j in range(o, w - chip + 1, step)]
        if check_bands is not None:
            ij = [(i, j) for i, j in ij if x[check_bands, i:i + chip, j:j + chip].sum(axis=(1, 2)).all()]
        if not ij:  # as in the original pipeline, all passes are required
            return None
        for s in range(0, len(ij), batch_size):
            batch = torch.stack([xt[:, i:i + chip, j:j + chip] for i, j in ij[s:s + batch_size]]).to(device)
            out = model(batch).float().cpu().numpy()
            for (i, j), p in zip(ij[s:s + batch_size], out):
                passes[k, :, i + b:i + b + step, j + b:j + b + step] = p[:n_out]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels
        bands = [np.nanmean(passes[:, 0], axis=0)]
        if n_out == 2:
            bands.append(np.sqrt(np.nanmean(passes[:, 1], axis=0)))
        bands.append(np.nanstd(passes[:, 0], axis=0))
    valid = (np.abs(x) > 1e-5).mean(axis=0)
    bands.append(np.where(np.isnan(bands[0]), np.nan, valid))
    return np.stack(bands).astype(np.float32)


def pixel_lonlat(src, window):
    """Pixel-center longitude/latitude (degrees) of a raster window: 1-D axes or 2-D grids."""
    t = src.window_transform(window)
    cols, rows = np.arange(window.width) + 0.5, np.arange(window.height) + 0.5
    if src.crs.is_geographic and t.b == 0 and t.d == 0:
        return t.c + cols * t.a, t.f + rows * t.e
    cc, rr = np.meshgrid(cols, rows)
    xs, ys = t * (cc.ravel(), rr.ravel())
    lon, lat = warp_transform(src.crs, "EPSG:4326", xs, ys)
    return np.reshape(lon, cc.shape), np.reshape(lat, cc.shape)


def predict(args):
    device = get_device(args.device)
    model = load_model(args.weights, monte_carlo=not args.no_mc, device=device)
    srcs = [rasterio.open(p) for p in args.input]
    ref = srcs[0]
    for s in srcs[1:]:
        if (s.width, s.height, s.transform) != (ref.width, ref.height, ref.transform):
            raise ValueError(f"{s.name} is not on the same grid as {ref.name}")
    n_bands = sum(s.count for s in srcs)
    geo = not args.no_geo
    if n_bands + 3 * geo != model.normalization.mean.shape[1]:
        raise ValueError(f"model expects {model.normalization.mean.shape[1]} channels, got {n_bands} bands"
                         + (" + 3 coordinate layers" if geo else ""))

    H, W = ref.height, ref.width
    full = Window(0, 0, W, H)
    if args.bbox:
        try:
            full = from_bounds(*args.bbox, transform=ref.transform).round_offsets().round_lengths().intersection(full)
        except rasterio.errors.WindowError:
            raise ValueError(f"--bbox {args.bbox} does not overlap {ref.name}") from None
    check = list(range(n_bands))  # all predictor layers, as in the original pipeline
    halo, blk = args.chip, args.block
    R0, C0 = full.row_off, full.col_off

    profile = dict(driver="GTiff", width=full.width, height=full.height, count=3 + (model.n_params == 2),
                   dtype="float32", crs=ref.crs, transform=ref.window_transform(full), nodata=NODATA,
                   tiled=True, blockxsize=256, blockysize=256, compress="deflate", BIGTIFF="IF_SAFER")
    print(f"## device={device} mc_dropout={not args.no_mc} offsets={args.offsets} window={full}")
    with rasterio.open(args.output, "w", **profile) as dst:
        names = BAND_NAMES if model.n_params == 2 else BAND_NAMES[:1] + BAND_NAMES[2:]
        for i, name in enumerate(names, 1):
            dst.set_band_description(i, name)
        blocks = [(r, c) for r in range(R0, R0 + full.height, blk) for c in range(C0, C0 + full.width, blk)]
        for n, (r, c) in enumerate(blocks, 1):
            rb, cb = min(r + blk, R0 + full.height), min(c + blk, C0 + full.width)  # block core end
            # read the block plus `halo` px of context on every side (clipped to the raster)
            r0, c0, r1, c1 = max(0, r - halo), max(0, c - halo), min(H, rb + halo), min(W, cb + halo)
            win = Window(c0, r0, c1 - c0, r1 - r0)
            x = np.concatenate([s.read(window=win, masked=True).astype(np.float32).filled(0) for s in srcs])
            out = None
            if np.nan_to_num(x).any(axis=(1, 2)).all():  # skip blocks with an empty layer
                if geo:
                    x = np.concatenate([x, geo_layers(*pixel_lonlat(ref, win))])
                out = predict_array(model, x, args.offsets, args.chip, args.batch_size, check, device)
            if out is None:
                continue
            # keep the block core, minus the raster's outer `halo` frame (not covered by every pass)
            rr0, cc0, rr1, cc1 = max(r, halo), max(c, halo), min(rb, H - halo), min(cb, W - halo)
            if rr1 <= rr0 or cc1 <= cc0:
                continue
            core = out[:, rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0]
            dst.write(np.nan_to_num(core, nan=NODATA), window=Window(cc0 - C0, rr0 - R0, cc1 - cc0, rr1 - rr0))
            print(f".. block {n}/{len(blocks)}", flush=True)
    print(f"## wrote {args.output}")


# --------------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------------
def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """One pass over `loader`; trains if an optimizer is given. Returns mean (loss, rmse, r2)."""
    model.train(optimizer is not None)
    totals = np.zeros(3)
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            totals += [loss.item(), mse(pred, y).item(), r2(pred, y).item()]
    loss, ms, rsq = totals / max(len(loader), 1)
    return loss, ms ** 0.5, rsq


def train(args):
    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    device = get_device(args.device)
    geo = not args.no_geo
    chips = load_chips(args.train)
    in_channels = chips["x"].shape[1] + 3 * geo
    n_params = 1 if args.loss == "mse" else 2

    pretrained = load_weights(args.weights) if args.weights else None
    start_filters = args.start_filters or (pretrained[0]["start_filters"] if pretrained else in_channels)
    mean, std = channel_stats(chips, geo)
    model = WSCIFusionNet(in_channels, start_filters, n_params, args.dropout, args.buffer, mean, std)
    if pretrained:
        # transfer every tensor whose shape matches; the pretrained normalization is kept unless --new-stats
        own = model.state_dict()
        keep = {k: v for k, v in pretrained[1].items() if k in own and own[k].shape == v.shape
                and not (args.new_stats and k.startswith("normalization."))}
        model.load_state_dict(keep, strict=False)
        print(f"## loaded {len(keep)}/{len(own)} tensors from {args.weights}")
    model.to(device)

    data = ChipDataset(chips, args.buffer, geo)
    if args.val:
        val_data = ChipDataset(load_chips(args.val), args.buffer, geo)
    else:  # random chip split; prefer spatially separate validation files (--val)
        idx = torch.randperm(len(data)).tolist()
        n_val = int(len(idx) * args.val_frac)
        if n_val < 1:
            raise ValueError("empty validation set: pass --val or a larger --val-frac")
        data, val_data = Subset(data, idx[n_val:]), Subset(data, idx[:n_val])
    kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(data, shuffle=True, **kw)
    val_loader = DataLoader(val_data, shuffle=True, **kw)  # shuffled: per-batch r2 spans regions

    if args.freeze:  # train the output layer only
        for name, prm in model.named_parameters():
            prm.requires_grad_(name.startswith("output_conv."))
    optimizer = torch.optim.Adam([prm for prm in model.parameters() if prm.requires_grad], lr=args.lr)
    milestones = sorted({max(1, int(args.epochs * f)) for f in (0.1, 0.2, 0.5)})
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    loss_fn = LOSSES[args.loss]
    config = dict(in_channels=in_channels, start_filters=start_filters, n_params=n_params,
                  dropout=args.dropout, buffer=args.buffer)
    print(f"## {len(data)} training / {len(val_data)} validation chips on {device}; config={config}")

    best, best_epoch = float("inf"), -1
    for epoch in range(args.epochs):
        tr = run_epoch(model, train_loader, loss_fn, device, optimizer)
        va = run_epoch(model, val_loader, loss_fn, device)  # MC dropout stays on, as in the original
        scheduler.step()
        print(f"epoch {epoch + 1:3d}/{args.epochs} lr={scheduler.get_last_lr()[0]:.1e} "
              f"train loss={tr[0]:.4f} rmse={tr[1]:.4f} r2={tr[2]:.3f} | "
              f"val loss={va[0]:.4f} rmse={va[1]:.4f} r2={va[2]:.3f}", flush=True)
        if va[0] < best:
            best, best_epoch = va[0], epoch
            hparams = {k: v for k, v in vars(args).items() if k not in ("func", "train", "val")}
            save_weights(model, args.output, config, training=dict(
                hparams, best_epoch=epoch, val_loss=va[0], val_rmse=va[1], val_r2=va[2]))
        elif epoch - best_epoch > args.patience:
            print("## early stopping")
            break
    if best_epoch < 0:
        raise RuntimeError("validation loss was never finite; nothing saved")
    print(f"## best val loss {best:.4f} at epoch {best_epoch + 1} -> {args.output}")


# --------------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    pp = sub.add_parser("predict", help="map WSCI (+ uncertainty) over a raster stack")
    pp.add_argument("-i", "--input", nargs="+", required=True,
                    help="co-registered raster(s) whose bands, in order, form the input stack (7 bands for "
                         "the released model; coordinate layers are added automatically)")
    pp.add_argument("-o", "--output", required=True, help="output GeoTIFF")
    pp.add_argument("-w", "--weights", default=DEFAULT_WEIGHTS, help="JSON weights file")
    pp.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                    help="subset in the raster's CRS")
    pp.add_argument("--offsets", nargs="+", type=int, default=[8, 12, 16, 20, 24],
                    help="sliding-window offsets in pixels; one MC pass per offset")
    pp.add_argument("--no-mc", action="store_true", help="disable MC dropout (deterministic passes)")
    pp.add_argument("--no-geo", action="store_true", help="do not append cyclic coordinate layers")
    pp.add_argument("--chip", type=int, default=40, help="model input window size")
    pp.add_argument("--block", type=int, default=1600, help="pixels per processing block side")
    pp.add_argument("--batch-size", type=int, default=256)
    pp.add_argument("--device", default="auto", help="auto | cuda | cuda:N | cpu")
    pp.set_defaults(func=predict)

    pt = sub.add_parser("train", help="train from scratch or fine-tune on .npz chips")
    pt.add_argument("--train", nargs="+", required=True, help=".npz chip files (x, y, lon, lat)")
    pt.add_argument("--val", nargs="+", help=".npz validation chips (default: random --val-frac split)")
    pt.add_argument("--val-frac", type=float, default=0.2)
    pt.add_argument("-o", "--output", required=True, help="output JSON weights (best validation loss)")
    pt.add_argument("-w", "--weights", help="pretrained JSON weights to fine-tune (tensors with matching shapes)")
    pt.add_argument("--freeze", action="store_true", help="train only the output layer")
    pt.add_argument("--new-stats", action="store_true",
                    help="normalize with statistics of the training chips instead of the pretrained ones")
    pt.add_argument("--loss", default="gaussian_nll", choices=sorted(LOSSES))
    pt.add_argument("--epochs", type=int, default=50)
    pt.add_argument("--batch-size", type=int, default=96)
    pt.add_argument("--lr", type=float, default=1e-3, help="initial learning rate (x0.1 at 10%%, 20%%, 50%% of epochs)")
    pt.add_argument("--dropout", type=float, default=0.2)
    pt.add_argument("--buffer", type=int, default=4, help="output pixels cropped per border")
    pt.add_argument("--start-filters", type=int,
                    help="first conv width (default: pretrained value, else number of input channels)")
    pt.add_argument("--patience", type=int, default=10, help="early-stopping epochs without improvement")
    pt.add_argument("--no-geo", action="store_true", help="do not append cyclic coordinate layers")
    pt.add_argument("--workers", type=int, default=4)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--device", default="auto")
    pt.set_defaults(func=train)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
