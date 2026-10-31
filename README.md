# GEDI WSCI Fusion

Model architecture, weights and minimal PyTorch code behind the **GEDI L4C WSCI Fusion** product:
wall-to-wall, quarterly 25 m maps of the Waveform Structural Complexity Index (WSCI) predicted from
Sentinel-1 + ALOS-2 PALSAR-2 SAR and the Copernicus DEM, trained on GEDI L4C WSCI footprints
([de Conto et al. 2025](https://doi.org/10.1088/3049-4753/ae2c01)).

| file | content |
|---|---|
| `model.py` | network (`WSCIFusionNet`), `MCDropout`, `FixedNormalization`, JSON weight I/O |
| `losses.py` | masked Gaussian NLL (default), Laplace NLL, MSE |
| `data.py` | cyclic coordinate encoding, training-chip dataset, normalization stats |
| `wsci_fusion.py` | command line: `predict` (MC dropout + sliding windows) and `train` (scratch / fine-tune) |
| `weights/wsci_fusion_effnet2.json` | released model: config, metadata and all weights as plain-text JSON |
| `tests/test_predict.py` | `predict` regressions on synthetic rasters (`python -m pytest tests`) |

## Installation

```bash
conda env create -f environment.yml   # PyTorch, NumPy, rasterio
conda activate gedi_wsci_fusion
```
Inference runs on the first CUDA GPU when available and falls back to CPU (`--device cpu|cuda:N` to force).

## Input stack

10 channels, **in this order**, on an EPSG:4326 grid with 2.2457882e-4° pixels (~25 m at the equator;
the Sentinel-1 export grid). Values are fed raw: the network normalizes internally with the stored
statistics (`(x - mean) / std`).

| # | layer | source (Google Earth Engine) | values | time step | native res. → 25 m grid | mean | std |
|---|---|---|---|---|---|---|---|
| 1 | elevation | Copernicus DEM GLO-30 (`COPERNICUS/DEM/GLO30`) | m | static | 30 m → nearest | 703.59 | 915.71 |
| 2 | HH | JAXA Global PALSAR-2/PALSAR yearly mosaic v2.5 (`JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH`) | DN (uint16, γ⁰) | calendar year | 25 m, on the grid as exported | 4690.41 | 2527.36 |
| 3 | HV | same | DN (uint16, γ⁰) | calendar year | 25 m, on the grid as exported | 2282.23 | 1451.16 |
| 4 | angle | same, local incidence angle | degrees | calendar year | 25 m, on the grid as exported | 36.56 | 10.54 |
| 5 | VV | Sentinel-1 IW GRD (`COPERNICUS/S1_GRD`), mean of asc. + desc. scenes | dB (σ⁰) | 3 months | 10 m → bilinear (in GEE) | -10.85 | 3.46 |
| 6 | VH | same | dB (σ⁰) | 3 months | 10 m → bilinear (in GEE) | -17.78 | 4.16 |
| 7 | angle | same, ellipsoid incidence angle | degrees | 3 months | 10 m → bilinear (in GEE) | 38.49 | 2.24 |
| 8 | lat | pixel-center latitude | radians | – | computed on the fly | 0 | 1 |
| 9 | sin(lon) | pixel-center longitude | – | – | computed on the fly | 0 | 1 |
| 10 | cos(lon) | pixel-center longitude | – | – | computed on the fly | 0 | 1 |

- **Temporal pairing**: each quarter (Jan–Mar, Apr–Jun, Jul–Sep, Oct–Dec; end date exclusive) uses
  the Sentinel-1 composite of that quarter and the PALSAR-2 mosaic of the same calendar year.
- **Alignment**: the PALSAR-2 exports (EPSG:4326, 25 m, no resampling) have the Sentinel-1 grid's x origin
  and pixel size and are offset from it by a whole number of rows, so they already sit on the Sentinel-1
  grid. The DEM was snapped to the Sentinel-1 grid with nearest-neighbour resampling. One GDAL VRT per
  input over the full Sentinel-1 extent does both
  (`gdalbuildvrt -te <S1 bounds> -tr 2.2457882102988038e-4 2.2457882102988038e-4 -r nearest`). Keep the
  full extent: `--chip-phase` (below) counts columns and rows from the Sentinel-1 grid's corner.
- **No-data** is 0 (NaN and raster no-data values are set to 0 before the network).
- **Coordinates** (layers 8–10) are computed from pixel centers (`data.geo_layers`); `predict` adds them
  automatically, so input rasters only carry layers 1–7.
- **Screening at inference**: a 40×40 window is skipped if any of layers 1–7 is entirely 0/NaN; a
  processing block (1600×1600 px) is skipped if any layer is empty over the whole block.

## Training data (original model)

- **Target**: GEDI L4C WSCI footprints from GEDI **version 2 (v002)** (L4C quality filtering), further
  restricted to 3 m < RH98 < 150 m (`rh_098` 300–15000 cm), WSCI < 15 and ESA WorldCover v200 classes
  other than built-up (50), bare/sparse (60), snow/ice (70) and permanent water (80).
  Footprint WSCI was **averaged per ~25 m cell and quarter** (EASE-Grid 2.0, EPSG:6933) and brought to
  the chip grid by nearest neighbour. Pixels without footprints are NaN (→ 0, ignored by the loss).
- **Period**: 15 quarters, Apr 2019 – Dec 2022; global GEDI coverage (~52° S – 52° N).
- **Chips**: 40×40 px (~1×1 km) windows centered (± 20 px random jitter) on random footprint pixels,
  up to 300 chips per ~80×80 km EASE-Grid tile, ~5 M chips. A chip was kept only if it had
  ≥ 16 valid target pixels (1 %), a non-empty input stack and ≥ 16 non-zero values at target pixels in
  each of layers 1–7.
- **Split**: 80 / 20 training / validation by ~80 km tile (spatial blocks).
- **Chip grid**: chips are square 40×40 pixel windows extracted from a WGS84 (EPSG:4326) grid of ~25 m
  pixels, so their ground size varies slightly with location (geographic projection); the product was
  predicted on the 2.2458e-4° grid above, which is also the recommended grid for new chips.
- **Tooling**: GEDI data were extracted, gridded and spatially organized with the
  [`gedih3`](https://github.com/tiagodc/GEDI-H3) package, aggregating footprints with the
  [EGI indexing system](https://tiagodc.github.io/GEDI-H3/concepts/egi-indexing.html); this is the
  recommended tooling to reproduce the generation of training chips. EGI levels matching the scales above:

  | Scale | Size | EGI level |
  |-------|------|-----------|
  | Pixel | ~25 m | 3 |
  | Chip (40×40 px) | ~1 km | 6 |
  | Spatial block (train/val split tile) | ~80 km | 11 |

## Model

Fully convolutional EfficientNetV2-style network; input resolution is kept throughout (no strides).

```
FixedNormalization -> Conv3x3(10->F) BN ReLU6
-> FusedMBConv(F->F, 3x3) -> FusedMBConv(F->2F, 3x3)
-> MBConv(2F->2F, e4, 3x3) -> MBConv(2F->4F, e6, 5x5) -> MBConv(4F->4F, e6, 5x5)
-> MBConv(4F->8F, e6, 5x5) -> MBConv(8F->8F, e4, 3x3) -> MBConv(8F->8F, e4, 3x3)
-> Conv1x1(8F->4F) BN ReLU6 -> Conv1x1(4F->2) -> softplus -> crop `buffer` px per border
```

- `F = start_filters` = number of input layers by default (**10** in the released model); 365,682
  trainable parameters. MBConv blocks use squeeze-and-excitation (reduction 4).
- **MC dropout**: channel-wise dropout (p = 0.2) after every Fused-MBConv block and on the residual
  branch of MBConv blocks with matching in/out widths; active at train *and* inference time
  (`monte_carlo=True`).
- **Output**: per pixel (mean, variance) of WSCI, both softplus-positive.
- **Negative buffer**: the outer `buffer = 4` px of each output are dropped (edge artifacts), so a
  40×40 chip yields its central 32×32 prediction; the loss is computed on that center only, and
  `predict` tiles windows with stride `chip − 2·buffer`, reading `buffer` from the weights file. The
  network is fully convolutional, so other chip sizes work; the output is always `chip − 2·buffer`.

## Training hyperparameters (defaults of `train`)

| parameter | value |
|---|---|
| loss | masked Gaussian NLL, ½ log σ² + (y − μ)² / 2σ², over pixels with target > 0 |
| optimizer | Adam, lr 1e-3, no weight decay |
| schedule | lr × 0.1 at 10 %, 20 % and 50 % of the epochs |
| epochs / batch | 50 / 96 |
| dropout / buffer / chip | 0.2 / 4 px / 40×40 px |

Released model validation NLL: −0.321 (all validation chips, MC dropout on); RMSE 0.50 WSCI on the
independent test set (paper).

## Inference

`predict` reproduces the product algorithm: the image is processed in 1600×1600 px blocks (+40 px
context); within a block, 40×40 windows are tiled with stride 32 (outputs abut), and the grid is shifted
by **5 diagonal offsets (8, 12, 16, 20, 24 px)**. With MC dropout active each offset is also an
independent posterior sample. Per pixel:

| band | value |
|---|---|
| 1 | mean WSCI = average of the 5 predicted means |
| 2 | aleatoric std = √(average of the 5 predicted variances) |
| 3 | epistemic std = std (ddof 0) of the 5 predicted means |
| 4 | fraction of the 10 input layers that are non-zero |

Total uncertainty = √(aleatoric² + epistemic²). The outer 40 px of the processed area are left empty
(not covered by every pass).

**Chip grid.** Where the chips fall matters: moving the 32 px chip grid by one pixel changes mean WSCI
systematically by about 0.01. The product's block reads started on Sentinel-1-grid columns ≡ 23 and rows
≡ 15 (mod 32) south of 60.0° N, and on rows ≡ 24 north of it (from the origins of its prediction cubes).
`predict` starts every block read on `--chip-phase` (default `23 15`; this adds up to 31 px of context,
padded with no-data before the raster's edge), so the chips do not depend on `--bbox` and match the
product's when the inputs cover the full Sentinel-1 grid. Use `--chip-phase 23 24` north of 60.0° N.

**Numerics.** Inference is strict fp32. `predict` turns off TF32, which PyTorch enables by default for
cuDNN convolutions on Ampere and newer GPUs and which moves outputs by up to 2e-3 WSCI.

The published product differs from a plain `predict` run in four ways:
- **Pixel mask**: permanent water and no-data are masked with ESA WorldCover v200 (class 80 / 0,
  mode-resampled), as are pixels with mean WSCI ≤ 0.
- **Blocks**: only blocks that intersect land (world administrative boundaries) were predicted. North of
  60° N, and south of it in some quarters (most Q3/Q4 of 2015–2020), a block also needed ESA WorldCover
  vegetation somewhere inside it.
- **Grid**: the 1°×1° Cloud-Optimized GeoTIFFs are on a release grid that is not the Sentinel-1 grid:
  origin (−177.69170364307112, 75.113759062371685), pixel 2.2457866777844e-4° (6.8e-7 smaller in relative
  terms, up to 1.1 px of drift at the east edge). Each release pixel holds the value of the nearest
  Sentinel-1-grid pixel.
- **Band names**: band 2 of the 2015–2022 files is described as `Alleatoric Standard Deviation` (sic).

## Usage

**Predict** (inputs must share one grid; bands are concatenated in the given order):
```bash
python wsci_fusion.py predict -i dem.vrt palsar_2022.vrt s1_2022-07-01-to-2022-10-01.vrt -o wsci_2022Q3.tif \
    [--bbox W S E N] [--chip-phase 23 15] [--offsets 8 12 16 20 24] [--no-mc] [--batch-size 256] [--device auto]
```

**Train / fine-tune** on chips stored as `.npz` files with arrays
`x` (N, 7, 40, 40) raw layers 1–7, `y` (N, 40, 40) target (NaN or ≤ 0 = no data),
`lon` (N, 40) and `lat` (N, 40) pixel-center coordinates (coordinate layers are built on the fly):
```bash
# replicate the study (defaults = original hyperparameters)
python wsci_fusion.py train --train train/*.npz --val val/*.npz -o weights/my_model.json
# fine-tune the released model (all weights), or only its output layer
python wsci_fusion.py train --train ... -w weights/wsci_fusion_effnet2.json [--freeze] --epochs 10 --lr 1e-4 -o ft.json
# new target variable / scale: mean-only head with MSE, or a Laplace likelihood
python wsci_fusion.py train --train ... -w weights/wsci_fusion_effnet2.json --loss mse -o rh98.json
```
Any data pipeline works as long as the stack has the same layers in the same order; `--val` should be
spatially separate from `--train` (otherwise a random `--val-frac` of chips is used).
When fine-tuning, every pretrained tensor whose shape still matches is reused, including the stored
normalization (pass `--new-stats` to normalize with statistics of your chips, e.g. for different layers);
from scratch, statistics always come from the training chips. The learning rate drops ×0.1 at 10, 20 and
50 % of `--epochs`. Targets must be positive (the loss mask is `target > 0`); shift them if needed.

**Architecture only** (other inputs, layer counts or chip sizes):
```python
import torch
from model import WSCIFusionNet, load_model

# 12 layers (e.g. 9 predictors + 3 coordinates); pass your per-layer mean/std for normalization
net = WSCIFusionNet(in_channels=12, start_filters=12, n_params=2, dropout=0.2, buffer=4,
                    mean=[0.0] * 12, std=[1.0] * 12)
out = net(torch.randn(8, 12, 64, 64))                      # -> (8, 2, 56, 56): mean, variance

model = load_model("weights/wsci_fusion_effnet2.json")  # released model, MC dropout on
model.set_monte_carlo(False)                              # deterministic predictions
```
`load_weights()` in `model.py` returns the config and `state_dict` of any JSON weights file;
`save_weights()` writes one.

## Reproducibility check

`predict` with these weights (MC dropout on, GPU) over −51.2° to −49.8° E, −0.2° to 1.3° N, 2022 Q3. It
was sampled at the 28.8 M valid pixels of the published 1° tiles inside that box (nearest Sentinel-1
pixel) and compared with the published product. The run-to-run columns compare two independent runs of
`predict`, which is the MC-dropout noise floor:

| band | bias | RMSE | r | run-to-run RMSE | run-to-run r |
|---|---|---|---|---|---|
| mean WSCI | 0.0008 | 0.0617 | 0.998 | 0.0610 | 0.998 |
| aleatoric std | −0.0002 | 0.0238 | 0.990 | 0.0230 | 0.991 |
| epistemic std | −0.0005 | 0.0526 | 0.829 | 0.0526 | 0.830 |

The systematic part of the difference to the product, δ = √(RMSE² − run-to-run RMSE²), is 0.009 (mean),
0.006 (aleatoric) and 0.000 (epistemic). The input rasters were exported again after the product was
made. Before `--chip-phase`, chips started at the box's own corner, and δ was 0.103 for the mean. With MC
dropout off, CPU and GPU agree to 5e-6.

## References

de Conto, T., Armston, J., Dubayah, R. (2025). Scalable deep fusion of spaceborne lidar and synthetic
aperture radar for global forest structural complexity mapping. *Machine Learning: Earth* 2, 015002.
https://doi.org/10.1088/3049-4753/ae2c01

de Conto, T., Armston, J., Dubayah, R. (2024). Characterizing the structural complexity of the Earth's
forests with spaceborne lidar. *Nature Communications* 15, 8116. https://doi.org/10.1038/s41467-024-52468-2

GEDI L4C Footprint Level Waveform Structural Complexity Index, Version 2. ORNL DAAC.
https://doi.org/10.3334/ORNLDAAC/2338

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 University of Maryland.
This applies to the code and the model weights in this repository.
