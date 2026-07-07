# S2S Precipitation Downscaling GAN

Multi-variant GAN pipeline (SRGAN, Pix2Pix, Hybrid U-Net, STFA-GAN) for downscaling
ECMWF S2S forecasts to IMERG-resolution precipitation over West Africa, with a
config-driven variable system and automated test suite.

A configuration-driven GAN pipeline for statistical downscaling of ECMWF S2S
subseasonal forecasts to high-resolution IMERG precipitation fields. Supports
four interchangeable generator architectures (SRGAN, Pix2Pix, Hybrid U-Net, and
a Spatio-Topographic Feature-Attention GAN) selected via a single CLI flag,
with surface and atmospheric input variables declared in one config file — no
model code changes needed to add new predictors. Built for West African
rainfall applications (Ghana domain).

Verified working on CPU: `pytest tests/ -v` → 13 passed.

**Topics:** `gan` `precipitation-downscaling` `climate-deep-learning` `pytorch`
`s2s-forecasting` `west-africa` `imerg` `super-resolution`

## Structure
```
config.py              # variable + shape declarations (NetCDF mock/prototype)
config_grib.py         # variable + path declarations (real ECMWF S2S GRIB2 data)
train.py                # CLI entrypoint: --variant {1,2,3,4}
models/                 # generators, discriminator, factory router
utils/dataset.py        # unified NetCDF dataloader (prototype)
utils/grib_dataset.py   # unified GRIB dataloader (production, real data)
tests/                  # dataset, architecture (fwd pass), gradient (backward pass) checks
.github/workflows/      # CI: runs the test suite on every push/PR
inspect_grib.py          # standalone script to inspect a GRIB file's variables/dims
```

## Configuration

All user-facing settings live in two config files — no need to touch model
or dataloader code to change what you train on.

### `config.py` — mock/NetCDF prototype settings
Used by the original NetCDF-based test/demo pipeline (`utils/dataset.py`).
```python
SURFACE_VARIABLES = ["t2m", "u10", "v10", "msl"]
ATMOSPHERIC_VARIABLES = ["q", "r", "t", "u"]
STATIC_GEOGRAPHIC_VARIABLES = ["landmask", "topography"]
COARSE_SHAPE = (9, 9)      # input grid resolution
FINE_SHAPE = (128, 128)    # target grid resolution
```

### `config_grib.py` — real ECMWF S2S GRIB2 data settings
Used by the production dataloader (`utils/grib_dataset.py`).

| Setting | Purpose |
|---|---|
| `DATA_ROOT` | Resolved dynamically — set via `DOWNSCALING_DATA_ROOT` env var, or overridden per-call (e.g. `--data_dir`) |
| `FILENAME_TEMPLATES` | Maps each GRIB file group to its filename pattern (`{date}` placeholder) |
| `SURFACE_VARIABLES` | `{variable_name: file_group}` — single-level variables |
| `ATMOSPHERIC_VARIABLES` | `{variable_name: file_group}` — multi-level variables, auto-expanded across `PRESSURE_LEVELS` |
| `PRESSURE_LEVELS` | e.g. `[700, 500]` hPa |
| `STATIC_LAYERS_PATH` / `IMERG_FILENAME_TEMPLATE` | Paths to static context and target precipitation files |

**To add a new predictor variable:** add one line to `SURFACE_VARIABLES` or
`ATMOSPHERIC_VARIABLES` in `config_grib.py`. Channel counts propagate
automatically into the model factory (`models/__init__.py`) — no other file
needs to change.

**Setting your data path:**
```bash
export DOWNSCALING_DATA_ROOT="~/Desktop/home/ws/qu5630/Leo_FAST2Africa"
```

## IMPORTANT: known bug fixed in this version
Older duplicate files (`srgan_v1.py`, `pix2pix_v2.py`, `unet_v3.py`, `stfagan_v4.py`)
had a PixelShuffle channel mismatch (256 instead of 1024/512 pre-shuffle channels),
which crashes with:
```
RuntimeError: expected input[...] to have 64 channels, but got 16 channels instead
```
These files are NOT included here. Only the corrected `variant*_*.py` files are used.
Do not reintroduce the old `_v1`/`_v2`/`_v3`/`_v4`-suffixed files into this folder.

## Setup

Using `uv` (recommended):
```bash
uv sync
uv run pytest tests/ -v
```

Using plain `pip` (e.g. Kaggle notebooks):
```bash
pip install -r requirements.txt
pytest tests/ -v
```

## Run tests
```bash
make test           # full suite
make test-dataset   # dataloader checks only
make test-arch      # forward-pass shape checks only
make test-grad      # gradient-flow checks only
make test-v1 / test-v2 / test-v3 / test-v4   # scoped to one variant
```

## Run training (dry run example)
```bash
uv run train.py --data_dir ./data/ghana_grid --variant 4 --batch_size 4 --epochs 1
```
