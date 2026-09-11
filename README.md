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

Verified working on CPU: `pytest tests/ -v` → 23 passed.

**Topics:** `gan` `precipitation-downscaling` `climate-deep-learning` `pytorch`
`s2s-forecasting` `west-africa` `imerg` `super-resolution`

## Structure
```
configs/                # YAML configs: variables, region, normalisation, dates (extends-based)
train.py                # CLI entrypoint: --config/--val_config, --variant {1,2,3,4}, --dry_run
models/                 # generators, discriminator, factory router
utils/config_loader.py  # YAML loader (`extends` inheritance, `!env` tags)
utils/regions.py        # named region presets (africa / west_africa / ghana)
utils/normalisation.py  # per-variable zscore/log1p transforms
utils/additional_features.py  # computed static context channels (latlon, day_of_year)
utils/folder_dataset.py # unified per-variable-folder NetCDF dataloader (production, real data)
tests/                  # dataset, architecture (fwd pass), gradient (backward pass) checks
.github/workflows/      # CI: runs the test suite on every push/PR
```

## Configuration

All user-facing settings live in YAML configs under `configs/` — no need to
touch model or dataloader code to change what you train on.

### `configs/ghana_template.yaml` — variable + region declarations
The shared base that `configs/ghana_train.yaml` and `configs/ghana_val.yaml`
`extends:`, so training and validation can never drift apart on which
variables are used.

| Setting | Purpose |
|---|---|
| `region` | Named preset (`africa` / `west_africa` / `ghana`) or explicit `{lat_min, lat_max, lon_min, lon_max}` |
| `patch` | `{coarse_shape, fine_shape, coarse_resolution_deg}` — see below |
| `inputs` | List of `{name, path, normalisation, [levels], [scale]}` — single-level or pressure-level (via `levels`) dynamic predictors |
| `target` | The IMERG precipitation variable + its normalisation |
| `additional_data` / `additional_data_paths` | Static high-resolution context (e.g. `landmask`, `topography`), computed (`latlon`, `day_of_year`) or file-based |

### `configs/ghana_train.yaml` / `configs/ghana_val.yaml` / `configs/west_africa_*.yaml`
Each `extends:` its template (`west_africa_template.yaml` itself `extends:
ghana_template.yaml`, overriding only `region`, so both domains always train
on the same variable set) and overrides only `data_root` (via `!env
${DOWNSCALING_DATA_ROOT}`) and `start_date`/`end_date`.

**Expected data layout** (one folder per variable under `data_root`):
```
<data_root>/<path>/[number_<member>/][level_<N>/]<name>_<date>[-<HHMM>].nc
```
Every ensemble member is indexed as a **separate** training sample, not averaged.

**To add a new predictor variable:** add one entry to `inputs:` in
`configs/ghana_template.yaml`. Channel counts propagate automatically —
`train.py` probes a batch off the dataloader and passes the real channel
count into the model factory (`models/__init__.py`) — no other file needs
to change.

**IMPORTANT — the `patch` block:** all four generators need the static/target
grid to resolve to exactly 128×128, and variants 1/3/4 (`SRGANGenerator`,
`HybridGenerator`, `STFAGenerator`) additionally need the dynamic input to
resolve to exactly 9×9 (a hard-coded `PixelShuffle(4)` ×2 + `kernel_size=17`
crop). `coarse_resolution_deg: 2.5` fixes the *physical* size of one coarse
pixel to ERA5's real native grid spacing (the target/IMERG side is 0.1° —
finer than the coarse side, as it must be); `coarse_shape`/`fine_shape` are
the pixel counts the model needs. The region is then **tiled** into as many
non-overlapping patches of that physical size as fit — one patch spans
9 × 2.5° = 22.5° on a side, so `region: ghana` and `region: west_africa`
(21×34°, still smaller than one patch) both currently yield a single 1×1
patch; only a region wider than 22.5° in both directions actually benefits
from tiling (e.g. `region: africa`, 73×72° → 3×3 = 9 patches). See the
comment block at the top of `ghana_template.yaml` and
`utils/folder_dataset.py`'s module docstring for the full mechanics.

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

## Run training
```bash
# Smoke test -- synthetic data, no config or real files needed
uv run train.py --dry_run --variant 4 --batch_size 4 --epochs 1

# Real training, with validation + checkpointing
uv run train.py --config configs/ghana_train.yaml --val_config configs/ghana_val.yaml \
    --variant 4 --batch_size 4 --epochs 50 --warmup_epochs 5
```
