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
| `additional_data` / `additional_data_paths` | Static high-resolution context, computed (`latlon`, `day_of_year`) or file-based (`landmask`, `elevation`) |

Verified against a real archive at `data_root=~/era5_daily_cache`: ERA5
`cape`/`tp`/`u_850`/`v_850`/`msl`/`tcw`/`w_{200,700,850}` (daily, 0.25°
native, 2019-01-01 through 2021-12-31, no gaps) and IMERG `precipitation`
(daily, 0.1° native, same period). `tcw`/`w_*` were built from raw hourly
ERA5 (source: an external `era5_data` archive) via
`scripts/build_daily_cache.py`, daily-averaged to match the rest of the
archive's convention. No ensemble members. Static context: a real global
(90N-90S, 180W-180E, 0.1° native) `landmask`/`elevation` pair at
`data_root/static/` (covers any region, so no out-of-bounds risk) — both
`minmax`-normalized (see `utils/normalisation.py`) to land in the same
[-1, 1] range as `latlon`/`day_of_year`, using each file's TRUE GLOBAL
min/max (not stats from any one resampled region — linear interpolation
can't produce a value outside the source file's real range, so fitting on
the global range guarantees every possible patch stays within [-1, 1], not
just the ones already tried). Unlike `inputs`/`target`,
`additional_data_paths` entries are NOT normalized unless you add a
`normalisation` block yourself (the computed `latlon`/`day_of_year`
features are self-normalizing to [-1, 1] by construction, so they don't
need one). All 8 dynamic variables
and all 4 static
context channels are wired in by default.

### `configs/ghana_train.yaml` / `configs/ghana_val.yaml` / `configs/west_africa_*.yaml` / `configs/africa_*.yaml`
Each `extends:` its template (`west_africa_template.yaml` and
`africa_template.yaml` both `extend: ghana_template.yaml`, overriding only
`region`, so every domain always trains on the same variable set) and
overrides only `data_root` (via `!env ${DOWNSCALING_DATA_ROOT}`) and
`start_date`/`end_date`.

**Expected data layout** (one folder per variable under `data_root`):
```
<data_root>/<path>/[number_<member>/][level_<N>/]<name>_<date>[-<HHMM>].nc
```
Every ensemble member is indexed as a **separate** training sample, not
averaged. Dimension names (`latitude`/`longitude` vs. `lat`/`lon`) and a
leftover singleton `time` axis are normalized automatically per file (see
`utils/folder_dataset.py`'s `_standardize_dims`) — real ERA5 and IMERG files
don't actually agree with each other on either convention.

**To add a new predictor variable:** add one entry to `inputs:` in
`configs/ghana_template.yaml`. Channel counts propagate automatically —
`train.py` probes a batch off the dataloader and passes the real channel
count into the model factory (`models/__init__.py`) — no other file needs
to change.

**IMPORTANT — the `patch` block:** all four generators need the static/target
grid to resolve to exactly 128×128, and variants 1/3/4 (`SRGANGenerator`,
`HybridGenerator`, `STFAGenerator`) additionally need the dynamic input to
resolve to exactly 9×9 (a hard-coded `PixelShuffle(4)` ×2 + `kernel_size=17`
crop). `coarse_resolution_deg: 1.5` is a **deliberate coarsening** of ERA5's
native 0.25° — not simply "the real resolution" — chosen so the implied fine
side (one patch = 9 × 1.5 = 13.5°, over 128 fixed pixels → 0.106°/pixel)
lands just at IMERG's real 0.1° detail limit, instead of fabricating detail
IMERG never measured (which is what using ERA5's native 0.25° directly would
do — 2.25°/127 ≈ 0.018°/pixel, ~6× finer than IMERG really is). It also
makes the coarse input a more realistic stand-in for an actual coarse S2S
forecast grid than raw ERA5 reanalysis resolution would be. See the full
reasoning in the comment block at the top of `ghana_template.yaml`.

The region is then **tiled** into as many non-overlapping patches of that
13.5°×13.5° physical size as fit: `ghana` (7×5°) yields a single 1×1 patch,
`west_africa` (21×34°) yields 1×2 = 2, and `africa` (73×72°) yields 5×5 = 25
— see `utils/folder_dataset.py`'s module docstring for the full mechanics.

**Setting your data path:**
```bash
export DOWNSCALING_DATA_ROOT=~/era5_daily_cache
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

## Model design notes

**No output activation on any generator.** The target fed to the pixel loss
is the `log1p` normalization from `target:` in `configs/ghana_template.yaml`
— which, despite the name, is a *z-scored* log1p value
(`(log1p(x) - log1p_mean) / log1p_std`), not a plain log1p. That's negative
for any below-average log-precip pixel, including every dry pixel (`log1p(0)`
alone z-scores to about -0.78) — so on a semi-arid domain, most target pixels
are negative. Forcing a non-negative output activation (`ReLU`, and
`Softplus` doesn't fix it either — both were tried) makes most of that range
unreachable, and the generator collapses to a near-constant output within a
couple of epochs, freezing the training loss. Physical (mm) non-negativity
of the final prediction is already guaranteed by the `expm1` inverse
transform at denormalization time, so it doesn't need to be enforced a
second time in normalized training space.

**Checkpoint selection metric.** `validate()` in `train.py` denormalizes both
prediction and target back to physical mm before computing MAE (rather than
comparing raw z-scored log1p values), so `Val_MAE_mm` and the "best"
checkpoint are judged in a physically meaningful unit.

**`--save_best_only`.** Skips the periodic `--checkpoint_every` snapshots and
only writes `variant{N}_best.pt` when validation MAE improves. Requires
`--val_config` (otherwise nothing would ever be saved). Default behavior
(periodic + best) is unchanged if you don't pass it.

**`--plot_every N`** (default 20, `0` disables). Every `N` epochs, reloads
the *current best* checkpoint (not necessarily this epoch's live weights)
and predicts on a fixed validation sample — always the same one (val loader
index 0, deterministic since validation isn't shuffled) so you can visually
track how the same location/date improves across training, rather than
comparing against a different random patch each time. Saves a 1x3
prediction/target/difference plot (all denormalized to mm) to
`<checkpoint_dir>/plots/`. If the best checkpoint hasn't changed since the
last plot (validation hasn't improved in that interval), it skips instead of
re-rendering an identical plot.

**Console output.** Each epoch shows a live `tqdm` progress bar (with running
`Loss_D`/`Loss_G_adv`/`Loss_G_pixel` in the postfix) over the training
batches, then prints the epoch's summary line once it finishes. A handful of
benign `DeprecationWarning`/`RuntimeWarning` messages that xarray's netCDF4
backend fires on (almost) every file read are filtered out at the top of
`train.py` so they don't drown out the log.

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

## GPU quickstart: West Africa, all 4 variants

`train.py` already picks up CUDA automatically
(`torch.device("cuda" if torch.cuda.is_available() else "cpu")`) and pins
DataLoader memory / uses non-blocking transfers when a GPU is present — no
flag needed to opt in.

**1. Get the data onto the GPU machine.** `git pull` only brings the code —
the real archive (`~/era5_daily_cache`, ~4.1 GB: ERA5 `cape`/`tp`/`u_850`/
`v_850`/`msl`/`tcw`/`w_{200,700,850}`, IMERG `precipitation`, and the
global `static/` landmask+elevation pair) is gitignored and has to be
copied separately, e.g.:
```bash
rsync -avh --progress ~/era5_daily_cache gpu-machine:~/
```
Then on the GPU machine:
```bash
export DOWNSCALING_DATA_ROOT=~/era5_daily_cache
```

**2. Install dependencies** — `uv sync` (or `pip install -r requirements.txt`
if not using uv). Make sure whatever `torch` gets installed is a CUDA build
matching the GPU machine's CUDA version (the plain PyPI `torch` wheel is
sometimes CPU-only — check `python -c "import torch; print(torch.cuda.is_available())"`
after install).

**3. Train each variant** (50 epochs, batch_size 16 — lower `BATCH` if you
hit an out-of-memory error):
```bash
make train-west-africa-v1 BATCH=16 EPOCHS=50
make train-west-africa-v2 BATCH=16 EPOCHS=50
make train-west-africa-v3 BATCH=16 EPOCHS=50
make train-west-africa-v4 BATCH=16 EPOCHS=50
```
Each writes best/periodic checkpoints to `./checkpoints/variant{1,2,3,4}_*.pt`
(override with `--checkpoint_dir`, only available via the underlying
`uv run train.py --config configs/west_africa_train.yaml --val_config
configs/west_africa_val.yaml --variant N ...` if you need a flag the
Makefile targets don't expose). Resume an interrupted run with
`--resume_from <checkpoint.pt>`.

**Note on cost:** on this (CPU-only) machine, one epoch of `ghana` (half of
`west_africa`'s sample count) measured 4.5 min (variant 2) up to 72 min
(variant 4) — see the variant-by-variant benchmark earlier in this
project's history for the CPU numbers. A GPU should cut variants 1/3/4
dramatically, since their runtime is dominated by a `kernel_size=17`
convolution that GPUs handle far better than CPUs; variant 2 was already
fast on CPU and won't change as much.
