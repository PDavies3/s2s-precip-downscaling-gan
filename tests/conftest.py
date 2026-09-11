import os
import sys

import numpy as np
import pytest
import torch
import xarray as xr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

COARSE_SHAPE = (9, 9)
FINE_SHAPE = (128, 128)
DYNAMIC_CHANNELS = 6
STATIC_CHANNELS = 2


@pytest.fixture
def sample_batch():
    torch.manual_seed(0)
    batch_size = 2
    return {
        "dynamic_input": torch.rand(batch_size, DYNAMIC_CHANNELS, *COARSE_SHAPE),
        "static_input": torch.rand(batch_size, STATIC_CHANNELS, *FINE_SHAPE),
        "target_imerg": torch.rand(batch_size, 1, *FINE_SHAPE),
    }


def _write_nc(path, var_name, data, lat, lon):
    path.parent.mkdir(parents=True, exist_ok=True)
    da = xr.DataArray(data, dims=("latitude", "longitude"),
                       coords={"latitude": lat, "longitude": lon})
    xr.Dataset({var_name: da}).to_netcdf(path)


@pytest.fixture
def folder_tree_root(tmp_path):
    root = tmp_path / "mock_dataset"
    dates = ["2020-01-01", "2020-01-02", "2020-06-15", "2020-12-30", "2020-12-31"]
    lead_hours = [0, 24]
    members = [1, 2]
    u_levels = [700, 500]

    coarse_lat = np.linspace(11.5, 4.5, 10)  # descending, matching real S2S grids.
    coarse_lon = np.linspace(-3.5, 1.5, 12)

    fine_lat = np.linspace(4.5, 11.5, 20)
    fine_lon = np.linspace(-3.5, 1.5, 24)

    rng = np.random.default_rng(0)

    for date in dates:
        for lh in lead_hours:
            for m in members:
                data = rng.normal(300.0, 5.0, size=(len(coarse_lat), len(coarse_lon))).astype("float32")
                path = root / "t2m" / f"number_{m}" / f"t2m_{date}-{lh:04d}.nc"
                _write_nc(path, "t2m", data, coarse_lat, coarse_lon)

            for level in u_levels:
                u_data = rng.normal(0.0, 10.0, size=(len(coarse_lat), len(coarse_lon))).astype("float32")
                u_path = root / "u" / f"level_{level}" / f"u_{date}-{lh:04d}.nc"
                _write_nc(u_path, "u", u_data, coarse_lat, coarse_lon)

            target_data = rng.exponential(0.5, size=(len(fine_lat), len(fine_lon))).astype("float32")
            target_path = root / "imerg" / f"precipitation_{date}-{lh:04d}.nc"
            _write_nc(target_path, "precipitation", target_data, fine_lat, fine_lon)

    landmask = (rng.random(size=(len(fine_lat), len(fine_lon))) > 0.5).astype("float32")
    _write_nc(root / "static" / "landmask.nc", "landmask", landmask, fine_lat, fine_lon)

    topography = rng.normal(500.0, 300.0, size=(len(fine_lat), len(fine_lon))).astype("float32")
    _write_nc(root / "static" / "topography.nc", "topography", topography, fine_lat, fine_lon)

    return {"root": str(root), "dates": dates, "lead_hours": lead_hours, "members": members}


@pytest.fixture
def wide_folder_tree_root(tmp_path):
    # Native coverage comfortably wider than both the "ghana" and
    # "west_africa" region presets, so patches carved out of either one
    # (each possibly extending a bit past the named region -- see
    # utils/folder_dataset.py's module docstring) still interpolate cleanly
    # instead of hitting the out-of-bounds NaN check.
    root = tmp_path / "wide_mock_dataset"
    dates = ["2020-01-01"]
    lat = np.linspace(35.0, -5.0, 50)   # descending, matching real S2S grids.
    lon = np.linspace(-25.0, 25.0, 60)
    rng = np.random.default_rng(2)

    for date in dates:
        data = rng.normal(300.0, 5.0, size=(len(lat), len(lon))).astype("float32")
        _write_nc(root / "t2m" / f"t2m_{date}-0000.nc", "t2m", data, lat, lon)
        target = rng.exponential(0.5, size=(len(lat), len(lon))).astype("float32")
        _write_nc(root / "imerg" / f"precipitation_{date}-0000.nc", "precipitation", target, lat, lon)

    landmask = (rng.random(size=(len(lat), len(lon))) > 0.5).astype("float32")
    _write_nc(root / "static" / "landmask.nc", "landmask", landmask, lat, lon)
    topography = rng.normal(500.0, 300.0, size=(len(lat), len(lon))).astype("float32")
    _write_nc(root / "static" / "topography.nc", "topography", topography, lat, lon)

    return {"root": str(root), "dates": dates}
