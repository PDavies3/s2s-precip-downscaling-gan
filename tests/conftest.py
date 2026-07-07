import os
import sys
import pytest
import numpy as np
import xarray as xr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import config


@pytest.fixture(scope="session")
def mock_data_environment(tmp_path_factory):
    mock_dir = tmp_path_factory.mktemp("climate_mock")

    times = ["2026-08-01", "2026-08-02"]
    lats_coarse, lons_coarse = np.arange(config.COARSE_SHAPE[0]), np.arange(config.COARSE_SHAPE[1])
    lats_fine, lons_fine = np.arange(config.FINE_SHAPE[0]), np.arange(config.FINE_SHAPE[1])

    dyn_dict = {
        v: (["time", "lat", "lon"], np.random.rand(2, *config.COARSE_SHAPE).astype(np.float32))
        for v in config.SURFACE_VARIABLES + config.ATMOSPHERIC_VARIABLES
    }
    xr.Dataset(
        dyn_dict, coords={"time": times, "lat": lats_coarse, "lon": lons_coarse}
    ).to_netcdf(os.path.join(mock_dir, "dynamic_train.nc"))

    stat_dict = {
        v: (["lat", "lon"], np.random.rand(*config.FINE_SHAPE).astype(np.float32))
        for v in config.STATIC_GEOGRAPHIC_VARIABLES
    }
    xr.Dataset(
        stat_dict, coords={"lat": lats_fine, "lon": lons_fine}
    ).to_netcdf(os.path.join(mock_dir, "static_layers.nc"))

    xr.Dataset(
        {"precipitation": (["time", "lat", "lon"], np.random.rand(2, *config.FINE_SHAPE).astype(np.float32))},
        coords={"time": times, "lat": lats_fine, "lon": lons_fine}
    ).to_netcdf(os.path.join(mock_dir, "imerg_train.nc"))

    return str(mock_dir)


@pytest.fixture
def sample_batch(mock_data_environment):
    from utils.dataset import get_dataloader
    return next(iter(get_dataloader(data_dir=mock_data_environment, split="train", batch_size=2)))
