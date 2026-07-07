"""
utils/grib_dataset.py
----------------------
Dataloader for the real ECMWF S2S GRIB2 archive, indexed as
(init_date, leadtime, ensemble_member) -- each combination is one
training sample, per your instruction to treat ensemble members as
separate samples rather than averaging them.

Design notes / things that WILL need adjusting once you run
inspect_grib.py and confirm actual dims:
  - We assume each GRIB file's ensemble dim is called "number" and its
    lead-time dim is called "step" (standard cfgrib conventions). If your
    files use different names, update _LEADTIME_DIM / _MEMBER_DIM below.
  - We assume a single file can be opened directly via
    xr.open_dataset(path, engine="cfgrib"). If a file mixes incompatible
    hypercubes (e.g. different typeOfLevel / stepType in one .grib), that
    single call will raise -- the loader falls back to cfgrib.open_datasets
    and searches each hypercube for the variable you asked for.
  - Files are BIG (200-430MB). We open each file ONCE per init date and
    cache the xarray handle (backed by lazy dask/numpy views), so repeated
    __getitem__ calls for different leadtimes/members don't re-read the
    file from disk.
"""
import os
import glob
import re
import functools
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, DataLoader

import config_grib as cfg

_LEADTIME_DIM = "step"
_MEMBER_DIM = "number"


def _open_grib_var(path, shortname):
    """
    Open a GRIB file and return the DataArray for `shortname`, handling
    files that contain multiple incompatible hypercubes.
    """
    try:
        ds = xr.open_dataset(path, engine="cfgrib",
                              backend_kwargs={"indexpath": ""})
        if shortname in ds.data_vars:
            return ds[shortname]
    except Exception:
        pass

    # Fall back: split the file into its separate hypercubes and search each
    import cfgrib
    datasets = cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})
    for ds in datasets:
        if shortname in ds.data_vars:
            return ds[shortname]

    raise KeyError(
        f"Variable '{shortname}' not found in {path}. "
        f"Run inspect_grib.py against this file to see available variables."
    )


class S2SGribDataset(Dataset):
    def __init__(self, data_root=None, split="train"):
        self.data_root = os.path.expanduser(data_root or cfg.DATA_ROOT)
        self.split = split

        self.init_dates = self._discover_init_dates()
        if not self.init_dates:
            raise FileNotFoundError(
                f"No init-date file sets found under {self.data_root}. "
                f"Expected files matching pattern like "
                f"'ECMWF-s2s-Forecast_leadtime_sfc_instantaneous_enfh_pf_init_YYYY-MM-DD.grib'"
            )

        # Open (and cache) every file for every date ONCE up front.
        # {(date, file_group): xr.DataArray-returning accessor}
        self._file_cache = {}
        self._static_ds = None
        self._imerg_cache = {}

        # Determine n_leadtimes / n_members from the first date's first file
        probe_var, probe_group = next(iter(cfg.SURFACE_VARIABLES.items()))
        probe_da = self._get_var(self.init_dates[0], probe_group, probe_var)
        self.n_leadtimes = probe_da.sizes.get(_LEADTIME_DIM, 1)
        self.n_members = probe_da.sizes.get(_MEMBER_DIM, 1)

        # Build flat index: every (date, leadtime, member) triplet = one sample
        self.index = [
            (d, lt, m)
            for d in self.init_dates
            for lt in range(self.n_leadtimes)
            for m in range(self.n_members)
        ]

    def _discover_init_dates(self):
        # Use one filename template to find all available dates
        any_template = next(iter(cfg.FILENAME_TEMPLATES.values()))
        pattern = any_template.format(date="*")
        matches = glob.glob(os.path.join(self.data_root, pattern))
        dates = []
        date_re = re.compile(r"init_(\d{4}-\d{2}-\d{2})")
        for m in matches:
            match = date_re.search(os.path.basename(m))
            if match:
                dates.append(match.group(1))
        return sorted(set(dates))

    def _get_var(self, date, file_group, shortname):
        """Cached accessor: opens each (date, file_group) exactly once."""
        key = (date, file_group)
        if key not in self._file_cache:
            self._file_cache[key] = {}
        cache = self._file_cache[key]
        if shortname not in cache:
            filename = cfg.FILENAME_TEMPLATES[file_group].format(date=date)
            path = os.path.join(self.data_root, filename)
            cache[shortname] = _open_grib_var(path, shortname)
        return cache[shortname]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        date, leadtime_idx, member_idx = self.index[idx]

        # --- Surface (single-level) variables ---
        surface_layers = []
        for var, group in cfg.SURFACE_VARIABLES.items():
            da = self._get_var(date, group, var)
            sel = da.isel(**{_LEADTIME_DIM: leadtime_idx, _MEMBER_DIM: member_idx})
            surface_layers.append(np.asarray(sel.values))

        # --- Multi-level (pressure-level) variables ---
        atmos_layers = []
        for var, group in cfg.ATMOSPHERIC_VARIABLES.items():
            da = self._get_var(date, group, var)
            for level in cfg.PRESSURE_LEVELS:
                sel = da.isel(**{_LEADTIME_DIM: leadtime_idx, _MEMBER_DIM: member_idx}) \
                        .sel(isobaricInhPa=level)
                atmos_layers.append(np.asarray(sel.values))

        dynamic_tensor = torch.from_numpy(
            np.stack(surface_layers + atmos_layers, axis=0)
        ).float()

        # --- Static high-resolution context (opened once, reused) ---
        if self._static_ds is None:
            static_path = os.path.join(self.data_root, cfg.STATIC_LAYERS_PATH)
            self._static_ds = xr.open_dataset(static_path)
        static_tensor = torch.from_numpy(
            np.stack([self._static_ds[v].values for v in cfg.STATIC_GEOGRAPHIC_VARIABLES], axis=0)
        ).float()

        # --- Target IMERG precipitation, aligned to this date+leadtime ---
        if date not in self._imerg_cache:
            imerg_path = os.path.join(
                self.data_root, cfg.IMERG_FILENAME_TEMPLATE.format(date=date)
            )
            self._imerg_cache[date] = xr.open_dataset(imerg_path)
        ds_target = self._imerg_cache[date]
        target_tensor = torch.from_numpy(
            np.asarray(ds_target["precipitation"].isel(time=leadtime_idx).values)
        ).float().unsqueeze(0)

        return {
            "dynamic_input": dynamic_tensor,   # [C_dynamic, H_low, W_low]
            "static_input": static_tensor,      # [C_static, H_high, W_high]
            "target_imerg": target_tensor,      # [1, H_high, W_high]
            "meta": {"date": date, "leadtime": leadtime_idx, "member": member_idx},
        }


def get_grib_dataloader(data_root=None, split="train", batch_size=16, num_workers=None):
    if num_workers is None:
        num_workers = min(2, os.cpu_count() or 1)
    dataset = S2SGribDataset(data_root=data_root, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
