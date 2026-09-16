import glob
import os
import re

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from utils.additional_features import COMPUTED_FEATURES
from utils.normalisation import normalize
from utils.regions import resolve_region

_FILENAME_RE = re.compile(r".*_(\d{4}-\d{2}-\d{2})(?:-(\d{4}))?\.nc$")
_MEMBER_RE = re.compile(r"number_(\d+)")
_LEVEL_RE = re.compile(r"level_(-?\d+)")


def _crop_region(da, region):
    if region is None:
        return da

    lat_vals = da["latitude"].values
    lon_vals = da["longitude"].values

    # handles both ascending and descending coordinate ordering.
    lat_slice = (slice(region["lat_max"], region["lat_min"])
                 if lat_vals[0] > lat_vals[-1]
                 else slice(region["lat_min"], region["lat_max"]))
    lon_slice = (slice(region["lon_max"], region["lon_min"])
                 if lon_vals[0] > lon_vals[-1]
                 else slice(region["lon_min"], region["lon_max"]))

    cropped = da.sel(latitude=lat_slice, longitude=lon_slice)
    if cropped.sizes.get("latitude", 0) == 0 or cropped.sizes.get("longitude", 0) == 0:
        raise ValueError(
            f"Region crop produced an EMPTY array. Requested region={region}, "
            f"but this file's actual lat range is "
            f"[{lat_vals.min():.2f}, {lat_vals.max():.2f}] and lon range is "
            f"[{lon_vals.min():.2f}, {lon_vals.max():.2f}]. Check for overlap."
        )
    return cropped


def _resample_to_shape(da, region, shape):
    n_lat, n_lon = shape

    lat_vals = da["latitude"].values
    lon_vals = da["longitude"].values

    target_lat = (np.linspace(region["lat_max"], region["lat_min"], n_lat)
                  if lat_vals[0] > lat_vals[-1]
                  else np.linspace(region["lat_min"], region["lat_max"], n_lat))
    target_lon = (np.linspace(region["lon_max"], region["lon_min"], n_lon)
                  if lon_vals[0] > lon_vals[-1]
                  else np.linspace(region["lon_min"], region["lon_max"], n_lon))

    resampled = da.interp(latitude=target_lat, longitude=target_lon, method="linear")
    if np.any(np.isnan(resampled.values)):
        raise ValueError(
            f"Resampling to shape={shape} over region={region} produced NaN "
            f"values -- the target grid likely extends slightly beyond the "
            f"source file's actual coverage "
            f"(source lat range [{lat_vals.min():.3f}, {lat_vals.max():.3f}], "
            f"lon range [{lon_vals.min():.3f}, {lon_vals.max():.3f}]). Try "
            f"shrinking the region/patch slightly or check the source file's extent."
        )
    return resampled


def _resample_to_resolution(da, region, resolution_deg):
    if resolution_deg is None:
        return da
    n_lat = round((region["lat_max"] - region["lat_min"]) / resolution_deg) + 1
    n_lon = round((region["lon_max"] - region["lon_min"]) / resolution_deg) + 1
    return _resample_to_shape(da, region, (n_lat, n_lon))


def _patch_bounds(region, patch_extent_lat_deg, patch_extent_lon_deg, patch_i, patch_j):
    lat_min = region["lat_min"] + patch_i * patch_extent_lat_deg
    lon_min = region["lon_min"] + patch_j * patch_extent_lon_deg
    return {
        "lat_min": lat_min, "lat_max": lat_min + patch_extent_lat_deg,
        "lon_min": lon_min, "lon_max": lon_min + patch_extent_lon_deg,
    }


def _standardize_dims(da):
    """
    Real source files are NOT consistent about dimension naming or a
    leftover singleton time axis -- e.g. in one real ERA5/IMERG archive,
    ERA5 files use ('latitude', 'longitude') while IMERG files use
    ('time', 'lon', 'lat') with time of length 1 (one file = one date).
    Normalize both quirks so _crop_region/_resample_to_shape (which always
    key off 'latitude'/'longitude') work regardless of the source convention.
    """
    rename = {}
    if "lat" in da.dims and "latitude" not in da.dims:
        rename["lat"] = "latitude"
    if "lon" in da.dims and "longitude" not in da.dims:
        rename["lon"] = "longitude"
    if rename:
        da = da.rename(rename)
    if "time" in da.dims:
        if da.sizes["time"] != 1:
            raise ValueError(
                f"Expected exactly one time step per file (one file = one "
                f"date/lead_hours combination), but '{da.name}' at this path "
                f"has {da.sizes['time']}."
            )
        da = da.isel(time=0)
    return da


def _parse_path(path):
    fname = os.path.basename(path)
    m = _FILENAME_RE.match(fname)
    if not m:
        return None
    date_str, llll = m.groups()
    member_match = _MEMBER_RE.search(path)
    level_match = _LEVEL_RE.search(path)
    member = int(member_match.group(1)) if member_match else None
    level = int(level_match.group(1)) if level_match else None
    lead_hours = int(llll) if llll is not None else 0
    return date_str, lead_hours, member, level


class ConfigurableDownscalingDataset(Dataset):
    """
    Two resampling modes, controlled by whether config['patch'] is present:

    - No `patch` block (default): the whole `region` is resampled to one
      sample, using each spec's own `resolution_deg` (or native resolution
      if omitted). Fine for a region whose real extent is already close to
      what you want a single sample to cover.

    - `patch` block present: `coarse_resolution_deg` fixes the PHYSICAL size
      of one coarse pixel (e.g. matching the real S2S native grid spacing),
      and `coarse_shape`/`fine_shape` are the fixed pixel counts the chosen
      model variant requires (9x9 / 128x128 for variants 1/3/4). The region
      is tiled into a grid of non-overlapping patches, each exactly
      coarse_shape/fine_shape pixels at that fixed physical resolution --
      so a bigger region (e.g. west_africa) yields MORE samples at the same
      real resolution, rather than the same number of samples at coarser
      resolution. A region smaller than one patch's physical footprint
      (patch_extent_deg = coarse_shape[i] * coarse_resolution_deg) still
      yields exactly one patch, whose bounds extend past the named region
      -- that's inherent to the model's fixed receptive field, not a bug.
    """

    def __init__(self, config, split="train"):
        self.config = config
        self.data_root = os.path.expanduser(config["data_root"])
        self.region = resolve_region(config.get("region"))
        self.inputs_spec = config["inputs"]
        self.target_spec = config["target"]
        self.additional_data = config.get("additional_data", [])
        self.additional_data_paths = config.get("additional_data_paths", {})
        if not self.additional_data:
            raise ValueError(
                "config['additional_data'] must have at least one entry -- "
                "it provides the static high-resolution context that fixes "
                "the model's OUTPUT RESOLUTION."
            )
        for name in self.additional_data:
            if name not in COMPUTED_FEATURES and name not in self.additional_data_paths:
                raise ValueError(
                    f"additional_data entry '{name}' is not a registered "
                    f"computed feature ({list(COMPUTED_FEATURES.keys())}) "
                    f"and has no path in additional_data_paths."
                )

        self.patch = config.get("patch")
        self._n_lat_patches = self._n_lon_patches = 1
        self._patch_extent_lat_deg = self._patch_extent_lon_deg = None
        if self.patch is not None:
            self.coarse_shape = tuple(self.patch["coarse_shape"])
            self.fine_shape = tuple(self.patch["fine_shape"])
            self.coarse_resolution_deg = self.patch["coarse_resolution_deg"]
            self._patch_extent_lat_deg = self.coarse_shape[0] * self.coarse_resolution_deg
            self._patch_extent_lon_deg = self.coarse_shape[1] * self.coarse_resolution_deg
            self._n_lat_patches = max(1, int(
                (self.region["lat_max"] - self.region["lat_min"]) // self._patch_extent_lat_deg))
            self._n_lon_patches = max(1, int(
                (self.region["lon_max"] - self.region["lon_min"]) // self._patch_extent_lon_deg))

        self.start_date = config.get("start_date")
        self.end_date = config.get("end_date")
        for label, value in (("start_date", self.start_date), ("end_date", self.end_date)):
            if value is not None and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value)):
                raise ValueError(
                    f"config['{label}'] = {value!r} is not in 'YYYY-MM-DD' format."
                )

        self._file_index = {}
        for spec in self.inputs_spec + [self.target_spec]:
            self._index_variable(spec)

        probe_name = self.inputs_spec[0]["name"]
        keys = [k for k in self._file_index if k[0] == probe_name]
        base_samples = sorted({(k[1], k[2], k[3]) for k in keys})

        if not base_samples:
            raise FileNotFoundError(
                f"No samples discovered for '{probe_name}' under "
                f"{os.path.join(self.data_root, self.inputs_spec[0]['path'])}."
            )

        if self.start_date is not None or self.end_date is not None:
            n_before = len(base_samples)
            base_samples = [
                s for s in base_samples
                if (self.start_date is None or s[0] >= self.start_date)
                and (self.end_date is None or s[0] <= self.end_date)
            ]
            if not base_samples:
                discovered_dates = sorted({d for d, _, _ in
                                            sorted({(k[1], k[2], k[3]) for k in keys})})
                raise ValueError(
                    f"start_date/end_date filter [{self.start_date}, {self.end_date}] "
                    f"removed all {n_before} discovered samples. Discovered dates range "
                    f"from {discovered_dates[0]} to {discovered_dates[-1]}."
                )
            print(f"[ConfigurableDownscalingDataset] Period filter [{self.start_date or '-inf'}, "
                  f"{self.end_date or '+inf'}]: {n_before} -> {len(base_samples)} samples.")

        self.samples = [
            (date_str, lead_hours, member, patch_i, patch_j)
            for (date_str, lead_hours, member) in base_samples
            for patch_i in range(self._n_lat_patches)
            for patch_j in range(self._n_lon_patches)
        ]

    def _index_variable(self, spec):
        var_dir = os.path.join(self.data_root, spec["path"])
        matches = glob.glob(os.path.join(var_dir, "**", "*.nc"), recursive=True)
        for path in matches:
            parsed = _parse_path(path)
            if parsed is None:
                continue
            date_str, lead_hours, member, level = parsed
            self._file_index[(spec["name"], date_str, lead_hours, member, level)] = path

    def _resolve_path(self, var_name, date_str, lead_hours, member, level):
        key = (var_name, date_str, lead_hours, member, level)
        if key in self._file_index:
            return self._file_index[key]
        fallback_key = (var_name, date_str, lead_hours, None, level)
        if fallback_key in self._file_index:
            return self._file_index[fallback_key]
        raise KeyError(
            f"No file found for variable='{var_name}', date={date_str}, "
            f"lead_hours={lead_hours}, member={member}, level={level}"
        )

    def _patch_region(self, patch_i, patch_j):
        if self.patch is None:
            return self.region
        return _patch_bounds(self.region, self._patch_extent_lat_deg,
                              self._patch_extent_lon_deg, patch_i, patch_j)

    def _load_cropped(self, path, region, resolution_deg=None, shape=None):
        ds = xr.open_dataset(path)
        var_name = list(ds.data_vars)[0]
        da_full = _standardize_dims(ds[var_name])
        if shape is not None:
            return _resample_to_shape(da_full, region, shape)
        if resolution_deg is not None:
            return _resample_to_resolution(da_full, region, resolution_deg)
        return _crop_region(da_full, region)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        date_str, lead_hours, member, patch_i, patch_j = self.samples[idx]
        region = self._patch_region(patch_i, patch_j)

        channels = []
        for spec in self.inputs_spec:
            levels = spec.get("levels", [None])
            for level in levels:
                path = self._resolve_path(spec["name"], date_str, lead_hours, member, level)
                shape = self.coarse_shape if self.patch is not None else None
                da = self._load_cropped(path, region, spec.get("resolution_deg"), shape)
                arr = da.values
                scale = spec.get("scale")
                if scale is not None:
                    arr = arr * scale
                arr = normalize(arr, spec["normalisation"])
                channels.append(arr)
        dynamic_tensor = torch.from_numpy(np.stack(channels, axis=0)).float()

        target_shape = self.fine_shape if self.patch is not None else None
        target_path = self._resolve_path(self.target_spec["name"], date_str, lead_hours, member, None)
        target_da = self._load_cropped(target_path, region, self.target_spec.get("resolution_deg"), target_shape)
        target_arr = target_da.values
        target_scale = self.target_spec.get("scale")
        if target_scale is not None:
            target_arr = target_arr * target_scale
        target_arr = normalize(target_arr, self.target_spec["normalisation"])
        target_tensor = torch.from_numpy(target_arr).float().unsqueeze(0)

        valid_time = np.datetime64(date_str) + np.timedelta64(lead_hours, "h")
        ctx = {
            "valid_time": valid_time,
            "shape": target_da.shape,
            "lat": target_da["latitude"].values,
            "lon": target_da["longitude"].values,
        }
        additional_channels = []
        for name in self.additional_data:
            if name in COMPUTED_FEATURES:
                additional_channels.append(COMPUTED_FEATURES[name](ctx))
            else:
                entry = self.additional_data_paths[name]
                if isinstance(entry, dict):
                    path = os.path.join(self.data_root, entry["path"])
                    resolution_deg = entry.get("resolution_deg")
                    norm_config = entry.get("normalisation")
                else:
                    path = os.path.join(self.data_root, entry)
                    resolution_deg = None
                    norm_config = None
                da = self._load_cropped(path, region, resolution_deg, target_shape)
                arr = da.values
                if norm_config is not None:
                    arr = normalize(arr, norm_config)
                additional_channels.append(arr[np.newaxis, ...])
        static_tensor = torch.from_numpy(np.concatenate(additional_channels, axis=0)).float()

        return {
            "dynamic_input": dynamic_tensor,
            "static_input": static_tensor,
            "target": target_tensor,
            "lat": torch.from_numpy(ctx["lat"].copy()).float(),
            "lon": torch.from_numpy(ctx["lon"].copy()).float(),
            "meta": {"date": date_str, "lead_hours": lead_hours,
                     "member": member if member is not None else -1,
                     "patch": (patch_i, patch_j) if self.patch is not None else None},
        }
