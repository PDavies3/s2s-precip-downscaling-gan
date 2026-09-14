"""
scripts/build_daily_cache.py
-----------------------------
Aggregates raw hourly ERA5 files (one file per year, e.g.
'<source_root>/tcw/tcw_2019.nc' with an 8760-step 'valid_time' dimension)
into the daily-mean, per-date-file convention utils/folder_dataset.py
expects (e.g. '<dest_root>/ERA5/era5_tcw/tcw_2019-01-01.nc'), matching
exactly how the existing era5_cape/era5_msl/era5_u_850/era5_v_850 folders
in the real archive were built (verified: the existing daily cape value at
a given date/gridpoint equals the mean of that day's 24 hourly values, not
a max or a single-hour snapshot).

Uses a direct numpy reshape+mean rather than xarray's resample/groupby --
verified the source files have perfectly regular hourly spacing (8760 or
8784 steps, no gaps), so reshaping (n_days, 24, lat, lon) and averaging
axis=1 is safe and roughly two orders of magnitude faster than xarray's
generic groupby machinery for a year-long file.

Usage:
    python scripts/build_daily_cache.py --var tcw --years 2019 2020 2021
    python scripts/build_daily_cache.py --var w --level 850 --years 2019 2020 2021

Safe to re-run: existing output files are skipped, so an interrupted run
can just be restarted.
"""
import argparse
import os
import time

import numpy as np
import xarray as xr

SOURCE_ROOT = "/media/pdavies/T7 Shield/era5_data"
DEST_ROOT = os.path.expanduser("~/era5_daily_cache/ERA5")


def convert(var_name, years, level=None):
    if level is None:
        source_dir = os.path.join(SOURCE_ROOT, var_name)
        dest_dir = os.path.join(DEST_ROOT, f"era5_{var_name}")
    else:
        source_dir = os.path.join(SOURCE_ROOT, var_name, str(level))
        dest_dir = os.path.join(DEST_ROOT, f"era5_{var_name}_{level}")
    os.makedirs(dest_dir, exist_ok=True)

    for year in years:
        fname = f"{var_name}_{year}.nc" if level is None else f"{var_name}_{level}_{year}.nc"
        src_path = os.path.join(source_dir, fname)
        if not os.path.exists(src_path):
            print(f">> SKIP {src_path} (not found)")
            continue

        n_days = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
        date_index = np.array([np.datetime64(f"{year}-01-01") + np.timedelta64(d, "D")
                                for d in range(n_days)])
        out_paths = [os.path.join(dest_dir, f"{var_name}_{d}.nc") for d in date_index.astype(str)]
        if all(os.path.exists(p) for p in out_paths):
            print(f">> {year}: already fully converted, skipping.")
            continue

        t0 = time.perf_counter()
        print(f">> Opening {src_path} ...")
        ds = xr.open_dataset(src_path)
        da = ds[var_name]
        if "pressure_level" in da.dims:
            da = da.squeeze("pressure_level", drop=True)
        if "number" in da.dims:
            da = da.squeeze("number", drop=True)

        n_hours = da.sizes["valid_time"]
        if n_hours != n_days * 24:
            raise ValueError(
                f"{src_path}: expected {n_days * 24} hourly steps for {year}, got {n_hours}. "
                f"The fast reshape path assumes no gaps -- fall back to xarray resample if this fires."
            )

        print(f">> Loading {n_hours} hourly steps ({da.nbytes / 1e9:.2f} GB) and reshaping to daily means ...")
        values = da.transpose("valid_time", "latitude", "longitude").values  # (n_hours, lat, lon)
        lat = da["latitude"].values
        lon = da["longitude"].values
        ds.close()

        daily = values.reshape(n_days, 24, values.shape[1], values.shape[2]).mean(axis=1)
        print(f">> Reshaped in {time.perf_counter() - t0:.1f}s. Writing {n_days} daily files ...")

        written = 0
        for i, date_str in enumerate(date_index.astype(str)):
            out_path = out_paths[i]
            if os.path.exists(out_path):
                continue
            day_da = xr.DataArray(daily[i], dims=("latitude", "longitude"),
                                   coords={"latitude": lat, "longitude": lon})
            xr.Dataset({var_name: day_da}).to_netcdf(out_path)
            written += 1
        print(f">> {year}: wrote {written} new daily files to {dest_dir} "
              f"(total {time.perf_counter() - t0:.1f}s)\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--var", required=True, help="Variable name, e.g. tcw or w")
    parser.add_argument("--level", type=int, default=None,
                         help="Pressure level subfolder, e.g. 850 (only for multi-level vars like w)")
    parser.add_argument("--years", type=int, nargs="+", required=True)
    args = parser.parse_args()
    convert(args.var, args.years, level=args.level)


if __name__ == "__main__":
    main()
