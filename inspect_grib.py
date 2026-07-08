"""
inspect_grib.py
----------------
Run this locally against ONE of your ECMWF S2S .grib files to print out
its structure (variables, dimensions, coordinate values). Paste the
output back so the dataloader can be built against your real data
instead of assumptions.

Setup (one-time):
    pip install cfgrib xarray eccodes
    # On Windows, eccodes sometimes needs the conda-forge build instead:
    #   conda install -c conda-forge cfgrib eccodes

Usage:
    python inspect_grib.py "ECMWF-s2s-Forecast_leadtime_sfc_d2m_t2m_CAPE_TCW_enfh_pf_init_2025-05-01.grib"
"""
import sys
import xarray as xr


def inspect(path):
    print(f"\n{'='*70}\nFILE: {path}\n{'='*70}")

    # A single GRIB file can contain multiple "hypercubes" (e.g. different
    # level types or step types mixed together). cfgrib needs each one
    # opened separately if they don't share a common grid/dims.
    try:
        ds = xr.open_dataset(path, engine="cfgrib")
        _describe(ds)
    except Exception as e:
        print(f"Single open_dataset failed ({e}).")
        print("Trying open_datasets (cfgrib.open_datasets) to split by hypercube...\n")
        import cfgrib
        datasets = cfgrib.open_datasets(path)
        for i, ds in enumerate(datasets):
            print(f"--- Hypercube {i} ---")
            _describe(ds)


def _describe(ds):
    print("\nData variables:")
    for name, da in ds.data_vars.items():
        print(f"  {name:12s} dims={da.dims}  shape={da.shape}  units={da.attrs.get('units', '?')}")

    print("\nCoordinates:")
    for name, coord in ds.coords.items():
        vals = coord.values
        preview = vals if getattr(vals, "size", 1) <= 10 else f"{vals.flat[0]} ... {vals.flat[-1]} (n={vals.size})"
        print(f"  {name:12s} {preview}")

    print("\nFull repr:")
    print(ds)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_grib.py <path_to_grib_file>")
        sys.exit(1)
    inspect(sys.argv[1])
