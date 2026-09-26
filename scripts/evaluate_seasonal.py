"""
scripts/evaluate_seasonal.py
-----------------------------
Post-processes an inference.py run's output (predictions/<run>/netcdf/*.nc +
metrics.csv) into two evaluation deliverables:

1. Individual diagnostic plots for the N days with the most widespread
   rainfall (highest rain_coverage_frac), each a 1x4 panel: ERA5 input,
   downscaled, IMERG target, bias.
2. A 4(season) x 4(panel) grid of season-mean composites (DJF/MAM/JJA/SON),
   same 4 panels, one row per season.

Both are restricted to a single patch (--patch_region, default "ghana")
picked by which patch's saved lat/lon bounds contain that named region's
center -- averaging pixel values across patches with different lat/lon grids
wouldn't be spatially meaningful, and multi-patch runs (e.g. west_africa)
have exactly this problem.

Usage:
    python scripts/evaluate_seasonal.py --predictions_dir predictions/west_africa_2022
"""
import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xarray as xr

from utils.regions import resolve_region
from utils.visualization import plot_full_comparison, plot_seasonal_grid

SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]


def _read_metrics(metrics_path):
    with open(metrics_path, newline="") as f:
        return list(csv.DictReader(f))


def _patch_key(row):
    return (row["patch_i"], row["patch_j"])


def _pick_target_patch(rows, netcdf_dir, patch_region):
    target = resolve_region(patch_region)
    center_lat = (target["lat_min"] + target["lat_max"]) / 2
    center_lon = (target["lon_min"] + target["lon_max"]) / 2

    seen = set()
    for row in rows:
        key = _patch_key(row)
        if key in seen:
            continue
        seen.add(key)
        path = _row_to_path(row, netcdf_dir)
        if not os.path.exists(path):
            continue
        ds = xr.open_dataset(path)
        lat, lon = ds["latitude"].values, ds["longitude"].values
        if lat.min() <= center_lat <= lat.max() and lon.min() <= center_lon <= lon.max():
            return key
    raise ValueError(f"No patch among {sorted(seen)} contains '{patch_region}'s center "
                      f"({center_lat}, {center_lon}).")


def _row_to_path(row, netcdf_dir):
    matches = glob.glob(os.path.join(
        netcdf_dir, f"variant*_{row['date']}_{int(row['lead_hours']):04d}h*"
        f"p{row['patch_i']}-{row['patch_j']}.nc"))
    if not matches:
        matches = glob.glob(os.path.join(
            netcdf_dir, f"variant*_{row['date']}_{int(row['lead_hours']):04d}h.nc"))
    return matches[0] if matches else ""


def _load_fields(path):
    ds = xr.open_dataset(path)
    return (ds["precipitation_era5_coarse"].values, ds["precipitation_pred"].values,
            ds["precipitation_target"].values, ds["latitude"].values, ds["longitude"].values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions_dir", type=str, required=True,
                         help="An inference.py --output_dir (contains netcdf/ and metrics.csv)")
    parser.add_argument("--patch_region", type=str, default="ghana",
                         help="Named region (see utils/regions.py) whose center picks which "
                              "patch this evaluation restricts to.")
    parser.add_argument("--top_n_events", type=int, default=3,
                         help="Number of highest rain_coverage_frac days to plot individually.")
    parser.add_argument("--output_dir", type=str, default=None,
                         help="Default: <predictions_dir>/evaluation")
    args = parser.parse_args()

    netcdf_dir = os.path.join(args.predictions_dir, "netcdf")
    metrics_path = os.path.join(args.predictions_dir, "metrics.csv")
    output_dir = args.output_dir or os.path.join(args.predictions_dir, "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    rows = _read_metrics(metrics_path)
    target_patch = _pick_target_patch(rows, netcdf_dir, args.patch_region)
    print(f">> Restricting to patch {target_patch} (contains '{args.patch_region}')")
    rows = [r for r in rows if _patch_key(r) == target_patch]
    print(f">> {len(rows)} samples in this patch")

    # --- 1. Top-N widest-rainfall event days ---
    rows_by_coverage = sorted(rows, key=lambda r: float(r["rain_coverage_frac"]), reverse=True)
    for row in rows_by_coverage[:args.top_n_events]:
        path = _row_to_path(row, netcdf_dir)
        era5_mm, pred_mm, target_mm, lat, lon = _load_fields(path)
        out_path = os.path.join(output_dir, f"event_{row['date']}.png")
        plot_full_comparison(
            era5_mm, pred_mm, target_mm, lat, lon, out_path,
            title=f"{row['date']} ({row['season']}) -- rain coverage "
                  f"{float(row['rain_coverage_frac']):.0%} of patch, MAE={float(row['mae_mm']):.2f} mm",
        )
        print(f">> Saved event plot: {out_path}")

    # --- 2. Seasonal mean composites ---
    season_fields = []
    for season in SEASON_ORDER:
        season_rows = [r for r in rows if r["season"] == season]
        if not season_rows:
            print(f">> No samples for {season}, skipping.")
            continue
        era5_sum = pred_sum = target_sum = None
        lat = lon = None
        for row in season_rows:
            path = _row_to_path(row, netcdf_dir)
            era5_mm, pred_mm, target_mm, lat, lon = _load_fields(path)
            if era5_sum is None:
                era5_sum, pred_sum, target_sum = era5_mm.copy(), pred_mm.copy(), target_mm.copy()
            else:
                era5_sum += era5_mm
                pred_sum += pred_mm
                target_sum += target_mm
        n = len(season_rows)
        season_fields.append((season, era5_sum / n, pred_sum / n, target_sum / n))
        print(f">> {season}: composited {n} samples")

    seasonal_path = os.path.join(output_dir, "seasonal_comparison.png")
    plot_seasonal_grid(season_fields, lat, lon, seasonal_path,
                        title=f"Seasonal mean comparison -- patch {target_patch} ('{args.patch_region}')")
    print(f">> Saved seasonal comparison: {seasonal_path}")


if __name__ == "__main__":
    main()
