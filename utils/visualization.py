import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ = ccrs.PlateCarree()


def _robust_max(*arrays, pct=99):
    """Percentile-based (not true max) color-scale ceiling -- a handful of extreme/
    outlier pixels (common in real satellite precip products) would otherwise wash
    out the entire color scale and hide the real spatial structure everywhere else."""
    combined = np.concatenate([np.asarray(a).ravel() for a in arrays])
    return max(float(np.percentile(combined, pct)), 1e-6)


def _render_panel(fig, ax, data, lat, lon, extent, panel_title, cmap, vmin, vmax):
    ax.set_extent(extent, crs=PROJ)
    im = ax.pcolormesh(lon, lat, data, vmin=vmin, vmax=vmax, cmap=cmap, transform=PROJ, shading="auto")
    ax.coastlines(resolution="50m", linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    # ax.set_title() silently fails to render on a cartopy GeoAxes once
    # gridlines(draw_labels=True) is active (no error -- the text is just
    # missing from the saved figure). ax.text() in axes coordinates sidesteps it.
    ax.text(0.5, 1.06, panel_title, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=13, fontweight="bold")
    # _robust_max clips the color scale at a percentile, not the true max/min -- the
    # colorbar's extend arrow makes that clipping visible instead of silently hiding it.
    extend = "both" if cmap == "RdBu_r" else "max"
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, extend=extend)
    cbar.set_label("mm")


def _save(fig, save_path, top):
    # fig.tight_layout() is skipped deliberately: combined with gridlines(draw_labels=True)
    # it can hit a cartopy/shapely geometry bug (GEOSException on a degenerate gridline
    # label clip polygon). Fixed margins avoid it.
    fig.subplots_adjust(left=0.04, right=0.98, top=top, bottom=0.05, wspace=0.3, hspace=0.5)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_prediction_vs_target(pred_mm, target_mm, lat, lon, save_path, title=None):
    """1x3 georeferenced diagnostic plot: prediction, target, and their difference, all in mm."""
    diff_mm = pred_mm - target_mm
    vmax = _robust_max(pred_mm, target_mm)
    diff_max = _robust_max(np.abs(diff_mm))
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]

    panels = [
        (pred_mm, "Predicted", "Blues", 0.0, vmax),
        (target_mm, "Target", "Blues", 0.0, vmax),
        (diff_mm, "Bias (Predicted − Target)", "RdBu_r", -diff_max, diff_max),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), subplot_kw={"projection": PROJ})
    for ax, (data, panel_title, cmap, vmin, vmax_) in zip(axes, panels):
        _render_panel(fig, ax, data, lat, lon, extent, panel_title, cmap, vmin, vmax_)

    if title:
        fig.suptitle(title)
    _save(fig, save_path, top=0.85)


def plot_full_comparison(era5_coarse_mm, pred_mm, target_mm, lat, lon, save_path, title=None):
    """1x4 georeferenced diagnostic plot: ERA5 coarse input, prediction, target, and bias, all in mm."""
    diff_mm = pred_mm - target_mm
    vmax = _robust_max(era5_coarse_mm, pred_mm, target_mm)
    diff_max = _robust_max(np.abs(diff_mm))
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]

    panels = [
        (era5_coarse_mm, "ERA5 Input (coarse)", "Blues", 0.0, vmax),
        (pred_mm, "Downscaled", "Blues", 0.0, vmax),
        (target_mm, "IMERG (Target)", "Blues", 0.0, vmax),
        (diff_mm, "Bias (Downscaled − IMERG)", "RdBu_r", -diff_max, diff_max),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), subplot_kw={"projection": PROJ})
    for ax, (data, panel_title, cmap, vmin, vmax_) in zip(axes, panels):
        _render_panel(fig, ax, data, lat, lon, extent, panel_title, cmap, vmin, vmax_)

    if title:
        fig.suptitle(title)
    _save(fig, save_path, top=0.85)


def plot_seasonal_grid(season_fields, lat, lon, save_path, title=None):
    """4x4 georeferenced grid: one row per season (DJF/MAM/JJA/SON), columns
    [ERA5 Input, Downscaled, IMERG, Bias]. Each row is a mean composite over
    that season's samples, color-scaled independently per row so a dry
    season's (e.g. DJF) spatial pattern stays visible next to a wet one's.

    season_fields: ordered list of (season_name, era5_coarse_mm, pred_mm, target_mm) tuples.
    """
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    n_rows = len(season_fields)

    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows), subplot_kw={"projection": PROJ})
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, (season_name, era5_mm, pred_mm, target_mm) in enumerate(season_fields):
        diff_mm = pred_mm - target_mm
        vmax = _robust_max(era5_mm, pred_mm, target_mm)
        diff_max = _robust_max(np.abs(diff_mm))

        panels = [
            (era5_mm, f"{season_name}: ERA5 Input", "Blues", 0.0, vmax),
            (pred_mm, f"{season_name}: Downscaled", "Blues", 0.0, vmax),
            (target_mm, f"{season_name}: IMERG", "Blues", 0.0, vmax),
            (diff_mm, f"{season_name}: Bias", "RdBu_r", -diff_max, diff_max),
        ]
        for col, (data, panel_title, cmap, vmin, vmax_) in enumerate(panels):
            _render_panel(fig, axes[row, col], data, lat, lon, extent, panel_title, cmap, vmin, vmax_)

    if title:
        fig.suptitle(title, fontsize=16)
    _save(fig, save_path, top=0.93 if n_rows > 1 else 0.85)
