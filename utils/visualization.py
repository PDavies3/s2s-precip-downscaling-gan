import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_prediction_vs_target(pred_mm, target_mm, lat, lon, save_path, title=None):
    """1x3 georeferenced diagnostic plot: prediction, target, and their difference, all in mm."""
    diff_mm = pred_mm - target_mm
    vmax = max(float(pred_mm.max()), float(target_mm.max()), 1e-6)
    diff_max = max(float(np.abs(diff_mm).max()), 1e-6)

    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    proj = ccrs.PlateCarree()

    panels = [
        (pred_mm, "Predicted", "Blues", 0.0, vmax),
        (target_mm, "Target", "Blues", 0.0, vmax),
        (diff_mm, "Bias (Predicted − Target)", "RdBu_r", -diff_max, diff_max),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), subplot_kw={"projection": proj})
    for ax, (data, panel_title, cmap, vmin, panel_vmax) in zip(axes, panels):
        ax.set_extent(extent, crs=proj)
        im = ax.pcolormesh(lon, lat, data, vmin=vmin, vmax=panel_vmax, cmap=cmap,
                            transform=proj, shading="auto")
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
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("mm")

    if title:
        fig.suptitle(title)
    # fig.tight_layout() is skipped deliberately: combined with gridlines(draw_labels=True)
    # it can hit a cartopy/shapely geometry bug (GEOSException on a degenerate gridline
    # label clip polygon). Fixed margins avoid it.
    fig.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.08, wspace=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
