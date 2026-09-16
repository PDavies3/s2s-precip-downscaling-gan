import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_prediction_vs_target(pred_mm, target_mm, save_path, title=None):
    """1x3 diagnostic plot: prediction, target, and their difference, all in mm."""
    diff_mm = pred_mm - target_mm
    vmax = max(float(pred_mm.max()), float(target_mm.max()), 1e-6)
    diff_max = max(float(np.abs(diff_mm).max()), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(pred_mm, vmin=0, vmax=vmax, cmap="Blues")
    axes[0].set_title("Prediction (mm)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(target_mm, vmin=0, vmax=vmax, cmap="Blues")
    axes[1].set_title("Target (mm)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(diff_mm, vmin=-diff_max, vmax=diff_max, cmap="RdBu_r")
    axes[2].set_title("Prediction - Target (mm)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
