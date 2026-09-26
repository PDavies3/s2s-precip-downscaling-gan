import argparse
import csv
import os
import warnings

import numpy as np
import torch
import xarray as xr
from scipy.ndimage import zoom
from tqdm import tqdm

from models import get_models
from train import get_config_dataloader
from utils.normalisation import denormalize
from utils.visualization import plot_prediction_vs_target

_SEASON_BY_MONTH = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


def _season_for_date(date_str):
    month = int(date_str[5:7])
    return _SEASON_BY_MONTH[month]

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"xarray\..*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"xarray\..*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"cartopy\..*")


def _sample_meta(meta, i):
    date = meta["date"][i]
    lead_hours = int(meta["lead_hours"][i])
    member = int(meta["member"][i])
    patch = meta["patch"]
    if isinstance(patch, (list, tuple)):
        patch_i, patch_j = int(patch[0][i]), int(patch[1][i])
    else:
        patch_i = patch_j = None
    return date, lead_hours, member, patch_i, patch_j


def _output_stem(variant, date, lead_hours, member, patch_i, patch_j):
    stem = f"variant{variant}_{date}_{lead_hours:04d}h"
    if member is not None and member >= 0:
        stem += f"_m{member}"
    if patch_i is not None:
        stem += f"_p{patch_i}-{patch_j}"
    return stem


def _save_prediction_netcdf(path, pred_mm, target_mm, era5_coarse_mm, lat, lon, date, lead_hours, member):
    ds = xr.Dataset(
        {
            "precipitation_pred": (("latitude", "longitude"), pred_mm),
            "precipitation_target": (("latitude", "longitude"), target_mm),
            "precipitation_era5_coarse": (("latitude", "longitude"), era5_coarse_mm),
        },
        coords={"latitude": lat, "longitude": lon},
        attrs={"date": date, "lead_hours": lead_hours, "member": member, "units": "mm",
               "precipitation_era5_coarse_note": "ERA5 tp, native 9x9 resolution bilinear-upsampled "
                                                  "onto the fine grid for display only -- not a real "
                                                  "downscaling, just a visual reference for how coarse "
                                                  "the raw input is."},
    )
    ds.to_netcdf(path)


@torch.no_grad()
def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config, dataset, loader = get_config_dataloader(args.config, args.batch_size, region_override=args.region,
                                                      shuffle=False)
    target_norm = dataset.target_spec["normalisation"]
    tp_idx = next((i for i, spec in enumerate(dataset.inputs_spec) if spec["name"] == "tp"), None)
    if tp_idx is None:
        raise ValueError("inference.py needs a 'tp' input in the config (the coarse ERA5 "
                          "precipitation field) to save the ERA5-input comparison panel.")
    tp_norm = dataset.inputs_spec[tp_idx]["normalisation"]

    probe_batch = next(iter(loader))
    dynamic_channels = probe_batch["dynamic_input"].shape[1]
    static_channels = probe_batch["static_input"].shape[1]

    netG, _ = get_models(args.variant, dynamic_channels, static_channels, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    netG.load_state_dict(ckpt["generator_state_dict"])
    netG.eval()
    print(f">> Loaded generator from {args.checkpoint} (epoch {ckpt.get('epoch', '?')} of its source run)")

    netcdf_dir = os.path.join(args.output_dir, "netcdf")
    plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(netcdf_dir, exist_ok=True)
    if args.plot_every_n > 0:
        os.makedirs(plot_dir, exist_ok=True)

    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_file = open(metrics_path, "w", newline="")
    writer = csv.writer(metrics_file)
    writer.writerow(["date", "season", "lead_hours", "member", "patch_i", "patch_j",
                     "mae_mm", "rain_coverage_frac"])

    n_samples = 0
    total_mae_mm = 0.0

    for batch in tqdm(loader, desc="Running inference"):
        dynamics = batch["dynamic_input"].to(device, non_blocking=True)
        statics = batch["static_input"].to(device, non_blocking=True)
        target = batch["target"]

        pred = netG(dynamics, statics).cpu()

        batch_size = dynamics.shape[0]
        for i in range(batch_size):
            if args.max_samples is not None and n_samples >= args.max_samples:
                break

            date, lead_hours, member, patch_i, patch_j = _sample_meta(batch["meta"], i)
            pred_mm = denormalize(pred[i, 0].numpy(), target_norm)
            target_mm = denormalize(target[i, 0].numpy(), target_norm)
            lat = batch["lat"][i].numpy()
            lon = batch["lon"][i].numpy()

            tp_coarse_mm = denormalize(dynamics[i, tp_idx].cpu().numpy(), tp_norm)
            era5_coarse_mm = zoom(tp_coarse_mm, np.array(pred_mm.shape) / np.array(tp_coarse_mm.shape), order=1)

            mae_mm = float(np.abs(pred_mm - target_mm).mean())
            rain_coverage_frac = float((target_mm > args.rain_threshold_mm).mean())
            season = _season_for_date(date)
            writer.writerow([date, season, lead_hours, member, patch_i, patch_j,
                              f"{mae_mm:.4f}", f"{rain_coverage_frac:.4f}"])
            metrics_file.flush()
            total_mae_mm += mae_mm
            n_samples += 1

            stem = _output_stem(args.variant, date, lead_hours, member, patch_i, patch_j)
            _save_prediction_netcdf(os.path.join(netcdf_dir, f"{stem}.nc"),
                                     pred_mm, target_mm, era5_coarse_mm, lat, lon, date, lead_hours, member)

            if args.plot_every_n > 0 and n_samples % args.plot_every_n == 0:
                plot_prediction_vs_target(
                    pred_mm, target_mm, lat, lon,
                    os.path.join(plot_dir, f"{stem}.png"),
                    title=f"Variant {args.variant} -- {date} (+{lead_hours}h) -- MAE={mae_mm:.3f} mm",
                )

        if args.max_samples is not None and n_samples >= args.max_samples:
            break

    metrics_file.close()

    mean_mae = total_mae_mm / max(n_samples, 1)
    print(f">> Ran inference on {n_samples} samples. Mean MAE: {mean_mae:.4f} mm")
    print(f">> Predictions saved to {netcdf_dir}")
    print(f">> Per-sample metrics saved to {metrics_path}")
    if args.plot_every_n > 0:
        print(f">> Diagnostic plots saved to {plot_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run a trained generator checkpoint over a config-driven dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config (see configs/)")
    parser.add_argument("--variant", type=int, required=True, choices=[1, 2, 3, 4], help="Generator variant ID")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a checkpoint .pt file")
    parser.add_argument("--output_dir", type=str, default="./predictions")
    parser.add_argument("--region", type=str, default=None, help="Override the config's region")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None, help="Cap the number of samples processed")
    parser.add_argument("--plot_every_n", type=int, default=0,
                         help="Save a pred/target/diff diagnostic plot every Nth sample (0 disables)")
    parser.add_argument("--rain_threshold_mm", type=float, default=1.0,
                         help="Threshold (mm) a target pixel must exceed to count toward "
                              "rain_coverage_frac in metrics.csv")
    args = parser.parse_args()

    run_inference(args)


if __name__ == "__main__":
    main()
