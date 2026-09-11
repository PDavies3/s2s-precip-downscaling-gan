import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import get_models
from utils.config_loader import load_config
from utils.folder_dataset import ConfigurableDownscalingDataset

CRITERION_GAN = nn.BCEWithLogitsLoss()
CRITERION_PIXEL = nn.L1Loss()


def get_config_dataloader(config_path, batch_size, region_override=None, shuffle=True):
    config = load_config(config_path)
    if region_override is not None:
        print(f">> --region override: using '{region_override}' instead of the config's region")
        config["region"] = region_override
    dataset = ConfigurableDownscalingDataset(config)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=min(2, os.cpu_count() or 1),
                         pin_memory=torch.cuda.is_available())
    return config, dataset, loader


def get_dry_run_batch(batch_size, dynamic_channels, static_channels, coarse_shape, fine_shape, device):
    dynamics = torch.rand(batch_size, dynamic_channels, *coarse_shape, device=device)
    statics = torch.rand(batch_size, static_channels, *fine_shape, device=device)
    real_rain = torch.rand(batch_size, 1, *fine_shape, device=device)
    return dynamics, statics, real_rain


def _discriminator_inputs(variant, dynamics, statics, real_rain, fake_rain):
    if variant in (1, 3):
        return real_rain, fake_rain
    coarse_interp = F.interpolate(dynamics, size=real_rain.shape[-2:], mode="bilinear", align_corners=False)
    cond_context = torch.cat([coarse_interp, statics], dim=1)
    d_real_input = torch.cat([cond_context, real_rain], dim=1)
    d_fake_input = torch.cat([cond_context, fake_rain], dim=1)
    return d_real_input, d_fake_input


def discriminator_step(netD, netG, optimizer_D, variant, dynamics, statics, real_rain):
    with torch.no_grad():
        fake_rain = netG(dynamics, statics)

    optimizer_D.zero_grad()
    d_real_input, d_fake_input = _discriminator_inputs(variant, dynamics, statics, real_rain, fake_rain)

    logits_real = netD(d_real_input)
    logits_fake = netD(d_fake_input)
    loss_D_real = CRITERION_GAN(logits_real, torch.ones_like(logits_real))
    loss_D_fake = CRITERION_GAN(logits_fake, torch.zeros_like(logits_fake))
    loss_D = (loss_D_real + loss_D_fake) * 0.5
    loss_D.backward()
    optimizer_D.step()
    return loss_D.item()


def generator_step(netG, netD, optimizer_G, variant, dynamics, statics, real_rain, pixel_weight, use_adversarial):
    optimizer_G.zero_grad()
    fake_rain = netG(dynamics, statics)
    loss_G_pixel = CRITERION_PIXEL(fake_rain, real_rain)

    if use_adversarial:
        if variant in (1, 3):
            d_g_input = fake_rain
        else:
            coarse_interp = F.interpolate(dynamics, size=real_rain.shape[-2:], mode="bilinear", align_corners=False)
            d_g_input = torch.cat([coarse_interp, statics, fake_rain], dim=1)
        logits_g = netD(d_g_input)
        loss_G_adv = CRITERION_GAN(logits_g, torch.ones_like(logits_g))
    else:
        loss_G_adv = torch.tensor(0.0, device=fake_rain.device)

    loss_G = loss_G_adv + (pixel_weight * loss_G_pixel)
    loss_G.backward()
    optimizer_G.step()
    return loss_G.item(), loss_G_adv.item(), loss_G_pixel.item()


@torch.no_grad()
def validate(netG, val_loader, device):
    netG.eval()
    total_pixel = 0.0
    n_batches = 0
    for batch in val_loader:
        dynamics = batch["dynamic_input"].to(device, non_blocking=True)
        statics = batch["static_input"].to(device, non_blocking=True)
        real_rain = batch["target"].to(device, non_blocking=True)
        fake_rain = netG(dynamics, statics)
        total_pixel += CRITERION_PIXEL(fake_rain, real_rain).item()
        n_batches += 1
    netG.train()
    return {"pixel": total_pixel / max(n_batches, 1)}


def _save_checkpoint(path, netG, netD, variant, epoch, extra=None):
    payload = {
        "generator_state_dict": netG.state_dict(),
        "discriminator_state_dict": netD.state_dict(),
        "variant": variant,
        "epoch": epoch,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Orchestrating Model Variant Strategy {args.variant} execution pipeline on target accelerator: {device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.dry_run:
        print(">> DRY RUN: using synthetic data, no real files needed.")
        dynamic_channels, static_channels = 6, 2
        coarse_shape, fine_shape = (9, 9), (128, 128)

        netG, netD = get_models(args.variant, dynamic_channels, static_channels, device)
        optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))

        for epoch in range(args.epochs):
            use_adversarial = epoch >= args.warmup_epochs
            for _ in range(args.dry_run_steps):
                dynamics, statics, real_rain = get_dry_run_batch(
                    args.batch_size, dynamic_channels, static_channels, coarse_shape, fine_shape, device
                )
                if use_adversarial:
                    loss_D = discriminator_step(netD, netG, optimizer_D, args.variant, dynamics, statics, real_rain)
                else:
                    loss_D = 0.0
                loss_G, loss_G_adv, loss_G_pixel = generator_step(
                    netG, netD, optimizer_G, args.variant, dynamics, statics, real_rain,
                    args.pixel_weight, use_adversarial,
                )
            tag = "(warmup, no adversarial) " if not use_adversarial else ""
            print(f"Epoch [{epoch+1}/{args.epochs}] (dry run) {tag}"
                  f"Loss_D: {loss_D:.4f} | Loss_G_adv: {loss_G_adv:.4f} | Loss_G_pixel: {loss_G_pixel:.4f}")

        ckpt_path = os.path.join(args.checkpoint_dir, f"variant{args.variant}_dryrun.pt")
        _save_checkpoint(ckpt_path, netG, netD, args.variant, args.epochs)
        print(f">> Dry run complete. Checkpoint saved to {ckpt_path}")
        return

    print(f">> Using config-driven dataset: {args.config}")
    _, _, train_loader = get_config_dataloader(args.config, args.batch_size, region_override=args.region)

    probe_batch = next(iter(train_loader))
    dynamic_channels = probe_batch["dynamic_input"].shape[1]
    static_channels = probe_batch["static_input"].shape[1]

    val_loader = None
    if args.val_config:
        print(f">> Using validation config: {args.val_config}")
        _, val_dataset, val_loader = get_config_dataloader(
            args.val_config, args.batch_size, region_override=args.region, shuffle=False
        )
        print(f">> {len(val_dataset)} validation samples")

    netG, netD = get_models(args.variant, dynamic_channels, static_channels, device)

    if args.resume_from:
        print(f">> Resuming generator weights from {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        netG.load_state_dict(ckpt["generator_state_dict"])
        print(f"   Resumed successfully from epoch {ckpt.get('epoch', '?')} of the source run.")

    optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))

    best_score = float("inf")

    for epoch in range(args.epochs):
        use_adversarial = epoch >= args.warmup_epochs
        netG.train()
        netD.train()

        sum_loss_D = sum_loss_G = sum_loss_G_adv = sum_loss_G_pixel = 0.0
        n_batches = 0

        for batch in train_loader:
            dynamics = batch["dynamic_input"].to(device, non_blocking=True)
            statics = batch["static_input"].to(device, non_blocking=True)
            real_rain = batch["target"].to(device, non_blocking=True)

            if use_adversarial:
                loss_D = discriminator_step(netD, netG, optimizer_D, args.variant, dynamics, statics, real_rain)
            else:
                loss_D = 0.0

            loss_G, loss_G_adv, loss_G_pixel = generator_step(
                netG, netD, optimizer_G, args.variant, dynamics, statics, real_rain,
                args.pixel_weight, use_adversarial,
            )

            sum_loss_D += loss_D
            sum_loss_G += loss_G
            sum_loss_G_adv += loss_G_adv
            sum_loss_G_pixel += loss_G_pixel
            n_batches += 1

        n = max(n_batches, 1)
        tag = "(warmup, no adversarial) " if not use_adversarial else ""
        print(f"Epoch [{epoch+1}/{args.epochs}] {tag}"
              f"Loss_D: {sum_loss_D/n:.4f} | Loss_G_adv: {sum_loss_G_adv/n:.4f} | "
              f"Loss_G_pixel: {sum_loss_G_pixel/n:.4f}")

        if val_loader is not None:
            val_metrics = validate(netG, val_loader, device)
            print(f"           Val_pixel: {val_metrics['pixel']:.4f}")

            if val_metrics["pixel"] < best_score:
                best_score = val_metrics["pixel"]
                best_path = os.path.join(args.checkpoint_dir, f"variant{args.variant}_best.pt")
                _save_checkpoint(best_path, netG, netD, args.variant, epoch + 1,
                                  extra={"val_pixel_loss": val_metrics["pixel"]})
                print(f"           >> New best (val_pixel={best_score:.4f}). Saved to {best_path}")

        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.checkpoint_dir, f"variant{args.variant}_epoch{epoch+1}.pt")
            _save_checkpoint(ckpt_path, netG, netD, args.variant, epoch + 1)
            print(f">> Checkpoint saved to {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Meteorological Downscaling Operational Core Engine")
    parser.add_argument("--config", type=str, default=None, help="Path to a training YAML config (see configs/)")
    parser.add_argument("--val_config", type=str, default=None, help="Path to a validation YAML config")
    parser.add_argument("--region", type=str, default=None, help="Override the config's region")
    parser.add_argument("--variant", type=int, required=True, choices=[1, 2, 3, 4], help="Downscaling Model Variant ID")
    parser.add_argument("--batch_size", type=int, default=16, help="Optimization deployment batch allocation")
    parser.add_argument("--epochs", type=int, default=50, help="Total epochs loop passes")
    parser.add_argument("--lr", type=float, default=2e-4, help="Adam optimizer base learning rate")
    parser.add_argument("--pixel_weight", type=float, default=10.0, help="L1 pixel-loss weight in the generator loss")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                         help="Epochs of pixel loss only before the discriminator/adversarial term is introduced.")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint_every", type=int, default=5)
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint to resume generator weights from")
    parser.add_argument("--dry_run", action="store_true", help="Train on synthetic data -- no real config/files needed")
    parser.add_argument("--dry_run_steps", type=int, default=3)
    args = parser.parse_args()

    if not args.dry_run and args.config is None:
        parser.error("--config is required unless --dry_run is set")

    train(args)


if __name__ == "__main__":
    main()
