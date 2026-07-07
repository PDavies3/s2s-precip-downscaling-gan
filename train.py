import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils.dataset import get_dataloader
from models import get_models

def main():
    parser = argparse.ArgumentParser(description="Meteorological Downscaling Operational Core Engine")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory path targeting train/test netCDFs")
    parser.add_argument("--variant", type=int, required=True, choices=[1, 2, 3, 4], help="Downscaling Model Variant ID")
    parser.add_argument("--batch_size", type=int, default=16, help="Optimization deployment batch allocation")
    parser.add_argument("--epochs", type=int, default=50, help="Total epochs loop passes")
    parser.add_argument("--lr", type=float, default=2e-4, help="Adam optimizer base learning rate")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Orchestrating Model Variant Strategy {args.variant} execution pipeline on target accelerator: {device}")
    
    # Initialize Unified Dataloader Layer
    train_loader = get_dataloader(data_dir=args.data_dir, split="train", batch_size=args.batch_size)
    
    # Extract Models From the Factory Matrix Module
    netG, netD = get_models(variant_id=args.variant, device=device)
    
    optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_Pixel = nn.L1Loss()
    
    for epoch in range(args.epochs):
        netG.train()
        netD.train()
        
        for i, batch in enumerate(train_loader):
            dynamics = batch['dynamic_input'].to(device)
            statics = batch['static_input'].to(device)
            real_rain = batch['target_imerg'].to(device)
            
            # ------------------------------------------------------------------
            # DISCRIMINATOR TRACK OPTIMIZATION PASS
            # ------------------------------------------------------------------
            optimizer_D.zero_grad()
            fake_rain = netG(dynamics, statics)
            
            # Context-routing structural formatting logic per Variant criteria
            if args.variant in [1, 3]:
                d_real_input, d_fake_input = real_rain, fake_rain.detach()
            elif args.variant in [2, 4]:
                coarse_interp = F.interpolate(dynamics, size=(128, 128), mode='bilinear', align_corners=False)
                cond_context = torch.cat([coarse_interp, statics], dim=1)
                d_real_input = torch.cat([cond_context, real_rain], dim=1)
                d_fake_input = torch.cat([cond_context, fake_rain.detach()], dim=1)
                
            loss_D_real = criterion_GAN(netD(d_real_input), torch.ones_like(netD(d_real_input)))
            loss_D_fake = criterion_GAN(netD(d_fake_input), torch.zeros_like(netD(d_fake_input)))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            # ------------------------------------------------------------------
            # GENERATOR TRACK OPTIMIZATION PASS
            # ------------------------------------------------------------------
            optimizer_G.zero_grad()
            
            if args.variant in [1, 3]:
                d_g_input = fake_rain
            elif args.variant in [2, 4]:
                d_g_input = torch.cat([cond_context, fake_rain], dim=1)
                
            loss_G_adv = criterion_GAN(netD(d_g_input), torch.ones_like(netD(d_g_input)))
            loss_G_pixel = criterion_Pixel(fake_rain, real_rain)
            
            loss_G = loss_G_adv + (10.0 * loss_G_pixel) # Alpha balancing structural constraint weight
            loss_G.backward()
            optimizer_G.step()
            
        print(f"Epoch [{epoch+1}/{args.epochs}] Run Completed. Loss_D: {loss_D.item():.4f} | Loss_G: {loss_G.item():.4f}")

if __name__ == "__main__":
    main()