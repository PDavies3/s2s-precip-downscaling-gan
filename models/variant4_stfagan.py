import torch
import torch.nn as nn

class SpatialAttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))

class STFAGenerator(nn.Module):
    def __init__(self, dynamic_in_channels, static_in_channels, out_channels=1):
        super().__init__()
        self.feat_extract = nn.Sequential(nn.Conv2d(dynamic_in_channels, 64, kernel_size=3, padding=1), nn.PReLU())
        self.upscale = nn.Sequential(
            nn.Conv2d(64, 1024, kernel_size=3, padding=1), nn.PixelShuffle(4), nn.PReLU(),
            nn.Conv2d(64, 1024, kernel_size=3, padding=1), nn.PixelShuffle(4), nn.PReLU()
        )
        self.crop_conv = nn.Conv2d(64, 64, kernel_size=17, padding=0)
        self.static_encoder = nn.Sequential(
            nn.Conv2d(static_in_channels, 32, kernel_size=3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.LeakyReLU(0.2)
        )
        self.attention_gate = SpatialAttentionGate(F_g=64, F_l=64, F_int=32)
        # No activation on the final conv: the training target is a z-scored log1p
        # precip value (configs/ghana_template.yaml), negative for any below-average
        # log-precip -- including every dry pixel. A non-negative activation here
        # (ReLU/Softplus) makes most of the target range unreachable and the generator
        # collapses to a constant near 0. Physical (mm) non-negativity is guaranteed by
        # expm1 at denormalization time, not needed here.
        self.reconstruct = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
        
    def forward(self, dynamic_in, static_in):
        x_low = self.feat_extract(dynamic_in)
        x_high = self.crop_conv(self.upscale(x_low))
        static_feats = self.static_encoder(static_in)
        gated_static = self.attention_gate(g=x_high, x=static_feats)
        return self.reconstruct(torch.cat([x_high, gated_static], dim=1))