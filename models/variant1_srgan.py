import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x): return x + self.block(x)

class SRGANGenerator(nn.Module):
    def __init__(self, in_channels, out_channels=1, num_res_blocks=8):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.PReLU())
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])
        
        # Progressive sub-pixel upscaling pipeline transformations: 9x9 -> 36x36 -> 144x144
        self.upscale = nn.Sequential(
            nn.Conv2d(64, 1024, kernel_size=3, padding=1), nn.PixelShuffle(4), nn.PReLU(),
            nn.Conv2d(64, 1024, kernel_size=3, padding=1), nn.PixelShuffle(4), nn.PReLU()
        )
        # Symmetrical center boundary alignment compression map to exactly 128x128 
        self.final_crop = nn.Conv2d(64, out_channels, kernel_size=17, stride=1, padding=0)
        # No output activation: the training target is a z-scored log1p precip value
        # (configs/ghana_template.yaml), which is negative for any below-average
        # log-precip -- including every dry pixel (log1p(0) alone z-scores to about
        # -0.78). A non-negative activation here (ReLU/Softplus) makes most of the
        # target range unreachable, so the generator collapses to a constant near 0
        # within a couple of epochs and stops learning. Non-negativity of the
        # physical (mm) output is already guaranteed by expm1 at denormalization time.

    def forward(self, dynamic_in, static_in=None):
        x = self.conv1(dynamic_in)
        x = self.res_blocks(x) + x
        x = self.upscale(x)
        return self.final_crop(x)