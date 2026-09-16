import torch
import torch.nn as nn

class HybridGenerator(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.low_enc = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.PReLU())
        
        self.learned_upscale = nn.Sequential(
            nn.Conv2d(32, 512, kernel_size=3, padding=1), nn.PixelShuffle(4), nn.PReLU(),
            nn.Conv2d(32, 512, kernel_size=3, padding=1), nn.PixelShuffle(4), nn.PReLU()
        )
        self.crop_to_target = nn.Conv2d(32, 64, kernel_size=17, stride=1, padding=0)
        # No activation on the final conv -- see README.md "Model design notes".
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.PReLU(),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, dynamic_in, static_in=None):
        x = self.low_enc(dynamic_in)
        x = self.learned_upscale(x)
        x = self.crop_to_target(x)
        return self.spatial_refine(x)