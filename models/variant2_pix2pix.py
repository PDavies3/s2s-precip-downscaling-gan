import torch
import torch.nn as nn
import torch.nn.functional as F

class Pix2PixGenerator(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.down1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128))
        self.down3 = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(256))
        
        self.up1 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        # No output activation -- see README.md "Model design notes".
        self.final = nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1)
        
    def forward(self, dynamic_in, static_in):
        coarse_interp = F.interpolate(dynamic_in, size=(128, 128), mode='bilinear', align_corners=False)
        x = torch.cat([coarse_interp, static_in], dim=1)
        
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        
        u1 = torch.cat([self.up1(d3), d2], dim=1)
        u2 = torch.cat([self.up2(u1), d1], dim=1)
        return self.final(u2)