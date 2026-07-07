import os
import torch
import xarray as xr
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config import SURFACE_VARIABLES, ATMOSPHERIC_VARIABLES, STATIC_GEOGRAPHIC_VARIABLES

class UnifiedClimateDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        
        self.ds_dynamic = xr.open_dataset(os.path.join(data_dir, f"dynamic_{split}.nc"))
        self.ds_static = xr.open_dataset(os.path.join(data_dir, "static_layers.nc"))
        self.ds_target = xr.open_dataset(os.path.join(data_dir, f"imerg_{split}.nc"))
        self.timestamps = self.ds_dynamic.time.values
        
    def __len__(self):
        return len(self.timestamps)
        
    def __getitem__(self, idx):
        dynamic_layers = []
        
        # Stream Surface Configuration Matrix Items
        for var in SURFACE_VARIABLES:
            data = self.ds_dynamic[var].isel(time=idx).values
            if len(data.shape) > 2: data = data.mean(axis=0) # Collapse ensemble dimensions if present
            dynamic_layers.append(data)
            
        # Stream Upper Atmospheric Multi-Level Flight Deck Parameter Metrics
        for var in ATMOSPHERIC_VARIABLES:
            data = self.ds_dynamic[var].isel(time=idx).values
            if len(data.shape) == 3: # Handle multi-vertical level matrices safely [Levels, Lat, Lon]
                for lvl in range(data.shape[0]):
                    dynamic_layers.append(data[lvl])
            else:
                if len(data.shape) > 2: data = data.mean(axis=0)
                dynamic_layers.append(data)
                
        # Consolidate Stack Elements and transform into single low-resolution input block
        dynamic_tensor = torch.from_numpy(np.stack(dynamic_layers, axis=0)).float()
        
        # Capture High-Resolution Static Geographics Constraints
        static_layers = []
        for var in STATIC_GEOGRAPHIC_VARIABLES:
            static_layers.append(self.ds_static[var].values)
        static_tensor = torch.from_numpy(np.stack(static_layers, axis=0)).float()
        
        # Capture Destination Rainfall Matrix Truth Target Channel
        target_rain = self.ds_target['precipitation'].isel(time=idx).values
        target_tensor = torch.from_numpy(target_rain).float().unsqueeze(0)
        
        return {
            'dynamic_input': dynamic_tensor,
            'static_input': static_tensor,
            'target_imerg': target_tensor
        }

def get_dataloader(data_dir, split, batch_size, num_workers=None):
    if num_workers is None:
        # P.Davies: add — avoid DataLoader's "excessive worker creation" warning on
        # low-core machines (e.g. laptop dev); still allows 2 workers on multi-core nodes
        num_workers = min(2, os.cpu_count() or 1)
    dataset = UnifiedClimateDataset(data_dir=data_dir, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # P.Davies: add — avoid no-op pin_memory warning on CPU-only machines (e.g. local dev on ThinkPad), still pins on CUDA training nodes
    )
