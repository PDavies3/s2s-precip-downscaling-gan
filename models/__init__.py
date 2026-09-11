from models.variant1_srgan import SRGANGenerator
from models.variant2_pix2pix import Pix2PixGenerator
from models.variant3_unet import HybridGenerator
from models.variant4_stfagan import STFAGenerator
from models.discriminator import Pix2PixDiscriminator

def get_models(variant_id, num_dynamic_channels, num_static_channels, device):
    # Channel counts come from the caller (a probe batch off the configured
    # dataloader) rather than a shared config module -- adding a predictor
    # variable to a YAML config changes these automatically, no code here
    # needs to change.

    if variant_id == 1:
        netG = SRGANGenerator(in_channels=num_dynamic_channels)
        netD = Pix2PixDiscriminator(in_channels=1)

    elif variant_id == 2:
        total_inputs = num_dynamic_channels + num_static_channels
        netG = Pix2PixGenerator(in_channels=total_inputs)
        netD = Pix2PixDiscriminator(in_channels=total_inputs + 1)

    elif variant_id == 3:
        netG = HybridGenerator(in_channels=num_dynamic_channels)
        netD = Pix2PixDiscriminator(in_channels=1)

    elif variant_id == 4:
        netG = STFAGenerator(dynamic_in_channels=num_dynamic_channels, static_in_channels=num_static_channels)
        total_inputs = num_dynamic_channels + num_static_channels
        netD = Pix2PixDiscriminator(in_channels=total_inputs + 1)

    else:
        raise ValueError(f"Requested configuration target Variant '{variant_id}' is invalid.")

    return netG.to(device), netD.to(device)
