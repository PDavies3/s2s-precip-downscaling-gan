import pytest
import torch
import config
from models import get_models


@pytest.mark.parametrize("variant_id", [1, 2, 3, 4])
def test_generator_spatial_shapes(variant_id, sample_batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG, _ = get_models(variant_id=variant_id, device=device)

    out = netG(sample_batch['dynamic_input'].to(device), sample_batch['static_input'].to(device))
    assert out.shape == (sample_batch['dynamic_input'].shape[0], 1, *config.FINE_SHAPE)
    assert torch.all(out >= 0)


@pytest.mark.parametrize("variant_id", [1, 2, 3, 4])
def test_discriminator_forward_pass(variant_id, sample_batch):
    import torch.nn.functional as F
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG, netD = get_models(variant_id=variant_id, device=device)

    dynamics = sample_batch['dynamic_input'].to(device)
    statics = sample_batch['static_input'].to(device)
    real_rain = sample_batch['target_imerg'].to(device)

    fake_rain = netG(dynamics, statics)

    if variant_id in [1, 3]:
        d_real_input, d_fake_input = real_rain, fake_rain
    else:  # 2, 4
        coarse_interp = F.interpolate(dynamics, size=config.FINE_SHAPE, mode='bilinear', align_corners=False)
        cond_context = torch.cat([coarse_interp, statics], dim=1)
        d_real_input = torch.cat([cond_context, real_rain], dim=1)
        d_fake_input = torch.cat([cond_context, fake_rain], dim=1)

    logits_real = netD(d_real_input)
    logits_fake = netD(d_fake_input)
    assert logits_real.shape == logits_fake.shape
    assert logits_real.shape[0] == dynamics.shape[0]
