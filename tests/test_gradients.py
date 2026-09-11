import pytest
import torch
from models import get_models


@pytest.mark.parametrize("variant_id", [1, 2, 3, 4])
def test_graph_gradient_flows(variant_id, sample_batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG, _ = get_models(variant_id, sample_batch['dynamic_input'].shape[1],
                          sample_batch['static_input'].shape[1], device)

    out = netG(sample_batch['dynamic_input'].to(device), sample_batch['static_input'].to(device))
    loss = torch.mean(torch.abs(out - sample_batch['target_imerg'].to(device)))
    loss.backward()

    found_grad = False
    for name, param in netG.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isnan(param.grad).sum() == 0, f"NaN gradient at layer: {name}"
            found_grad = True

    assert found_grad, f"No gradient updates flowed back on variant {variant_id}"
