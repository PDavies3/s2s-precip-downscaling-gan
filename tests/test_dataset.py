import config
from utils.dataset import UnifiedClimateDataset


def test_dataloader_stack_integrity(mock_data_environment):
    dataset = UnifiedClimateDataset(data_dir=mock_data_environment, split="train")
    sample = dataset[0]

    expected_dyn_depth = len(config.SURFACE_VARIABLES) + len(config.ATMOSPHERIC_VARIABLES)
    expected_stat_depth = len(config.STATIC_GEOGRAPHIC_VARIABLES)

    assert sample['dynamic_input'].shape == (expected_dyn_depth, *config.COARSE_SHAPE)
    assert sample['static_input'].shape == (expected_stat_depth, *config.FINE_SHAPE)
    assert sample['target_imerg'].shape == (1, *config.FINE_SHAPE)
