import copy

import pytest

from utils.folder_dataset import ConfigurableDownscalingDataset


def _base_config(root):
    return {
        "data_root": root,
        "region": "ghana",
        "inputs": [
            {"name": "t2m", "path": "t2m",
             "normalisation": {"type": "zscore", "stats": {"mean": 300.0, "std": 5.0}}},
        ],
        "target": {"name": "precipitation", "path": "imerg",
                   "normalisation": {"type": "log1p", "stats": {"log1p_mean": 0.08, "log1p_std": 0.27}}},
        "additional_data": ["landmask", "topography"],
        "additional_data_paths": {
            "landmask": "static/landmask.nc",
            "topography": "static/topography.nc",
        },
    }


def test_discovers_correct_sample_count(folder_tree_root):
    config = _base_config(folder_tree_root["root"])
    ds = ConfigurableDownscalingDataset(config)
    expected = len(folder_tree_root["dates"]) * len(folder_tree_root["lead_hours"]) * len(folder_tree_root["members"])
    assert len(ds) == expected


def test_adding_a_variable_via_config_only_changes_channel_count(folder_tree_root):
    config = _base_config(folder_tree_root["root"])
    ds_before = ConfigurableDownscalingDataset(config)
    assert ds_before[0]["dynamic_input"].shape[0] == 1

    config_with_u = copy.deepcopy(config)
    config_with_u["inputs"].append({
        "name": "u", "path": "u", "levels": [700, 500],
        "normalisation": {"type": "zscore", "stats": {"mean": 0.0, "std": 10.0}},
    })
    ds_after = ConfigurableDownscalingDataset(config_with_u)
    assert ds_after[0]["dynamic_input"].shape[0] == 3


def test_static_fields_loaded_and_cropped_consistently(folder_tree_root):
    config = _base_config(folder_tree_root["root"])
    ds = ConfigurableDownscalingDataset(config)
    sample = ds[0]
    assert sample["static_input"].shape[0] == 2
    assert sample["static_input"].shape[-2:] == sample["target"].shape[-2:]


def test_empty_additional_data_list_raises_clear_error(folder_tree_root):
    config = _base_config(folder_tree_root["root"])
    config["additional_data"] = []
    with pytest.raises(ValueError, match="additional_data"):
        ConfigurableDownscalingDataset(config)


def test_missing_data_root_raises_clear_error():
    config = _base_config("/nonexistent/path/that/does/not/exist")
    with pytest.raises(FileNotFoundError):
        ConfigurableDownscalingDataset(config)


def test_start_date_excludes_earlier_samples(folder_tree_root):
    config = _base_config(folder_tree_root["root"])
    config["start_date"] = "2020-06-15"
    ds = ConfigurableDownscalingDataset(config)
    dates_used = {s[0] for s in ds.samples}
    assert "2020-01-01" not in dates_used
    assert dates_used == {"2020-06-15", "2020-12-30", "2020-12-31"}


def test_end_date_excludes_later_samples(folder_tree_root):
    config = _base_config(folder_tree_root["root"])
    config["end_date"] = "2020-06-15"
    ds = ConfigurableDownscalingDataset(config)
    dates_used = {s[0] for s in ds.samples}
    assert "2020-12-31" not in dates_used
    assert dates_used == {"2020-01-01", "2020-01-02", "2020-06-15"}


def test_period_filter_removing_all_samples_raises_clear_error(folder_tree_root):
    config = _base_config(folder_tree_root["root"])
    config["start_date"] = "2099-01-01"
    with pytest.raises(ValueError, match="removed all"):
        ConfigurableDownscalingDataset(config)


def _patch_config(root, region):
    return {
        "data_root": root,
        "region": region,
        "patch": {"coarse_shape": [2, 2], "fine_shape": [8, 8], "coarse_resolution_deg": 1.0},
        "inputs": [
            {"name": "t2m", "path": "t2m",
             "normalisation": {"type": "zscore", "stats": {"mean": 300.0, "std": 5.0}}},
        ],
        "target": {"name": "precipitation", "path": "imerg",
                   "normalisation": {"type": "log1p", "stats": {"log1p_mean": 0.08, "log1p_std": 0.27}}},
        "additional_data": ["landmask", "topography"],
        "additional_data_paths": {
            "landmask": "static/landmask.nc",
            "topography": "static/topography.nc",
        },
    }


def test_patch_shapes_match_configured_coarse_and_fine_shape(wide_folder_tree_root):
    config = _patch_config(wide_folder_tree_root["root"], "ghana")
    ds = ConfigurableDownscalingDataset(config)
    sample = ds[0]
    assert sample["dynamic_input"].shape[-2:] == (2, 2)
    assert sample["static_input"].shape[-2:] == (8, 8)
    assert sample["target"].shape[-2:] == (8, 8)


def test_bigger_region_yields_more_patches_at_same_resolution(wide_folder_tree_root):
    ghana_ds = ConfigurableDownscalingDataset(_patch_config(wide_folder_tree_root["root"], "ghana"))
    west_africa_ds = ConfigurableDownscalingDataset(_patch_config(wide_folder_tree_root["root"], "west_africa"))
    # ghana: 7x5 deg / 2x2 deg patches -> 3x2 = 6. west_africa: 21x34 deg -> 10x17 = 170.
    assert ghana_ds._n_lat_patches * ghana_ds._n_lon_patches == 6
    assert west_africa_ds._n_lat_patches * west_africa_ds._n_lon_patches == 170
    assert len(west_africa_ds) > len(ghana_ds)
    # both still resolve to the SAME configured shape -- more samples, not coarser ones.
    assert ghana_ds[0]["dynamic_input"].shape == west_africa_ds[0]["dynamic_input"].shape


def test_region_smaller_than_one_patch_still_yields_a_single_patch(wide_folder_tree_root):
    config = _patch_config(wide_folder_tree_root["root"], "ghana")
    config["patch"]["coarse_resolution_deg"] = 10.0  # patch footprint (20x20 deg) > ghana's 7x5 deg box
    ds = ConfigurableDownscalingDataset(config)
    assert ds._n_lat_patches == 1
    assert ds._n_lon_patches == 1
    sample = ds[0]
    assert sample["dynamic_input"].shape[-2:] == (2, 2)
