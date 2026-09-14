# ==============================================================================
# Downscaling Pipeline — dev shortcuts (uses uv under the hood)
# ==============================================================================

CONFIG     ?= configs/ghana_train.yaml
VAL_CONFIG ?= configs/ghana_val.yaml
BATCH      ?= 16
EPOCHS     ?= 50

.PHONY: sync test test-fast test-dataset test-arch test-grad \
        test-v1 test-v2 test-v3 test-v4 \
        train-v1 train-v2 train-v3 train-v4 \
        train-west-africa-v1 train-west-africa-v2 train-west-africa-v3 train-west-africa-v4 \
        smoke clean

## Install/sync all dependencies into local .venv
sync:
	uv sync

## Run the full test suite (verbose)
test:
	uv run pytest tests/ -v

## Run the full test suite, quiet, stop on first failure
test-fast:
	uv run pytest tests/ -x -q

## Isolated: dataloader / NetCDF channel-stacking checks only
test-dataset:
	uv run pytest tests/test_dataset.py -v

## Isolated: forward-pass shape checks for all 4 generators + discriminators
test-arch:
	uv run pytest tests/test_architectures.py -v

## Isolated: backward-pass / gradient-flow checks for all 4 variants
test-grad:
	uv run pytest tests/test_gradients.py -v

## Run every test scoped to a single variant (any test file)
test-v1:
	uv run pytest tests/ -k "1" -v
test-v2:
	uv run pytest tests/ -k "2" -v
test-v3:
	uv run pytest tests/ -k "3" -v
test-v4:
	uv run pytest tests/ -k "4" -v

## One-epoch, tiny-batch dry run per variant — sanity check before a real job.
## Uses synthetic data (--dry_run), so no config or real files are needed.
smoke:
	@for v in 1 2 3 4; do \
		echo ">> Smoke test variant $$v"; \
		uv run train.py --dry_run --variant $$v --batch_size 2 --epochs 1 || exit 1; \
	done
	@echo ">> All 4 variants completed a 1-epoch dry run without error."

## Real training runs (override CONFIG / VAL_CONFIG / BATCH / EPOCHS as needed)
train-v1:
	uv run train.py --config $(CONFIG) --val_config $(VAL_CONFIG) --variant 1 --batch_size $(BATCH) --epochs $(EPOCHS)
train-v2:
	uv run train.py --config $(CONFIG) --val_config $(VAL_CONFIG) --variant 2 --batch_size $(BATCH) --epochs $(EPOCHS)
train-v3:
	uv run train.py --config $(CONFIG) --val_config $(VAL_CONFIG) --variant 3 --batch_size $(BATCH) --epochs $(EPOCHS)
train-v4:
	uv run train.py --config $(CONFIG) --val_config $(VAL_CONFIG) --variant 4 --batch_size $(BATCH) --epochs $(EPOCHS)

## Real training runs on the west_africa domain (override BATCH/EPOCHS as needed).
## Requires DOWNSCALING_DATA_ROOT set to wherever ~/era5_daily_cache lives.
train-west-africa-v1:
	uv run train.py --config configs/west_africa_train.yaml --val_config configs/west_africa_val.yaml --variant 1 --batch_size $(BATCH) --epochs $(EPOCHS)
train-west-africa-v2:
	uv run train.py --config configs/west_africa_train.yaml --val_config configs/west_africa_val.yaml --variant 2 --batch_size $(BATCH) --epochs $(EPOCHS)
train-west-africa-v3:
	uv run train.py --config configs/west_africa_train.yaml --val_config configs/west_africa_val.yaml --variant 3 --batch_size $(BATCH) --epochs $(EPOCHS)
train-west-africa-v4:
	uv run train.py --config configs/west_africa_train.yaml --val_config configs/west_africa_val.yaml --variant 4 --batch_size $(BATCH) --epochs $(EPOCHS)

## Clean caches
clean:
	rm -rf .pytest_cache **/__pycache__ .venv
