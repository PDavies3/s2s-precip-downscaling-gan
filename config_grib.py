"""
config_grib.py
--------------
Configuration for the REAL ECMWF S2S GRIB2 data (as opposed to config.py,
which was written for the earlier NetCDF-mock prototype).

Update the variable short names below once you've confirmed them with
inspect_grib.py. The short names guessed here are standard ECMWF S2S
conventions but MUST be verified against your actual files -- GRIB short
names can vary by center/product and this is the #1 place a silent
mismatch will bite you.
"""
import os

# DATA_ROOT is resolved dynamically so this works for any user/machine
# without editing this file. Resolution order (first one that's set wins):
#   1. DOWNSCALING_DATA_ROOT environment variable
#   2. --data_dir CLI flag (train.py passes this through and overrides
#      DATA_ROOT at runtime -- see get_grib_dataloader(data_root=...))
#   3. Fallback default: ./data relative to wherever the script is run
DATA_ROOT = os.path.expanduser(
    os.environ.get("DOWNSCALING_DATA_ROOT", os.path.join(os.getcwd(), "data"))
)

# Filename template -- {date} gets substituted with e.g. "2025-05-01"
FILENAME_TEMPLATES = {
    "sfc_d2m_t2m_cape_tcw": "ECMWF-s2s-Forecast_leadtime_sfc_d2m_t2m_CAPE_TCW_enfh_pf_init_{date}.grib",
    "sfc_instantaneous":    "ECMWF-s2s-Forecast_leadtime_sfc_instantaneous_enfh_pf_init_{date}.grib",
    "sfc_t_max":            "ECMWF-s2s-Forecast_leadtime_sfc_T_max_enfh_pf_init_{date}.grib",
    "u_700_500":            "ECMWF-s2s-Forecast_leadtime_u_700_500_enfh_pf_init_{date}.grib",
    "w_700_500":            "ECMWF-s2s-Forecast_leadtime_w_700_500_enfh_pf_init_{date}.grib",
}

# Surface (single-level) variables you want to train on, and which file
# group each lives in. shortName is the cfgrib/eccodes short name -- CONFIRM
# these against inspect_grib.py output before trusting this list.
SURFACE_VARIABLES = {
    "t2m":  "sfc_d2m_t2m_cape_tcw",   # 2m temperature
    "d2m":  "sfc_d2m_t2m_cape_tcw",   # 2m dewpoint temperature
    "cape": "sfc_d2m_t2m_cape_tcw",   # Convective available potential energy
    "tciw": "sfc_d2m_t2m_cape_tcw",   # Total column water -- confirm exact shortName (tciw/tcw/tcwv?)
    "msl":  "sfc_instantaneous",     # Mean sea level pressure
    "u10":  "sfc_instantaneous",     # 10m U wind
    "v10":  "sfc_instantaneous",     # 10m V wind
    "mx2t": "sfc_t_max",              # Maximum 2m temperature over the step
}

# Multi-level (pressure-level) variables. Each expands into
# len(PRESSURE_LEVELS) channels automatically.
ATMOSPHERIC_VARIABLES = {
    "u": "u_700_500",   # U wind component at pressure levels
    "w": "w_700_500",   # Vertical velocity at pressure levels
}
PRESSURE_LEVELS = [700, 500]  # hPa -- confirm against isobaricInhPa coord values

# Static high-resolution context (unchanged from the mock prototype --
# point this at your real landmask/topography file)
STATIC_GEOGRAPHIC_VARIABLES = ["landmask", "topography"]
STATIC_LAYERS_PATH = "static_layers.nc"

# Target precipitation (IMERG), one NetCDF per init date, variable "precipitation"
IMERG_FILENAME_TEMPLATE = "imerg_{date}.nc"

# Grid geometry
COARSE_SHAPE = (9, 9)      # confirm against actual lat/lon dims in the GRIB files
FINE_SHAPE = (128, 128)

# Ensemble handling: per your instruction, every member is a SEPARATE
# training sample (not averaged).
ENSEMBLE_MODE = "separate_samples"
