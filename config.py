# ==============================================================================
# METEOROLOGICAL VARIABLE SELECTION DECLARATION WORKSPACE
# ==============================================================================

# Coarse-Grid Surface Parameters (Expected shape from NC: [Time, Lat_Coarse, Lon_Coarse])
SURFACE_VARIABLES = [
    't2m',     # 2 Metre Temperature
    'u10',     # 10 Metre U-Wind Component
    'v10',     # 10 Metre V-Wind Component
    'msl'      # Mean Sea Level Pressure
]

# Coarse-Grid Multi-Level Atmospheric Parameters
ATMOSPHERIC_VARIABLES = [
    'q',       # Specific Humidity
    'r',       # Relative Humidity
    't',       # Temperature
    'u'        # U-component of Wind Velocity
]

# High-Resolution Contextual Fields (Target-grid domain resolution anchoring)
STATIC_GEOGRAPHIC_VARIABLES = [
    'landmask',   # Coastal-sea boundary index
    'topography'  # Orographic elevation relief tracking (DEM)
]

# Dimensional Constraints (Matches S2S 9x9 coarse resolution to IMERG 128x128 destination)
COARSE_SHAPE = (9, 9)
FINE_SHAPE = (128, 128)