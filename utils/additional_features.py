import numpy as np


def compute_day_of_year(ctx):
    vt = np.datetime64(ctx["valid_time"])
    doy = (vt - vt.astype("datetime64[Y]")).astype("timedelta64[D]").astype(int) + 1
    angle = 2 * np.pi * doy / 365.0
    sin_channel = np.full(ctx["shape"], np.sin(angle), dtype=np.float32)
    cos_channel = np.full(ctx["shape"], np.cos(angle), dtype=np.float32)
    return np.stack([sin_channel, cos_channel], axis=0)


def compute_latlon(ctx):
    lon_grid, lat_grid = np.meshgrid(ctx["lon"], ctx["lat"])
    lat_channel = (lat_grid / 90.0).astype(np.float32)
    lon_channel = (lon_grid / 180.0).astype(np.float32)
    return np.stack([lat_channel, lon_channel], axis=0)


COMPUTED_FEATURES = {
    "day_of_year": compute_day_of_year,
    "latlon": compute_latlon,
}
