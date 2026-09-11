
REGIONS = {
    "africa": {
        "lat_min": -35.0, "lat_max": 38.0,
        "lon_min": -20.0, "lon_max": 52.0,
    },
    "west_africa": {
        "lat_min": 4.0, "lat_max": 25.0,
        "lon_min": -18.0, "lon_max": 16.0,
    },
    "ghana": {
        "lat_min": 4.5, "lat_max": 11.5,
        "lon_min": -3.5, "lon_max": 1.5,
    },
}


def resolve_region(region):
    if isinstance(region, str):
        if region not in REGIONS:
            raise ValueError(
                f"Unknown named region '{region}'. Available presets: "
                f"{list(REGIONS.keys())}. Or provide explicit bounds as a "
                f"dict: {{lat_min, lat_max, lon_min, lon_max}}."
            )
        return REGIONS[region]
    return region
