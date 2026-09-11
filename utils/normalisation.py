import numpy as np


def zscore_transform(x, stats):
    return (x - stats["mean"]) / stats["std"]


def zscore_inverse(x, stats):
    return x * stats["std"] + stats["mean"]


def log1p_transform(x, stats):
    log_x = np.log1p(np.clip(x, a_min=0, a_max=None))
    return (log_x - stats["log1p_mean"]) / stats["log1p_std"]


def log1p_inverse(x, stats):
    log_x = x * stats["log1p_std"] + stats["log1p_mean"]
    return np.expm1(log_x)


_TRANSFORMS = {
    "zscore": (zscore_transform, zscore_inverse),
    "log1p": (log1p_transform, log1p_inverse),
}


def normalize(x, norm_config):
    transform_fn, _ = _TRANSFORMS[norm_config["type"]]
    return transform_fn(x, norm_config["stats"])


def denormalize(x, norm_config):
    _, inverse_fn = _TRANSFORMS[norm_config["type"]]
    return inverse_fn(x, norm_config["stats"])
