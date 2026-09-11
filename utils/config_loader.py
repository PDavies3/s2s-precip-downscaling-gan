import os
import warnings

import yaml


class _EnvTagLoader(yaml.SafeLoader):
    pass


def _env_constructor(loader, node):
    value = loader.construct_scalar(node)
    if "${" in value and value.startswith("${") and value.endswith("}"):
        var_name = value[2:-1]
        resolved = os.environ.get(var_name)
        if resolved is None:
            warnings.warn(
                f"!env reference '${{{var_name}}}' has no matching environment "
                f"variable set -- using the literal string as a fallback. "
                f"Set it with: export {var_name}=/your/path"
            )
            return value
        return resolved
    return value


_EnvTagLoader.add_constructor("!env", _env_constructor)


def load_config(path):
    path = os.path.abspath(path)
    with open(path) as f:
        config = yaml.load(f, Loader=_EnvTagLoader)

    if config is None:
        config = {}

    if "extends" in config:
        base_path = os.path.join(os.path.dirname(path), config["extends"])
        base_config = load_config(base_path)  # recursive -- resolves multi-level `extends` chains
        merged = _deep_merge(base_config, config)
        merged.pop("extends", None)
        return merged

    return config


def _deep_merge(base, override):
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
