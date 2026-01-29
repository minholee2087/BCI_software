import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if "dataset" not in cfg:
        raise ValueError("Config must define 'dataset'")
    if "modalities" not in cfg:
        raise ValueError("Config must define 'modalities'")

    return cfg
