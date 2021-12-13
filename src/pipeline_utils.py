import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ConfigName = str
ConfigParams = dict[str, str]
ConfigInfo = dict[ConfigName, ConfigParams]


def _make_param_dict(yaml_info: dict[str, Any]) -> ConfigParams:
    return {
        "mem": yaml_info["mem"],
        "time": yaml_info["time"],
        "partition": yaml_info.get("partition", "short"),
    }


def get_configuration_information(config_file: Path) -> ConfigInfo:
    with open(config_file, "r") as file:
        config_info = {x["name"]: _make_param_dict(x) for x in yaml.safe_load(file)}
    return config_info


def make_replicates_configuration(config: ConfigInfo, n_reps: int) -> ConfigInfo:
    new_configs: ConfigInfo = {}
    for name, params in config.items():
        for i in range(1, n_reps + 1):
            _config = params.copy()
            _config["name"] = name
            _config["rep"] = str(i)
            new_configs[f"{name}__{i}"] = _config
    return new_configs


# ---- Utilities


def get_theano_compdir(*args: Any, **kwargs: Any) -> str:
    return (
        f"THEANO_FLAGS='compiledir={tempfile.gettempdir()}/{np.random.randint(10000)}'"
    )
