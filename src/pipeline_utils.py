import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ConfigName = str
ConfigParams = dict[str, str]


def _make_param_dict(yaml_info: dict[str, Any]) -> ConfigParams:
    return {
        "mem": yaml_info["mem"],
        "time": yaml_info["time"],
        "partition": yaml_info.get("partition", "short"),
    }


def get_configuration_information(config_file: Path) -> dict[ConfigName, ConfigParams]:
    with open(config_file, "r") as file:
        config_info = {x["name"]: _make_param_dict(x) for x in yaml.safe_load(file)}
    return config_info


# ---- Utilities


def get_theano_compdir(*args: Any, **kwargs: Any) -> str:
    return (
        f"THEANO_FLAGS='compiledir={tempfile.gettempdir()}/{np.random.randint(10000)}' "
    )
