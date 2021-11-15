import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def get_configuration_names(config_file: Path) -> list[str]:
    with open(config_file, "r") as file:
        configuration_names = [config["name"] for config in yaml.safe_load(file)]
    return configuration_names


# ---- Utilities


def get_theano_compdir(*args: Any, **kwargs: Any) -> str:
    return (
        f"THEANO_FLAGS='compiledir={tempfile.gettempdir()}/{np.random.randint(100)}' "
    )
