#!/usr/bin/env python3

import os
from pathlib import Path
from typing import NoReturn, Optional

import pandas as pd
from typer import Typer

from models import hierarchical_model as hier
from models import simple_linear_regression as lm
from src.configuration import Model, get_model_configuration

app = Typer()


def assert_never(value: NoReturn) -> NoReturn:
    """Force runtime and static enumeration exhaustiveness.
    Args:
        value (NoReturn): Some value passed as an enum value.
    Returns:
        NoReturn: Nothing.
    """
    assert False, f"Unhandled value: {value} ({type(value).__name__})"  # noqa: B011


def _read_file_sizes(dir: Path) -> pd.DataFrame:
    files = list(dir.iterdir())
    sizes = pd.Series([os.path.getsize(f) for f in files])
    sizes_mb = sizes / 1000000.0
    return pd.DataFrame(
        {
            "file": files,
            "name": [f.name.replace(f.suffix, "") for f in files],
            "bytes": sizes,
            "mb": sizes_mb,
        }
    )


@app.command()
def model_result_sizes(dir: Path, output: Path) -> None:
    """Create a CSV of model file sizes.

    Args:
        dir (Path): Directory containing the pickled files.
        output (Path): Path to output a CSV.
    """
    _read_file_sizes(dir).assign(
        mb_lbl=lambda d: [f"{int(x)} MB" for x in d.mb],
    ).to_csv(output)


@app.command()
def fit(config_name: str, config_file: Optional[Path] = None) -> None:
    """Fit a model from the configuration file.

    Args:
        config_name (str): Name of the model.
        config_file (Optional[Path], optional): Configuration file. Defaults to None.
    """
    if config_file is None:
        config_file = Path("model-configs.yaml")
    mdl_config = get_model_configuration(config_name, config_file)

    if mdl_config.model is Model.SIMPLE_PYMC3:
        lm.simple_pymc3_model(config_name, config_kwargs=mdl_config.config)
    elif mdl_config.model is Model.SIMPLE_STAN:
        lm.simple_stan_model(config_name, config_kwargs=mdl_config.config)
    elif mdl_config.model is Model.HIERARCHICAL_PYMC3:
        hier.hierarchical_pymc3_model(config_name, config_kwargs=mdl_config.config)
    elif mdl_config.model is Model.HIERARCHICAL_STAN:
        hier.hierarchical_stan_model(config_name, config_kwargs=mdl_config.config)
    else:
        assert_never(mdl_config.model)

    return None


if __name__ == "__main__":
    app()
