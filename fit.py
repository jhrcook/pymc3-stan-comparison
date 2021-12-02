#!/usr/bin/env python3

import os
import pickle
from pathlib import Path
from typing import Optional

import arviz as az
import pandas as pd
from dotenv import load_dotenv
from typer import Typer

from src import configuration as config

load_dotenv()

app = Typer()


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
    ).to_csv(output, index=False)


def _save_model_posterior(
    name: str, mdl_post: az.InferenceData, dir: Optional[Path]
) -> None:
    if dir is None:
        return None

    if not dir.exists():
        dir.mkdir()

    out_path = dir / f"{name}.pkl"
    with open(out_path, "wb") as file:
        pickle.dump(mdl_post, file)
    return None


def _check_config_file(config_file: Optional[Path]) -> Path:
    if config_file is not None:
        return config_file
    if (default_config_file := config.default_config()) is None:
        raise FileNotFoundError("Could not find a configuration file.")
    return default_config_file


@app.command()
def fit(
    config_name: str, config_file: Optional[Path], save_dir: Optional[Path] = None
) -> None:
    """Fit a model from the configuration file.

    Args:
        config_name (str): Name of the model.
        config_file (Optional[Path]): Path to the configuration file. If not exists, a
        default will be looked for in the environment, else an error is raised.
    """
    config_file = _check_config_file(config_file)
    mdl_config = config.get_model_configuration(config_name, config_file)
    res = config.get_model_callable(mdl_config)(mdl_config.config)
    _save_model_posterior(name=config_name, mdl_post=res, dir=save_dir)
    return None


@app.command()
def check_config(config_file: Optional[Path] = None) -> None:
    config_file = _check_config_file(config_file)
    _ = config.read_configuration_file(config_file)
    print("Configuration file successfully read and parsed.")
    return None


if __name__ == "__main__":
    app()
