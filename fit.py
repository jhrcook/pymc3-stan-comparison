#!/usr/bin/env python3

import os
from itertools import product
from pathlib import Path
from time import time
from typing import Optional

import arviz as az
import pandas as pd
from colorama import Style
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

    out_path = dir / f"{name}.netcdf"
    mdl_post.to_netcdf(str(out_path))
    return None


def _check_config_file(config_file: Optional[Path]) -> Path:
    if config_file is not None:
        return config_file
    if (default_config_file := config.default_config()) is None:
        raise FileNotFoundError("Could not find a configuration file.")
    return default_config_file


def _write_function_call_time(fpath: Path, call_time: float) -> None:
    if not fpath.parent.exists():
        fpath.parent.mkdir(parents=True)
    with open(fpath, "a") as file:
        file.write(str(call_time))
        file.write("\n")
    return None


def _make_call_time_benchmark_fname(name: str) -> Path:
    return Path("benchmarks") / (name + ".fxntime")


@app.command()
def fit(
    name: str,
    config_name: str,
    config_file: Optional[Path] = None,
    save_dir: Optional[Path] = None,
) -> None:
    """Fit a model from the configuration file.

    Args:
        name (str): Unique name for this replicate.
        config_name (str): Name of the model.
        config_file (Optional[Path], optional): Path to the configuration file. If not
        provided, a default will be looked for in the environment, else an error is
        raised.
        save_dir (Optional[Path], optional): Directory to which model posterior files
        should be saved. Defaults to None and the model is not saved.
    """
    config_file = _check_config_file(config_file)
    mdl_config = config.get_model_configuration(config_name, config_file)
    tic = time()
    res = config.get_model_callable(mdl_config)(mdl_config.config)
    toc = time()
    _write_function_call_time(
        fpath=_make_call_time_benchmark_fname(name), call_time=toc - tic
    )
    _save_model_posterior(name=name, mdl_post=res, dir=save_dir)
    return None


@app.command()
def check_config(config_file: Optional[Path] = None) -> None:
    config_file = _check_config_file(config_file)
    _ = config.read_configuration_file(config_file)
    print("Configuration file successfully read and parsed.")
    return None


def _get_file_names(dir: Path) -> set[str]:
    return set([f.name.replace(f.suffix, "") for f in dir.iterdir()])


def _print_missing_files(name: str, mdl_names: set[str]) -> None:
    if len(mdl_names) == 0:
        return None
    print(Style.BRIGHT + f"Missing {name}:" + Style.RESET_ALL)
    _mdl_names = list(mdl_names)
    _mdl_names.sort()
    for mdl_name in _mdl_names:
        print("  " + mdl_name)


def _remove_files_with_name(mdl_names: set[str], dirs: list[Path]) -> None:
    all_files: set[Path] = set()
    n_removed: int = 0
    for _dir in dirs:
        all_files = all_files.union(set(_dir.iterdir()))
    for mdl_name, file in product(mdl_names, all_files):
        if mdl_name in file.name:
            n_removed += 1
            file.unlink(missing_ok=True)
    print(Style.BRIGHT + f"Removed {n_removed} files." + Style.RESET_ALL)
    return None


@app.command()
def check_benchmarks_and_model_files(
    benchmark_dir: Path,
    model_file_dir: Path,
    config_file: Optional[Path] = None,
    prune: bool = False,
) -> None:
    """Check that a benchmark and posterior file exist for each model configuration.

    Args:
        benchmark_dir (Path): Directory with benchmark files.
        model_file_dir (Path): Directory with model result files.
        config_file (Optional[Path], optional): Path to the configuration file. If not
        exists, a default will be looked for in the environment, else an error is
        raised.
        prune (bool, optional): Should files for a model be removed if it is missing one
        of the checked files? Defaults to False.

    Raises:
        FileNotFoundError: Raised if any models are missing a required file.
    """
    config_file = _check_config_file(config_file)
    mdl_configs = config.read_configuration_file(config_file)
    all_benchmarks = _get_file_names(benchmark_dir)
    all_model_files = _get_file_names(model_file_dir)

    missing_benchmarks: set[str] = set()
    missing_model_files: set[str] = set()

    for mdl_config in mdl_configs.configurations:
        name = mdl_config.name
        if name not in all_benchmarks:
            missing_benchmarks.add(name)
        if name not in all_model_files:
            missing_model_files.add(name)

    _print_missing_files("benchmarks", missing_benchmarks)
    _print_missing_files("model files", missing_model_files)
    if prune:
        _remove_files_with_name(
            missing_benchmarks.union(missing_model_files),
            [benchmark_dir, model_file_dir],
        )
    if len(missing_benchmarks) > 0 or len(missing_model_files) > 0:
        raise FileNotFoundError(
            "Missing benchmarks or model files -- see output above."
        )


if __name__ == "__main__":
    app()
