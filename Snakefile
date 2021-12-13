import os
from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml
from snakemake.io import Wildcards, touch
from dotenv import load_dotenv

from src.pipeline_utils import (
    get_theano_compdir,
    get_configuration_information,
    make_replicates_configuration,
)

load_dotenv()

# ---- Configure ----

N_PROFILE_REPS: int = int(os.environ.get("N_PROFILE_REPS", 5))
CONFIG_FILE: Path = Path(os.environ["CONFIG_FILE"])
MODEL_FILES_DIR: Path = Path(os.environ.get("MODEL_FILES_DIR", None))

# ---- Setup ----

if not MODEL_FILES_DIR.exists():
    MODEL_FILES_DIR.mkdir()

configurations = get_configuration_information(CONFIG_FILE)
configurations = make_replicates_configuration(configurations, N_PROFILE_REPS)
configuration_names = list(configurations.keys())


def _get_config_params(w: Wildcards) -> dict[str, str]:
    return configurations[w.name]


def get_config_name(w: Wildcards) -> str:
    return _get_config_params(w)["name"]


def get_config_replicate_num(w: Wildcards) -> str:
    return _get_config_params(w)["rep"]


def get_config_mem(w: Wildcards) -> str:
    return _get_config_params(w)["mem"]


def get_config_time(w: Wildcards) -> str:
    return _get_config_params(w)["time"]


def get_config_partition(w: Wildcards) -> str:
    return _get_config_params(w)["partition"]


# ---- Rules ----


# Steps to run locally.
localrules:
    all,
    model_result_sizes,
    notebook,


rule all:
    input:
        notebook_html="docs/index.html",


rule fit_model:
    output:
        res=f"{MODEL_FILES_DIR}/{{name}}.netcdf",
    conda:
        "environment.yaml"
    params:
        config_name=lambda w: get_config_name(w),
        replicate=lambda w: get_config_replicate_num(w),
        mem=lambda w: get_config_mem(w),
        time=lambda w: get_config_time(w),
        partition=lambda w: get_config_partition(w),
        theano_dir=get_theano_compdir,
    shell:
        (
            "{params.theano_dir} ./fit.py fit {wildcards.name} {params.config_name} "
            + f"--config-file={CONFIG_FILE} --save-dir={MODEL_FILES_DIR}"
        )


rule model_result_sizes:
    input:
        model_results=expand(
            str(MODEL_FILES_DIR / "{name}.netcdf"), name=configuration_names
        ),
    output:
        csv="model-result-file-sizes.csv",
    conda:
        "environment.yaml"
    shell:
        f"./fit.py model-result-sizes {MODEL_FILES_DIR} {{output.csv}}"


rule notebook:
    input:
        model_results=expand(
            str(MODEL_FILES_DIR / "{name}.netcdf"), name=configuration_names
        ),
        nb="docs/index.ipynb",
        model_sizes=rules.model_result_sizes.output.csv,
    output:
        html="docs/index.html",
    conda:
        "environment.yaml"
    shell:
        "jupyter nbconvert --to notebook --execute --inplace {input.nb} && "
        "jupyter nbconvert --to html {input.nb}"
