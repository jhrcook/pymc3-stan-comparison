import os
from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml
from snakemake.io import Wildcards, touch
from dotenv import load_dotenv

from src.pipeline_utils import get_theano_compdir, get_configuration_information

load_dotenv()

# ---- Configure ----

N_PROFILE_REPS: int = int(os.environ.get("N_PROFILE_REPS", 5))
CONFIG_FILE: Path = Path(os.environ["CONFIG_FILE"])
MODEL_FILES_DIR: Path = Path(os.environ.get("MODEL_FILES_DIR", None))

# ---- Setup ----

if not MODEL_FILES_DIR.exists():
    MODEL_FILES_DIR.mkdir()

configurations = get_configuration_information(CONFIG_FILE)
configuration_names = list(configurations.keys())


def _get_config_params(w: Wildcards) -> dict[str, str]:
    return configurations[w.name]


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
    check_model_outputs,


rule all:
    input:
        notebook_html="docs/index.html",


rule fit_model:
    output:
        res=f"{MODEL_FILES_DIR}/{{name}}.pkl",
    benchmark:
        repeat("benchmarks/{name}.tsv", N_PROFILE_REPS)
    conda:
        "environment.yaml"
    params:
        mem=lambda w: get_config_mem(w),
        time=lambda w: get_config_time(w),
        partition=lambda w: get_config_partition(w),
        theano_dir=get_theano_compdir,
    shell:
        f"{{params.theano_dir}} ./fit.py fit {{wildcards.name}} --config-file={CONFIG_FILE} --save-dir={MODEL_FILES_DIR}"


rule check_model_outputs:
    input:
        model_results=expand(
            str(MODEL_FILES_DIR / "{name}.pkl"), name=configuration_names
        ),
        config=CONFIG_FILE,
    conda:
        "environment.yaml"
    output:
        touch_file=touch(".check-model-outputs.touch"),
    shell:
        f"./fit.py check-benchmarks-and-model-files benchmarks {MODEL_FILES_DIR} --config-file={{input.config}} --no-prune"


rule model_result_sizes:
    input:
        mdl_file_check=rules.check_model_outputs.output.touch_file,
        model_results=expand(
            str(MODEL_FILES_DIR / "{name}.pkl"), name=configuration_names
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
            str(MODEL_FILES_DIR / "{name}.pkl"), name=configuration_names
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
