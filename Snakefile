from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml
from snakemake.io import Wildcards

from src.pipeline_utils import get_theano_compdir, get_configuration_information

# ---- Configure ----

N_PROFILE_REPS: Final[int] = 10
CONFIG_FILE = Path("model-configs.yaml")

# ---- Setup ----

configurations = get_configuration_information(CONFIG_FILE)
configuration_names = list(configurations.keys())


def _get_config_params(w: Wildcards) -> dict[str, str]:
    return configurations[w.name]


def get_config_mem(w: Wildcards) -> str:
    return _get_config_params(w)["mem"]


def get_config_time(w: Wildcards) -> str:
    return _get_config_params(w)["time"]


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
        res="model-results/{name}.pkl",
    benchmark:
        repeat("benchmarks/{name}.tsv", N_PROFILE_REPS)
    conda:
        "environment.yml"
    params:
        mem=lambda w: get_config_mem(w),
        time=lambda w: get_config_time(w),
        theano_dir=get_theano_compdir,
    shell:
        "{params.theano_dir}" + "./fit.py fit {wildcards.name}"


rule model_result_sizes:
    input:
        model_results=expand("model-results/{name}.pkl", name=configuration_names),
    output:
        csv="model-result-file-sizes.csv",
    conda:
        "environment.yml"
    shell:
        "./fit.py model-result-sizes 'model-results' {output.csv}"


rule notebook:
    input:
        model_results=expand("model-results/{name}.pkl", name=configuration_names),
        nb="docs/index.ipynb",
        model_sizes=rules.model_result_sizes.output.csv,
    output:
        html="docs/index.html",
    conda:
        "environment.yml"
    shell:
        "jupyter nbconvert --to notebook --execute --inplace {input.nb} && "
        "jupyter nbconvert --to html {input.nb}"
