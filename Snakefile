from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml

from src.pipeline_utils import get_theano_compdir, get_configuration_names

# ---- Configure ----

N_PROFILE_REPS: Final[int] = 10
CONFIG_FILE = Path("model-configs.yaml")

# ---- Setup ----

configuration_names = get_configuration_names(CONFIG_FILE)

# ---- Rules ----


# Steps to run locally.
localrules:
    all,
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
        theano_dir=get_theano_compdir,
    shell:
        "{params.theano_dir}" + "./fit.py {wildcards.name}"


rule notebook:
    input:
        model_results=expand("model-results/{name}.pkl", name=configuration_names),
        nb="docs/index.ipynb",
    output:
        html="docs/index.html",
    conda:
        "environment.yml"
    shell:
        "jupyter nbconvert --to notebook --inplace {input.nb} && "
        "jupyter nbconvert --to html {input.nb}"
