from pathlib import Path

import yaml

CONFIG_FILE = Path("model-configs.yaml")

with open(CONFIG_FILE, "r") as file:
    configuration_names = [config["name"] for config in yaml.safe_load(file)]


rule all:
    input:
        expand("model-results/{name}.pkl", name=configuration_names),


rule fit_model:
    output:
        res="model-results/{name}.pkl",
    benchmark:
        repeat("benchmarks/{name}.tsv", 3)
    conda:
        "environment.yml"
    shell:
        "./fit.py {wildcards.name}"
