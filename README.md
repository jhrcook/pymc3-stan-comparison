# Comparing performance of PyMC3 and Stan

The goal of this project is to compare the performance between two popular probabilistic programming languages, [Stan](https://mc-stan.org) and [PyMC3](https://docs.pymc.io/en/v3/).

**The results can be found here: [jhrcook.github.io/pymc3-stan-comparison/](https://jhrcook.github.io/pymc3-stan-comparison/)**

**Contributions are welcome!**
To add a new type of model, please see the guide below and feel free to ask for [help](https://github.com/jhrcook/pymc3-stan-comparison/issues).
You can also contribute to the data analysis by editing the analysis notebook: [docs/index.ipynb](docs/index.ipynb).

> This project is functional, but still a work in progress.

## Table of Contents

1. [Process overview](#process-overview)
1. [Contributing](#contributing)
1. [Running the pipeline](#running-the-pipeline)

---

## Process overview

(TODO) - describe the pipeline and configuration system; using snakemake to profile which uses `psutil`.

## Contributing

Any contributions are welcome, particularly for different model types.
Once you have the development environment setup, there are just a few steps to adding a new model to the pipeline.

### Overview

The pipeline uses the configurations in [model-configs.yaml](model-configs.yaml) to know which models to run.
Each model configuration has five parts:

1. `name`: a unique, identifiable name for the configuration
1. `model`: the model that will be run (has multiple configuration options)
1. `mem`: memory (in bytes) to allocate for running the model
1. `time`: time (in `HH:MM:SS`) to allocate for running the model - **max 12 hours**
1. `config`: an arbitrary keyword argument dictionary for configuring the model

The `model` parameter determines which PyMC3 or Stan model to run and the `config` dictionary will be used to configure the data and model.
The `mem` and `time` parameters are for the pipeline to use when profiling the models-fitting processes.

To run an individual model configuration once, pass the name of the configuration to the `fit` command in "fit.py" CLI.
The example below runs the simplest linear regression PyMC3 model:

```bash
./fit.py fit "simple_pymc3_100"
```

### Setup

Setup your Python virtual environment using `conda` with the command below:

```bash
conda env create -f environment.yaml
```

It is recommended to try running the two simplest PyMC3 and Stan models to help check your system is ready:

```bash
./fit.py fit "simple_pymc3_100"
./fit.py fit "simple_stan_100"
```

If either of these fail, please open an [issue](https://github.com/jhrcook/pymc3-stan-comparison/issues) on GitHub.

I recommend creating a new git branch and working on there.
Please give the branch a descriptive name (e.g. if you are adding Gaussian process models name it `gaussian-process`).

```bash
git checkout -b <new-branch-name>
```

### Define a new model

If you stick to a few design guidelines in coding your model, adding it to the pipeline is trivial.
The simplest example of a model is the [simple linear regression](models/simple_linear_regression.py) model – I recommend using this as a guide.

Each Stan and PyMC3 model will have a configuration class and a function called to fit the model.

#### Model configuration class

I decided to use ['pydantic'](https://pydantic-docs.helpmanual.io) for all of the configuration classes to make data parsing and validation easy.
There are several ways to define the configuration classes, but I have found the following pattern to work well and adhere to the [DRY](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself) principle.

First, create a class with the adjustable parameters for your data.
For example, for the simple linear regression model, there is a single parameter `size` that determines the number of data points.

```python3
from pydantic import BaseModel, PositiveInt

class SimpleLinearRegressionDataConfig(BaseModel):
    """Configuration for the data for the simple linear regression model."""

    size: PositiveInt
```

This is one class because the adjustable parameters will be used by both the PyMC3 and Stan models.

Then, use this data configuration class to create configuration classes for each model.
I have created two classes (one for each library) with the basic parameters already included (such as `tune` and `draws`).
Sub-classing from these means that the new configuration class automatically inherits those parameters.


Below are the configuration classes for the PyMC3 and Stan simple linear regression models.
Note that the ellipses `...` are actually used in the code because there are no additional parameters to specify – everything is inherited from `BasePymc3Configuration` and `SimpleLinearRegressionDataConfig`.

```python3
from .sampling_configurations import BasePymc3Configuration, BaseStanConfiguration


class SimplePymc3ModelConfiguration(
    BasePymc3Configuration, SimpleLinearRegressionDataConfig
):
    """Configuration for the Simple PyMC3 model."""

    ...


class SimpleStanModelConfiguration(
    BaseStanConfiguration, SimpleLinearRegressionDataConfig
):
    """Configuration for the Simple PyMC3 model."""

    ...
```

## Running the pipeline

### Setup

```bash
conda env create -f pipeline-environment.yaml
```

On O2, I can run the following command:

```bash
# Made for O2, only.
sbatch run-pipeline.sh
```

Or to run locally:

```bash
snakemake --cores 1 --use-conda
```
