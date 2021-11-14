#!/usr/bin/env python3

from enum import Enum, unique
from pathlib import Path
from typing import Any, NoReturn, Optional

import numpy as np
import pymc3 as pm
import yaml
from pydantic import BaseModel, Field, PositiveInt
from typer import Typer

app = Typer()


def assert_never(value: NoReturn) -> NoReturn:
    """Force runtime and static enumeration exhaustiveness.
    Args:
        value (NoReturn): Some value passed as an enum value.
    Returns:
        NoReturn: Nothing.
    """
    assert False, f"Unhandled value: {value} ({type(value).__name__})"  # noqa: B011


class SimplePymc3ModelConfiguration(BaseModel):
    """Configuration for the Simple PyMC3 model."""

    size: PositiveInt
    tune: PositiveInt
    draws: PositiveInt


def simple_pymc3_model(config: SimplePymc3ModelConfiguration) -> None:
    size = config.size

    x_data = np.random.normal(0, 1, size=size)
    y_data = 2.4 + 3 * x_data + np.random.normal(0, 0.2, size=size)

    with pm.Model():
        a = pm.Normal("a", 0, 5)
        b = pm.Normal("b", 0, 5)
        mu = pm.Deterministic("mu", a + b * x_data)
        sigma = pm.HalfNormal("sigma", 5)
        y = pm.Normal("y", mu, sigma, observed=y_data)  # noqa: F841

        trace = pm.sample(  # noqa: F841
            draws=config.draws, tune=config.tune, return_inferencedata=True
        )
    return None


@unique
class Model(Enum):
    """Available models."""

    SIMPLE_PYMC3 = "SIMPLE_PYMC3"
    SIMPLE_STAN = "SIMPLE_STAN"


class ModelConfiguration(BaseModel):
    """Model configuration format."""

    name: str
    model: Model
    config: dict[str, Any] = Field(default_factory=dict)


class ModelConfigurations(BaseModel):
    """Format of model configuration file."""

    configurations: list[ModelConfiguration] = Field(default_factory=list)


def read_configuration_file(config_file: Path) -> ModelConfigurations:
    with open(config_file, "r") as file:
        yaml_data = yaml.safe_load(file)
        configs = ModelConfigurations(
            configurations=[ModelConfiguration(**x) for x in yaml_data]
        )
    return configs


def get_model_configuration(name: str, config_file: Path) -> ModelConfiguration:
    configs = read_configuration_file(config_file=config_file)
    config = [c for c in configs.configurations if c.name == name]
    if len(config) == 1:
        return config[0]
    else:
        raise BaseException(f"Found {len(config)} configurations with name '{name}'.")


@app.command()
def main(config_name: str, config_file: Optional[Path] = None) -> None:
    if config_file is None:
        config_file = Path("model-configs.yaml")
    mdl_config = get_model_configuration(config_name, config_file)

    if mdl_config.model is Model.SIMPLE_PYMC3:
        simple_pymc3_model(SimplePymc3ModelConfiguration(**mdl_config.config))
    elif mdl_config.model is Model.SIMPLE_STAN:
        raise NotImplementedError()
    else:
        assert_never(mdl_config.model)

    return None


if __name__ == "__main__":
    app()
