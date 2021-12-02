"""Configuration of the models."""

import os
import warnings
from enum import Enum, unique
from pathlib import Path
from typing import Any, Callable, Final, Optional

import arviz as az
import yaml
from pydantic import BaseModel, Field

from models import hierarchical_model as hier
from models import simple_linear_regression as lm
from models import two_teir_hierarchical as tth


@unique
class Model(Enum):
    """Available models."""

    SIMPLE_PYMC3 = "SIMPLE_PYMC3"
    SIMPLE_STAN = "SIMPLE_STAN"
    HIERARCHICAL_PYMC3 = "HIERARCHICAL_PYMC3"
    HIERARCHICAL_STAN = "HIERARCHICAL_STAN"
    TWOTIER_PYMC3 = "TWOTIER_PYMC3"
    TWOTIER_STAN = "TWOTIER_STAN"


ModelCallable = Callable[[dict[str, Any]], az.InferenceData]

MODEL_CALLER_MAP: Final[dict[Model, ModelCallable]] = {
    Model.SIMPLE_PYMC3: lm.simple_pymc3_model,
    Model.SIMPLE_STAN: lm.simple_stan_model,
    Model.HIERARCHICAL_PYMC3: hier.hierarchical_pymc3_model,
    Model.HIERARCHICAL_STAN: hier.hierarchical_stan_model,
    Model.TWOTIER_PYMC3: tth.two_tier_hierarchical_pymc3_model,
    Model.TWOTIER_STAN: tth.two_tier_hierarchical_stan_model,
}


class ModelNotInCallerLookupMap(BaseException):
    """Model not in caller lookup map."""

    def __init__(self, model: Model) -> None:
        msg = f"Need to add '{model.value}' to the lookup dict 'MODEL_CALLER_MAP'."
        super().__init__(msg)
        return None


class ModelConfiguration(BaseModel):
    """Model configuration format."""

    name: str
    model: Model
    config: dict[str, Any] = Field(default_factory=dict)


class ModelConfigurations(BaseModel):
    """Format of model configuration file."""

    configurations: list[ModelConfiguration] = Field(default_factory=list)


class UnexpectedNumberOfConfigurations(BaseException):
    """A non-1 number of configurations were located."""

    def __init__(self, n: int, name: str) -> None:
        msg = f"Found {n} configurations with name '{name}'."
        super().__init__(msg)
        return None


def read_configuration_file(config_file: Path) -> ModelConfigurations:
    """Read in a configuration file.

    Args:
        config_file (Path): Path to the configuration file.

    Returns:
        ModelConfigurations: Model configurations from file.
    """
    with open(config_file, "r") as file:
        yaml_data = yaml.safe_load(file)
        configs = ModelConfigurations(
            configurations=[ModelConfiguration(**x) for x in yaml_data]
        )
    return configs


def get_model_configuration(name: str, config_file: Path) -> ModelConfiguration:
    """Get a model configuration from a config file.

    Args:
        name (str): Name of the configuration.
        config_file (Path): Path to the configuration file.

    Raises:
        BaseException: If multiple configurations have the name in the file.

    Returns:
        ModelConfiguration: Model configuration from the config file.
    """
    configs = read_configuration_file(config_file=config_file)
    config = [c for c in configs.configurations if c.name == name]
    if len(config) == 1:
        return config[0]
    else:
        raise UnexpectedNumberOfConfigurations(n=len(config), name=name)


def get_model_callable(mdl_config: ModelConfiguration) -> ModelCallable:
    """Get a callable function for a model.

    Args:
        mdl_config (ModelConfiguration): Model configuration.

    Raises:
        ModelNotInCallerLookupMap: If the requested model is not in the lookup table.

    Returns:
        ModelCallable: Function to call to fit the model in the configuration.
    """
    if mdl_config.model not in MODEL_CALLER_MAP:
        raise ModelNotInCallerLookupMap(mdl_config.model)
    return MODEL_CALLER_MAP[mdl_config.model]


def default_config() -> Optional[Path]:
    """Get the configuration file path from an environment variable.

    Returns:
        Optional[Path]: Path to the configuration file if it exists.
    """
    p = os.getenv("CONFIG_FILE")
    if p is None:
        return p
    warnings.warn("Using default configuration file.")
    return Path(p)
