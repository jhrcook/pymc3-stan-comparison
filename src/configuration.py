from enum import Enum, unique
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


@unique
class Model(Enum):
    """Available models."""

    SIMPLE_PYMC3 = "SIMPLE_PYMC3"
    SIMPLE_STAN = "SIMPLE_STAN"
    HIERARCHICAL_PYMC3 = "HIERARCHICAL_PYMC3"
    HIERARCHICAL_STAN = "HIERARCHICAL_STAN"


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
