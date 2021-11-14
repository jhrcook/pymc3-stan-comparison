#!/usr/bin/env python3

from pathlib import Path
from typing import NoReturn, Optional

from typer import Typer

from models.simple_pymc3_model import SimplePymc3ModelConfiguration, simple_pymc3_model
from src.configuration import Model, get_model_configuration

app = Typer()


def assert_never(value: NoReturn) -> NoReturn:
    """Force runtime and static enumeration exhaustiveness.
    Args:
        value (NoReturn): Some value passed as an enum value.
    Returns:
        NoReturn: Nothing.
    """
    assert False, f"Unhandled value: {value} ({type(value).__name__})"  # noqa: B011


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
