"""Model utilities."""

from pathlib import Path

import httpstan.cache
from stan.model import Model as StanModel


def delete_stan_build(mdl: StanModel) -> None:
    httpstan.cache.delete_model_directory(mdl.model_name)
    return None


def read_stan_code(name: str) -> str:
    """Read Stan code from file to a string.

    Args:
        name (str): Name of the Stan code file (no extension, assumed to be '.stan').

    Raises:
        FileNotFoundError: If Stan file is not found.

    Returns:
        str: Stan code as a string.
    """
    stan_dir_path = Path(__file__).parent
    stan_file_path = stan_dir_path / f"{name}.stan"

    if not stan_file_path.exists():
        raise FileNotFoundError(f"No Stan file: '{str(stan_file_path)}")

    with open(stan_file_path, "r") as file:
        code = "".join(list(file))
    return code
