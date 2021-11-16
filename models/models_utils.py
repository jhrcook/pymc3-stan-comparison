import pickle
from pathlib import Path
from typing import Union

import arviz as az
import httpstan.cache
import stan.fit
from stan.model import Model as StanModel


def delete_stan_build(mdl: StanModel) -> None:
    httpstan.cache.delete_model_directory(mdl.model_name)
    return None


def write_results(name: str, posterior: Union[stan.fit.Fit, az.InferenceData]) -> None:
    out_path = Path("model-results") / f"{name}.pkl"
    with open(out_path, "wb") as file:
        pickle.dump(posterior, file)
    return None
