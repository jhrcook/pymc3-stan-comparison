"""Simple linear regression PyMC3 and Stan models."""

from typing import Any, Union

import arviz as az
import numpy as np
import pymc3 as pm
import stan
import stan.fit
from pydantic import BaseModel, PositiveInt

from .models_utils import read_stan_code
from .sampling_configurations import BasePymc3Configuration, BaseStanConfiguration

# ---- Data generation ----


def _generate_data(size: int) -> dict[str, np.ndarray]:
    x = np.random.normal(0, 5, size=size)
    m = np.random.normal(5, 2.5)
    b = np.random.normal(0, 2.5)
    y = b + (m * x) + np.random.normal(0, 1, size=size)
    return {"x": x, "y": y}


class SimpleLinearRegressionDataConfig(BaseModel):
    """Configuration for the data for the simple linear regression model."""

    size: PositiveInt


# ---- PyMC3 ----


class SimplePymc3ModelConfiguration(
    BasePymc3Configuration, SimpleLinearRegressionDataConfig
):
    """Configuration for the Simple PyMC3 model."""

    ...


def simple_pymc3_model(config_kwargs: dict[str, Any]) -> az.InferenceData:
    config = SimplePymc3ModelConfiguration(**config_kwargs)
    data = _generate_data(config.size)

    with pm.Model():
        a = pm.Normal("a", 0, 5)
        b = pm.Normal("b", 0, 5)
        mu = a + b * data["x"]
        sigma = pm.HalfNormal("sigma", 5)
        y = pm.Normal("y", mu, sigma, observed=data["y"])  # noqa: F841

        trace = pm.sample(
            draws=config.draws,
            tune=config.tune,
            init=config.init,
            n_init=config.n_init,
            chains=config.chains,
            cores=config.cores,
            return_inferencedata=True,
        )
    assert isinstance(trace, az.InferenceData)
    return trace


# ---- Stan ----


class SimpleStanModelConfiguration(
    BaseStanConfiguration, SimpleLinearRegressionDataConfig
):
    """Configuration for the Simple PyMC3 model."""

    ...


def _stan_code() -> str:
    return read_stan_code("simple_linear_regression")


def _stan_idata() -> dict[str, Any]:
    return {
        "posterior_predictive": "y_hat",
        "observed_data": ["y"],
        "constant_data": ["x"],
        "log_likelihood": {"y": "log_lik"},
    }


def simple_stan_model(config_kwargs: dict[str, Any]) -> az.InferenceData:
    config = SimpleStanModelConfiguration(**config_kwargs)
    data = _generate_data(config.size)
    stan_data: dict[str, Union[int, np.ndarray]] = {"N": config.size}
    for p in ["x", "y"]:
        stan_data[p] = data[p].tolist()

    model = stan.build(_stan_code(), data=stan_data)
    fit = model.sample(
        num_chains=config.chains, num_samples=config.draws, num_warmup=config.tune
    )
    trace = az.from_pystan(posterior=fit, **_stan_idata())
    return trace
