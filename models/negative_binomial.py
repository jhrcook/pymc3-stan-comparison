"""Negative binomial models in PyMC3 and Stan."""

from dataclasses import dataclass
from typing import Any, Optional, Union

import arviz as az
import numpy as np
import pymc3 as pm
import pymc3.math as pmmath
import stan
from pydantic import BaseModel, PositiveFloat, PositiveInt
from scipy import stats

from .models_utils import read_stan_code
from .sampling_configurations import BasePymc3Configuration, BaseStanConfiguration


class NegBinomDataConfig(BaseModel):
    """Configuration for the data generation for the negative binomial models."""

    N: PositiveInt  # number of data points
    K: PositiveInt  # number of covariates
    alpha: PositiveFloat


class NegBinomPymc3ModelConfiguration(NegBinomDataConfig, BasePymc3Configuration):
    """Configuration for the PyMC3 negative binomial model."""

    ...


class NegBinomStanModelConfiguration(NegBinomDataConfig, BaseStanConfiguration):
    """Configuration for the Stan negative binomial model."""

    ...


def _get_nb_vals(
    mu: Union[float, np.ndarray], alpha: float, size: Optional[float] = None
) -> np.ndarray:
    """Generate negative binomially distributed samples.

    Generate negative binomially distributed samples by drawing a sample from a gamma
    distribution with mean `mu` and shape parameter `alpha', then drawing from a Poisson
    distribution whose rate parameter is given by the sampled gamma variable.

    Source: PyMC3 GLM negative binomial regression example.

    Args:
        mu (Union[float, np.ndarray]): Centrality of the distribution.
        alpha (float): Dispersion value.
        size (Optional[float]): Number of data points. If None, then the length of `mu`
        is used.

    Returns:
        np.ndarray: Array of samples from the NB distribution.
    """
    if size is None:
        if isinstance(mu, float):
            size = 1
        else:
            size = len(mu)

    g = stats.gamma.rvs(alpha, scale=mu / alpha, size=size)
    return stats.poisson.rvs(g)


@dataclass
class NegativeBinomialData:
    """Data for the negative binomial models."""

    X: np.ndarray
    y: np.ndarray


def _generate_data(config: NegBinomDataConfig) -> NegativeBinomialData:
    N = config.N
    K = config.K
    X = np.random.normal(0, 1, size=(N, K))
    X[:, 0] = 1.0
    beta = np.random.uniform(-3, 10, size=(K, 1))
    mu = np.exp(np.dot(X, beta)).flatten()
    y = _get_nb_vals(mu, alpha=config.alpha)
    return NegativeBinomialData(X=X, y=y)


def negbinom_pymc3_model(config_kwargs: dict[str, Any]) -> az.InferenceData:
    config = NegBinomPymc3ModelConfiguration(**config_kwargs)
    data = _generate_data(config)

    with pm.Model():
        beta = pm.Normal("beta", 0, 5, shape=(config.K, 1))
        eta = pm.Deterministic("eta", pmmath.dot(data.X, beta))
        mu = pmmath.exp(eta)
        alpha = pm.HalfCauchy("alpha", 10)
        y = pm.NegativeBinomial("y", mu, alpha, observed=data.y)  # noqa: F841

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


def _stan_code() -> str:
    return read_stan_code("negative_binomial")


def _stan_idata() -> dict[str, Any]:
    return {
        "posterior_predictive": "y_hat",
        "observed_data": ["y"],
        "constant_data": ["X"],
        "log_likelihood": {"y": "log_lik"},
    }


def negbinom_stan_model(config_kwargs: dict[str, Any]) -> az.InferenceData:
    config = NegBinomStanModelConfiguration(**config_kwargs)
    data = _generate_data(config)

    stan_data: dict[str, Union[int, np.ndarray]] = {
        "N": config.N,
        "K": config.K,
        "X": data.X,
        "y": data.y,
    }
    model = stan.build(_stan_code(), data=stan_data)
    fit = model.sample(
        num_chains=config.chains, num_samples=config.draws, num_warmup=config.tune
    )
    trace = az.from_pystan(posterior=fit, **_stan_idata())
    return trace
