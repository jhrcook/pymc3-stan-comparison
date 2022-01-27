"""Negative binomial models in PyMC3 and Stan."""

from typing import Any, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc3 as pm
import pymc3.math as pmmath
import stan
from pydantic import BaseModel, PositiveFloat, PositiveInt
from scipy import stats

from .models_utils import read_stan_code
from .sampling_configurations import BasePymc3Configuration, BaseStanConfiguration


class NegBinomDataConfig(BaseModel):
    """Configuration for the data generation for the negative binomial models."""

    N: PositiveInt
    alpha: PositiveFloat
    scaler: PositiveFloat = 1.0


class NegBinomPymc3ModelConfiguration(NegBinomDataConfig, BasePymc3Configuration):
    """Configuration for the PyMC3 negative binomial model."""

    ...


class NegBinomStanModelConfiguration(NegBinomDataConfig, BaseStanConfiguration):
    """Configuration for the Stan negative binomial model."""

    ...


def _get_nb_vals(mu: float, alpha: float, size: float) -> np.ndarray:
    """Generate negative binomially distributed samples.

    Generate negative binomially distributed samples by drawing a sample from a gamma
    distribution with mean `mu` and shape parameter `alpha', then drawing from a Poisson
    distribution whose rate parameter is given by the sampled gamma variable.

    Source: PyMC3 GLM negative binomial regression example.

    Args:
        mu (float): Centrality of the distribution.
        alpha (float): Dispersion value.
        size (float): Number of data points.

    Returns:
        np.ndarray: Array of samples from the NB distribution.
    """
    g = stats.gamma.rvs(alpha, scale=mu / alpha, size=size)
    return stats.poisson.rvs(g)


def _generate_data(config: NegBinomDataConfig) -> pd.DataFrame:
    alpha = config.alpha
    N = config.N
    theta_noalcohol_meds = 1 * config.scaler  # no alcohol, took an antihist
    theta_alcohol_meds = 3 * config.scaler  # alcohol, took an antihist
    theta_noalcohol_nomeds = 6 * config.scaler  # no alcohol, no antihist
    theta_alcohol_nomeds = 36 * config.scaler  # alcohol, no antihist
    df = pd.DataFrame(
        {
            "nsneeze": np.concatenate(
                (
                    _get_nb_vals(theta_noalcohol_meds, alpha, N),
                    _get_nb_vals(theta_alcohol_meds, alpha, N),
                    _get_nb_vals(theta_noalcohol_nomeds, alpha, N),
                    _get_nb_vals(theta_alcohol_nomeds, alpha, N),
                )
            ),
            "alcohol": np.concatenate(
                (
                    np.repeat(False, N),
                    np.repeat(True, N),
                    np.repeat(False, N),
                    np.repeat(True, N),
                )
            ),
            "nomeds": np.concatenate(
                (
                    np.repeat(False, N),
                    np.repeat(False, N),
                    np.repeat(True, N),
                    np.repeat(True, N),
                )
            ),
        }
    )
    return df


def _model_matrix(data: pd.DataFrame) -> np.ndarray:
    return (
        data.copy()
        .assign(intercept=1, alch_nomeds=lambda d: d.alcohol * d.nomeds)
        .astype(int)[["intercept", "alcohol", "nomeds", "alch_nomeds"]]
        .values
    )


def negbinom_pymc3_model(config_kwargs: dict[str, Any]) -> az.InferenceData:
    config = NegBinomPymc3ModelConfiguration(**config_kwargs)
    data = _generate_data(config)
    X = _model_matrix(data)

    with pm.Model():
        beta = pm.Normal("beta", 0, 5, shape=(4, 1))
        eta = pm.Deterministic("eta", pmmath.dot(X, beta))
        mu = pmmath.exp(eta)
        alpha = pm.HalfCauchy("alpha", 10)
        y = pm.NegativeBinomial("y", mu, alpha, observed=data.nsneeze)  # noqa: F841

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
    X = _model_matrix(data)

    stan_data: dict[str, Union[int, np.ndarray]] = {
        "N": data.shape[0],
        "K": 4,
        "X": X,
        "y": np.array(data["nsneeze"].values),
    }
    model = stan.build(_stan_code(), data=stan_data)
    fit = model.sample(
        num_chains=config.chains, num_samples=config.draws, num_warmup=config.tune
    )
    trace = az.from_pystan(posterior=fit, **_stan_idata())
    return trace
