"""A relatively simple hierarchical model."""

from typing import Any, Union

import arviz as az
import numpy as np
import pymc3 as pm
import stan
from pydantic import BaseModel, PositiveInt

from .models_utils import read_stan_code
from .sampling_configurations import BasePymc3Configuration, BaseStanConfiguration


class HierarchicalDataConfig(BaseModel):
    N: PositiveInt  # number of data points per group
    J: PositiveInt  # number of groups
    K: PositiveInt = 1  # number of features


class HierarchicalPymc3ModelConfiguration(
    HierarchicalDataConfig, BasePymc3Configuration
):

    ...


def _generate_data(config: HierarchicalDataConfig) -> dict[str, np.ndarray]:
    N = config.N  # number of data points per group
    J = config.J  # number of groups
    K = config.K  # number of features

    idx = np.repeat(np.arange(J), N)
    mu_beta = np.random.uniform(-3, 3, K)
    sigma_beta = np.random.uniform(1, 3, K)
    sigma = np.random.gamma(2.0, 2.0, 1)

    beta = np.hstack(
        [np.random.normal(g, t, (J, 1)) for g, t in zip(mu_beta, sigma_beta)]
    )

    total_data_pts = N * J
    X = np.hstack(
        [
            np.ones((total_data_pts, 1)),
            np.random.uniform(-2, 2, (total_data_pts, K - 1)),
        ]
    )
    y = np.hstack(
        [
            np.random.normal(np.dot(X[i, :], beta[idx[i], :]), sigma, 1)
            for i in range(total_data_pts)
        ]
    ).flatten()
    return {"X": X, "y": y, "idx": idx}


def hierarchical_pymc3_model(config_kwargs: dict[str, Any]) -> az.InferenceData:
    config = HierarchicalPymc3ModelConfiguration(**config_kwargs)
    data = _generate_data(config)
    idx = data["idx"]
    beta_size = (config.J, config.K)
    with pm.Model():
        mu_beta = pm.Normal("mu_beta", 0, 5, shape=config.K)
        sigma_beta = pm.HalfCauchy("sigma_beta", 2.5, shape=config.K)
        sigma = pm.Gamma("sigma", 2, 0.1)

        delta_beta = pm.Normal("delta_beta", 0, 1, shape=beta_size)
        beta = pm.Deterministic("beta", mu_beta + sigma_beta * delta_beta)

        mu = data["X"] * beta[idx, :]
        y = pm.Normal("y", mu.sum(axis=1), sigma, observed=data["y"])  # noqa: F841

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


class HierarchicalStanModelConfiguration(HierarchicalDataConfig, BaseStanConfiguration):

    ...


def _stan_code() -> str:
    return read_stan_code("hierarchical_model")


def _stan_idata() -> dict[str, Any]:
    return {
        "posterior_predictive": "y_hat",
        "observed_data": ["y"],
        "constant_data": ["x"],
        "log_likelihood": {"y": "log_lik"},
    }


def hierarchical_stan_model(config_kwargs: dict[str, Any]) -> az.InferenceData:
    config = HierarchicalStanModelConfiguration(**config_kwargs)
    data = _generate_data(config)
    stan_data: dict[str, Union[int, np.ndarray]] = {
        "N": int(config.N * config.J),
        "J": config.J,
        "K": config.K,
        "idx": (data["idx"] + 1).astype(int),
    }
    for p in ["X", "y"]:
        stan_data[p] = data[p]

    model = stan.build(_stan_code(), data=stan_data)
    fit = model.sample(  # noqa: F841
        num_chains=config.chains, num_samples=config.draws, num_warmup=config.tune
    )
    trace = az.from_pystan(posterior=fit, **_stan_idata())
    return trace
