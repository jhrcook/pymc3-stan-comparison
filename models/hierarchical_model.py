"""A relatively simple hierarchical model."""

from typing import Any, Union

import arviz as az
import numpy as np
import pymc3 as pm
import stan
from pydantic import BaseModel, PositiveInt

from .models_utils import write_results
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
    mu_beta = np.random.uniform(-10, 10, K)
    sigma_beta = np.random.uniform(1, 10, K)
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


def hierarchical_pymc3_model(name: str, config_kwargs: dict[str, Any]) -> None:
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
    write_results(name, trace)
    return None


_stan_model = """
data {
    int<lower=1> N;               // number of observations total
    int<lower=1> J;               // number of groups
    int<lower=1> K;               // number of features
    int<lower=1,upper=J> idx[N];  // group indices
    matrix[N,K] X;                // model matrix
    vector[N] y;                  // response variable
}

parameters {
    vector[K] mu_beta;
    vector<lower=0>[K] mu_sigma;
    vector[K] delta_beta[J];
    real<lower=0> sigma;
}

transformed parameters {
    //* Non-centered parameterization.
    vector[K] beta[J];
    for(j in 1:J) {
        beta[j] = mu_beta + mu_sigma .* delta_beta[j];
    }
}

model {
    vector[N] mu;
    mu_beta ~ normal(0, 5);
    mu_sigma ~ cauchy(0, 2.5);
    sigma ~ gamma(2, 0.1);
    for(j in 1:J) {
        delta_beta[j] ~ normal(0, 1);
    }
    for(n in 1:N) {
        mu[n] = X[n] * beta[idx[n]];
    }
    y ~ normal(mu, sigma);
}
"""


class HierarchicalStanModelConfiguration(HierarchicalDataConfig, BaseStanConfiguration):

    ...


def hierarchical_stan_model(name: str, config_kwargs: dict[str, Any]) -> None:
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

    model = stan.build(_stan_model, data=stan_data)
    trace = model.sample(
        num_chains=config.chains, num_samples=config.draws, num_warmup=config.tune
    )
    write_results(name, trace)
    return None
