"""2-tier hierarchical model with 2-dimensional parameters."""

from itertools import product
from typing import Any, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc3 as pm
import stan
from pydantic import BaseModel, PositiveInt

from .sampling_configurations import BasePymc3Configuration, BaseStanConfiguration

# ---- Data ----


class TwoTierHierarchicalDataConfig(BaseModel):
    N: PositiveInt  # number of data points per group
    dims: list[PositiveInt]

    def get_dims(self) -> tuple[int, int, int, int]:
        return self.dims[0], self.dims[1], self.dims[2], self.dims[3]


def _generate_data(config: TwoTierHierarchicalDataConfig) -> pd.DataFrame:
    n_data_pts = config.N
    a, b, c, d = config.dims

    assert a > c, "There must be more 'a' groups than 'c' groups."
    assert b > d, "There must be more 'b' groups than 'd' groups."

    mu_mu_alpha = np.random.randn(1)
    sigma_mu_alpha = np.abs(np.random.randn(1))
    mu_alpha = np.random.normal(mu_mu_alpha, sigma_mu_alpha, size=(c, d))
    sigma_alpha = np.abs(np.random.randn(1))
    sigma = np.abs(np.random.normal(0, 0.2))

    a_to_c_idx = np.tile(np.arange(c), int(np.ceil(a / c)))[:a]
    a_to_c_idx.sort()
    b_to_d_idx = np.tile(np.arange(d), int(np.ceil(b / d)))[:b]
    b_to_d_idx.sort()

    alpha = np.empty(shape=(a, b), dtype=float)
    for i, c_idx in enumerate(a_to_c_idx):
        for j, d_idx in enumerate(b_to_d_idx):
            alpha[i, j] = np.random.normal(mu_alpha[c_idx, d_idx], sigma_alpha)

    values = np.array(
        [
            np.random.normal(alpha[i, j], sigma, size=n_data_pts)
            for i, j in product(range(a), range(b))
        ]
    )

    df = pd.DataFrame({"y": values.flatten()})
    a_col = np.array([np.repeat(i, n_data_pts) for i, _ in product(range(a), range(b))])
    df["a"] = a_col.flatten()
    b_col = np.array([np.repeat(j, n_data_pts) for _, j in product(range(a), range(b))])
    df["b"] = b_col.flatten()
    df["c"] = [a_to_c_idx[i] for i in df["a"]]
    df["d"] = [b_to_d_idx[i] for i in df["b"]]
    return df


class DataIndices(BaseModel):

    a_idx: list[int]
    b_idx: list[int]
    a_c_idx: list[int]
    b_d_idx: list[int]


def _get_data_indices(data: pd.DataFrame, idx_plus: int = 0) -> DataIndices:
    def _prep(x: pd.Series) -> list[int]:
        return (x.values + idx_plus).flatten().astype(int).tolist()

    a_idx = data["a"]
    b_idx = data["b"]
    a_c_idx = data.copy()[["a", "c"]].drop_duplicates().sort_values("a")["c"]

    b_d_idx = data.copy()[["b", "d"]].drop_duplicates().sort_values("b")["d"]

    return DataIndices(
        a_idx=_prep(a_idx),
        b_idx=_prep(b_idx),
        a_c_idx=_prep(a_c_idx),
        b_d_idx=_prep(b_d_idx),
    )


# ---- PyMC3 model ----


class TwoTierHierPymc3ModelConfiguration(
    TwoTierHierarchicalDataConfig, BasePymc3Configuration
):

    ...


def two_tier_hierarchical_pymc3_model(
    config_kwargs: dict[str, Any]
) -> az.InferenceData:
    config = TwoTierHierPymc3ModelConfiguration(**config_kwargs)
    data = _generate_data(config)

    a, b, c, d = config.get_dims()
    idx = _get_data_indices(data.copy())

    with pm.Model():
        mu_mu_alpha = pm.Normal("mu_mu_alpha", 0, 1)
        sigma_mu_alpha = pm.HalfNormal("sigma_mu_alpha", 1)
        mu_alpha = pm.Normal("mu_alpha", mu_mu_alpha, sigma_mu_alpha, shape=(c, d))
        sigma_alpha = pm.HalfNormal("sigma_alpha", 1)
        alpha = pm.Normal(
            "alpha", mu_alpha[idx.a_c_idx, :][:, idx.b_d_idx], sigma_alpha, shape=(a, b)
        )
        mu = pm.Deterministic("mu", alpha[idx.a_idx, idx.b_idx])
        sigma = pm.HalfNormal("sigma", 2)
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


# ---- Stan model ----


class TwoTierHierStanModelConfiguration(
    TwoTierHierarchicalDataConfig, BaseStanConfiguration
):

    ...


_stan_model = """
data {
    int<lower=1> N;  // number of observations total

    int<lower=1> A;
    int<lower=1> B;
    int<lower=1> C;
    int<lower=1> D;
    int<lower=1,upper=A> a_idx[N];
    int<lower=1,upper=B> b_idx[N];
    int<lower=1,upper=C> a_c_idx[A];
    int<lower=1,upper=D> b_d_idx[B];

    vector[N] y;
}

parameters {
    real mu_mu_alpha;
    real<lower=0> sigma_mu_alpha;

    matrix[C, D] mu_alpha;
    real<lower=0> sigma_alpha;

    matrix[A, B] alpha;
    real<lower=0> sigma;
}

model {
    vector[N] mu;

    mu_mu_alpha ~ normal(0, 1);
    sigma_mu_alpha ~ normal(0, 1);
    sigma_alpha ~ normal(0, 1);
    sigma ~ normal(0, 2);

    for (c in 1:C) {
        for (d in 1:D) {
            mu_alpha[c, d] ~ normal(mu_mu_alpha, sigma_mu_alpha);
        }
    }

    for (a in 1:A) {
        for (b in 1:B) {
            alpha[a, b] ~ normal(mu_alpha[a_c_idx[a], b_d_idx[b]], sigma_alpha);
        }
    }

    for(n in 1:N) {
        mu[n] = alpha[a_idx[n], b_idx[n]];
    }

    y ~ normal(mu, sigma);
}
"""


def two_tier_hierarchical_stan_model(config_kwargs: dict[str, Any]) -> az.InferenceData:
    config = TwoTierHierStanModelConfiguration(**config_kwargs)
    data = _generate_data(config)

    stan_data: dict[str, Union[int, np.ndarray]] = {
        "N": data.shape[0],
        "y": np.array(data["y"].values),
    }

    for n, value in zip(["A", "B", "C", "D"], config.get_dims()):
        stan_data[n] = value

    for n, idx in _get_data_indices(data, idx_plus=1).dict().items():
        stan_data[n] = idx

    model = stan.build(_stan_model, data=stan_data)
    fit = model.sample(
        num_chains=config.chains, num_samples=config.draws, num_warmup=config.tune
    )
    trace = az.from_pystan(posterior=fit)
    return trace
