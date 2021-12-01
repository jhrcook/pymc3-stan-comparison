"""2-tier hierarchical model with 2-dimensional parameters."""

from dataclasses import dataclass
from itertools import product
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc3 as pm
from pydantic import BaseModel, PositiveInt

from .models_utils import write_results
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
    sigma = np.abs(np.random.normal(0, 0.5))

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


@dataclass
class DataIndices:

    a_idx: np.ndarray
    b_idx: np.ndarray
    a_c_idx: np.ndarray
    b_d_idx: np.ndarray


def _get_data_indices(data: pd.DataFrame) -> DataIndices:
    a_idx = np.array(data["a"].values.flatten())
    b_idx = np.array(data["b"].values.flatten())
    a_c_idx = np.array(
        data.copy()[["a", "c"]].drop_duplicates().sort_values("a")["c"].values.flatten()
    )
    b_d_idx = np.array(
        data.copy()[["b", "d"]].drop_duplicates().sort_values("b")["d"].values.flatten()
    )
    return DataIndices(a_idx=a_idx, b_idx=b_idx, a_c_idx=a_c_idx, b_d_idx=b_d_idx)


# ---- PyMC3 model ----


class TwoTierHierPymc3ModelConfiguration(
    TwoTierHierarchicalDataConfig, BasePymc3Configuration
):

    ...


def two_tier_hierarchical_pymc3_model(name: str, config_kwargs: dict[str, Any]) -> None:
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
    write_results(name, trace)

    return None


# ---- Stan model ----


class TwoTierHierStanModelConfiguration(
    TwoTierHierarchicalDataConfig, BaseStanConfiguration
):

    ...


def two_tier_hierarchical_stan_model(name: str, config_kwargs: dict[str, Any]) -> None:
    config = TwoTierHierarchicalDataConfig(**config_kwargs)
    data = _generate_data(config)
    print(data)
    raise NotImplementedError()
    return None
