import numpy as np
import pymc3 as pm
from pydantic import BaseModel, PositiveInt


class SimplePymc3ModelConfiguration(BaseModel):
    """Configuration for the Simple PyMC3 model."""

    size: PositiveInt
    tune: PositiveInt
    draws: PositiveInt


def simple_pymc3_model(config: SimplePymc3ModelConfiguration) -> None:
    size = config.size

    x_data = np.random.normal(0, 1, size=size)
    y_data = 2.4 + 3 * x_data + np.random.normal(0, 0.2, size=size)

    with pm.Model():
        a = pm.Normal("a", 0, 5)
        b = pm.Normal("b", 0, 5)
        mu = pm.Deterministic("mu", a + b * x_data)
        sigma = pm.HalfNormal("sigma", 5)
        y = pm.Normal("y", mu, sigma, observed=y_data)  # noqa: F841

        trace = pm.sample(  # noqa: F841
            draws=config.draws, tune=config.tune, return_inferencedata=True
        )
    return None
