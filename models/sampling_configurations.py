"""Model sampling configuration."""


from pydantic import BaseModel, PositiveInt


class BaseMcmcSamplingConfiguration(BaseModel):
    """Base MCMC sampling parameters."""

    tune: PositiveInt = 1000
    draws: PositiveInt = 1000
    chains: PositiveInt = 4
    cores: PositiveInt = 4


class BasePymc3Configuration(BaseMcmcSamplingConfiguration):
    """Base configuration for PyMC3 models."""

    init: str = "auto"
    n_init: int = 200000


class BaseStanConfiguration(BaseMcmcSamplingConfiguration):
    """Base configuration for Stan models."""

    ...
