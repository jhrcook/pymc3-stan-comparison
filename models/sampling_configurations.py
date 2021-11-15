"""Model sampling configuration."""


from pydantic import BaseModel, PositiveInt


class BaseMcmcSamplingConfiguration(BaseModel):
    """Base MCMC sampling parameters."""

    tune: PositiveInt
    draws: PositiveInt
    chains: PositiveInt = 4
    cores: PositiveInt = 2


class BasePymc3Configuration(BaseMcmcSamplingConfiguration):
    """Base configuration for PyMC3 models."""

    init: str = "auto"
    n_init: int = 200000


class BaseStanConfiguration(BaseMcmcSamplingConfiguration):
    """Base configuration for Stan models."""

    remove_cache: bool = False
