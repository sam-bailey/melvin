from typing import Any, Callable, Protocol

from jax import random
from jax.interpreters.xla import _DeviceArray as DeviceArray
from jax.scipy import stats

from .object_tagger import ObjectTagger


class DistributionType(Protocol):
    def rvs(self, prng_key: DeviceArray, n_samples: int) -> DeviceArray:
        "Must be able to sample from a distribution"

    def logpdf(self, x: DeviceArray) -> DeviceArray:
        "Must be able to calculate the logpdf of a distribution"


distributions = ObjectTagger()


@distributions("normal")
class MultivariateNormalDistribution:
    def __init__(self, mean: DeviceArray, cov: DeviceArray):
        self._mean = mean
        self._cov = cov

    def rvs(self, prng_key: DeviceArray, n_samples: int) -> DeviceArray:
        return random.multivariate_normal(
            key=prng_key,
            mean=self._mean,
            cov=self._cov,
            shape=(n_samples,),
        )

    def logpdf(self, x: DeviceArray) -> DeviceArray:
        return stats.multivariate_normal.logpdf(
            x=x,
            mean=self._mean,
            cov=self._cov,
        )


@distributions("cauchy")
class CauchyDistribution:
    def __init__(self, mean: DeviceArray, cov: DeviceArray):
        self._mean = mean
        self._cov = cov

    def rvs(self, prng_key: DeviceArray, n_samples: int) -> DeviceArray:
        pass

    def logpdf(self, x: DeviceArray) -> DeviceArray:
        pass
