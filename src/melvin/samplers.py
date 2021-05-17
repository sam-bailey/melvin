from typing import Callable, Type

import jax
import jax.numpy as jnp
from jax import random
from jaxlib.xla_extension import DeviceArray

from .distributions import DistributionType
from .model_parameters import ModelParameters
from .object_tagger import ObjectTagger

samplers = ObjectTagger()


@samplers("simple")
class SimpleSampler:
    def __init__(
        self,
        params: Type[ModelParameters],
        base_distribution: Type[DistributionType],
        true_log_prob_fn: Callable[[DeviceArray], DeviceArray],
    ):
        """
        This sampler ignores the true_log_prob_fn and just samples from the base distribution
        """
        self._params = params
        self._base_distribution = base_distribution(
            mean=self._params.raw,
            cov=self._params.cov_raw,
        )
        self._true_log_prob_fn = true_log_prob_fn

    def __call__(self, prng_key: DeviceArray, n_samples: int) -> DeviceArray:
        raw_params_rvs = self._base_distribution.rvs(
            prng_key=prng_key, n_samples=n_samples
        )
        vec_transform = jax.vmap(self._params.transform)
        params_rvs = vec_transform(raw_params_rvs)
        return params_rvs


@samplers("importance")
class ImportanceSampler(SimpleSampler):
    """
    This sampler performs importance sampling, which samples from the base distribution and then re-samples
    based on weights calculated from the true_log_prob_fn
    """

    def _laplace_logpdf(self, params: DeviceArray) -> DeviceArray:
        grad_det = jnp.abs(jnp.linalg.det(self._params.inverse_transform_jac(params)))
        log_grad_det = jnp.log(grad_det)

        raw_params_logpdf = self._base_distribution.logpdf(
            x=self._params.inverse_transform(params),
        )
        return raw_params_logpdf + log_grad_det

    def _get_raw_importance_weights(self, samples: DeviceArray) -> DeviceArray:
        true_log_prob = self._true_log_prob_fn(samples).reshape(-1)
        laplace_logpdf_vec = jax.vmap(self._laplace_logpdf)
        base_log_prob = laplace_logpdf_vec(samples).reshape(-1)

        weights = jnp.exp(
            true_log_prob
            - base_log_prob
            - jnp.mean(true_log_prob)
            + jnp.mean(base_log_prob)
        )
        return weights

    def _truncate_importance_weights(self, weights: DeviceArray) -> DeviceArray:
        weights /= jnp.mean(weights)
        max_weight = 1.0 / jnp.sqrt(len(weights))
        truncated_weights = jnp.minimum(weights, max_weight)
        return truncated_weights

    def _resample(
        self, prng_key: DeviceArray, samples: DeviceArray, weights: DeviceArray
    ) -> DeviceArray:
        idx = jnp.arange(len(weights))
        rand_idx = random.choice(key=prng_key, a=idx, shape=idx.shape, p=weights)

        return samples[rand_idx, :]

    def __call__(self, prng_key: DeviceArray, n_samples: int) -> DeviceArray:
        prng_key_1, prng_key_2 = random.split(prng_key)

        simple_samples = super().__call__(prng_key=prng_key_1, n_samples=n_samples)

        weights = self._get_raw_importance_weights(simple_samples)
        truncated_weights = self._truncate_importance_weights(weights)

        importance_samples = self._resample(
            prng_key=prng_key_2, samples=simple_samples, weights=truncated_weights
        )

        return importance_samples
