from __future__ import annotations

from typing import Any, Callable, Dict, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pylab as plt
from jax import random
from jax.experimental.optimizers import Optimizer, adam
from jax.interpreters.xla import _DeviceArray as DeviceArray
from tqdm import trange

from .model import BasicModel


class MaximumLikelihoodModel(BasicModel):
    def __init__(
        self,
        name: str,
        model_fn: Callable[[DeviceArray, DeviceArray, DeviceArray], DeviceArray],
        log_likelihood_fn: Callable[
            [DeviceArray, DeviceArray, DeviceArray, DeviceArray], DeviceArray
        ],
        initial_params: DeviceArray,
        fixed_params: DeviceArray,
        optimizer: Optimizer = adam,
        optimizer_kwargs: Dict[str, Any] = {"step_size": 0.1},
    ):
        self.log_likelihood_fn = log_likelihood_fn

        def loss_fn(
            y: DeviceArray,
            y_pred: DeviceArray,
            params: DeviceArray,
            fixed_params: DeviceArray,
        ) -> DeviceArray:
            return -1 * log_likelihood_fn(y, y_pred, params, fixed_params)

        super().__init__(
            name=name,
            model_fn=model_fn,
            loss_fn=loss_fn,
            initial_params=initial_params,
            fixed_params=fixed_params,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
        )

    @property
    def cov(self) -> DeviceArray:
        return jnp.linalg.inv(self.hessian_fn(self.params, self.X_train, self.y_train))

    @property
    def var(self) -> DeviceArray:
        return jnp.diag(self.cov)

    @property
    def sig(self) -> DeviceArray:
        return jnp.sqrt(self.var)

    def params_rvs(self, prng_key: DeviceArray, n_samples: int) -> DeviceArray:
        return random.multivariate_normal(
            key=prng_key, mean=self.params, cov=self.cov, shape=(n_samples, 1)
        )

    def predict_rvs(
        self, prng_key: DeviceArray, n_samples: int, X: DeviceArray
    ) -> DeviceArray:
        prng_key_1, prng_key_2 = random.split(prng_key, 2)
        params_rvs = self.params_rvs(prng_key_1, n_samples)
        return params_rvs, prng_key_2
        # Need to broadcast predict so I get a shape of (n_samples, len_x)

    @property
    def max_log_likelihood(self):
        -1 * self.train_loss

    def likelihood_ratio_test(self, fixed_params):
        pass
