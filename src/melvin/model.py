from __future__ import annotations

from typing import Any, Callable, Dict

import jax
import matplotlib.pylab as plt
from jax.experimental.optimizers import Optimizer, adam
from jax.interpreters.xla import _DeviceArray as DeviceArray
from jaxlib.xla_client import Device
from matplotlib.axes._subplots import Axes
from melvin.minimization_problem import MinimizationProblem
from tqdm import trange


class BasicModel(MinimizationProblem):
    def __init__(
        self,
        name: str,
        model_fn: Callable[[DeviceArray, DeviceArray, DeviceArray], DeviceArray],
        loss_fn: Callable[
            [DeviceArray, DeviceArray, DeviceArray, DeviceArray], DeviceArray
        ],
        initial_params: DeviceArray,
        fixed_params: DeviceArray,
        optimizer: Optimizer = adam,
        optimizer_kwargs: Dict[str, Any] = {"step_size": 0.1},
    ):
        """
        loss_fn must be a callable which takes in three jax arrays (y_true, y_pred, params) and returns the model loss.
        model_fn is a function that defines how the model combines the data and parameters into a prediction.
        The arguments to the model_fn are two jax arrays. The first jax array is the parameters of the model,
        and the second is the data. If using batched data it is one batch of data.
        """

        def objective_fn(
            params: DeviceArray,
            fixed_params: DeviceArray,
            X: DeviceArray,
            y: DeviceArray,
        ) -> DeviceArray:
            y_pred = model_fn(params, X, fixed_params)
            return loss_fn(y, y_pred, params, fixed_params)

        super().__init__(
            name=name,
            objective_fn=objective_fn,
            initial_params=initial_params,
            fixed_params=fixed_params,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
        )

        self.model_fn = model_fn
        self.loss_fn = loss_fn

    def fit(self, n_steps: int, X: DeviceArray, y: DeviceArray) -> None:
        self.X_train = X
        self.y_train = y
        super().fit(n_steps=n_steps, X=X, y=y)

    def predict(self, X: DeviceArray) -> DeviceArray:
        return self.model_fn(self.params, X, self.fixed_params)

    def evaluate(self, X: DeviceArray, y: DeviceArray) -> DeviceArray:
        return self.objective_fn(self.params, self.fixed_params, X=X, y=y)

    @property
    def train_loss(self):
        return self.history[-1]
