from __future__ import annotations

import copy
from typing import Callable, Dict

import jax
import jax.numpy as jnp
import matplotlib.pylab as plt
from jax.experimental.optimizers import Optimizer, adam
from jax.interpreters.xla import _DeviceArray as DeviceArray
from matplotlib.axes._subplots import Axes
from tqdm import trange

LikelihoodFnType = Callable[[DeviceArray, DeviceArray], DeviceArray]


class MaximumLikelihoodModels(object):
    def __init__(
        self,
        name: str,
        log_likelihood_fn: LikelihoodFnType,
        x0: DeviceArray,
        optimizer: Optimizer = adam(step_size=0.1),
    ):
        self.optimizer = optimizer
        self.log_likelihood_fn = {name: jax.jit(log_likelihood_fn)}

        @jax.jit
        def _loss_fn(x: DeviceArray, data: DeviceArray) -> DeviceArray:
            return -1 * log_likelihood_fn(x=x, data=data)

        self.loss_fn = {name: log_likelihood_fn}
        self.value_and_grad_fn = {name: jax.value_and_grad(_loss_fn)}
        self.hessian_fn = {name: jax.jacfwd(jax.jacrev(_loss_fn))}

        self.x = {name: x0}
        self.loss_history = {name: []}
        self.model_names = [name]

    def _fit_one_model(
        self,
        model_name: str,
        n_steps: int,
        data: DeviceArray,
        reset_history: bool = False,
    ) -> None:
        if reset_history:
            self.loss_history[model_name] = []

        opt_init, opt_update, get_params = self.optimizer
        opt_state = opt_init(self.x[model_name])

        for step in trange(n_steps, desc=model_name):
            x_i = get_params(opt_state)
            value, grads = self.value_and_grad_fn[model_name](x_i, data)
            self.loss_history[model_name].append(value)
            opt_state = opt_update(step, grads, opt_state)

        self.x[model_name] = get_params(opt_state)

    def fit(self, n_steps: int, data: DeviceArray, reset_history: bool = False) -> None:
        self.train_data = data
        for name in self.model_names:
            self._fit_one_model(name, n_steps, data, reset_history)

    def plot_loss_history(self, ax: Axes = None) -> Axes:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        for model_name, loss_history in self.loss_history.items():
            ax.plot(loss_history, label=model_name)

        ax.set_title("Loss History")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.legend()

        return ax

    @property
    def cov(self) -> Dict[str, DeviceArray]:
        return {
            name: jnp.linalg.inv(self.hessian_fn[name](self.x[name], self.train_data))
            for name in self.model_names
        }

    @property
    def var(self) -> Dict[str, DeviceArray]:
        return {name: jnp.diag(cov) for name, cov in self.cov.items()}

    @property
    def sig(self) -> Dict[str, DeviceArray]:
        return {name: jnp.sqrt(var) for name, var in self.var.items()}

    def x_rvs(self, shape):
        pass

    @property
    def max_likelihood(self):
        pass

    def likelihood_ratio_test(self, fixed_params):
        pass

    def merge_model(self, other: MaximumLikelihoodModels) -> None:
        self.log_likelihood_fn.update(other.log_likelihood_fn)
        self.loss_fn.update(other.loss_fn)
        self.value_and_grad_fn.update(other.value_and_grad_fn)
        self.hessian_fn.update(other.hessian_fn)
        self.x.update(other.x)
        self.loss_history.update(other.loss_history)
        self.model_names += other.model_names

    def __or__(self, other: MaximumLikelihoodModels) -> MaximumLikelihoodModels:
        new_model = copy.deepcopy(self)
        new_model.merge_model(other)
        return new_model
