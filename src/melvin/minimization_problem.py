from __future__ import annotations

import jax
from jax.interpreters.xla import _DeviceArray as DeviceArray
from jax.experimental.optimizers import adam, Optimizer
from typing import Callable, Dict, Any
from tqdm import trange
import matplotlib.pylab as plt
from matplotlib.axes._subplots import Axes


class MinimizationProblem:
    def __init__(
        self,
        name: str, 
        objective_fn: Callable[[DeviceArray, Any], DeviceArray], 
        initial_params: DeviceArray, 
        optimizer: Optimizer = adam,
        optimizer_kwargs: Dict[str, Any] = {"step_size": 0.1}
        ):

        self.update_optimizer(optimizer, **optimizer_kwargs)
        self.name = name
        self.objective_fn = jax.jit(objective_fn)
        self.value_and_grad_fn = jax.value_and_grad(self.objective_fn)
        self.hessian_fn = jax.jacfwd(jax.jacrev(self.objective_fn))
        self.params = initial_params
        self.reset_history()

    def update_optimizer(self, new_optimizer: Optimizer, **new_optimizer_kwargs: Any) -> None:
        self.optimizer = new_optimizer(**new_optimizer_kwargs)

    def reset_history(self):
        self.history = []
    
    def fit(self, n_steps: int, **objective_fn_kwargs: Any) -> None:
        opt_init, opt_update, get_params = self.optimizer
        opt_state = opt_init(self.params)

        for step in trange(n_steps, desc=self.name):
            x_i = get_params(opt_state)
            value, grads = self.value_and_grad_fn(x_i, **objective_fn_kwargs)
            self.history.append(value)
            opt_state = opt_update(step, grads, opt_state)
        
        self.params = get_params(opt_state)

    def plot_history(self, ax: Axes = None) -> Axes:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        ax.plot(self.history, label=self.name)

        ax.set_title("History")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.legend()

        return ax

    def objective_value(self, **objective_fn_kwargs: Any):
        return self.objective_fn(self.params, **objective_fn_kwargs)