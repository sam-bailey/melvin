from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from warnings import warn

import jax
import jax.numpy as jnp
import matplotlib.pylab as plt
from jax import grad, hessian, jit, random
from jax.interpreters.xla import _DeviceArray as DeviceArray
from jax.scipy.optimize import minimize as minimize_jax
from scipy.optimize import minimize
from tqdm import trange


def _greater_than(x: DeviceArray, low: DeviceArray) -> DeviceArray:
    return jnp.exp(x) + low


def _greater_than_inv(x: DeviceArray, low: DeviceArray) -> DeviceArray:
    return jnp.log(x - low)


def _less_than(x: DeviceArray, upp: DeviceArray) -> DeviceArray:
    return upp - jnp.exp(x)


def _less_than_inv(x: DeviceArray, upp: DeviceArray) -> DeviceArray:
    return jnp.log(upp - x)


def _between(x: DeviceArray, low: DeviceArray, upp: DeviceArray) -> DeviceArray:
    return 1.0 / (1.0 + jnp.exp(-x)) * (upp - low) + low


def _between_inv(x: DeviceArray, low: DeviceArray, upp: DeviceArray) -> DeviceArray:
    return (jnp.log(x / (1.0 - x)) - low) / (upp - low)


def _transform(
    x: DeviceArray, low: Optional[DeviceArray], upp: Optional[DeviceArray]
) -> DeviceArray:
    if (low is not None) & (upp is not None):
        return _between(x, low, upp)
    elif low is not None:
        return _greater_than(x, low)
    elif upp is not None:
        return _less_than(x, upp)
    else:
        return x


def _transform_inv(
    x: DeviceArray, low: Optional[DeviceArray], upp: Optional[DeviceArray]
) -> DeviceArray:
    if (low is not None) & (upp is not None):
        return _between_inv(x, low, upp)
    elif low is not None:
        return _greater_than_inv(x, low)
    elif upp is not None:
        return _less_than_inv(x, upp)
    else:
        return x


class LaplaceApproximation:
    jax_minimize_methods = ["BFGS"]
    param_bounds: Optional[List[Tuple[float]]] = None

    def __init__(
        self,
        name: str,
        initial_params: DeviceArray,
        fixed_params: DeviceArray,
        minimize_method: str = "BFGS",
        minimize_kwargs: Dict[str, Any] = {},
        use_jax_minimize: bool = True,
    ):
        self.name = name
        self.params = initial_params
        self._raw_params_cov = None
        self._raw_params_sig = None
        self.fixed_params = fixed_params

        self.use_jax_minimize = (
            minimize_method in self.jax_minimize_methods
        ) & use_jax_minimize
        self.minimize_method = minimize_method
        self.minimize_kwargs = minimize_kwargs

        self.attempted_fit = False
        self.successful_fit = False

    def __str__(self) -> str:
        output_lst = [
            f"Laplace Approximation: {self.name}",
            f"Fixed Parameters: {self.fixed_params}",
        ]
        if not self.attempted_fit:
            output_lst.append(
                f"Fit has not been run. Initial Parameters: {self.params}"
            )
        elif self.successful_fit:
            output_lst.append("Fit converged successfully")
            output_lst.append("Fitted Parameters: ")
        else:
            output_lst.append("*** WARN: Fit did not converge successfully ***")
            output_lst.append("Current Parameters: ")

        if self.attempted_fit:
            for param, sig, (low, upp) in zip(
                self.params, self.params_sig, self.param_bounds
            ):
                params_line = f"\t {param} +/- {sig}"
                if low is not None:
                    params_line += f"\t [Lower Bound = {low}]"
                if upp is not None:
                    params_line += f"\t [Upper Bound = {upp}]"
                output_lst.append(params_line)

        return "\n".join(output_lst)

    def _transform_params(self, x: DeviceArray) -> DeviceArray:
        if self.param_bounds is None:
            return x
        else:
            return jnp.array(
                [_transform(x, low, upp) for x, (low, upp) in zip(x, self.param_bounds)]
            )

    def _inverse_transform_params(self, x: DeviceArray) -> DeviceArray:
        if self.param_bounds is None:
            return x
        else:
            return jnp.array(
                [
                    _transform_inv(x, low, upp)
                    for x, (low, upp) in zip(x, self.param_bounds)
                ]
            )

    @property
    def params(self) -> DeviceArray:
        return self._transform_params(self._raw_params)

    @params.setter
    def params(self, x: DeviceArray) -> None:
        self._raw_params = self._inverse_transform_params(x)

    @property
    def params_cov(self) -> DeviceArray:
        if self._raw_params_cov is None:
            return None
        else:
            transform_jac = jax.jacfwd(self._transform_params)(self._raw_params)
            return self._raw_params_cov * transform_jac ** 2

    @property
    def params_sig(self) -> DeviceArray:
        if self.params_cov is None:
            return None
        else:
            return jnp.sqrt(jnp.diag(self.params_cov))

    def posterior_log_prob(
        self, params: DeviceArray, y: DeviceArray, y_pred: DeviceArray
    ) -> DeviceArray:
        # You can use self.fixed_params too
        raise NotImplementedError("Must implement the posterior_log_prob() method")

    def model(self, params: DeviceArray, X: DeviceArray) -> DeviceArray:
        # You can use self.fixed_params too
        raise NotImplementedError("Must implement the model() method")

    def fit(self, X: DeviceArray, y: DeviceArray) -> None:

        # Record the data that was used to fit, for use later
        self._X_fit = X
        self._y_fit = y

        # Make the functions for fitting (objective, jacobian and hessian)
        @jit
        def _objective(
            raw_params: DeviceArray, X: DeviceArray, y: DeviceArray
        ) -> DeviceArray:
            params = self._transform_params(raw_params)
            y_pred = self.model(params=params, X=X)
            log_prob = self.posterior_log_prob(params=params, y=y, y_pred=y_pred)
            return -1 * log_prob

        @jit
        def objective_grad(raw_params: DeviceArray, X: DeviceArray, y: DeviceArray):
            # gradient of the negative log posterior
            return grad(_objective)(raw_params, X, y)

        @jit
        def objective_hess(raw_params: DeviceArray, X: DeviceArray, y: DeviceArray):
            return hessian(_objective)(raw_params, X, y)

        # Perform the fitting, using jax or regular scipy as specified
        if self.use_jax_minimize:
            result = minimize_jax(
                fun=_objective,
                x0=self._raw_params,
                args=(X, y),
                method=self.minimize_method,
                **self.minimize_kwargs,
            )

            self._raw_params = result.x
            self._minimize_result = result
            self.attempted_fit = True
            self.successful_fit = result.success.item()

        else:
            result = minimize(
                fun=_objective,
                jac=objective_grad,
                hess=objective_hess,
                x0=self._raw_params,
                args=(X, y),
                method=self.minimize_method,
                **self.minimize_kwargs,
            )

            self._raw_params = jnp.array(result.x)
            self._minimize_result = result
            self.attempted_fit = True
            self.successful_fit = result.success

        if not self.successful_fit:
            print(
                "Warning: Minimize did not converge. Make sure you are using 64bit jax and "
                "check self._minimize_result for more info"
            )

        # Store the resulting covariance and uncertainty on the raw parameters
        self._raw_params_cov = jnp.linalg.inv(objective_hess(self._raw_params, X, y))
        self._raw_params_sig = jnp.sqrt(jnp.diag(self._raw_params_cov))

    def params_rvs(self, prng_key: DeviceArray, n_samples: int) -> DeviceArray:
        raw_params_rvs = random.multivariate_normal(
            key=prng_key,
            mean=self._raw_params,
            cov=self._raw_params_cov,
            shape=(n_samples,),
        )
        vec_transform = jax.vmap(self._transform_params)
        return vec_transform(raw_params_rvs)

    @property
    def max_posterior_log_prob(self) -> float:
        y_pred = self.model(params=self.params, X=self._X_fit)
        log_prob = self.posterior_log_prob(
            params=self.params, y=self._y_fit, y_pred=y_pred
        )
        return log_prob.item()
