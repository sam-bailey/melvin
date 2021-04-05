from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from warnings import warn

import jax
import jax.numpy as jnp
import matplotlib.pylab as plt
from jax import grad, hessian, jit, random
from jax._src.scipy.optimize.minimize import OptimizeResults as OptimizeResultsJax
from jax.interpreters.xla import Device
from jax.interpreters.xla import _DeviceArray as DeviceArray
from jax.scipy.optimize import minimize as minimize_jax
from scipy.optimize import minimize
from scipy.optimize.optimize import OptimizeResult
from scipy.stats import gaussian_kde

OptimizeResultType = Union[OptimizeResult, OptimizeResultsJax]


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


def sample_entropy_estimate(x: DeviceArray) -> DeviceArray:
    """
    This method uses the splitting data estimate described here:
    http://jimbeck.caltech.edu/summerlectures/references/Entropy%20estimation.pdf
    """
    n_samples = x.shape[0]
    n_params = x.shape[1]

    if n_params == 1:
        kde_samples = x[: n_samples // 2, :].reshape(-1)
        est_samples = x[n_samples // 2 :, :].reshape(-1)
    else:
        kde_samples = x[: n_samples // 2, :].T
        est_samples = x[n_samples // 2 :, :].T

    kde = gaussian_kde(dataset=kde_samples)
    return jnp.mean(-1 * jnp.log(kde(est_samples)))


class LaplaceApproximation:
    jax_minimize_methods = ["BFGS"]
    sample_methods = ["laplace", "gaussian_importance"]  # Add t-distribution importance
    param_bounds: Optional[List[Tuple[float]]] = None

    def __init__(
        self,
        name: str,
        initial_params: DeviceArray,
        X: DeviceArray = jnp.array([]),
        y: DeviceArray = jnp.array([]),
        fixed_params: DeviceArray = jnp.array([]),
        minimize_method: str = "BFGS",
        minimize_kwargs: Dict[str, Any] = {},
        use_jax_minimize: bool = True,
    ):
        self.name = name

        self._X = X
        self._y = y

        self.fixed_params = fixed_params
        self.params = initial_params

        self._prep_minimizer(use_jax_minimize, minimize_method, minimize_kwargs)
        self._fit()

    def _prep_minimizer(
        self,
        use_jax_minimize: bool,
        minimize_method: str,
        minimize_kwargs: Dict[str, Any],
    ) -> None:
        self._use_jax_minimize = (
            minimize_method in self.jax_minimize_methods
        ) & use_jax_minimize
        self._minimize_method = minimize_method
        self._minimize_kwargs = minimize_kwargs

    def __str__(self) -> str:
        output_lst = [
            f"Laplace Approximation: {self.name}",
            f"Fixed Parameters: {self.fixed_params}",
        ]

        if self._successful_fit:
            output_lst.append("Fit converged successfully")
            output_lst.append("Fitted Parameters: ")
        else:
            output_lst.append("*** WARN: Fit did not converge successfully ***")
            output_lst.append("Current Parameters: ")

        for param, sig, (low, upp) in zip(
            self.params, self.params_sig, self.param_bounds
        ):
            params_line = f"\t {param} +/- {sig}"
            if low is not None:
                params_line += f"\t [Lower Bound = {low}]"
            if upp is not None:
                params_line += f"\t [Upper Bound = {upp}]"
            output_lst.append(params_line)

        output_lst.append(f"Log Posterior Prob = {self.max_posterior_log_prob}")

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
        return self._transform_params(self.raw_params)

    @params.setter
    def params(self, x: DeviceArray) -> None:
        self.raw_params = self._inverse_transform_params(x)

    @property
    def raw_params(self) -> DeviceArray:
        return self._raw_params

    @raw_params.setter
    def raw_params(self, x: DeviceArray) -> DeviceArray:
        self._raw_params = x
        self._raw_params_cov = jnp.linalg.inv(self._objective_hess(x))
        self._raw_params_sig = jnp.sqrt(jnp.diag(self._raw_params_cov))

    @property
    def raw_params_cov(self) -> DeviceArray:
        return self._raw_params_cov

    @property
    def raw_params_sig(self) -> DeviceArray:
        return self._raw_params_sig

    @property
    def params_cov(self) -> DeviceArray:
        if self.raw_params_cov is None:
            return None
        else:
            transform_jac = jax.jacfwd(self._transform_params)(self.raw_params)
            return self.raw_params_cov * transform_jac ** 2

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

    def _objective(self, raw_params: DeviceArray) -> DeviceArray:
        params = self._transform_params(raw_params)
        return self._objective_from_params(params)

    def _objective_from_params(self, params: DeviceArray) -> DeviceArray:
        y_pred = self.model(params=params, X=self._X)
        log_prob = self.posterior_log_prob(params=params, y=self._y, y_pred=y_pred)
        return -1 * log_prob

    @property
    def _objective_grad(self) -> Callable[[DeviceArray], DeviceArray]:
        # gradient of the negative log posterior
        return jax.jit(grad(self._objective))

    @property
    def _objective_hess(self) -> Callable[[DeviceArray], DeviceArray]:
        # gradient of the negative log posterior
        return jax.jit(hessian(self._objective))

    def _store_result(
        self,
        raw_params: DeviceArray,
        minimize_result: OptimizeResultType,
        successful_fit: bool,
    ) -> None:
        self.raw_params = raw_params
        self._minimize_result = minimize_result
        self._successful_fit = successful_fit

        if not self._successful_fit:
            print(
                "Warning: Minimize did not converge. Make sure you are using 64bit jax and "
                "check self._minimize_result for more info"
            )

    def _fit_with_jax_minimize(self):
        result = minimize_jax(
            fun=self._objective,
            x0=self.raw_params,
            method=self._minimize_method,
            **self._minimize_kwargs,
        )

        self._store_result(
            raw_params=result.x,
            minimize_result=result,
            successful_fit=result.success.item(),
        )

    def _fit_with_scipy_minimize(self):
        result = minimize(
            fun=self._objective,
            jac=self._objective_grad,
            hess=self._objective_hess,
            x0=self.raw_params,
            method=self._minimize_method,
            **self._minimize_kwargs,
        )

        self._store_result(
            raw_params=jnp.array(result.x),
            minimize_result=result,
            successful_fit=result.success,
        )

    def _fit(self) -> None:
        # Perform the fitting, using jax or regular scipy as specified
        if self._use_jax_minimize:
            self._fit_with_jax_minimize()
        else:
            self._fit_with_scipy_minimize()

    def _laplace_logpdf(self, params: DeviceArray) -> DeviceArray:
        inverse_transform_params_grad_fn = jax.jacfwd(self._inverse_transform_params)
        grad_det = jnp.abs(jnp.linalg.det(inverse_transform_params_grad_fn(params)))
        log_grad_det = jnp.log(grad_det)

        raw_params_logpdf = jax.scipy.stats.multivariate_normal.logpdf(
            x=self._inverse_transform_params(params),
            mean=self.raw_params,
            cov=self.raw_params_cov,
        )
        return raw_params_logpdf + log_grad_det

    def sample_params(
        self,
        prng_key: DeviceArray,
        n_samples: int,
        method: str = "laplace",
    ) -> DeviceArray:
        assert (
            method in self.sample_methods
        ), f"Method must be one of {self.sample_methods}"

        prng_key_1, prng_key_2 = random.split(prng_key)

        raw_params_rvs = random.multivariate_normal(
            key=prng_key_1,
            mean=self.raw_params,
            cov=self.raw_params_cov,
            shape=(n_samples,),
        )
        vec_transform = jax.vmap(self._transform_params)
        params = vec_transform(raw_params_rvs)

        if method == "gaussian_importance":
            params = self._importance_sample(prng_key_2, params)

        return params

    def _importance_sample(
        self, prng_key: DeviceArray, params: DeviceArray
    ) -> DeviceArray:
        model_vmap = jax.vmap(self.model, (0, None))
        posterior_log_prob_vmap = jax.vmap(self.posterior_log_prob, (0, None, None))
        laplace_logpdf_vec = jax.vmap(self._laplace_logpdf)

        y_pred = model_vmap(params, self._X)
        true_log_prob = posterior_log_prob_vmap(params, self._y, y_pred)

        weights = jnp.exp(
            true_log_prob.reshape(-1) - laplace_logpdf_vec(params).reshape(-1)
        )
        weights /= jnp.sum(weights)
        max_weight = 1.0 / jnp.sqrt(len(weights))

        truncated_weights = jnp.minimum(weights, max_weight)

        idx = jnp.arange(len(truncated_weights))
        rand_idx = random.choice(
            key=prng_key, a=idx, shape=idx.shape, p=truncated_weights
        )

        return params[rand_idx, :]

    def sample_params_map(
        self,
        prng_key: DeviceArray,
        n_samples: int,
        func: Callable[[DeviceArray, Any], DeviceArray],
        args: Tuple,
        method: str = "laplace",
    ) -> DeviceArray:
        params_rvs = self.sample_params(
            prng_key=prng_key, n_samples=n_samples, method=method
        )

        n_args = len(args)
        func_map = jax.vmap(
            func, (0,) + (None,) * n_args
        )  # Map over the first dimension only

        return func_map(params_rvs, *args)

    def predict(self, X: DeviceArray) -> DeviceArray:
        return self.model(params=self.params, X=X)

    def sample_predict(
        self,
        X: DeviceArray,
        prng_key: DeviceArray,
        n_samples: int,
        method: str = "laplace",
    ) -> DeviceArray:
        params_rvs = self.sample_params(
            prng_key=prng_key, n_samples=n_samples, method=method
        )

        model_func_map = jax.vmap(self.model, (0, None))  # Map over the params, not X

        return model_func_map(params_rvs, X)

    @property
    def max_posterior_log_prob(self) -> float:
        return -1 * self._objective(self.raw_params).item()

    def evaluate_samples(self, samples: DeviceArray) -> Tuple[float, float]:
        objective_func_map = jax.vmap(self._objective_from_params)
        log_posterior_samples = -1 * objective_func_map(samples)
        average_log_posterior = jnp.mean(log_posterior_samples)
        entropy_estimate = sample_entropy_estimate(samples)

        return average_log_posterior + entropy_estimate
