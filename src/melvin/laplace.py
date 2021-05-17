from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from warnings import warn

import jax
import jax.numpy as jnp
import matplotlib.pylab as plt
from jax import jit, random
from jax.api import jacfwd
from jaxlib.xla_extension import DeviceArray
from scipy.optimize import minimize
from scipy.optimize.optimize import OptimizeResult
from scipy.stats import gaussian_kde

from .distributions import DistributionType, distributions
from .model_parameters import ModelParameters
from .samplers import samplers


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
    _default_minimize_kwargs = {"method": "BFGS"}
    param_bounds: Optional[List[Tuple[float]]] = None
    _max_samples_for_auto = 10_000

    def __init__(
        self,
        name: str,
        initial_params: DeviceArray,
        X: DeviceArray = jnp.array([]),
        y: DeviceArray = jnp.array([]),
        fixed_params: DeviceArray = jnp.array([]),
        minimize_kwargs: Dict[str, Any] = {},
        base_distribution: str = "normal",
    ):
        self.name = name

        self._X = X
        self._y = y

        self.fixed_params = fixed_params
        self.params = ModelParameters(x=initial_params, bounds=self.param_bounds)

        self._minimize_kwargs = self._default_minimize_kwargs
        self._minimize_kwargs.update(minimize_kwargs)

        self._base_distribution: DistributionType = distributions.get(base_distribution)
        self._base_distribution_name = base_distribution

        self._fit()

    def __str__(self) -> str:
        output_lst = [
            f"Laplace Approximation: {self.name}",
            f"Base distribution: {self._base_distribution_name}",
            f"Fixed Parameters: {self.fixed_params}",
        ]

        if self._successful_fit:
            output_lst.append("Fit converged successfully")
            output_lst.append("Fitted Parameters: ")
        else:
            output_lst.append("*** WARN: Fit did not converge successfully ***")
            output_lst.append("Current Parameters: ")

        output_lst.append(str(self.params))

        output_lst.append(f"MAP Posterior Prob = {self.max_posterior_log_prob}")

        return "\n".join(output_lst)

    def log_prior(self, params: DeviceArray) -> DeviceArray:
        # You can use self.fixed_params too
        raise NotImplementedError("Must implemented the log_prior() method")

    def log_likelihood(
        self, params: DeviceArray, y: DeviceArray, y_pred: DeviceArray
    ) -> DeviceArray:
        # You can use self.fixed_params too
        raise NotImplementedError("Must implement the log_likelihood() method")

    def _log_posterior(
        self, params: DeviceArray, y: DeviceArray, y_pred: DeviceArray
    ) -> DeviceArray:
        # You can use self.fixed_params too
        return self.log_prior(params) + self.log_likelihood(
            params=params, y=y, y_pred=y_pred
        )

    def model(self, params: DeviceArray, X: DeviceArray) -> DeviceArray:
        # You can use self.fixed_params too
        return jnp.array([jnp.nan])

    def _objective(self, params_raw: DeviceArray) -> DeviceArray:
        params = self.params.transform(params_raw)
        return self._objective_from_params(params)

    def _objective_from_params(self, params_x: DeviceArray) -> DeviceArray:
        y_pred = self.model(params=params_x, X=self._X)
        log_prob = self._log_posterior(params=params_x, y=self._y, y_pred=y_pred)
        return -1 * log_prob

    @property
    def _objective_jac(self) -> Callable[[DeviceArray], DeviceArray]:
        # jacobian of the negative log posterior
        return jit(jacfwd(self._objective))

    @property
    def _objective_hess(self) -> Callable[[DeviceArray], DeviceArray]:
        # hessian of the negative log posterior
        return jit(jacfwd(jacfwd(self._objective)))

    def _store_result(self, minimize_result: OptimizeResult) -> None:

        raw_params = jnp.array(minimize_result.x)
        self.params.update_raw(
            raw=raw_params, cov_raw=jnp.linalg.inv(self._objective_hess(raw_params))
        )
        self.laplace_log_posterior = self.params.transform_logpdf(
            self._base_distribution(
                mean=self.params.raw, cov=self.params.cov_raw
            ).logpdf
        )

        self._minimize_result = minimize_result
        self._successful_fit = minimize_result.success

        if not self._successful_fit:
            print(
                "Warning: Minimize did not converge. Make sure you are using 64bit jax and "
                "check self._minimize_result for more info"
            )

    def _fit(self):
        result = minimize(
            fun=self._objective,
            jac=self._objective_jac,
            hess=self._objective_hess,
            x0=self.params.raw,
            **self._minimize_kwargs,
        )

        self._store_result(result)

    def sample_params(
        self,
        prng_key: DeviceArray,
        n_samples: int,
        method: str = "auto",
        verbose: bool = False,
    ) -> DeviceArray:

        objective_vmap = jax.vmap(self._objective_from_params)

        def _true_log_prob_fn(params: DeviceArray) -> DeviceArray:
            return -1 * objective_vmap(params)

        if method == "auto":
            best_perf = -1 * jnp.inf
            for sampler_name, sampler_cls in samplers:
                sampler = sampler_cls(
                    params=self.params,
                    base_distribution=self._base_distribution,
                    true_log_prob_fn=_true_log_prob_fn,
                )
                samples = sampler(prng_key=prng_key, n_samples=n_samples)
                n_samples_for_eval = min([n_samples, self._max_samples_for_auto])
                perf = self.evaluate_samples(samples[:n_samples_for_eval, :])

                if verbose:
                    print(f"Method = {sampler_name},\t Perf = {perf}")

                if perf > best_perf:
                    best_perf = perf
                    best_method = sampler_name
                    final_samples = samples

            if verbose:
                print(f"**Best Method = {best_method}**")

        else:
            if verbose:
                print(f"Using method = {method}")
            sampler_cls = samplers.get(method)
            sampler = sampler_cls(
                params=self.params,
                base_distribution=self._base_distribution,
                true_log_prob_fn=_true_log_prob_fn,
            )
            final_samples = sampler(prng_key=prng_key, n_samples=n_samples)

        return final_samples

    def sample_params_map(
        self,
        prng_key: DeviceArray,
        n_samples: int,
        func: Callable[[DeviceArray, Any], DeviceArray],
        args: Tuple,
        method: str = "auto",
        verbose: bool = True,
    ) -> DeviceArray:
        params_rvs = self.sample_params(
            prng_key=prng_key, n_samples=n_samples, method=method, verbose=verbose
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
        method: str = "auto",
        verbose: bool = True,
    ) -> DeviceArray:
        params_rvs = self.sample_params(
            prng_key=prng_key, n_samples=n_samples, method=method, verbose=verbose
        )

        model_func_map = jax.vmap(self.model, (0, None))  # Map over the params, not X

        return model_func_map(params_rvs, X)

    @property
    def max_posterior_log_prob(self) -> float:
        return -1 * self._objective(self.params.raw).item()

    def evaluate_samples(self, samples: DeviceArray) -> Tuple[float, float]:
        try:
            objective_func_map = jax.vmap(self._objective_from_params)
            log_posterior_samples = -1 * objective_func_map(samples)
            average_log_posterior = jnp.mean(log_posterior_samples)
            entropy_estimate = sample_entropy_estimate(samples)

            return average_log_posterior + entropy_estimate
        except ValueError:
            return -1 * jnp.inf
