from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from warnings import warn

import jax
import jax.numpy as jnp
import matplotlib.pylab as plt
from jax import hessian, jacfwd, jit
from jax._src.scipy.optimize.minimize import OptimizeResults as OptimizeResultsJax
from jax.interpreters.xla import _DeviceArray as DeviceArray


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


def _transform(x: DeviceArray, bounds: Optional[DeviceArray]) -> DeviceArray:
    if bounds is None:
        return x

    low = bounds[:, 0]
    upp = bounds[:, 1]

    return jnp.where(
        jnp.isfinite(low) & jnp.isfinite(upp),
        _between(x, low, upp),
        jnp.where(
            jnp.isfinite(low),
            _greater_than(x, low),
            jnp.where(jnp.isfinite(upp), _less_than(x, upp), x),
        ),
    )


def _transform_inv(x: DeviceArray, bounds: Optional[DeviceArray]) -> DeviceArray:
    if bounds is None:
        return x

    low = bounds[:, 0]
    upp = bounds[:, 1]

    return jnp.where(
        jnp.isfinite(low) & jnp.isfinite(upp),
        _between_inv(x, low, upp),
        jnp.where(
            jnp.isfinite(low),
            _greater_than_inv(x, low),
            jnp.where(jnp.isfinite(upp), _less_than_inv(x, upp), x),
        ),
    )


class ModelParameters:
    def __init__(
        self,
        x: DeviceArray,
        cov_x: Optional[DeviceArray] = None,
        bounds: Optional[DeviceArray] = None,
    ):
        self.bounds = bounds
        self.update_x(x=x, cov_x=cov_x)

    def update_x(self, x: DeviceArray, cov_x: Optional[DeviceArray] = None):
        self._x = x
        self._raw = self.inverse_transform(x)
        self._dx_draw = self.transform_jac(self._raw)
        self._draw_dx = self.inverse_transform_jac(self._x)

        if cov_x is None:
            self._cov_x = None
            self._cov_raw = None
        else:
            self._cov_x = cov_x
            self._cov_raw = self.draw_dx @ cov_x @ self.draw_dx.T

    def update_raw(self, raw: DeviceArray, cov_raw: Optional[DeviceArray] = None):
        self._raw = raw
        self._x = self.transform(raw)
        self._dx_draw = self.transform_jac(self._raw)
        self._draw_dx = self.inverse_transform_jac(self._x)

        if cov_raw is None:
            self._cov_x = None
            self._cov_raw = None
        else:
            self._cov_x = self.dx_draw @ cov_raw @ self.dx_draw.T
            self._cov_raw = cov_raw

    @classmethod
    def from_raw(
        cls,
        raw: DeviceArray,
        cov_raw: Optional[DeviceArray] = None,
        bounds: Optional[DeviceArray] = None,
    ):
        params = cls(
            x=jnp.zeros_like(raw), cov_x=jnp.zeros_like(cov_raw), bounds=bounds
        )
        params.update_raw(raw=raw, cov_raw=cov_raw)
        return params

    def transform(self, raw: DeviceArray) -> DeviceArray:
        return _transform(raw, self.bounds)

    def inverse_transform(self, x: DeviceArray) -> DeviceArray:
        return _transform_inv(x, self.bounds)

    @property
    def transform_jac(self) -> Callable[[DeviceArray], DeviceArray]:
        return jax.jit(jacfwd(self.transform))

    @property
    def inverse_transform_jac(self) -> Callable[[DeviceArray], DeviceArray]:
        return jax.jit(jacfwd(self.inverse_transform))

    @property
    def x(self) -> DeviceArray:
        return self._x

    @property
    def raw(self) -> DeviceArray:
        return self._raw

    @property
    def dx_draw(self) -> DeviceArray:
        return self._dx_draw

    @property
    def draw_dx(self) -> DeviceArray:
        return self._draw_dx

    @property
    def cov_x(self) -> Optional[DeviceArray]:
        return self._cov_x

    @property
    def cov_raw(self) -> Optional[DeviceArray]:
        return self._cov_raw

    @property
    def var_x(self) -> Optional[DeviceArray]:
        if self.cov_x is None:
            return None
        else:
            return jnp.diag(self.cov_x)

    @property
    def sig_x(self) -> Optional[DeviceArray]:
        if self.var_x is None:
            return None
        else:
            return jnp.sqrt(self.var_x)

    @property
    def var_raw(self) -> Optional[DeviceArray]:
        if self.cov_raw is None:
            return None
        else:
            return jnp.diag(self.cov_raw)

    @property
    def sig_raw(self) -> Optional[DeviceArray]:
        if self.var_raw is None:
            return None
        else:
            return jnp.sqrt(self.var_raw)

    def transform_logpdf(
        self, logpdf_fn: Callable[[DeviceArray], DeviceArray]
    ) -> DeviceArray:
        def _transformed_logpdf(x: DeviceArray) -> DeviceArray:
            grad_det = jnp.abs(jnp.linalg.det(self.inverse_transform_jac(x)))
            log_grad_det = jnp.log(grad_det)

            raw_logpdf = logpdf_fn(
                x=self.inverse_transform(x),
            )
            return raw_logpdf + log_grad_det

        return _transformed_logpdf

    def __str__(self) -> str:
        def _param_idx_to_str(idx: int) -> str:
            param = self.x[idx]

            if self.sig_x is None:
                sig = None
            else:
                sig = self.sig_x[idx]

            if self.bounds is None:
                low = None
                upp = None
            else:
                low = self.bounds[idx, 0]
                upp = self.bounds[idx, 1]

            params_str = f"  {param}"
            if sig is not None:
                params_str += f" +/- {sig}"
            params_str += ","
            if low is not None:
                if jnp.isfinite(low):
                    params_str += f"\t [Lower Bound = {low}]"
            if upp is not None:
                if jnp.isfinite(upp):
                    params_str += f"\t [Upper Bound = {upp}]"

            return params_str

        output_lst = ["["]
        for idx in range(len(self.x)):
            output_lst.append(_param_idx_to_str(idx))
        output_lst.append("]")

        return "\n".join(output_lst)
