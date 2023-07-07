from typing import Callable, Tuple

from functools import partial

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike

from bayesian_sde_solver._utils import insert

def sde_solver(
        key,
        drift: Callable,
        sigma: Callable,
        x0: ArrayLike,
        bm: Callable,
        delta: float,
        N: int,
        ode_int: Callable,
) -> Tuple[ArrayLike, ArrayLike]:
    init = x0
    get_coeffs, eval_fn = bm()

    def body(x, inp):
        key_k, t_k = inp
        bm_key, sample_key = jax.random.split(key_k, 2)
        coeffs_k = get_coeffs(bm_key, delta)
        func = lambda t: eval_fn(t, delta, *coeffs_k)
        vector_field = lambda z, t: drift(z, t + t_k) + sigma(z, t + t_k) @ jax.jacfwd(func)(t)
        next_x = ode_int(sample_key, init=x, vector_field=vector_field, T=delta)
        return next_x, next_x

    keys = jax.random.split(key, N)
    ts = jnp.linspace(0, N * delta, N + 1)

    inps = keys, ts[:-1]
    _, samples = jax.lax.scan(body, init, inps)
    samples = insert(samples, 0, init, axis=0)
    return ts, samples
