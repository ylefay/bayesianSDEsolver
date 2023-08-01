from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike

from bayesian_sde_solver._utils import insert


def sde_solver(
        key,
        drift: Callable,
        sigma: Callable,
        x0: ArrayLike,
        vf_gen: Callable,
        delta: float,
        N: int,
        ode_int: Callable,
) -> Tuple[ArrayLike, ArrayLike, Tuple[ArrayLike]]:
    init = x0
    get_coeffs, vf = vf_gen()

    def body(x, inp):
        key_k, t_k = inp
        bm_key, sample_key = jax.random.split(key_k, 2)
        coeffs_k = get_coeffs(bm_key, delta)
        vector_field = vf(drift, sigma, delta, t_k, *coeffs_k)
        next_x = ode_int(sample_key, init=x, vector_field=vector_field, T=delta)
        return next_x, (next_x, *coeffs_k)

    keys = jax.random.split(key, N)
    ts = jnp.linspace(0, N * delta, N + 1)

    inps = keys, ts[:-1]
    _, samples = jax.lax.scan(body, init, inps)
    traj, *coeffs = samples
    traj = insert(traj, 0, init, axis=0)
    return ts, traj, *coeffs
